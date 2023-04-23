import argparse

import os, sys
import numpy as np
import time
import random

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.builder import (
    build_model, build_dataloader, 
    build_loss, build_scheduler
)
from src.models import LatentCodesDiscriminator

from configs import get_config
from utils import (
    create_logger, _denormalize, 
    disable_gradient, enable_gradient,
    dict2string, eta_format,
    get_first_order_diff_norm
)

from timm.utils import AverageMeter

# Some constants
_CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def parse_option():
    parser = argparse.ArgumentParser('Editing', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')

    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--output_dir', default='outputs', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    # distributed training
    parser.add_argument("--local_rank", type=int, required=False, help='local rank for DistributedDataParallel')

    parser.add_argument("--datablob", type=str, required=True, help='root for all data and pretrained models')

    args, _ = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    model, generator, latent_avg = build_model(config)
    train_loader, _ = build_dataloader(config)
    loss_fn = build_loss(config).cuda()

    model.cuda()
    generator.cuda()
    generator.eval()

    disable_gradient(generator)

    if not config.MODEL.USE_VISUAL:
        disable_gradient(model.encoder.visual)
        disable_gradient(model.mapper.v_mapper)
    
    # latent_avg.requires_grad = False

    # set optimizer (different parts with different learning rate)
    optimizer = torch.optim.Adam([
        {'params': filter(lambda p: p.requires_grad, model.encoder.parameters()), 
         'lr': config.TRAIN.OPTIMIZER.BASE_LR,
         'betas': (0.0, 0.999)},
        {'params': filter(lambda p: p.requires_grad, model.rnn.parameters()), 
         'lr': config.TRAIN.OPTIMIZER.RNN_LR,
         'betas': (0.0, 0.999)},
        {'params': filter(lambda p: p.requires_grad, model.mapper.parameters()), 
         'lr': config.TRAIN.OPTIMIZER.MAPPER_LR,
         'betas': (0.0, 0.999)},
    ])

    # if config.AMP_OPT_LEVEL != "O0":
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=config.AMP_OPT_LEVEL)

    # distributed training
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[dist.get_rank()], find_unused_parameters=True)
    
    # generator: stylegan2
    # generator = torch.nn.parallel.DistributedDataParallel(
    #     generator, device_ids=[dist.get_rank()], find_unused_parameters=True)
    
    face_pool = torch.nn.AdaptiveAvgPool2d(config.DATA.IMG_SIZE)

    lr_scheduler = build_scheduler(config, optimizer)

    if config.TRAIN.LOSS.lambda_wdis > 0:
        discriminator = LatentCodesDiscriminator(512, 4).cuda()
        discriminator = torch.nn.parallel.DistributedDataParallel(
                discriminator, device_ids=[dist.get_rank()])
        discriminator_optimizer = torch.optim.Adam(
                list(discriminator.parameters()), 
                lr=config.TRAIN.OPTIMIZER.W_DISCRIMINTOR_LR)
    
    # some meters
    data_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()

    _iter = 0
    _data_start = time.time()
    for databatch in train_loader:

        _iter += 1
        if _iter > config.TRAIN.TOTAL_ITERS:
            break
        
        data_time_meter.update(time.time() - _data_start)

        if config.MODEL.USE_VISUAL:
            img, txt = databatch['img'], databatch['tokenized_texts']
            img, txt = img.cuda(non_blocking=True), txt.cuda(non_blocking=True)
            latent_code = databatch['latent_code'].cuda(non_blocking=True) \
                if config.MODEL.USE_LATENT else 0
        
            # denormalize the input image first (CLIP mean, std)
            # then normalize the image to [-1, 1]
            _img = img.clone()
            _denormalize(_img, _CLIP_MEAN, _CLIP_STD)
            _img = 2.0 * _img - 1.0

            img_ws, txt_ws = model(img, txt)
        
        else:
            txt = databatch['tokenized_texts']
            txt = txt.cuda(non_blocking=True)
            latent_code = 0
            _img = None
            
            _, txt_ws = model(None, txt)
            sample_z = torch.randn(config.DATA.BATCH_SIZE, 512).cuda() # + latent_avg.cuda()
            img_ws = generator.get_latent(sample_z).detach().unsqueeze(1)

        txt_ws = txt_ws.view(config.DATA.BATCH_SIZE, config.MODEL.N_FRAMES, -1, 512)

        img_seq = []
        loss = 0.0
        loss_fn.add_loss_item("total_loss", 0.0)
        # loss += 1.0 * F.mse_loss(txt_ws[:, 0, :, :], txt_ws[:, -1, :, :], reduction='mean')
        if config.TRAIN.LOSS.lambda_w_2rd > 0:
            loss_2rd = config.TRAIN.LOSS.lambda_w_2rd * F.mse_loss(
                txt_ws[:, 2:, :, :] - txt_ws[:, 1:(config.MODEL.N_FRAMES-1), :, :], 
                txt_ws[:, 1:(config.MODEL.N_FRAMES-1), :, :] - txt_ws[:, 0:(config.MODEL.N_FRAMES-2), :, :], 
                reduction='mean')
            loss += loss_2rd

            loss_fn.add_loss_item("2rd_diff", loss_2rd.item())

        if config.TRAIN.LOSS.lambda_w_reg > 0:
            loss_reg = config.TRAIN.LOSS.lambda_w_reg * F.mse_loss(
                txt_ws[:,  0, :, :],
                txt_ws[:, -1, :, :],
                reduction='mean')
            loss += loss_reg

            loss_fn.add_loss_item("loss_reg", loss_reg.item())

        if config.TRAIN.LOSS.lambda_w_1st > 0:
            loss_1st = config.TRAIN.LOSS.lambda_w_1st * F.mse_loss(
                txt_ws[:, 0:(config.MODEL.N_FRAMES-1), :, :],
                txt_ws[:, 1:, :, :],
                reduction='mean')
            loss_fn.add_loss_item("1st_diff", loss_1st.item())

        for t in range(config.MODEL.N_FRAMES):
            edit_latent_code = img_ws + txt_ws[:, t, :, :] + latent_code
            img_edited, _ = generator(
                [edit_latent_code], 
                input_is_latent=True, 
                randomize_noise=False)
            img_seq.append(img_edited.detach().cpu())
            if t == 0:
                _loss, _ = loss_fn(face_pool(img_edited), _img, text=None)
                loss += _loss
            elif t == config.MODEL.N_FRAMES - 1:
                # _used_text = random.sample(_TEXT_DESCRIPTION, edit_latent_code.shape[0])
                _loss, _ = loss_fn(
                        face_pool(img_edited), _img, 
                        text=databatch['raw_texts'],
                        use_specific_text=(config.SPECIFIC_TEXT is not None))
                loss += _loss
                
            if config.MODEL.USE_LATENT:
                loss_latent_reg = config.TRAIN.LOSS.lambda_w_l1 * F.mse_loss(edit_latent_code, latent_code, reduction='mean')
                loss += loss_latent_reg
                loss_fn.add_loss_item("latent_reg", loss_latent_reg.item())

            elif not config.MODEL.USE_VISUAL:
                loss_latent_reg = config.TRAIN.LOSS.lambda_w_l1 * F.mse_loss(edit_latent_code, img_ws, reduction='mean')
                loss += loss_latent_reg
                loss_fn.add_loss_item("latent_reg", loss_latent_reg.item())
        
        if config.TRAIN.LOSS.lambda_wdis > 0:
            fake_w = edit_latent_code.view(-1, 512)
            fake_pred = discriminator(fake_w)
            gen_fake_loss = config.TRAIN.LOSS.lambda_wdis * F.softplus(-fake_pred).mean()
            loss_fn.add_loss_item("gen_fake_loss", gen_fake_loss.item())
            loss += gen_fake_loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print(get_first_order_diff_norm(txt_ws).data)

        # train discrimintor
        if config.TRAIN.LOSS.lambda_wdis > 0:
            enable_gradient(discriminator)
            sample_z = torch.randn(img_ws.size(0), 512).cuda() # + latent_avg.cuda()
            real_w = generator.get_latent(sample_z)
            real_pred = discriminator(real_w.detach())
            fake_pred = discriminator((edit_latent_code).detach())
            real_loss = F.softplus(-real_pred).mean()
            fake_loss = F.softplus(fake_pred).mean()

            dis_loss = real_loss + fake_loss
            discriminator_optimizer.zero_grad()
            dis_loss.backward()
            discriminator_optimizer.step()

            disable_gradient(discriminator)
        
        # update learning rate in optimizer
        if lr_scheduler:
            lr_scheduler.step_update(_iter)
        torch.cuda.synchronize()
        
        # log information and save temp image
        if _iter % config.PRINT_FREQ == 0 and global_rank == 0:
            save_image(
                tensor=torch.cat(img_seq, dim=3),
                fp=os.path.join(config.OUTPUT_DIR, "imgs", f"iter_{_iter:06d}_{global_rank}.png"),
                normalize=True,
                value_range=(-1, 1),
                nrow=1,
            )

            # Just for printing log, no other specific use
            lr = optimizer.param_groups[0]['lr'] * 1e6

            logger.info(
                f"Iter {_iter:06d} | {dict2string(loss_fn.loss_dict)}"
                f"\t| LR {lr:.4f}e-6"
                f"\t| ETA {eta_format(batch_time_meter.avg * (config.TRAIN.TOTAL_ITERS - _iter))}"
                f"\t| batch_time {batch_time_meter.val:.03f}({batch_time_meter.avg:.03f})"
                f"\t| data_time {data_time_meter.val:.03f}({data_time_meter.avg:.03f})"
                f"\t| saved_id {databatch['filename']}"
                f"\t| used_text {databatch['raw_texts']}")
        
        # save checkpoint
        if _iter % config.SAVE_FREQ == 0 and global_rank == 0:
            fp = os.path.join(config.OUTPUT_DIR, 'ckpt', f'Iter_{_iter:06d}.pth')
            torch.save({
                    'iter': _iter,
                    'arch': config.MODEL.BACKBONE,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }, f=fp)

            logger.info(f"Saving checkpoint at {fp}")

        batch_time_meter.update(time.time() - _data_start)
        
        _data_start = time.time()


# @torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    
    _, config = parse_option()

    # if config.AMP_OPT_LEVEL != "O0":
    #     assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    linear_scaled_lr = config.TRAIN.OPTIMIZER.BASE_LR

    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS

    config.defrost()
    config.TRAIN.OPTIMIZER.BASE_LR = linear_scaled_lr
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_DIR, "imgs"), exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_DIR, "ckpt"), exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT_DIR, dist_rank=dist.get_rank(), name=f"{config.EXP_NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT_DIR, "config.yaml")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())
    main(config)