import argparse

import os, sys
from statistics import mode
import numpy as np
import time
import random
from PIL import Image

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from torchvision.io import write_video

# for dimension reduction
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.builder import (
    build_model, build_dataloader, 
    build_loss, build_scheduler
)

from src.datasets import CelebAHQ, _transform
from src.datasets.celeba_hq import build_vocabulary

from utils import (
    create_logger, _denormalize, 
    disable_gradient, enable_gradient,
    dict2string, eta_format,
    truncate)
from configs import get_config

from timm.utils import AverageMeter
from tqdm import tqdm
import clip

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

    parser.add_argument("--datablob", type=str, required=True, help='local rank for DistributedDataParallel')
    args, _ = parser.parse_known_args()

    config = get_config(args)

    return args, config


def get_first_order_diff_norm(x):
    b, t = x.shape[0:2]
    diff = F.mse_loss(
        x, torch.roll(x, shifts=1, dims=1), 
        reduction='none')
    diff = diff.view(b, t, -1)
    return diff.mean(-1)


def pca_projection_visualize(x : torch.Tensor, saved_path : str):
    # change tensor to numpy array
    x = x.detach().cpu().numpy()

    pca = PCA(n_components=2)
    reduced_x = pca.fit_transform(x)

    for i in range(len(reduced_x)):
        if i < len(reduced_x) - 1:
            plt.plot(reduced_x[i:(i+2), 0], reduced_x[i:(i+2), 1])
        plt.scatter(reduced_x[i:(i+1), 0], reduced_x[i:(i+1), 1])

    plt.savefig(fname=saved_path)
    plt.close()


@torch.no_grad()
def main(config):

    model, generator, latent_avg = build_model(config, is_train=False)
    # _, val_loader = build_dataloader(config)

    val_ds = CelebAHQ(data_dir=config.DATA.DATA_PATH, is_valid=True, debug_num=250)

    model.cuda()
    # latent_avg = latent_avg.cuda()
    generator.cuda()
    generator.eval()

    for param in generator.parameters():
        param.requires_grad = False

    # distributed inference
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[dist.get_rank()], find_unused_parameters=True)

    # load parameters
    pretrained_ckpts = os.listdir(os.path.join(config.OUTPUT_DIR, 'ckpt'))
    latest_ckpt = None
    if len(pretrained_ckpts) > 0:
        pretrained_ckpts.sort()
        latest_ckpt = os.path.join(os.path.join(config.OUTPUT_DIR, 'ckpt', pretrained_ckpts[-1]))
    pretrained_path = config.PRETRAINED.OUR_MODEL or latest_ckpt
    assert pretrained_path is not None
    pretrained_params = torch.load(
        pretrained_path, map_location='cuda:{}'.format(dist.get_rank()))
    model.load_state_dict(pretrained_params['state_dict'])

    val_dir = os.path.join(
        config.OUTPUT_DIR, 
        f"val_{os.path.basename(pretrained_path).split('.')[0]}_d3")
    # viz_dir = os.path.join(
    #     config.OUTPUT_DIR, 
    #     f"viz_{os.path.basename(pretrained_path).split('.')[0]}_new")
    os.makedirs(val_dir, exist_ok=True)
    # os.makedirs(viz_dir, exist_ok=True)

    logger.info(f"Successfully load pre-trained parameters from {pretrained_path}")

    # generator: stylegan2
    # generator = torch.nn.parallel.DistributedDataParallel(
    #     generator, device_ids=[dist.get_rank()], find_unused_parameters=True)
    
    # some meters
    data_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()

    Texts = build_vocabulary(config.GEN_TYPE)
    
    # Texts = [_t for _t in Texts if 'happy' in _t or 'ear' in _t]

    _data_start = time.time()

    for idx in tqdm(range(dist.get_rank(), len(val_ds), dist.get_world_size())):
    # for databatch in tqdm(val_loader, total=len(val_loader)):
        databatch = val_ds[idx]
        data_time_meter.update(time.time() - _data_start)
        # img, txt = databatch['img'], databatch['tokenized_texts']
        # img, txt = img.cuda(non_blocking=True), txt.cuda(non_blocking=True)
        # latent_code = databatch['latent_code'].cuda(non_blocking=True) \
        #     if config.MODEL.USE_LATENT else 0
        
        # # denormalize the input image first (CLIP mean, std)
        # # then normalize the image to [-1, 1]
        # _img = img.clone()
        # _denormalize(_img, _CLIP_MEAN, _CLIP_STD)
        # _img = 2.0 * _img - 1.0
        if config.MODEL.USE_VISUAL:
            img = databatch['img'].unsqueeze(0).cuda(non_blocking=True)
            latent_code = databatch['latent_code'].unsqueeze(0).cuda(non_blocking=True) \
                if config.MODEL.USE_LATENT else 0
        
            # denormalize the input image first (CLIP mean, std)
            # then normalize the image to [-1, 1]
            _img = img.clone()
            _denormalize(_img, _CLIP_MEAN, _CLIP_STD)
            _img = 2.0 * _img - 1.0

        else:
            latent_code = 0
            _img = img = None
            sample_z = torch.randn(1, 512).cuda() # + latent_avg.cuda()
            img_ws = generator.get_latent(sample_z).detach().unsqueeze(1)
        
        for text_string in Texts:
            img_seq = []
            latent_seq = []

            txt = clip.tokenize(text_string)
            txt = txt.cuda(non_blocking=True)
            _img_ws, txt_ws = model(img, txt, is_inference=True, choice='linear')
            img_ws = _img_ws if _img_ws is not None else img_ws
            # img_ws, txt_ws = model(img, txt)
            txt_ws = txt_ws.view(img_ws.shape[0], config.MODEL.N_INFERENCE_FRAMES, -1, 512)

            for t in range(config.MODEL.N_INFERENCE_FRAMES):
                edit_latent_code = img_ws + txt_ws[:, t, :, :] + latent_code
                # edit_latent_code = truncate(edit_latent_code, latent_avg, True)
                img_edited, _ = generator(
                    [edit_latent_code], 
                    input_is_latent=True, 
                    randomize_noise=False)
                img_seq.append(img_edited.cpu())
                latent_seq.append(torch.mean(edit_latent_code, dim=1, keepdim=True))

            latent_seq = torch.cat(latent_seq, dim=1)

            # save image 
            img_tensor = torch.cat(img_seq, dim=0)
            img_tensor.clamp_(-1, 1)

            img_tensor = ((img_tensor + 1.) / 2. * 255).type(torch.uint8).permute(0, 2, 3, 1)
            write_video(
                filename=os.path.join(
                    val_dir, 
                    f"{databatch['filename']}_{text_string.replace(' ', '_')}.mp4"),
                video_array=img_tensor,
                fps=10
            )
            
            # save_image(
            #     img_tensor[i],
            #     fp=os.path.join(
            #         val_dir, 
            #         f"{databatch['filename'][i]}_{databatch['raw_texts'][i].replace(' ', '_')}_{dist.get_rank()}.png"),
            #     normalize=True,
            #     value_range=(-1, 1),
            #     nrow=1,
            # )

        batch_time_meter.update(time.time() - _data_start)
        
        torch.cuda.synchronize()

        _data_start = time.time()
    
    logger.info(f"Inference and io time for one batch: {batch_time_meter.avg:.4f}s")


@torch.no_grad()
def compare_with_mocogan(config):

    model, generator, latent_avg = build_model(config, is_train=False)

    model.cuda()
    # latent_avg = latent_avg.cuda()
    generator.cuda()
    generator.eval()

    for param in generator.parameters():
        param.requires_grad = False

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[dist.get_rank()], find_unused_parameters=True)

    # load latents from mocogan-hd
    latents_mocogan_hd = torch.load(
        f"{config.DATABLOB}/eccv2022-outputs/baselines/mocogan-hd/used_latents.pt")

    # load parameters
    pretrained_ckpts = os.listdir(os.path.join(config.OUTPUT_DIR, 'ckpt'))
    latest_ckpt = None
    if len(pretrained_ckpts) > 0:
        pretrained_ckpts.sort()
        latest_ckpt = os.path.join(
            os.path.join(config.OUTPUT_DIR, 'ckpt', pretrained_ckpts[-1]))
    pretrained_path = config.PRETRAINED.OUR_MODEL or latest_ckpt
    assert pretrained_path is not None
    pretrained_params = torch.load(
        pretrained_path, map_location='cuda:{}'.format(dist.get_rank()))
    try:
        model.load_state_dict(pretrained_params['state_dict'])
    except:
        param_copy = {}

        for key in pretrained_params['state_dict']:
            param_copy[key.replace('module.', '')] = pretrained_params['state_dict'][key]
        model.load_state_dict(param_copy)
    
    val_dir = os.path.join(
        config.OUTPUT_DIR, 
        f"val_{os.path.basename(pretrained_path).split('.')[0]}_mocogan-hd_new")
    viz_dir = os.path.join(
        config.OUTPUT_DIR, 
        f"viz_{os.path.basename(pretrained_path).split('.')[0]}_mocogan-hd")
    os.makedirs(val_dir, exist_ok=True)
    # os.makedirs(viz_dir, exist_ok=True)

    logger.info(f"Successfully load pre-trained parameters from {pretrained_path}")

    # some meters
    data_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()

    Texts = build_vocabulary("face_pmt")

    _data_start = time.time()
    keys = list(latents_mocogan_hd.keys())
    for idx in tqdm(range(dist.get_rank(), len(keys), dist.get_world_size())):
    # for key in sampled_keys:
        data_time_meter.update(time.time() - _data_start)

        latent_code = latents_mocogan_hd[keys[idx]].cuda()

        # for text_string in Texts:
        for text_string in [random.choice(Texts)]:
            txt = clip.tokenize(text_string)
            txt = txt.cuda(non_blocking=True)
        
            _img = None
            
            _, txt_ws = model(None, txt, is_inference=True, choice='linear')
            img_ws = generator.get_latent(latent_code).detach().unsqueeze(1)

            # img_ws, txt_ws = model(img, txt)
            txt_ws = txt_ws.view(img_ws.shape[0], config.MODEL.N_INFERENCE_FRAMES, -1, 512)

            img_seq = []
            latent_seq = []

            for t in range(config.MODEL.N_INFERENCE_FRAMES):
                # print(img_ws.shape)
                edit_latent_code = img_ws + txt_ws[:, t, :, :] # + latent_code
                img_edited, _ = generator(
                    [edit_latent_code], 
                    input_is_latent=True, 
                    randomize_noise=False)
                img_seq.append(img_edited.cpu())
                latent_seq.append(torch.mean(edit_latent_code, dim=1, keepdim=True))

            latent_seq = torch.cat(latent_seq, dim=1)

            img_seq = torch.cat(img_seq, dim=0)
            # [T, H, W, C]
            img_seq.clamp_(-1.0, 1.0)
            img_seq = ((img_seq + 1.) / 2. * 255).type(torch.uint8).permute(0, 2, 3, 1)
            write_video(
                filename=os.path.join(
                    val_dir, 
                    f"{keys[idx]}_{text_string.replace(' ', '_')}.mp4"),
                video_array=img_seq,
                fps=10
            )            

        batch_time_meter.update(time.time() - _data_start)
        _data_start = time.time()
    
    logger.info(f"Inference and io time for one batch: {batch_time_meter.avg:.4f}s")


if __name__ == '__main__':
    
    _, config = parse_option()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend='nccl', init_method='env://', world_size=world_size, rank=rank)
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
    logger = create_logger(
        output_dir=config.OUTPUT_DIR, dist_rank=dist.get_rank(), name=f"{config.EXP_NAME}_val")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT_DIR, "config.yaml")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())
    main(config)
