"""
Sample the latent code for the first frame and generated results.
"""


import argparse

import os, sys
import numpy as np
import time

import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image

import clip

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.builder import (
    build_model, build_dataloader, 
    build_loss, build_scheduler
)

from utils import (
    create_logger, _denormalize, 
    disable_gradient, enable_gradient,
    dict2string, eta_format)
from configs import get_config

from timm.utils import AverageMeter
from tqdm import tqdm

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
    parser.add_argument('--output_dir', default='eccv2022-outputs', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    parser.add_argument("--datablob", type=str, required=True, help='local rank for DistributedDataParallel')
    args, _ = parser.parse_known_args()

    config = get_config(args)

    return args, config

@torch.no_grad()
def main(config):

    model, generator, latent_avg = build_model(config, is_train=False)

    model.cuda()
    generator.cuda()
    generator.eval()

    for param in generator.parameters():
        param.requires_grad = False
    latent_avg.requires_grad = False

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

    logger.info(f"Successfully load pre-trained parameters from {pretrained_path}")

    # generator: stylegan2
    # generator = torch.nn.parallel.DistributedDataParallel(
    #     generator, device_ids=[dist.get_rank()], find_unused_parameters=True)
    
    # some meters
    data_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()

    batchsize = 1
    num_samples = 100

    raw_text = ["The person is smiling"] * batchsize
    tokenized_text = clip.tokenize(raw_text)
    
    _data_start = time.time()
    for i in tqdm(range(num_samples)):
        data_time_meter.update(time.time() - _data_start)

        # without real image as input
        _, txt_ws = model(None, tokenized_text)
        txt_ws = txt_ws.view(batchsize, config.MODEL.N_INFERENCE_FRAMES, -1, 512)
        
        # sample latent code for first frame
        sample_z = torch.randn(batchsize, 512).cuda() + latent_avg.cuda()
        img_ws = generator.get_latent(sample_z)
        
        img_seq = []

        for t in range(config.MODEL.N_INFERENCE_FRAMES):
            edit_latent_code = img_ws + txt_ws[:, t, :, :]
            img_edited, _ = generator(
                [edit_latent_code], 
                input_is_latent=True, 
                randomize_noise=False)
            img_seq.append(img_edited.cpu())

        # save image 
        img_tensor = torch.cat(img_seq, dim=3)

        save_image(
            img_tensor,
            fp=os.path.join(
                config.OUTPUT_DIR, 
                "val_random_sample", 
                f"{i:05d}_{dist.get_rank()}.png"),
            normalize=True,
            value_range=(-1, 1),
            nrow=1,
        )

        batch_time_meter.update(time.time() - _data_start)
        
        torch.cuda.synchronize()
        _data_start = time.time()
    
    logger.info(f"Inference and io time for one batch: {batch_time_meter.avg:.4f}s")


@torch.no_grad()
def one_image_with_different_text(config):

    os.makedirs(os.path.join(
        config.OUTPUT_DIR, 
        "val_one_image_with_different_text"),
        exist_ok=True
    )

    model, generator, latent_avg = build_model(config, is_train=False)

    model.cuda()
    generator.cuda()
    generator.eval()

    for param in generator.parameters():
        param.requires_grad = False
    latent_avg.requires_grad = False

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

    logger.info(f"Successfully load pre-trained parameters from {pretrained_path}")

    # generator: stylegan2
    # generator = torch.nn.parallel.DistributedDataParallel(
    #     generator, device_ids=[dist.get_rank()], find_unused_parameters=True)
    
    # some meters
    data_time_meter = AverageMeter()
    batch_time_meter = AverageMeter()

    batchsize = 1

    _TEXT_DESCRIPTION = [
        "The person is smiling.",      #0
        "The person is angry.",        #1
        "The person is closing eyes.", #2
        "The person is winking",       #3
        "The person is shocked.",      #4
        "The person is pouting.",      #5
        "The person is crying.",       #6
        "The person is disgusted.",    #7
        "The person is appalled.",     #8
        "The person is opening mouth.",#9
        "The person is hatred.",       #10
        "The person is surprised.",    #11
    ]

    # sample latent code for first frame
    sample_z = torch.randn(batchsize, 512).cuda() / 10. + latent_avg.cuda()
    img_ws = generator.get_latent(sample_z)

    tokenized_text = clip.tokenize(_TEXT_DESCRIPTION)
    
    _data_start = time.time()
    for i in tqdm(range(len(_TEXT_DESCRIPTION))):
        data_time_meter.update(time.time() - _data_start)

        # without real image as input
        _, txt_ws = model(None, tokenized_text[i].unsqueeze(0).repeat(batchsize, 1))
        txt_ws = txt_ws.view(batchsize, config.MODEL.N_INFERENCE_FRAMES, -1, 512)
        
        img_seq = []

        for t in range(config.MODEL.N_INFERENCE_FRAMES):
            edit_latent_code = img_ws + txt_ws[:, t, :, :]
            img_edited, _ = generator(
                [edit_latent_code], 
                input_is_latent=True, 
                randomize_noise=False)
            img_seq.append(img_edited.cpu())

        # save image 
        img_tensor = torch.cat(img_seq, dim=3)

        save_image(
            img_tensor,
            fp=os.path.join(
                config.OUTPUT_DIR, 
                "val_one_image_with_different_text", 
                f"{i:05d}_{dist.get_rank()}.png"),
            normalize=True,
            value_range=(-1, 1),
            nrow=1,
        )

        batch_time_meter.update(time.time() - _data_start)
        
        torch.cuda.synchronize()
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

    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()
    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    linear_scaled_lr = config.TRAIN.OPTIMIZER.BASE_LR

    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS

    config.defrost()
    config.TRAIN.OPTIMIZER.BASE_LR = linear_scaled_lr

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_DIR, "val_random_sample"), exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT_DIR, dist_rank=dist.get_rank(), name=f"{config.EXP_NAME}_val")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT_DIR, "config.yaml")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    logger.info(config.dump())
    one_image_with_different_text(config)