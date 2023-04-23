import torch
import torch.utils.data as data

from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler


from .models import Generator, UniEditModelv1
from .datasets import CelebAHQ, DataCollator
from .criteria import LossFactoryBase
from .lr_schedulers import LinearLRScheduler


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def build_model(cfg, is_train=True):
    if is_train:
        n_frames = cfg.MODEL.N_FRAMES
    else:
        n_frames = cfg.MODEL.N_INFERENCE_FRAMES
    model = UniEditModelv1(
        feat_backbone=cfg.MODEL.BACKBONE, 
        target_resolution=cfg.TARGET_RESOLUTION,
        n_frames=n_frames)
    generator = Generator(
        cfg.TARGET_RESOLUTION, 512, 8, channel_multiplier=2)
    model_params = torch.load(f=cfg.PRETRAINED.STYLEGAN2, map_location='cpu')
    generator.load_state_dict(model_params['g_ema'])
    if 'latent_avg' not in model_params:
        latent_avg = None
    else:
        latent_avg = model_params['latent_avg']
    return model, generator, latent_avg


def build_dataloader(cfg):
    train_ds = CelebAHQ(
        data_dir=cfg.DATA.DATA_PATH, 
        is_valid=False, 
        debug_num=cfg.DATA.DEBUG_NUM,
        specific_text=cfg.SPECIFIC_TEXT,
    )

    datacollator = DataCollator(cfg.SPECIFIC_TEXT, _type=cfg.GEN_TYPE)

    train_loader = data.DataLoader(
        dataset=train_ds,
        batch_size=cfg.DATA.BATCH_SIZE,
        num_workers=cfg.DATA.NUM_WORKERS,
        shuffle=cfg.DATA.SHUFFLE,
        sampler=None,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=cfg.DATA.DROP_LAST,
        prefetch_factor=cfg.DATA.PREFETCH_FACTOR,
        persistent_workers=cfg.DATA.PERSISTENT_WORKERS,
        collate_fn=datacollator.collate_batch,
    )

    # here we only use 20 examples for validation
    val_ds = CelebAHQ(data_dir=cfg.DATA.DATA_PATH, is_valid=True, debug_num=20)
    val_loader = data.DataLoader(
        dataset=val_ds,
        batch_size=cfg.DATA.BATCH_SIZE,
        num_workers=cfg.DATA.NUM_WORKERS,
        shuffle=False,
        sampler=None,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=cfg.DATA.PREFETCH_FACTOR,
        collate_fn=datacollator.collate_batch,
    )

    train_loader = sample_data(train_loader)

    return train_loader, val_loader


def build_loss(cfg):
    loss_fn = LossFactoryBase(
        lambda_l1         = cfg.TRAIN.LOSS.lambda_l1   , 
        lambda_l2         = cfg.TRAIN.LOSS.lambda_l2   , 
        lambda_lpips      = cfg.TRAIN.LOSS.lambda_lpips,
        lambda_id         = cfg.TRAIN.LOSS.lambda_id   ,
        lambda_clip       = cfg.TRAIN.LOSS.lambda_clip ,
        id_weight_path    = cfg.PRETRAINED.ID_WEIGHT,
        resolution        = cfg.DATA.IMG_SIZE,
        contrastive       = cfg.TRAIN.LOSS.CONTRASTIVE,
    )

    return loss_fn


def build_scheduler(cfg, optimizer):
    num_steps = int(cfg.TRAIN.TOTAL_ITERS)
    warmup_steps = int(cfg.TRAIN.WARMUP_STEPS)
    decay_steps = int(cfg.TRAIN.DECAY_STEPS)

    lr_scheduler = None
    if cfg.TRAIN.LR_SCHEDULER.NAME == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            t_mul=1.,
            lr_min=cfg.TRAIN.MIN_LR,
            warmup_lr_init=cfg.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif cfg.TRAIN.LR_SCHEDULER.NAME == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=cfg.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif cfg.TRAIN.LR_SCHEDULER.NAME == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=cfg.TRAIN.LR_SCHEDULER.DECAY_RATE,
            warmup_lr_init=cfg.TRAIN.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    else:
        lr_scheduler = None

    return lr_scheduler