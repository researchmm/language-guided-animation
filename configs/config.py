import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

_C.DATABLOB = "/home/data/tiankai"

# --------------------
#        data
# --------------------

_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'CelebAHQ'
# Input image size
_C.DATA.IMG_SIZE = 256
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8
_C.DATA.DROP_LAST = True
_C.DATA.SHUFFLE = True
_C.DATA.PREFETCH_FACTOR = 4
_C.DATA.PERSISTENT_WORKERS = True
_C.DATA.DEBUG_NUM = -1

# --------------------
#        model
# --------------------
_C.MODEL = CN()
_C.MODEL.TYPE = "modelv1"
_C.MODEL.BACKBONE = 'RN50'
# number of frames used in training
_C.MODEL.N_FRAMES = 10
# number of frames used in inference
_C.MODEL.N_INFERENCE_FRAMES = 10
_C.MODEL.USE_LATENT = True
_C.MODEL.USE_VISUAL = True

# --------------------
#        train
# --------------------
_C.TRAIN = CN()
_C.TRAIN.ACCUMULATION_STEPS = 1

_C.TRAIN.TOTAL_ITERS = 500000
_C.TRAIN.WARMUP_STEPS = 0
_C.TRAIN.DECAY_STEPS = 0
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.TYPE = 'Adam'
_C.TRAIN.OPTIMIZER.BASE_LR = 1e-3
_C.TRAIN.OPTIMIZER.MAPPER_LR = 1e-3
_C.TRAIN.OPTIMIZER.RNN_LR = 1e-3
_C.TRAIN.OPTIMIZER.W_DISCRIMINTOR_LR = 1e-4

_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6

# learning scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# loss weight
_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.lambda_l1 =     0.0
_C.TRAIN.LOSS.lambda_l2 =     0.0 
_C.TRAIN.LOSS.lambda_lpips =  0.0
_C.TRAIN.LOSS.lambda_id =     0.0
_C.TRAIN.LOSS.lambda_clip =   0.0
_C.TRAIN.LOSS.lambda_wdis =   0.0
_C.TRAIN.LOSS.lambda_w_1st =  0.0
_C.TRAIN.LOSS.lambda_w_2rd =  0.0
_C.TRAIN.LOSS.lambda_w_reg =  0.0
_C.TRAIN.LOSS.lambda_w_l1  =  1.0

# to use contrastive clip loss
_C.TRAIN.LOSS.CONTRASTIVE  = True

_C.TARGET_RESOLUTION = 1024

# --------------------
#     pretrained
# --------------------
_C.PRETRAINED = CN()
_C.PRETRAINED.ID_WEIGHT = "datasets/pretrained_models/model_ir_se50.pth"
_C.PRETRAINED.VGG16     = "datasets/pretrained_models/vgg16.pth"
_C.PRETRAINED.STYLEGAN2 = "datasets/pretrained_models/stylegan2-ffhq-config-f.pt"
_C.PRETRAINED.STYLEGAN3 = "datasets/pretrained_models/stylegan3_pretrained/stylegan3-t-ffhq-1024x1024.pkl"
_C.PRETRAINED.OUR_MODEL = None

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT_DIR = "outputs"
_C.EXP_NAME = ""
_C.STYLE_IMG = ""
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False

_C.SPECIFIC_TEXT = None

# generation type, "face" or "dog"
_C.GEN_TYPE = "face"


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()

def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, config.EXP_NAME)

    # add datablob
    if args.datablob:
        config.DATABLOB = args.datablob
    
    config.OUTPUT_DIR           = os.path.join(config.DATABLOB, config.OUTPUT_DIR)
    config.DATA.DATA_PATH       = os.path.join(config.DATABLOB, config.DATA.DATA_PATH)
    config.PRETRAINED.ID_WEIGHT = os.path.join(config.DATABLOB, config.PRETRAINED.ID_WEIGHT)
    config.PRETRAINED.VGG16     = os.path.join(config.DATABLOB, config.PRETRAINED.VGG16    )
    config.PRETRAINED.STYLEGAN2 = os.path.join(config.DATABLOB, config.PRETRAINED.STYLEGAN2)
    config.PRETRAINED.STYLEGAN3 = os.path.join(config.DATABLOB, config.PRETRAINED.STYLEGAN3)

    if config.PRETRAINED.OUR_MODEL is not None:
        config.PRETRAINED.OUR_MODEL = os.path.join(
            config.OUTPUT_DIR, "ckpt", config.PRETRAINED.OUR_MODEL)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config