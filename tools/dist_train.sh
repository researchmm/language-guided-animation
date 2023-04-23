set -ex

CONFIG=$1
GPUS=$2
MASTER_PORT=$4
DATABLOB=$3

pip install -r extra_requirements.txt

# wheather the file exists, if not, download it
if [ ! -f "$DATABLOB/datasets/pretrained_models/stylegan2-ffhq-config-f.pt" ]; then
    mkdir -p $DATABLOB/datasets/pretrained_models;
    wget https://github.com/researchmm/language-guided-animation/releases/download/v0.0.0/stylegan2-ffhq-config-f.pt -O $DATABLOB/datasets/pretrained_models/stylegan2-ffhq-config-f.pt;
fi

if [ ! -f "$DATABLOB/datasets/CelebAMask-HQ/meta_train.pkl" ]; then
    mkdir -p $DATABLOB/datasets/CelebAMask-HQ;
    wget https://github.com/researchmm/language-guided-animation/releases/download/v0.0.0/meta_train.pkl -O $DATABLOB/datasets/CelebAMask-HQ/meta_train.pkl;
fi

if [ ! -f "$DATABLOB/datasets/CelebAMask-HQ/meta_test.pkl" ]; then
    mkdir -p $DATABLOB/datasets/CelebAMask-HQ;
    wget https://github.com/researchmm/language-guided-animation/releases/download/v0.0.0/meta_test.pkl -O $DATABLOB/datasets/CelebAMask-HQ/meta_test.pkl;
fi

if [ ! -f "$DATABLOB/datasets/CelebAMask-HQ/celeba_e4e_latents.pt" ]; then
    mkdir -p $DATABLOB/datasets/CelebAMask-HQ;
    wget https://github.com/researchmm/language-guided-animation/releases/download/v0.0.0/celeba_e4e_latents.pt -O $DATABLOB/datasets/CelebAMask-HQ/celeba_e4e_latents.pt;
fi

if [ ! -f "$DATABLOB/datasets/pretrained_models/stylegan2_pretrained/stylegan2-awesome-network-snapshot-metfaces2.pt" ]; then
    mkdir -p $DATABLOB/datasets/pretrained_models/stylegan2_pretrained/;
    wget https://github.com/researchmm/language-guided-animation/releases/download/v0.0.0/stylegan2-awesome-network-snapshot-metfaces2.pt -O $DATABLOB/datasets/pretrained_models/stylegan2_pretrained/stylegan2-awesome-network-snapshot-metfaces2.pt;
fi

# train
torchrun --nproc_per_node $GPUS \
    --master_port $MASTER_PORT tools/dist_train.py \
    --opts MODEL.N_INFERENCE_FRAMES 16 \
           DATA.BATCH_SIZE 4 \
    --cfg configs/$CONFIG.yaml \
    --datablob $DATABLOB

# inference
torchrun --nproc_per_node $GPUS \
    --master_port $MASTER_PORT tools/dist_inference.py \
    --opts MODEL.N_INFERENCE_FRAMES 16 \
           PRETRAINED.OUR_MODEL "Iter_002000.pth" \
    --cfg configs/$CONFIG.yaml \
    --datablob $DATABLOB
