set -ex

CONFIG=$1
GPUS=$2

pip install -r extra_requirements.txt

python tools/inference.py \
    --opts MODEL.N_INFERENCE_FRAMES 16 \
           PRETRAINED.OUR_MODEL "Iter_005000.pth" \
    --cfg configs/$CONFIG.yaml \
    --local_rank 0 \
    --datablob $3 \
    --opts DATA.BATCH_SIZE 1
