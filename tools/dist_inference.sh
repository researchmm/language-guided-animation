set -ex

CONFIG=$1
GPUS=$2

pip install -r extra_requirements.txt

python -m torch.distributed.launch --nproc_per_node $GPUS \
    --master_port 12345 tools/dist_inference.py \
    --opts MODEL.N_INFERENCE_FRAMES 16 \
           PRETRAINED.OUR_MODEL "Iter_002000.pth" \
    --cfg configs/$CONFIG.yaml \
    --datablob $3
