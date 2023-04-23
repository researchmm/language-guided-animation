DATA_DIR=$1

if [ -z $CUDA_VISIBLE_DEVICES ]; then
   CUDA_VISIBLE_DEVICES='all'
fi
docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
   --mount src=/,dst=/home,type=bind \
   -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
   -w /home tiankaihang/azureml_docker:torch2.0 \
   bash -c "bash"