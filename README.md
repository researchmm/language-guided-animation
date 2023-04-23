# Language-Guided Face Animation by Recurrent StyleGAN-based Generator

Tiankai Hang, Huan Yang, Bei Liu, Jianlong Fu, Xin Geng, and Baining Guo.

## Demo video
https://user-images.githubusercontent.com/52823230/183278083-488ee277-775f-4022-ab36-9a30a24feaf1.mp4

## Abstract 
Recent works on language-guided image manipulation have shown great power of language in providing rich semantics, especially for face images. However, the other natural information, motions, in language is less explored. In this paper, we leverage the motion information and study a novel task, language-guided face animation, that aims to animate a static face image with the help of languages. To better utilize both semantics and motions from languages, we propose a simple yet effective framework. Specifically, we propose a recurrent motion generator to extract a series of semantic and motion information from the language and feed it along with visual information to a pre-trained StyleGAN to generate high-quality frames. To optimize the proposed framework, three carefully designed loss functions are proposed including a regularization loss to keep the face identity, a path length regularization loss to ensure motion smoothness, and a contrastive loss to enable video synthesis with various language guidance in one single model. Extensive experiments with both qualitative and quantitative evaluations on diverse domains (e.g., human face, anime face, and dog face) demonstrate the superiority of our model in generating high-quality and realistic videos from one still image with the guidance of language.

## Environment
The main package we require is `torch>=1.10` and the matched CUDA version. (You should check that by `nvcc -V`). Other packages should be easily installed by `pip install -r extra_requirements.txt`.

## Prepare Data and Pre-trained StyleGAN

Set the path `DATABLOB` for your datasets, pre-trained checkpoints, and outputs. Then put the required data under the path.

```bash
git clone https://github.com/researchmm/language-guided-animation.git
cd language-guided-animation
# current folder
DATABLOB="."
mkdir -p $DATABLOB/datasets/pretrained_models;
wget https://github.com/researchmm/language-guided-animation/releases/download/v0.0.0/stylegan2-ffhq-config-f.pt -O $DATABLOB/datasets/pretrained_models/stylegan2-ffhq-config-f.pt;
mkdir -p $DATABLOB/datasets/CelebAMask-HQ;
wget https://github.com/researchmm/language-guided-animation/releases/download/v0.0.0/meta_train.pkl -O $DATABLOB/datasets/CelebAMask-HQ/meta_train.pkl;
wget https://github.com/researchmm/language-guided-animation/releases/download/v0.0.0/meta_test.pkl -O $DATABLOB/datasets/CelebAMask-HQ/meta_test.pkl;
wget https://github.com/researchmm/language-guided-animation/releases/download/v0.0.0/celeba_e4e_latents.pt -O $DATABLOB/datasets/CelebAMask-HQ/celeba_e4e_latents.pt;
```

Download the `CelebAMask-HQ` dataset from [`switchablenorms/CelebAMask-HQ`](https://github.com/switchablenorms/CelebAMask-HQ#celebamask-hq-dataset-downloads).
The data structure should be
```
DATABLOB
    datasets
        CelebAMask-HQ
            CelebA-HQ-img
                0.jpg
                1.jpg
                ...
```

`meta_train.pkl` and `meta_test.pkl` are built mete information. `celeba_e4e_latents.pt` is encoded latents from [`e4e`](https://github.com/omertov/encoder4editing).

## Training 

```bash
# Number of GPUs
GPUS=4
DATABLOB="."
MASTER_PORT=12345
CONFIG="celeba-hq/celeba_1st0_2rd1_clip05_constantLR_fused_5k_pmt_bs4"
torchrun --nproc_per_node $GPUS \
    --master_port $MASTER_PORT tools/dist_train.py \
    --opts MODEL.N_INFERENCE_FRAMES 16 \
           DATA.BATCH_SIZE 4 \
    --cfg configs/$CONFIG.yaml \
    --datablob $DATABLOB
```
or you could directly run 
```bash
GPUS=4
DATABLOB="."
MASTER_PORT=12345
CONFIG="celeba-hq/celeba_1st0_2rd1_clip05_constantLR_fused_5k_pmt_bs4"
bash tools/dist_train.sh $CONFIG $GPUS $DATABLOB $MASTER_PORT
```
The [`tools/dist_train.sh`](tools/dist_train.sh) will automatically download the pre-trained StyleGAN, conduct training and inference.

## Inference
```bash
torchrun --nproc_per_node $GPUS \
    --master_port $MASTER_PORT tools/dist_inference.py \
    --opts MODEL.N_INFERENCE_FRAMES 16 \
           PRETRAINED.OUR_MODEL "Iter_002000.pth" \
    --cfg configs/$CONFIG.yaml \
    --datablob $DATABLOB
```

## To Do
- [ ] Share pretrained models.

## Citation
If you find our work useful for your research, please consider citing our paper.
```bibtex
@ARTICLE{hang2023lang,
  author={Hang, Tiankai and Yang, Huan and Liu, Bei and Fu, Jianlong and Geng, Xin and Guo, Baining},
  journal={IEEE Transactions on Multimedia}, 
  title={Language-Guided Face Animation by Recurrent StyleGAN-Based Generator}, 
  year={2023},
  volume={},
  number={},
  pages={1-12},
  doi={10.1109/TMM.2023.3248143}
}
```