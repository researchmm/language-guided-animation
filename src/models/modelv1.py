import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import clip

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR) # models/

from stylegan2.model import EqualLinear, PixelNorm
from clip_openai import CLIP_MODEL_INFO
from rnn import LatentRecurrent

_INIT_RES = 4


class LatentCodesDiscriminator(nn.Module):
    def __init__(self, style_dim, n_mlp):
        super().__init__()

        self.style_dim = style_dim

        layers = []
        for _ in range(n_mlp - 1):
            layers.append(
                nn.Linear(style_dim, style_dim)
            )
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(512, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, w):
        return self.mlp(w)


class UniEditModelv1(nn.Module):
    def __init__(self, 
                 feat_backbone="RN50", 
                 target_resolution=1024, 
                 n_frames=10, 
                 num_layers=None):
        super().__init__()

        self.n_frames = n_frames
        self.encoder, _ = clip.load(feat_backbone, device="cpu")

        self.embed_dim = CLIP_MODEL_INFO[feat_backbone]['embed_dim']
        self.input_resolution = CLIP_MODEL_INFO[feat_backbone]['resolution']
        self.mapper = Mapper(
            target_resolution, 
            embed_dim=self.embed_dim,
            num_layers=num_layers)

        self.rnn = LatentRecurrent(z_dim=self.embed_dim)

    def forward(self, img, txt, is_inference=False, choice=None):
        """
        img: input image with shape [B, 3, H, W]
        txt: tokenized input text with shape [B, 77]
        """

        # the text embedding is with the shape [B, embed_dim]
        txt_embed = self.encoder.encode_text(txt)
        txt_embed = F.normalize(txt_embed, dim=1)

        # the text embedding is with the shape [B, embed_dim] or None
        # None means that there is no image as input, just sample in 
        # the latent space directly
        if img is not None:
            img_embed = self.encoder.encode_image(img)
            img_embed = F.normalize(img_embed, dim=1)

            # rnn: to generate motion latent from text, shape [B, n_frames, embed_dim]
            # Baseline: takes txt_embed as input 
            # Improve:  takes txt_embed and img_embed as input 
            # text_motion_embed = self.rnn(txt_embed, self.n_frames)
            # text_motion_embed = self.rnn(txt_embed + img_embed, self.n_frames)
            text_motion_embed = self.rnn(
                txt_embed + img_embed.detach(), 
                self.n_frames, 
                is_inference=is_inference,
                choice=choice)
            img_ws, text_motion_embed = self.mapper(img_embed, text_motion_embed)
        else:
            text_motion_embed = self.rnn(
                txt_embed, 
                self.n_frames, 
                is_inference=is_inference,
                choice=choice)
            img_ws, text_motion_embed = self.mapper(None, text_motion_embed)
        
        return img_ws, text_motion_embed


class Mapper(nn.Module):
    def __init__(self, resolution=1024, embed_dim=1024, num_layers=None):
        """
        resolution: the resolution of generated image (default: 1024)
        """
        super().__init__()
        self.num_layers = num_layers if num_layers is not None \
            else int(np.log2(resolution // _INIT_RES * 2)) * 2
        self.v_mapper = nn.Sequential(
            PixelNorm(),
            EqualLinear(embed_dim, 1024, activation='fused_lrelu'),
            EqualLinear(1024, 1024, activation='fused_lrelu'),
            EqualLinear(1024, 512 * self.num_layers, activation='fused_lrelu'),
        )
        self.t_mapper = nn.Sequential(
            PixelNorm(),
            EqualLinear(embed_dim, 1024, activation='fused_lrelu'),
            EqualLinear(1024, 1024, activation='fused_lrelu'),
            EqualLinear(1024, 512 * self.num_layers, activation='fused_lrelu'),
        )

    def forward(self, vis, txt):
        if vis is not None:
            vis_wp  = self.v_mapper(vis).view(-1, self.num_layers, 512)
        else:
            vis_wp = None
        
        txt_dwp = self.t_mapper(txt).view(-1, self.num_layers, 512)
        return vis_wp, txt_dwp


if __name__ == '__main__':
    model = UniEditModelv1(target_resolution=1024, n_frames=4).cuda()
    img, txt = torch.randn(8, 3, 224, 224).cuda(), torch.randn(8, 77).cuda().int()
    txt[txt < 0] = 0
    
    for i in range(500):
        img_ws, text_motion_embed = model(img, txt)
