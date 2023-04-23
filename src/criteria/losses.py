import os, sys

import torch
import torch.nn.functional as F
import clip

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'models')
sys.path.append(MODEL_DIR)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_irse import Backbone
from .lpips import LPIPS


class CLIPLoss(torch.nn.Module):

    def __init__(self, resolution=224):
        super(CLIPLoss, self).__init__()
        self.size = (resolution, resolution)
        self.model, _ = clip.load("ViT-B/32", device="cpu")
        self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image, text, value_range=[1, -1]):
        if value_range[0] == 1 and value_range[-1] == -1:
            image = (image + 1.0) / 2.0
            image = (image - self.mean) / self.std

        image_features = F.interpolate(image, size=self.size, mode='bilinear', align_corners=True)
        image_features = self.model.encode_image(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [B, C]

        text_features = torch.cat([clip.tokenize(text)]).cuda()
        text_features = self.model.encode_text(text_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # [B, L]

        similarity = torch.bmm(image_features.unsqueeze(1), text_features.unsqueeze(-1))
        score = (1 - similarity).mean()
        return score


class CLIPLoss_modified(torch.nn.Module):

    def __init__(self, resolution=224, temperature=0.07, contrastive=True):
        super(CLIPLoss_modified, self).__init__()
        self.size = (resolution, resolution)
        self.model, _ = clip.load("ViT-B/32", device="cpu")
        self.register_buffer('mean', torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        for param in self.model.parameters():
            param.requires_grad = False

        self.temperature = temperature
        self.contrastive = contrastive

    def forward(self, image, text, value_range=[1, -1]):
        if value_range[0] == 1 and value_range[-1] == -1:
            image = (image + 1.0) / 2.0
            image = (image - self.mean) / self.std

        image_features = F.interpolate(image, size=self.size, mode='bilinear', align_corners=True)
        image_features = self.model.encode_image(image_features)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [B, C]

        text_features = torch.cat([clip.tokenize(text)]).cuda()
        text_features = self.model.encode_text(text_features)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)    # [B, C]

        if isinstance(text, list) and self.contrastive:
            similarity = torch.mm(image_features, text_features.permute(1, 0))
            score = torch.log(
                1 + torch.exp(
                    (similarity - torch.diag(similarity).view(-1, 1)) / self.temperature
                ).sum(dim=1)).mean()
        elif isinstance(text, str) or (not self.contrastive):
            if text_features.shape[0] == 1:
                text_features = text_features.repeat(image_features.shape[0], 1)
            
            similarity = torch.bmm(image_features.unsqueeze(1), text_features.unsqueeze(-1))
            score = (1 - similarity).mean()
        else:
            raise ValueError(f"Such input type {type(text)} is not supported!")
        # score = (1 - similarity[:, 0]).mean() + (similarity[:, 1:]).mean()
        return score


class IDLoss(torch.nn.Module):
    def __init__(self, weight_path):
        super(IDLoss, self).__init__()
        print('>>>>>> Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(weight_path, map_location='cpu'))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        for param in self.facenet.parameters():
            param.requires_grad = False

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        y_feats = self.extract_feats(y).detach()  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)

        loss = (1 - torch.bmm(y_hat_feats.unsqueeze(1), y_feats.unsqueeze(-1))).mean()
        return loss


class LossFactoryBase(torch.nn.Module):
    def __init__(self, 
                 lambda_l1=0.0, 
                 lambda_l2=0.0, 
                 lambda_lpips=0.0,
                 lambda_id=0.0,
                 lambda_clip=0.0,
                 id_weight_path=None,
                 resolution=224,
                 contrastive=True):
        super().__init__()

        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_lpips = lambda_lpips
        self.lambda_id = lambda_id
        self.lambda_clip = lambda_clip

        self.loss_rec_l1 = F.l1_loss  if lambda_l1 > 0 else None
        self.loss_rec_l2 = F.mse_loss if lambda_l2 > 0 else None
        
        if id_weight_path is not None and lambda_id > 0:
            self.loss_id = IDLoss(weight_path=id_weight_path)
        else:
            self.loss_id = None
        
        self.loss_clip = CLIPLoss_modified(
            resolution=resolution,
            contrastive=contrastive) \
                if lambda_clip > 0 else None
        self.loss_lpips = LPIPS() if lambda_lpips > 0 else None

        self.loss_dict = {'total_loss': 0.0}

    
    def add_loss_item(self, name, loss_data):
        self.loss_dict[name] = loss_data
        self.loss_dict['total_loss'] += loss_data

    def forward(self, pred, target=None, text=None, use_specific_text=False):
        loss = 0.0

        if self.loss_rec_l1 and target is not None:
            l1 = self.loss_rec_l1(pred, target) * self.lambda_l1
            loss += l1
            self.loss_dict['loss_rec_l1'] = l1.item()
        if self.loss_rec_l2 and target is not None:
            l2 = self.loss_rec_l2(pred, target) * self.lambda_l2
            loss += l2
            self.loss_dict['loss_rec_l2'] = l2.item()
        if self.loss_id and target is not None:
            id = self.loss_id(pred, target) * self.lambda_id
            loss += id
            self.loss_dict['loss_id'] = id.item()
        if self.loss_clip and text is not None:
            if use_specific_text:
                l_clip = self.loss_clip(pred, text[0]) * self.lambda_clip
            else:
                l_clip = self.loss_clip(pred, text) * self.lambda_clip
            loss += l_clip
            self.loss_dict['loss_clip'] = l_clip.item()
        if self.loss_lpips and target is not None:
            l_lpips = self.loss_lpips(pred, target).mean() * self.lambda_lpips
            loss += l_lpips
            self.loss_dict['loss_lpips'] = l_lpips.item()
        
        try:
            self.loss_dict['total_loss'] += loss.item()
        except:
            self.loss_dict['total_loss'] += loss
        
        return loss, self.loss_dict


# ref: https://github.com/snap-research/MoCoGAN-HD/blob/main/models/losses.py
def loss_hinge_dis(dis_fake, dis_real):
    loss_real = torch.mean(F.relu(1. - dis_real))
    loss_fake = torch.mean(F.relu(1. + dis_fake))
    return loss_real, loss_fake


def loss_hinge_gen(dis_fake):
    loss = -torch.mean(dis_fake)
    return loss


def compute_gradient_penalty_T(real_B, fake_B, modelD, opt):
    alpha = torch.rand(list(real_B.size())[0], 1, 1, 1, 1)
    alpha = alpha.expand(real_B.size()).cuda(real_B.get_device())

    interpolates = alpha * real_B.data + (1 - alpha) * fake_B.data
    interpolates = torch.tensor(interpolates, requires_grad=True)

    pred_interpolates = modelD(interpolates)

    gradient_penalty = 0
    if isinstance(pred_interpolates, list):
        for cur_pred in pred_interpolates:
            gradients = torch.autograd.grad(outputs=cur_pred[-1],
                                            inputs=interpolates,
                                            grad_outputs=torch.ones(
                                                cur_pred[-1].size()).cuda(
                                                    real_B.get_device()),
                                            create_graph=True,
                                            retain_graph=True,
                                            only_inputs=True)[0]

            gradient_penalty += ((gradients.norm(2, dim=1) - 1)**2).mean()
    else:
        sys.exit('output is not list!')

    gradient_penalty = (gradient_penalty / opt.num_D) * 10
    return gradient_penalty


class GANLoss(torch.nn.Module):
    def __init__(self,
                 use_lsgan=True,
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = torch.nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None)
                            or (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = torch.tensor(real_tensor,
                                                   requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None)
                            or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = torch.tensor(fake_tensor,
                                                   requires_grad=False)
            target_tensor = self.fake_label_var

        if input.is_cuda:
            target_tensor = target_tensor.cuda()
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class Relativistic_Average_LSGAN(GANLoss):
    '''
        Relativistic average LSGAN
    '''
    def __call__(self, input_1, input_2, target_is_real):
        if isinstance(input_1[0], list):
            loss = 0
            for input_i, _input_i in zip(input_1, input_2):
                pred = input_i[-1]
                _pred = _input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred - torch.mean(_pred), target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input_1[-1], target_is_real)
            return self.loss(input_1[-1] - torch.mean(input_2[-1]),
                             target_tensor)


if __name__ == '__main__':
    import time

    loss_fn = LossFactoryBase(
        lambda_l1   =1.0, 
        lambda_l2   =1.0, 
        lambda_lpips=1.0,
        lambda_id   =1.0,
        lambda_clip =1.0,
        id_weight_path="/home/data/tiankai/datasets/pretrained_models/model_ir_se50.pth",
    ).cuda()

    bs = 16

    x = torch.randn(bs, 3, 224, 224).cuda()
    y = torch.randn(bs, 3, 224, 224).cuda()
    text = ["hello"] * bs

    start_time = time.time()
    total_iters = 1000
    for i in range(total_iters):
        loss = loss_fn(x, y, text)
    time_per_iter = (time.time() - start_time) / total_iters
    print(f"time used per iter: {time_per_iter * 1000}ms")

    """
    time used: 
    l1 0.033976078033447266ms
    l2 0.039466142654418945ms
    id 20.167252779006958ms
    lpips 19.74770212173462ms
    clip 15.133862495422363ms
    total: 49.85308480262756ms (3019 MiB)

    if 1 second per iteration, 3600 x 24 = 86400
    """
