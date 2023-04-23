import random

import torch
import torch.nn.functional as F


# transform seconds to standard format
def eta_format(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    d, h, m, s = int(d), int(h), int(m), int(s)
    _ret_string = ""
    if d > 0:
        _ret_string += f"{d}d:"
    _ret_string += f"{h:02d}:{m:02d}:{s:02d}"
    return _ret_string


def dict2string(_dict: dict):
    _str = ""
    total_loss = 0.0
    for _key in _dict:
        _str += f"{_key.replace('loss_', '')} {_dict[_key]:.03f} "

    return _str


def enable_gradient(model):
    for param in model.parameters():
        param.requires_grad = True


def disable_gradient(model):
    for param in model.parameters():
        param.requires_grad = False


def _debug_tensor(tensor):
    """
    input data type: `torch.Tensor` or `list of torch.Tensor` or `tuple of torch.Tensor`  
    """
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        for _t in tensor:
            _debug_tensor(_t)
    else:
        print(
            f"======================\n"
            f"shape:  {tensor.shape}\n"
            f"min:    {tensor.min()}\n"
            f"max:    {tensor.max()}\n"
            f"mean:   {tensor.mean()}\n"
            f"std:    {tensor.std()}\n"
            f"device: {tensor.device}\n"
            f"======================\n"
        )


def _denormalize(img, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    for i in range(3):
        img[:, i, :, :] *= std[i]
        img[:, i, :, :] += mean[i]


def get_first_order_diff_norm(x):
    b, t = x.shape[0:2]
    diff = F.mse_loss(
        x, torch.roll(x, shifts=1, dims=1), 
        reduction='none')
    diff = diff.view(b, t, -1)
    return diff.mean(-1)


def truncate(latent, latent_avg, truncate=False):
    if truncate:
        rate = random.uniform(0.5, 1.0)
        return latent_avg + rate * (latent - latent_avg)
    else:
        return latent

