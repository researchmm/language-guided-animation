"""
implementation for Recurrent latent code generation，
"""

import math
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import random


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
            f"device: {tensor.device}"
        )


class RNNModule(nn.Module):
    def __init__(self,
                 z_dim=512,
                 h_dim=384,
                 w_residual=0.2):
        super(RNNModule, self).__init__()
        
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.w_residual = w_residual

        self.enc_cell = nn.LSTMCell(z_dim, h_dim)
        self.cell = nn.LSTMCell(z_dim, h_dim)
        self.w = nn.Parameter(torch.FloatTensor(h_dim, z_dim))
        self.b = nn.Parameter(torch.FloatTensor(z_dim))
        self.fc1 = nn.Linear(h_dim * 2, z_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(z_dim, z_dim)

        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if (isinstance(module, nn.LSTMCell)):
                for name, param in module.named_parameters():
                    if ('weight_ih' in name) or ('weight_hh' in name):
                        mul = param.shape[0] // 4
                        for idx in range(4):
                            init.orthogonal_(param[idx * mul:(idx + 1) * mul])
                    elif 'bias' in name:
                        param.data.fill_(0)
            if (isinstance(module, nn.Linear)):
                init.orthogonal_(module.weight)

        nn.init.normal_(self.w, std=0.02)
        self.b.data.fill_(0.0)

    def forward(self, z, n_frame):

        out = [z]
        h_, c_ = self.enc_cell(z)
        h = [h_]
        c = [c_]
        e = []
        
        for i in range(n_frame - 1):
            # e_ = self.get_initial_state_z(z.shape[0])
            e_ = out[-1] # B x z_dim
            h_, c_ = self.cell(e_, (h[-1], c[-1]))
            mul = torch.matmul(h_, self.w) + self.b
            mul = torch.tanh(mul)
            e.append(e_)
            h.append(h_)
            c.append(c_)
            out_ = out[-1] + self.w_residual * mul
            out.append(out_)

        out = [item.unsqueeze(1) for item in out]

        out = torch.cat(out, dim=1).view(-1, self.z_dim)

        e = [item.unsqueeze(1) for item in e]
        e = torch.cat(e, dim=1).view(-1, self.z_dim)

        hh = h[1:]
        hh = [item.unsqueeze(1) for item in hh]
        hh = torch.cat(hh, dim=1).view(-1, self.h_dim)

        cc = c[1:]
        cc = [item.unsqueeze(1) for item in cc]
        cc = torch.cat(cc, dim=1).view(-1, self.h_dim)

        hc = torch.cat((hh, cc), dim=1)
        e_rec = self.fc2(self.relu(self.fc1(hc)))

        return out, e, e_rec

    def get_initial_state_z(self, batchSize):
        return torch.cuda.FloatTensor(batchSize, self.z_dim).normal_()


# ------------------------------------------
# Recurrent generation like that in basicVSR
# ------------------------------------------
class LatentRecurrent(nn.Module):
    def __init__(self, z_dim=512, h_dim=384, w_residual=0.2):
        super().__init__()
        
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.w_residual = w_residual * 4.0

        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim * 2, z_dim)
        self.fc3 = nn.Linear(h_dim * 2, h_dim)

        self.lrelu = nn.LeakyReLU()
    

    def truncate(self, z0, z):
        z0 = z0.clone().detach()
        rate = random.uniform(0.5, 1.0)
        return z0 + rate * (z - z0)


    def decay(self, t, T, choice='linear'):
        # t ranges from 1 to T - 1
        if choice == 'linear':
            return (T - t) / T
        elif choice == 'cosine':
            pass
        elif choice == 'sqrt':
            return math.sqrt((T - t) / T)
        elif choice == 'exp':
            return 1 - math.exp(t / T - 1)
        elif choice == 'const' or choice is None:
            return 1.0
        else:
            raise NotImplementedError(f"Such choice {choice} is not supported")


    def forward(self, z, n_frames=10, is_inference=False, choice='const'):
        """
        Generate motion code

        Args
        -----------
        z: initial latent code with shape [B, z_dim]
        n_frames: number of frames

        Return
        -----------
        out: a list of latent codes
        """
        bs = z.shape[0]

        out = [z]
        # h_init = torch.zeros(bs, self.h_dim).cuda(z.get_device())
        h_init = torch.randn(bs, self.h_dim).cuda(z.get_device()) * 0.01
        h = [h_init]
        for t in range(1, n_frames):
            _z = out[-1]
            _h = h[-1]

            # [B, 2 * h_dim]
            _temp = torch.cat((self.lrelu(self.fc1(_z)), _h), dim=-1)
            # [B, z_dim]
            _out = self.lrelu(self.fc2(_temp))
            if is_inference:
                r = self.w_residual / n_frames * self.decay(t, n_frames, choice)
            else:
                r = self.w_residual / n_frames
            _out = r * _out + out[-1]

            out.append(_out)
            h.append(self.lrelu(self.fc3(_temp)) * r + h[-1])
        
        out = [(item - z).unsqueeze(1) for item in out]
        out = torch.cat(out, dim=1).view(-1, self.z_dim) # [B * n_frames, z_dim]
        return out

if __name__ == '__main__':
    z_dim = 512
    bs = 8
    rnn = LatentRecurrent(z_dim=512).cuda()
    z = torch.randn(bs, z_dim).cuda()

    for i in range(5000):
        out = rnn(z, 10)

    _debug_tensor(out)

    for i in range(5000):
        out = rnn(z, 50)
    
    _debug_tensor(out)