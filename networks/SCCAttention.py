import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import numpy as np
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.ln(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout):
        super().__init__()
        inner_dim = heads * head_dim
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, head_dim, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        out = x
        for att, ffn in self.layers:
            out = out + att(out)
            out = out + ffn(out)
        return out


class LGSAttention(nn.Module):
    def __init__(self, in_channel=3, dim=512, kernel_size=3, patch_size=2):
        super().__init__()
        self.ph, self.pw = patch_size, patch_size
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channel, dim, kernel_size=1)
        self.trans = Transformer(dim=dim, depth=3, heads=8, head_dim=64, mlp_dim=1024)
        self.conv3 = nn.Conv2d(dim, in_channel, kernel_size=1)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel)
        )

    def forward(self, x):
        ## Local Representation
        y = self.conv2(self.conv1(x))  # bs,dim,h,w

        ## Global Representation
        _, _, h, w = y.shape
        y = rearrange(y, 'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim', ph=self.ph, pw=self.pw)  # bs,h,w,dim
        y = self.trans(y)
        y = rearrange(y, 'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)', ph=self.ph, pw=self.pw, nh=h // self.ph,
                      nw=w // self.pw)  # bs,dim,h,w

        x0 = self.conv3(y)  # bs,dim,h,w
        x1 = self.conv1x1(x)
        x_spatial = torch.cat([x0, x1], 1)
        return x_spatial


class SECAttention(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_channel = self.sse(x) * x
        return x_channel


class SCCAttention(nn.Module):
    def __init__(self, input_resolution, in_channel=3, dim=512, kernel_size=3, patch_size=2):
        super().__init__()
        self.input_resolution = input_resolution
        self.ph, self.pw = patch_size, patch_size
        self.SECAttention = SECAttention(in_channel=in_channel)
        self.LGSAttention = LGSAttention(in_channel=in_channel, dim=dim, kernel_size=kernel_size, patch_size=patch_size)
        self.conv = nn.Conv2d(4 * in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        h, w = self.input_resolution
        b, l, c = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(b, c, h, w)

        x_spatial = self.SECAttention(x)
        x_channel = self.LGSAttention(x)
        y = torch.cat([x, x_spatial, x_channel], 1)  # bs,2*dim,h,w
        y = self.conv(y)  # bs,c,h,w

        y = y.reshape(b, c, -1)
        y = y.permute(0, 2, 1)

        return y