"""
Neighborhood Attention 1D PyTorch Module

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_
from .functional import natten1dqkrpb, natten1dav


class NeighborhoodAttention1D(nn.Module):
    """
    Neighborhood Attention 1D Module
    """
    def __init__(self, dim, kernel_size, num_heads,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 dilation=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        assert kernel_size > 1 and kernel_size % 2 == 1, \
            f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.rpb = nn.Parameter(torch.zeros(num_heads, (2 * kernel_size - 1)))
        trunc_normal_(self.rpb, std=.02, mean=0., a=-2., b=2.)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, Lp, C = x.shape
        L = Lp
        pad_l = pad_r = 0
        if L < self.window_size:
            pad_r = max(0, self.window_size - L)
            x = pad(x, (0, 0, pad_l, pad_r))
            _, L, _ = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = natten1dqkrpb(q, k, self.rpb, self.dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten1dav(attn, v, self.dilation)
        x = x.permute(0, 2, 1, 3).reshape(B, L, C)
        if pad_r:
            x = x[:, :Lp, :]

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return f'kernel_size={self.kernel_size}, dilation={self.dilation}, head_dim={self.head_dim}, num_heads={self.num_heads}'

