#################################################################################################
# Copyright (c) 2023 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################
import torch
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_

from .functional import natten3dav, natten3dqkrpb


class NeighborhoodAttention3D(nn.Module):
    """
    Neighborhood Attention 3D Module
    """

    def __init__(
        self,
        dim,
        num_heads,
        kernel_size,
        kernel_size_d=None,
        dilation=1,
        dilation_d=None,
        bias=True,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        kernel_size_d = kernel_size_d or kernel_size
        dilation = dilation or 1
        dilation_d = dilation_d or dilation
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert (
            dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        assert (
            dilation_d >= 1
        ), f"Dilation (depth) must be greater than or equal to 1, got {dilation_d}."
        self.kernel_size = kernel_size
        self.kernel_size_d = kernel_size_d
        self.dilation = dilation
        self.dilation_d = dilation_d

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(
                    num_heads,
                    (2 * kernel_size_d - 1),
                    (2 * kernel_size - 1),
                    (2 * kernel_size - 1),
                )
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, D, H, W, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, D, H, W, 3, self.num_heads, self.head_dim)
            .permute(4, 0, 5, 1, 2, 3, 6)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = natten3dqkrpb(
            q,
            k,
            self.rpb,
            self.kernel_size_d,
            self.kernel_size,
            self.dilation_d,
            self.dilation,
        )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = natten3dav(
            attn,
            v,
            self.kernel_size_d,
            self.kernel_size,
            self.dilation_d,
            self.dilation,
        )
        x = x.permute(0, 2, 3, 4, 1, 5).reshape(B, D, H, W, C)

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size_d={self.kernel_size_d}, dilation_d={self.dilation_d}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"rel_pos_bias={self.rpb is not None}"
        )
