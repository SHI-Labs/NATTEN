#################################################################################################
# Copyright (c) 2022-2024 Ali Hassani.
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

from typing import Any

import torch
from torch.profiler import profile as torch_profile

from .utils import extract_na_ops, profile_with_torch

try:
    from natten.libnatten import (  # type: ignore
        na1d_av_backward as na1d_av_d,
        na1d_av_forward as na1d_av,
        na1d_qk_backward as na1d_qk_d,
        na1d_qk_forward as na1d_qk,
    )
except:
    from natten._C import (  # type: ignore
        natten1dav_backward as na1d_av_d,
        natten1dav_forward as na1d_av,
        natten1dqkrpb_backward as na1d_qk_d,
        natten1dqkrpb_forward as na1d_qk,
    )


_NA_1D_C_KEYWORDS = {
    "pn_keywords": ["NA1dPN", "OperatorE0", "gemm::PointwiseNeighborhood1D"],
    "nn_keywords": ["NA1dNN", "OperatorE1", "gemm::NeighborhoodNeighborhood1D"],
    "in_keywords": ["NA1dIN", "OperatorE2", "gemm::InverseNeighborhood1D"],
    "rpb_keywords": ["rpb_forward"],
    "rpbgrad_keywords": [
        "rpb_backward",
        "rpb_grad",
        "natten1drpb_cuda_backward",
        "na1d_rpb_cuda_backward",
        "rel_pos_bias_gradient_1d",
        "naive::RelPosBiasGradient1D",
    ],
    "qkrpb_keywords": ["na1d_qkrpb_cuda_forward", "natten1dqkrpb_cuda_forward"],
    "av_keywords": ["na1d_av_cuda_forward", "natten1dav_cuda_forward"],
    "qgrad_keywords": ["na1d_q_cuda_backward", "natten1dq_cuda_backward"],
    "kgrad_keywords": ["na1d_k_cuda_backward", "natten1dk_cuda_backward"],
    "vgrad_keywords": ["na1d_v_cuda_backward", "natten1dv_cuda_backward"],
    "agrad_keywords": ["na1d_a_cuda_backward", "natten1da_cuda_backward"],
    "legacy_pn_keywords": [
        "pointwise_neighborhood_1d",
        "naive::PointwiseNeighborhood1D",
    ],
    "legacy_nn_keywords": [
        "neighborhood_neighborhood_1d",
        "naive::NeighborhoodNeighborhood1D",
    ],
    "legacy_in_keywords": ["inverse_neighborhood_1d", "naive::InverseNeighborhood1D"],
}


def _profile_na1d_with_torch(
    batch_size: int,
    heads: int,
    length: int,
    dim: int,
    kernel_size: int,
    dilation: int,
    dtype: Any,
    warmup_steps: int = 10,
    enable_bias: bool = False,
) -> torch_profile:
    q = torch.randn(
        (batch_size, heads, length, dim), requires_grad=False, dtype=dtype
    ).cuda(0)
    k = torch.randn(
        (batch_size, heads, length, dim), requires_grad=False, dtype=dtype
    ).cuda(0)
    v = torch.randn(
        (batch_size, heads, length, dim), requires_grad=False, dtype=dtype
    ).cuda(0)
    o = torch.empty(
        (batch_size, heads, length, dim), requires_grad=False, dtype=dtype
    ).cuda(0)
    a = torch.empty(
        [batch_size, heads, length, kernel_size], device=q.device, dtype=dtype
    )
    d_a = torch.empty_like(a)
    d_q = torch.empty_like(q)
    d_k = torch.empty_like(k)
    d_v = torch.empty_like(v)
    if enable_bias:
        rpb = torch.randn(
            (heads, kernel_size * 2 - 1), requires_grad=False, dtype=dtype
        ).cuda(0)
        d_rpb = torch.zeros_like(rpb)
    else:
        rpb, d_rpb = None, None

    return profile_with_torch(
        na1d_qk,
        na1d_av,
        na1d_qk_d,
        na1d_av_d,
        q,
        k,
        v,
        a,
        o,
        d_a,
        d_q,
        d_k,
        d_v,
        rpb,
        d_rpb,
        kernel_size,
        dilation,
        warmup_steps,
    )


def profile_na1d(
    batch_size: int,
    heads: int,
    length: int,
    dim: int,
    kernel_size: int,
    dilation: int,
    dtype: Any,
    warmup_steps: int = 10,
    enable_bias: bool = False,
):
    profile_result = _profile_na1d_with_torch(
        batch_size=batch_size,
        heads=heads,
        length=length,
        dim=dim,
        kernel_size=kernel_size,
        dilation=dilation,
        dtype=dtype,
        warmup_steps=warmup_steps,
        enable_bias=enable_bias,
    )
    out = extract_na_ops(profile_result, _NA_1D_C_KEYWORDS)
    exp_num_ops = 7 if enable_bias else 6
    captured_num_ops = 0 if not out else len(out)
    i = 0
    while captured_num_ops != exp_num_ops and i < 50:
        profile_result = _profile_na1d_with_torch(
            batch_size=batch_size,
            heads=heads,
            length=length,
            dim=dim,
            kernel_size=kernel_size,
            dilation=dilation,
            dtype=dtype,
            warmup_steps=warmup_steps,
            enable_bias=enable_bias,
        )
        out = extract_na_ops(profile_result, _NA_1D_C_KEYWORDS)
        captured_num_ops = 0 if not out else len(out)
        i += 1
    assert out, f"Profiler keeps failing after {i} iters, exiting..."
    return out
