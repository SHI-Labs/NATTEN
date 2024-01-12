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

from natten.functional import na2d_av, na2d_qk
from torch.profiler import profile as torch_profile

from .utils import extract_na_ops, profile_extra_tokens_with_torch, profile_with_torch

try:
    from natten.libnatten import (  # type: ignore
        na2d_av_backward as na2d_av_backward_op,
        na2d_av_forward as na2d_av_op,
        na2d_qk_backward as na2d_qk_backward_op,
        na2d_qk_forward as na2d_qk_op,
    )
except:
    from natten._C import (  # type: ignore
        natten2dav_backward as na2d_av_backward_op,
        natten2dav_forward as na2d_av_op,
        natten2dqkrpb_backward as na2d_qk_backward_op,
        natten2dqkrpb_forward as na2d_qk_op,
    )


_NA_2D_C_KEYWORDS = {
    "pn_keywords": ["NA2dPN", "OperatorE0", "gemm::PointwiseNeighborhood2D"],
    "nn_keywords": ["NA2dNN", "OperatorE1", "gemm::NeighborhoodNeighborhood2D"],
    "in_keywords": ["NA2dIN", "OperatorE2", "gemm::InverseNeighborhood2D"],
    "rpb_keywords": ["rpb_forward"],
    "rpbgrad_keywords": [
        "rpb_backward",
        "rpb_grad",
        "natten2drpb_cuda_backward",
        "na2d_rpb_cuda_backward",
        "rel_pos_bias_gradient_2d",
        "naive::RelPosBiasGradient2D",
    ],
    "qkrpb_keywords": ["na2d_qkrpb_cuda_forward", "natten2dqkrpb_cuda_forward"],
    "av_keywords": ["na2d_av_cuda_forward", "natten2dav_cuda_forward"],
    "qgrad_keywords": ["na2d_q_cuda_backward", "natten2dq_cuda_backward"],
    "kgrad_keywords": ["na2d_k_cuda_backward", "natten2dk_cuda_backward"],
    "vgrad_keywords": ["na2d_v_cuda_backward", "natten2dv_cuda_backward"],
    "agrad_keywords": ["na2d_a_cuda_backward", "natten2da_cuda_backward"],
    "legacy_pn_keywords": [
        "pointwise_neighborhood_2d",
        "naive::PointwiseNeighborhood2D",
    ],
    "legacy_nn_keywords": [
        "neighborhood_neighborhood_2d",
        "naive::NeighborhoodNeighborhood2D",
    ],
    "legacy_in_keywords": ["inverse_neighborhood_2d", "naive::InverseNeighborhood2D"],
}


def _profile_na2d_with_torch(
    batch_size: int,
    heads: int,
    height: int,
    width: int,
    dim: int,
    kernel_size: int,
    dilation: int,
    dtype: Any,
    warmup_steps: int = 10,
    enable_bias: bool = False,
) -> torch_profile:
    q = torch.randn(
        (batch_size, heads, height, width, dim), requires_grad=False, dtype=dtype
    ).cuda(0)
    k = torch.randn(
        (batch_size, heads, height, width, dim), requires_grad=False, dtype=dtype
    ).cuda(0)
    v = torch.randn(
        (batch_size, heads, height, width, dim), requires_grad=False, dtype=dtype
    ).cuda(0)
    o = torch.empty(
        (batch_size, heads, height, width, dim), requires_grad=False, dtype=dtype
    ).cuda(0)
    a = torch.empty(
        [batch_size, heads, height, width, kernel_size**2],
        device=q.device,
        dtype=dtype,
    )
    d_a = torch.empty_like(a)
    d_q = torch.empty_like(q)
    d_k = torch.empty_like(k)
    d_v = torch.empty_like(v)
    if enable_bias:
        rpb = torch.randn(
            (heads, kernel_size * 2 - 1, kernel_size * 2 - 1),
            requires_grad=False,
            dtype=dtype,
        ).cuda(0)
        d_rpb = torch.zeros_like(rpb)
    else:
        rpb, d_rpb = None, None

    return profile_with_torch(
        na2d_qk_op,
        na2d_av_op,
        na2d_qk_backward_op,
        na2d_av_backward_op,
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


def profile_na2d(
    batch_size: int,
    heads: int,
    height: int,
    width: int,
    dim: int,
    kernel_size: int,
    dilation: int,
    dtype: Any,
    warmup_steps: int = 10,
    enable_bias: bool = False,
):
    profile_result = _profile_na2d_with_torch(
        batch_size=batch_size,
        heads=heads,
        height=height,
        width=width,
        dim=dim,
        kernel_size=kernel_size,
        dilation=dilation,
        dtype=dtype,
        warmup_steps=warmup_steps,
        enable_bias=enable_bias,
    )
    out = extract_na_ops(profile_result, _NA_2D_C_KEYWORDS)
    exp_num_ops = 7 if enable_bias else 6
    captured_num_ops = 0 if not out else len(out)
    i = 0
    while captured_num_ops < exp_num_ops and i < 50:
        profile_result = _profile_na2d_with_torch(
            batch_size=batch_size,
            heads=heads,
            height=height,
            width=width,
            dim=dim,
            kernel_size=kernel_size,
            dilation=dilation,
            dtype=dtype,
            warmup_steps=warmup_steps,
            enable_bias=enable_bias,
        )
        out = extract_na_ops(profile_result, _NA_2D_C_KEYWORDS)
        captured_num_ops = 0 if not out else len(out)
        i += 1
    assert out, f"Profiler keeps failing after {i} iters, exiting..."
    return out


def _profile_na2d_extra_tokens_with_torch(
    batch_size: int,
    heads: int,
    height: int,
    width: int,
    dim: int,
    kernel_size: int,
    dilation: int,
    num_extra_tokens: int,
    dtype: Any,
    warmup_steps: int = 10,
    disable_concat_fusion: bool = False,
    broadcast_batch: bool = False,
) -> torch_profile:
    q = torch.randn(
        (batch_size, heads, height, width, dim), requires_grad=False, dtype=dtype
    ).cuda(0)
    k = torch.randn(
        (batch_size, heads, height, width, dim), requires_grad=False, dtype=dtype
    ).cuda(0)
    v = torch.randn(
        (batch_size, heads, height, width, dim), requires_grad=False, dtype=dtype
    ).cuda(0)
    k_ext = torch.randn(
        (1 if broadcast_batch else batch_size, heads, num_extra_tokens, dim),
        requires_grad=False,
        dtype=dtype,
    ).cuda(0)
    v_ext = torch.randn(
        (1 if broadcast_batch else batch_size, heads, num_extra_tokens, dim),
        requires_grad=False,
        dtype=dtype,
    ).cuda(0)
    if broadcast_batch:
        k_ext = k_ext.expand(batch_size, heads, num_extra_tokens, dim)
        v_ext = v_ext.expand(batch_size, heads, num_extra_tokens, dim)

    return profile_extra_tokens_with_torch(
        na2d_qk,
        na2d_av,
        q,
        k,
        v,
        k_ext,
        v_ext,
        kernel_size,
        dilation,
        warmup_steps=warmup_steps,
        disable_concat_fusion=disable_concat_fusion,
    )


def profile_na2d_extra_tokens(
    batch_size: int,
    heads: int,
    height: int,
    width: int,
    dim: int,
    kernel_size: int,
    dilation: int,
    num_extra_tokens: int,
    dtype: Any,
    warmup_steps: int = 10,
    disable_concat_fusion: bool = False,
    broadcast_batch: bool = False,
):
    profile_result = _profile_na2d_extra_tokens_with_torch(
        batch_size=batch_size,
        heads=heads,
        height=height,
        width=width,
        dim=dim,
        kernel_size=kernel_size,
        dilation=dilation,
        num_extra_tokens=num_extra_tokens,
        dtype=dtype,
        warmup_steps=warmup_steps,
        disable_concat_fusion=disable_concat_fusion,
        broadcast_batch=broadcast_batch,
    )
    out = extract_na_ops(profile_result, _NA_2D_C_KEYWORDS)
    exp_num_ops = 3
    captured_num_ops = 0 if not out else len(out)
    i = 0
    while captured_num_ops < exp_num_ops and i < 50:
        profile_result = _profile_na2d_extra_tokens_with_torch(
            batch_size=batch_size,
            heads=heads,
            height=height,
            width=width,
            dim=dim,
            kernel_size=kernel_size,
            dilation=dilation,
            num_extra_tokens=num_extra_tokens,
            dtype=dtype,
            warmup_steps=warmup_steps,
            disable_concat_fusion=disable_concat_fusion,
            broadcast_batch=broadcast_batch,
        )
        out = extract_na_ops(profile_result, _NA_2D_C_KEYWORDS)
        captured_num_ops = 0 if not out else len(out)
        i += 1
    assert out, f"Profiler keeps failing after {i} iters, exiting..."
    return out
