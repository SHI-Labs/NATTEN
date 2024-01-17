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

from typing import Optional, Tuple

import torch

from natten.autotuner import autotune_fna
from natten.utils import check_all_args
from torch import Tensor
from torch.profiler import profile as torch_profile, ProfilerActivity, record_function

from .formatting import extract_na_ops
from .mappings import fav2, fmha, get_ops

from .problem import Problem


def init_qkv_tensors(problem: Problem) -> Tuple[Tensor, Tensor, Tensor]:
    q = torch.randn(
        problem.get_flattened_tensor_shape(True),
        device="cuda",
        dtype=problem.dtype,
        requires_grad=False,
    )
    k, v = torch.randn_like(q), torch.randn_like(q)
    return q, k, v


def init_tensors(
    problem: Problem, fuse: bool
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    q = torch.randn(
        problem.get_tensor_shape(fuse),
        device="cuda",
        dtype=problem.dtype,
        requires_grad=False,
    )
    attn = None
    bias = None
    if not fuse:
        # TODO: pass true when layout is changed
        attn = torch.randn(
            problem.get_attn_tensor_shape(False),
            device="cuda",
            dtype=problem.dtype,
            requires_grad=False,
        )
    k, v, o = torch.randn_like(q), torch.randn_like(q), torch.randn_like(q)
    if problem.has_bias:
        bias = torch.randn(
            problem.get_bias_shape(),
            device="cuda",
            dtype=problem.dtype,
            requires_grad=False,
        )
    return q, k, v, o, attn, bias


def init_grad_tensors(
    problem: Problem, fuse: bool
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    d_q = torch.randn(
        problem.get_tensor_shape(fuse),
        device="cuda",
        dtype=problem.dtype,
        requires_grad=False,
    )
    d_attn = None
    d_bias = None
    if not fuse:
        # TODO: pass true when layout is changed
        d_attn = torch.randn(
            problem.get_attn_tensor_shape(False),
            device="cuda",
            dtype=problem.dtype,
            requires_grad=False,
        )
    d_k, d_v = torch.randn_like(d_q), torch.randn_like(d_q)
    if problem.has_bias:
        d_bias = torch.randn(
            problem.get_bias_shape(),
            device="cuda",
            dtype=problem.dtype,
            requires_grad=False,
        )
    return d_q, d_k, d_v, d_attn, d_bias


def _profile_na_with_torch(
    problem: Problem,
    warmup_steps: int,
    fuse: bool = False,
    disable_backward: bool = False,
) -> torch_profile:
    qk_op, qk_backward_op, av_op, av_backward_op, fused_op, fused_backward_op = get_ops(
        problem.na_dim
    )
    kernel_size, dilation, is_causal = check_all_args(
        problem.na_dim, problem.kernel_size, problem.dilation, problem.is_causal
    )

    query, key, value, out, attn, bias = init_tensors(problem, fuse)
    d_query, d_key, d_value, d_attn, d_bias = init_grad_tensors(problem, fuse)

    if fuse:
        tiling_config = autotune_fna(
            problem.na_dim, query, kernel_size, dilation, is_causal
        )
        assert fused_op is not None and callable(fused_op)

        def run_ops(
            query,
            key,
            value,
            attn,
            out,
            bias,
            d_query,
            d_key,
            d_value,
            d_attn,
            d_bias,
            kernel_size,
            dilation,
            is_causal,
            disable_backward,
        ):
            fused_op(
                out,
                query,
                key,
                value,
                bias,
                kernel_size,
                dilation,
                is_causal,
                1.0,
                *tiling_config,
            )
            if not disable_backward:
                raise NotImplementedError()

    else:

        def run_ops(
            query,
            key,
            value,
            attn,
            out,
            bias,
            d_query,
            d_key,
            d_value,
            d_attn,
            d_bias,
            kernel_size,
            dilation,
            is_causal,
            disable_backward,
        ):
            qk_op(attn, query, key, bias, kernel_size, dilation, is_causal)
            attn = attn.softmax(dim=-1)
            av_op(out, attn, value, kernel_size, dilation, is_causal)
            if not disable_backward:
                av_backward_op(
                    d_attn, d_value, out, attn, value, kernel_size, dilation, is_causal
                )
                qk_backward_op(
                    d_query,
                    d_key,
                    d_bias,
                    d_attn,
                    query,
                    key,
                    kernel_size,
                    dilation,
                    is_causal,
                )

    with torch.no_grad():
        for _ in range(warmup_steps):
            run_ops(
                query,
                key,
                value,
                attn,
                out,
                bias,
                d_query,
                d_key,
                d_value,
                d_attn,
                d_bias,
                kernel_size,
                dilation,
                is_causal,
                disable_backward,
            )

        with torch_profile(
            activities=[ProfilerActivity.CUDA], record_shapes=True
        ) as prof:
            with record_function("model_inference"):
                run_ops(
                    query,
                    key,
                    value,
                    attn,
                    out,
                    bias,
                    d_query,
                    d_key,
                    d_value,
                    d_attn,
                    d_bias,
                    kernel_size,
                    dilation,
                    is_causal,
                    disable_backward,
                )

    return prof


def profile_na_with_torch(
    problem: Problem,
    warmup_steps: int = 10,
    fuse: bool = False,
    disable_backward: bool = False,
):
    profile_result = _profile_na_with_torch(
        problem,
        warmup_steps=warmup_steps,
        fuse=fuse,
        disable_backward=disable_backward,
    )
    out = extract_na_ops(profile_result, problem.na_dim)
    assert out is not None

    if not disable_backward:
        exp_num_ops = (8 if problem.has_bias else 7) if not fuse else 2
    else:
        exp_num_ops = 3 if not fuse else 1

    captured_num_ops = len(out)
    i = 0
    while captured_num_ops != exp_num_ops and i < 50:
        _ll = [f"{x.op_str}: {x.time_str}" for x in out]
        print(
            f"Not captured; trying again... {i=} {captured_num_ops=}, {exp_num_ops=}, {_ll}"
        )
        profile_result = _profile_na_with_torch(
            problem=problem,
            warmup_steps=warmup_steps,
            fuse=fuse,
            disable_backward=disable_backward,
        )
        out = extract_na_ops(profile_result, problem.na_dim)
        assert out is not None
        captured_num_ops = len(out)
        i += 1
    assert out, f"Profiler keeps failing after {i} iters, exiting..."
    return out


def _profile_fmha_with_torch(
    problem: Problem,
    warmup_steps: int,
    xformers: Optional[bool] = False,
    disable_backward: Optional[bool] = False,
) -> torch_profile:
    if not disable_backward:
        raise NotImplementedError()
    op = fmha if xformers else fav2
    query, key, value = init_qkv_tensors(problem)

    def run_ops(query, key, value):
        return op(query, key, value)

    with torch.no_grad():
        for _ in range(warmup_steps):
            run_ops(query, key, value)

        with torch_profile(
            activities=[ProfilerActivity.CUDA], record_shapes=True
        ) as prof:
            with record_function("model_inference"):
                run_ops(query, key, value)

    return prof


def profile_fmha_with_torch(
    problem: Problem,
    warmup_steps: int = 10,
    xformers: bool = False,
    disable_backward: Optional[bool] = False,
):
    profile_result = _profile_fmha_with_torch(
        problem,
        warmup_steps=warmup_steps,
        xformers=xformers,
        disable_backward=disable_backward,
    )
    out = extract_na_ops(profile_result, problem.na_dim)
    assert out is not None

    exp_num_ops = 1 if disable_backward else 2

    captured_num_ops = len(out)
    i = 0
    while captured_num_ops < exp_num_ops and i < 50:
        _ll = [f"{x.op_str}: {x.time_str}" for x in out]
        print(
            f"Not captured; trying again... {i=} {captured_num_ops=}, {exp_num_ops=}, {_ll}"
        )
        profile_result = _profile_fmha_with_torch(
            problem=problem,
            warmup_steps=warmup_steps,
            xformers=xformers,
            disable_backward=disable_backward,
        )
        out = extract_na_ops(profile_result, problem.na_dim)
        assert out is not None
        captured_num_ops = len(out)
        i += 1
    assert out, f"Profiler keeps failing after {i} iters, exiting..."
    return out
