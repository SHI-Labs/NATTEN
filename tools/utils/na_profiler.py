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

import math
from typing import Optional, Tuple

import torch

from natten.utils import check_all_args
from torch import Tensor
from torch.profiler import profile as torch_profile, ProfilerActivity, record_function

from .formatting import extract_na_ops
from .mappings import fav2, fmha, get_ops

from .problem import Problem


def init_tensors(
    problem: Problem, flatten_sequence: bool, heads_last: bool
) -> Tuple[
    Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tuple[Tensor, Tensor]]
]:
    q = torch.randn(
        (
            problem.get_flattened_tensor_shape(True)
            if flatten_sequence
            else problem.get_tensor_shape(heads_last)
        ),
        device="cuda",
        dtype=problem.dtype,
        requires_grad=False,
    )
    k, v, d_out = torch.randn_like(q), torch.randn_like(q), torch.randn_like(q)
    bias = None
    if problem.has_bias:
        bias = torch.randn(
            problem.get_bias_shape(),
            device="cuda",
            dtype=problem.dtype,
            requires_grad=False,
        )

    add_kv = None
    if problem.has_additional_kv:
        assert not flatten_sequence
        add_k = torch.randn(
            (problem.get_additional_kv_shape(heads_last)),
            device="cuda",
            dtype=problem.dtype,
            requires_grad=False,
        )
        add_v = torch.randn_like(add_k)
        add_kv = (add_k, add_v)

    return q, k, v, d_out, bias, add_kv


def _profile_na_with_torch(
    problem: Problem,
    warmup_steps: int,
    fuse: bool = False,
    disable_backward: bool = False,
) -> torch_profile:
    qk_op, av_op, fused_op = get_ops(problem.na_dim)
    kernel_size, dilation, is_causal = check_all_args(
        problem.na_dim, problem.kernel_size, problem.dilation, problem.is_causal
    )

    query, key, value, d_out, bias, additional_kv = init_tensors(
        problem, flatten_sequence=False, heads_last=fuse
    )

    if bias is not None and fuse and not disable_backward:
        raise NotImplementedError("FNA does not support bias in backward pass.")

    if fuse:

        def run_ops(
            query,
            key,
            value,
            bias,
            d_out,
            kernel_size,
            dilation,
            is_causal,
            disable_backward,
            additional_kv,
        ):
            query.requires_grad = not disable_backward
            key.requires_grad = not disable_backward
            value.requires_grad = not disable_backward
            additional_keys, additional_values = (
                (None, None) if additional_kv is None else additional_kv
            )
            if additional_kv is not None:
                additional_keys.requires_grad = not disable_backward
                additional_values.requires_grad = not disable_backward
            out = fused_op(
                query,
                key,
                value,
                kernel_size=kernel_size,
                dilation=dilation,
                is_causal=is_causal,
                rpb=bias,
                additional_keys=additional_keys,
                additional_values=additional_values,
            )
            if not disable_backward:
                out.backward(d_out)

    else:

        def run_ops(
            query,
            key,
            value,
            bias,
            d_out,
            kernel_size,
            dilation,
            is_causal,
            disable_backward,
            additional_kv,
        ):
            query.requires_grad = not disable_backward
            key.requires_grad = not disable_backward
            value.requires_grad = not disable_backward
            if bias is not None:
                bias.requires_grad = not disable_backward

            additional_keys, additional_values = (
                (None, None) if additional_kv is None else additional_kv
            )
            if additional_kv is not None:
                additional_keys.requires_grad = not disable_backward
                additional_values.requires_grad = not disable_backward
            attn = qk_op(
                query,
                key,
                kernel_size=kernel_size,
                dilation=dilation,
                is_causal=is_causal,
                rpb=bias,
                additional_keys=additional_keys,
            )
            # NOTE: not multiplying by attention scale, since can be trivially fused
            # into any kernel or op.
            attn = attn.softmax(dim=-1)
            out = av_op(
                attn,
                value,
                kernel_size=kernel_size,
                dilation=dilation,
                is_causal=is_causal,
                additional_values=additional_values,
            )
            if not disable_backward:
                out.backward(d_out)

    for _ in range(warmup_steps):
        with torch.no_grad():
            q, k, v, do, rpb = (
                query.clone(),
                key.clone(),
                value.clone(),
                d_out.clone(),
                None if bias is None else bias.clone(),
            )
            add_kv = (
                None
                if additional_kv is None
                else (additional_kv[0].clone(), additional_kv[1].clone())
            )
        run_ops(
            q, k, v, rpb, do, kernel_size, dilation, is_causal, disable_backward, add_kv
        )

    with torch.no_grad():
        q, k, v, do, rpb = (
            query.clone(),
            key.clone(),
            value.clone(),
            d_out.clone(),
            None if bias is None else bias.clone(),
        )
        add_kv = (
            None
            if additional_kv is None
            else (additional_kv[0].clone(), additional_kv[1].clone())
        )

    with torch_profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            run_ops(
                q,
                k,
                v,
                rpb,
                do,
                kernel_size,
                dilation,
                is_causal,
                disable_backward,
                add_kv,
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
    while captured_num_ops < exp_num_ops and i < 50:
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
    op = fmha if xformers else fav2
    assert (
        not problem.has_additional_kv
    ), "Profiling FMHA/FA with additional KV is not supported."
    query, key, value, d_out, bias, additional_kv = init_tensors(
        problem, flatten_sequence=True, heads_last=True
    )

    if bias is not None:
        raise NotImplementedError("Profiling FMHA/FAv2 with bias is not supported.")

    def run_ops(query, key, value, d_out, window_size=None):
        query.requires_grad = not disable_backward
        key.requires_grad = not disable_backward
        value.requires_grad = not disable_backward
        out = op(query, key, value, window_size=window_size)
        if not disable_backward:
            out.backward(d_out)

    window_size_ = math.prod(problem.kernel_size)
    window_size = None if window_size_ < 1 else window_size_
    print(f"Running fused attention baseline with {window_size=}")
    for _ in range(warmup_steps):
        run_ops(
            query.clone(),
            key.clone(),
            value.clone(),
            d_out.clone(),
            window_size=window_size,
        )

    with torch_profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            run_ops(
                query.clone(),
                key.clone(),
                value.clone(),
                d_out.clone(),
                window_size=window_size,
            )

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
