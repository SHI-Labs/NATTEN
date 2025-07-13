#################################################################################################
# Copyright (c) 2022-2025 Ali Hassani.
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

import time
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.profiler import profile as torch_profile, ProfilerActivity

from natten.functional import neighborhood_attention_generic
from natten.types import DimensionType, KernelSchedule
from natten.utils import log
from natten.utils.checks import check_all_args

from .formatting import convert_to_natten_profiler_ops, Result
from .ops import sdpa

from .problem import Problem

logger = log.get_logger(__name__)

IS_CUDA = torch.cuda.is_available()
torch_device = "cuda" if IS_CUDA else "cpu"
profiler_activity_tag = ProfilerActivity.CUDA if IS_CUDA else ProfilerActivity.CPU


def init_tensors(
    problem: Problem, flatten_sequence: bool, heads_last: bool
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tuple[Tensor, Tensor]]]:
    q, k, v, d_out = problem.make_qkvo_tensors(
        device=torch_device,
        requires_grad=False,
        heads_last=heads_last,
        flatten=flatten_sequence,
    )

    add_kv = None
    if problem.has_additional_kv:
        assert not flatten_sequence
        add_k, add_v = problem.make_additional_kv_tensors(
            device=torch_device, requires_grad=False, heads_last=heads_last
        )
        add_kv = (add_k, add_v)

    return q, k, v, d_out, add_kv


# Doesn't profile; uses cuda events and only reports a runtime
def measure_natten_runtime(
    problem: Problem,
    warmup_steps: int,
    backend: Optional[str] = None,
    fmha_backend: Optional[str] = None,
    disable_backward: bool = False,
    # Perf-related args
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    backward_q_tile_shape: Optional[DimensionType] = None,
    backward_kv_tile_shape: Optional[DimensionType] = None,
    backward_kv_splits: Optional[DimensionType] = None,
    backward_use_pt_reduction: bool = False,
    run_persistent_kernel: bool = True,
    kernel_schedule: Optional[Union[str, KernelSchedule]] = None,
    torch_compile: bool = False,
    # FMHA args
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
    backward_q_tile_size: Optional[int] = None,
    backward_kv_tile_size: Optional[int] = None,
) -> float:

    window_size, stride, dilation, is_causal = check_all_args(
        problem.na_dim,
        problem.window_size,
        problem.stride,
        problem.dilation,
        problem.is_causal,
    )

    query, key, value, d_out, additional_kv = init_tensors(
        problem, flatten_sequence=False, heads_last=True
    )

    torch.set_grad_enabled(not disable_backward)

    def run_ops(
        query,
        key,
        value,
        d_out,
        window_size,
        stride,
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

        out = neighborhood_attention_generic(
            query,
            key,
            value,
            kernel_size=window_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            additional_keys=additional_keys,
            additional_values=additional_values,
            backend=backend,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            backward_q_tile_shape=backward_q_tile_shape,
            backward_kv_tile_shape=backward_kv_tile_shape,
            backward_kv_splits=backward_kv_splits,
            backward_use_pt_reduction=backward_use_pt_reduction,
            run_persistent_kernel=run_persistent_kernel,
            kernel_schedule=kernel_schedule,
            torch_compile=torch_compile,
            attention_kwargs={
                "backend": fmha_backend,
                "kernel_schedule": kernel_schedule,
                "q_tile_size": q_tile_size,
                "kv_tile_size": kv_tile_size,
                "backward_q_tile_size": backward_q_tile_size,
                "backward_kv_tile_size": backward_kv_tile_size,
            },
        )

        if not disable_backward:
            out.backward(d_out)

    for _ in range(warmup_steps):
        with torch.no_grad():
            q, k, v, do = (
                query.clone(),
                key.clone(),
                value.clone(),
                d_out.clone(),
            )
            add_kv = (
                None
                if additional_kv is None
                else (additional_kv[0].clone(), additional_kv[1].clone())
            )
        run_ops(
            q,
            k,
            v,
            do,
            window_size,
            stride,
            dilation,
            is_causal,
            disable_backward,
            add_kv,
        )

    with torch.no_grad():
        q, k, v, do = (
            query.clone(),
            key.clone(),
            value.clone(),
            d_out.clone(),
        )
        add_kv = (
            None
            if additional_kv is None
            else (additional_kv[0].clone(), additional_kv[1].clone())
        )

    if IS_CUDA:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.time()

    run_ops(
        q,
        k,
        v,
        do,
        window_size,
        stride,
        dilation,
        is_causal,
        disable_backward,
        add_kv,
    )

    if IS_CUDA:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
    else:
        elapsed_time_ms = (time.time() - start_time) * 1e3

    return elapsed_time_ms


def _profile_na_with_torch(
    problem: Problem,
    warmup_steps: int,
    backend: Optional[str] = None,
    fmha_backend: Optional[str] = None,
    disable_backward: bool = False,
    # Perf-related args
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    backward_q_tile_shape: Optional[DimensionType] = None,
    backward_kv_tile_shape: Optional[DimensionType] = None,
    backward_kv_splits: Optional[DimensionType] = None,
    backward_use_pt_reduction: bool = False,
    run_persistent_kernel: bool = True,
    kernel_schedule: Optional[Union[str, KernelSchedule]] = None,
    torch_compile: bool = False,
    # FMHA args
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
    backward_q_tile_size: Optional[int] = None,
    backward_kv_tile_size: Optional[int] = None,
) -> torch_profile:

    window_size, stride, dilation, is_causal = check_all_args(
        problem.na_dim,
        problem.window_size,
        problem.stride,
        problem.dilation,
        problem.is_causal,
    )

    query, key, value, d_out, additional_kv = init_tensors(
        problem, flatten_sequence=False, heads_last=True
    )

    torch.set_grad_enabled(not disable_backward)

    def run_ops(
        query,
        key,
        value,
        d_out,
        window_size,
        stride,
        dilation,
        is_causal,
        disable_backward,
        additional_kv,
    ):
        if IS_CUDA:
            torch.cuda.synchronize()
        query.requires_grad = not disable_backward
        key.requires_grad = not disable_backward
        value.requires_grad = not disable_backward
        additional_keys, additional_values = (
            (None, None) if additional_kv is None else additional_kv
        )
        if additional_kv is not None:
            additional_keys.requires_grad = not disable_backward
            additional_values.requires_grad = not disable_backward

        out = neighborhood_attention_generic(
            query,
            key,
            value,
            kernel_size=window_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            additional_keys=additional_keys,
            additional_values=additional_values,
            backend=backend,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            backward_q_tile_shape=backward_q_tile_shape,
            backward_kv_tile_shape=backward_kv_tile_shape,
            backward_kv_splits=backward_kv_splits,
            backward_use_pt_reduction=backward_use_pt_reduction,
            run_persistent_kernel=run_persistent_kernel,
            kernel_schedule=kernel_schedule,
            torch_compile=torch_compile,
            attention_kwargs={
                "backend": fmha_backend,
                "kernel_schedule": kernel_schedule,
                "q_tile_size": q_tile_size,
                "kv_tile_size": kv_tile_size,
                "backward_q_tile_size": backward_q_tile_size,
                "backward_kv_tile_size": backward_kv_tile_size,
            },
        )

        if IS_CUDA:
            torch.cuda.synchronize()
        if not disable_backward:
            out.backward(d_out)
            if IS_CUDA:
                torch.cuda.synchronize()

    for _ in range(warmup_steps):
        with torch.no_grad():
            q, k, v, do = (
                query.clone(),
                key.clone(),
                value.clone(),
                d_out.clone(),
            )
            add_kv = (
                None
                if additional_kv is None
                else (additional_kv[0].clone(), additional_kv[1].clone())
            )
        run_ops(
            q,
            k,
            v,
            do,
            window_size,
            stride,
            dilation,
            is_causal,
            disable_backward,
            add_kv,
        )

    with torch.no_grad():
        q, k, v, do = (
            query.clone(),
            key.clone(),
            value.clone(),
            d_out.clone(),
        )
        add_kv = (
            None
            if additional_kv is None
            else (additional_kv[0].clone(), additional_kv[1].clone())
        )

    with torch_profile(activities=[profiler_activity_tag]) as prof:
        run_ops(
            q,
            k,
            v,
            do,
            window_size,
            stride,
            dilation,
            is_causal,
            disable_backward,
            add_kv,
        )

    return prof


def profile_na_with_torch(
    problem: Problem,
    warmup_steps: int = 10,
    backend: Optional[str] = None,
    fmha_backend: Optional[str] = None,
    disable_backward: bool = False,
    # Perf-related args
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    backward_q_tile_shape: Optional[DimensionType] = None,
    backward_kv_tile_shape: Optional[DimensionType] = None,
    backward_kv_splits: Optional[DimensionType] = None,
    backward_use_pt_reduction: bool = False,
    run_persistent_kernel: bool = True,
    kernel_schedule: Optional[Union[str, KernelSchedule]] = None,
    torch_compile: bool = False,
    # Debug
    debug_report_prof_result: bool = False,
    # FMHA args
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
    backward_q_tile_size: Optional[int] = None,
    backward_kv_tile_size: Optional[int] = None,
) -> Optional[List[Result]]:
    profile_result = _profile_na_with_torch(
        problem,
        warmup_steps=warmup_steps,
        backend=backend,
        fmha_backend=fmha_backend,
        disable_backward=disable_backward,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
        backward_q_tile_shape=backward_q_tile_shape,
        backward_kv_tile_shape=backward_kv_tile_shape,
        backward_kv_splits=backward_kv_splits,
        backward_use_pt_reduction=backward_use_pt_reduction,
        run_persistent_kernel=run_persistent_kernel,
        kernel_schedule=kernel_schedule,
        torch_compile=torch_compile,
        q_tile_size=q_tile_size,
        kv_tile_size=kv_tile_size,
        backward_q_tile_size=backward_q_tile_size,
        backward_kv_tile_size=backward_kv_tile_size,
    )

    if debug_report_prof_result:
        logger.debug(profile_result.key_averages().table())

    out = convert_to_natten_profiler_ops(profile_result)
    assert out is not None

    return out


def _profile_fmha_with_torch(
    problem: Problem,
    warmup_steps: int,
    backend: Optional[str] = "cudnn",
    disable_backward: Optional[bool] = False,
) -> torch_profile:
    torch.set_grad_enabled(not disable_backward)

    assert (
        not problem.has_additional_kv
    ), "Profiling SDPA with additional KV is not supported."

    query, key, value, d_out, additional_kv = init_tensors(
        problem, flatten_sequence=True, heads_last=False  # torch SDPA is heads first :(
    )

    def run_ops(query, key, value, d_out, backend):
        query.requires_grad = not disable_backward
        key.requires_grad = not disable_backward
        value.requires_grad = not disable_backward
        out = sdpa(query, key, value, backend=backend)
        if not disable_backward:
            out.backward(d_out)

    logger.warning(
        "Running standard self attention; ignoring NA parameters window size, stride, dilation, causal, etc."
    )

    for _ in range(warmup_steps):
        run_ops(
            query.clone(),
            key.clone(),
            value.clone(),
            d_out.clone(),
            backend=backend,
        )

    with torch_profile(activities=[profiler_activity_tag]) as prof:
        run_ops(
            query.clone(),
            key.clone(),
            value.clone(),
            d_out.clone(),
            backend=backend,
        )

    return prof


def profile_fmha_with_torch(
    problem: Problem,
    warmup_steps: int = 10,
    backend: Optional[str] = "cudnn",
    disable_backward: Optional[bool] = False,
    # Debug
    debug_report_prof_result: bool = False,
) -> Optional[List[Result]]:
    profile_result = _profile_fmha_with_torch(
        problem,
        warmup_steps=warmup_steps,
        backend=backend,
        disable_backward=disable_backward,
    )

    if debug_report_prof_result:
        logger.debug(profile_result.key_averages().table())

    out = convert_to_natten_profiler_ops(profile_result)
    assert out is not None

    return out
