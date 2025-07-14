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

import argparse
import math
import time
from collections.abc import Sequence
from functools import partial
from typing import List, Optional, Union

import torch

from . import (
    allow_flex_compile,
    set_memory_usage_preference,
    use_kv_parallelism_in_fused_na,
)

from .profiling_utils import (
    generate_problem,
    print_table,
    profile_fmha_with_torch,
    profile_na_with_torch,
)
from .profiling_utils.dry_run import dry_run as dry_run_fn, optimize as optimize_fn
from .profiling_utils.formatting import Result
from .profiling_utils.problem import Problem
from .types import CausalArgType, DimensionType, KernelSchedule
from .utils import log

logger = log.get_logger(__name__)

__all__ = []  # type: ignore

SUPPORTED_DTYPES = ["fp32", "bf16", "fp16"]
NATTEN_BACKENDS = ["cutlass-fna", "blackwell-fna", "hopper-fna", "flex-fna"]
NATTEN_FMHA_BACKENDS = ["cutlass-fmha", "blackwell-fmha", "hopper-fmha", "flex-fmha"]
SDPA_BACKENDS = ["xformers", "cudnn", "fav2"]

SCHEDULE_MAP = {
    "non": KernelSchedule.NonPersistent,
    "coop": KernelSchedule.WarpSpecializedCooperative,
    "pp": KernelSchedule.WarpSpecializedPingpong,
}

DEFAULT_DTYPE = "fp16"
DEFAULT_BACKEND = "fna"


def do_profile(
    problem: Problem,
    backend: Optional[str] = None,
    fmha_backend: Optional[str] = None,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    backward_q_tile_shape: Optional[DimensionType] = None,
    backward_kv_tile_shape: Optional[DimensionType] = None,
    persistent: bool = True,
    torch_compile: bool = False,
    kernel_schedule: Optional[Union[str, KernelSchedule]] = None,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
    backward_q_tile_size: Optional[int] = None,
    backward_kv_tile_size: Optional[int] = None,
    warmup_steps: int = 5,
    backprop: bool = False,
    max_retries: int = 5,
    fail_on_capture_error: bool = True,
):
    if backend is None or backend in NATTEN_BACKENDS:
        func = partial(
            profile_na_with_torch,
            backend=backend,
            fmha_backend=fmha_backend,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            backward_q_tile_shape=backward_q_tile_shape,
            backward_kv_tile_shape=backward_kv_tile_shape,
            run_persistent_kernel=persistent,
            torch_compile=torch_compile,
            kernel_schedule=kernel_schedule,
            q_tile_size=q_tile_size,
            kv_tile_size=kv_tile_size,
            backward_q_tile_size=backward_q_tile_size,
            backward_kv_tile_size=backward_kv_tile_size,
        )

    elif backend in SDPA_BACKENDS:
        func = partial(profile_fmha_with_torch, backend=backend)

    else:
        raise NotImplementedError(f"Unrecognized backend {backend}.")

    logged_ops: Optional[List[Result]] = func(
        problem=problem,
        warmup_steps=warmup_steps,
        disable_backward=not backprop,
    )

    # Temp solution for an intermittent bug where trace is empty (kernel launch, device synch, and
    # all that stuff is still there, but for some reason no kernels.... I've seen this happen in
    # many profiling tools, so I think it may be a driver bug or something.)
    retries = 0
    while logged_ops is None or len(logged_ops) < 1:
        logger.debug(
            "Profiler trace was empty, retrying after device sync and short sleep."
        )
        torch.cuda.synchronize()
        time.sleep(1)
        torch.cuda.synchronize()

        retries += 1
        logger.debug(f"Retrying [attempt {retries} / {max_retries}] ...")

        logged_ops = func(
            problem=problem,
            warmup_steps=warmup_steps,
            disable_backward=not backprop,
            debug_report_prof_result=True,
        )

        if retries >= max_retries:
            break

    if fail_on_capture_error and (logged_ops is None or len(logged_ops) < 1):
        raise RuntimeError(
            f"Nothing was measured during profiling after {max_retries} retries. "
            "Please open an issue and copy paste the following information:\n"
            f"{problem=},\n"
            f"{backend=},\n"
            f"{fmha_backend=},\n"
            f"{q_tile_shape=},\n"
            f"{kv_tile_shape=},\n"
            f"{backward_q_tile_shape=},\n"
            f"{backward_kv_tile_shape=},\n"
            f"{persistent=},\n"
            f"{torch_compile=},\n"
            f"{kernel_schedule=},\n"
            f"{warmup_steps=},\n"
            f"{backprop=},\n"
            f"{logged_ops=},\n"
        )

    return logged_ops


def check_input_size(input_size: Sequence) -> DimensionType:
    if input_size is None:
        raise ValueError(f"Input size is a required argument, got {input_size}.")

    if len(input_size) < 1 or len(input_size) > 3:
        raise ValueError(
            f"Input size must be a tuple of either 1, 2 or 3 integers, got {len(input_size)}."
        )

    return tuple(x for x in input_size)


def check_window_size(
    input_size: Sequence, window_size: Optional[Sequence] = None
) -> DimensionType:
    assert isinstance(input_size, Sequence) and len(input_size) in [1, 2, 3]
    na_dim = len(input_size)

    if window_size is None:
        return tuple(x for x in input_size)  # type: ignore[return-value]

    if len(window_size) != na_dim:
        raise ValueError(
            f"Invalid window size {window_size} for input size {input_size}. "
            "They must match in number of elements."
        )

    for w, x in zip(window_size, input_size):
        if w > x:
            raise ValueError(
                f"Invalid window size {window_size} for input size {input_size}. "
                "Window size cannot be greater than input size along any dimension."
            )

    return tuple(w for w in window_size)


def check_dilation(
    input_size: Sequence, window_size: Sequence, dilation: Optional[Sequence] = None
) -> DimensionType:
    assert isinstance(input_size, Sequence) and len(input_size) in [1, 2, 3]
    assert isinstance(window_size, Sequence) and len(window_size) == len(input_size)
    na_dim = len(input_size)

    if dilation is None:
        return tuple(1 for _ in range(na_dim))  # type: ignore[return-value]

    if len(dilation) != na_dim:
        raise ValueError(
            f"Invalid dilation {dilation} for input size {input_size}. "
            "They must match in number of elements."
        )

    for d, w, x in zip(dilation, window_size, input_size):
        if d * w > x:
            raise ValueError(
                f"Invalid dilation {dilation} for input size {input_size} and window size {window_size}. "
                "The product of window size and dilation cannot be greater than input size along any dimension."
            )

    return tuple(d for d in dilation)


def check_stride(
    input_size: Sequence,
    window_size: Sequence,
    dilation: Sequence,
    stride: Optional[Sequence] = None,
) -> DimensionType:
    assert isinstance(input_size, Sequence) and len(input_size) in [1, 2, 3]
    assert isinstance(window_size, Sequence) and len(window_size) == len(input_size)
    assert isinstance(dilation, Sequence) and len(dilation) == len(input_size)
    na_dim = len(input_size)

    if stride is None:
        return tuple(1 for _ in range(na_dim))  # type: ignore[return-value]

    if len(stride) != na_dim:
        raise ValueError(
            f"Invalid stride {stride} for input size {input_size}. "
            "They must match in number of elements."
        )

    for s, d, w, x in zip(stride, dilation, window_size, input_size):
        if s > w:
            raise ValueError(
                f"Invalid stride {stride} for window size {window_size}. "
                "Stride cannot be larger than window size along any dimension."
            )

    return tuple(s for s in stride)


def check_causal(
    input_size: Sequence, causal: Optional[Sequence] = None
) -> CausalArgType:
    assert isinstance(input_size, Sequence) and len(input_size) in [1, 2, 3]
    na_dim = len(input_size)

    if causal is None:
        return tuple(False for _ in range(na_dim))  # type: ignore[return-value]

    return tuple(c for c in causal)


def get_args():
    parser = argparse.ArgumentParser(
        description="NATTEN profiling toolkit. Profile different scenarios with NATTEN and "
        "see accurate measurements of operation runtime under different backends and configurations."
    )

    parser.add_argument(
        "-b", "--batch-size", type=int, default=1, help="QKV batch size."
    )
    parser.add_argument(
        "-n", "--heads", type=int, default=1, help="QKV number of heads (no GQA/MQA)."
    )
    parser.add_argument(
        "-d", "--dim", type=int, default=64, help="QK (and optionally V) head dim."
    )
    parser.add_argument(
        "--dim-value",
        type=int,
        default=None,
        help="Head dim for the V tensor, if different from Q and K. Defaults to the value of "
        "`-d`/`--dim`.",
    )

    parser.add_argument(
        "-i",
        "--input-size",
        type=int,
        nargs="+",
        help="QKV token layout shape (i.e. sequence length in 1-D, "
        "height and width in 2-D, depth, height, and width in 3-D.)",
    )

    parser.add_argument(
        "-w",
        "--window-size",
        type=int,
        nargs="*",
        help="Neighborhood attention window size (shape), also referred to "
        "as kernel size. This must be a tuple with the same number of elements "
        "as in --input-size. Defaults to --input-size (self attention).",
    )

    parser.add_argument(
        "-s",
        "--stride",
        type=int,
        nargs="*",
        help="Neighborhood attention stride values. This must be a tuple with the "
        "same number of elements as in --input-size. Defaults to 1s (standard "
        "sliding window).",
    )

    parser.add_argument(
        "--dilation",
        type=int,
        nargs="*",
        help="Neighborhood attention dilation values. This must be a tuple with the "
        "same number of elements as in --input-size. Defaults to 1s (standard "
        "sliding window).",
    )

    parser.add_argument(
        "-c",
        "--causal",
        type=bool,
        nargs="*",
        help="Causal masking values. This must be a boolean tuple with the "
        "same number of elements as in --input-size. Defaults to all `False`s ("
        "bi-directional in all dimensions).",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default=DEFAULT_DTYPE,
        choices=SUPPORTED_DTYPES,
        help=f"Element (data) type. Choices: {', '.join(SUPPORTED_DTYPES)}",
    )

    parser.add_argument(
        "--backprop",
        action="store_true",
        help="Profile backward pass as well as forward pass.",
    )

    parser.add_argument(
        "--add-kv",
        type=int,
        default=0,
        help="Number of additional KV tokens. Defaults to 0.",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=NATTEN_BACKENDS + SDPA_BACKENDS,
        help="Backend / kernel to run."
        "Choices: "
        f"NATTEN backends: {', '.join(NATTEN_BACKENDS)}.\n"
        f"Torch SDPA backends (can only perform self attention): {', '.join(SDPA_BACKENDS)}.",
    )

    parser.add_argument(
        "--fmha-backend",
        type=str,
        default=None,
        choices=NATTEN_FMHA_BACKENDS,
        help="Backend / kernel for cross-attention (additional KV) and fast-path self attention in NATTEN. "
        "Choices: "
        f"{', '.join(NATTEN_FMHA_BACKENDS)}.",
    )

    parser.add_argument(
        "--q-tile",
        type=int,
        nargs="*",
        help="Q tile shape in the forward pass kernel (varies between different backends). Try "
        "--dry-run to see available backends and tile shapes for your use case.",
    )

    parser.add_argument(
        "--kv-tile",
        type=int,
        nargs="*",
        help="KV tile shape in the forward pass kernel (varies between different backends). Try "
        "--dry-run to see available backends and tile shapes for your use case.",
    )

    parser.add_argument(
        "--backward-q-tile",
        type=int,
        nargs="*",
        help="Q tile shape in the backward pass kernel (only respected by `cutlass-fna` backend). "
        "Try --dry-run to see available backends and tile shapes for your use case.",
    )

    parser.add_argument(
        "--backward-kv-tile",
        type=int,
        nargs="*",
        help="KV tile shape in the backward pass kernel (only respected by `cutlass-fna` backend). "
        "Try --dry-run to see available backends and tile shapes for your use case.",
    )

    parser.add_argument(
        "--schedule",
        type=str,
        default=None,
        choices=SCHEDULE_MAP.keys(),
        help="Kernel schedule (hopper-fna and hopper-fmha only). "
        "Choices: non (non-persistent), coop (cooperative), pp (ping-pong).",
    )

    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Use persistent scheduling in the kernel (blackwell-fna and blackwell-fmha only).",
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile Flex Attention block sparse mask and kernel (flex backend only).",
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of warmup steps. Defaults to 10.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display valid forward and backward pass configurations for this use case and exit.",
    )

    parser.add_argument(
        "--max-configs",
        type=int,
        default=10,
        help="Maximum number of tile configurations to display in dry run. Defaults to 10. If set "
        "to 0 shows all valid configurations.",
    )

    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Find the best configuration (and backend if unspecified) for your use case, by "
        "profiling all available choices, and selecting the fastest one.",
    )

    parser.add_argument(
        "--optimize-warmup-steps",
        type=int,
        default=5,
        help="Number of warmup steps for optimizer. Defaults to 5.",
    )

    args = parser.parse_args()

    args.dim_value = args.dim_value or args.dim

    args.input_size = check_input_size(args.input_size)
    args.window_size = check_window_size(
        input_size=args.input_size, window_size=args.window_size
    )
    args.dilation = check_dilation(
        input_size=args.input_size, window_size=args.window_size, dilation=args.dilation
    )
    args.stride = check_stride(
        input_size=args.input_size,
        window_size=args.window_size,
        dilation=args.dilation,
        stride=args.stride,
    )
    args.causal = check_causal(input_size=args.input_size, causal=args.causal)

    return args


def profile(
    batch_size: int,
    heads: int,
    input_size,
    dim: int,
    dim_value: int,
    window_size: DimensionType,
    stride: DimensionType,
    dilation: DimensionType,
    dtype: str,
    causal: CausalArgType,
    warmup_steps: int,
    backend: str,
    fmha_backend: str,
    backprop: bool,
    add_kv: int,
    q_tile: Optional[DimensionType],
    kv_tile: Optional[DimensionType],
    backward_q_tile: Optional[DimensionType],
    backward_kv_tile: Optional[DimensionType],
    persistent: bool,
    schedule: Optional[str],
    compile: bool,
    dry_run: bool,
    max_configs: int,
    optimize: bool,
    optimize_warmup_steps: int,
):
    if dry_run and optimize:
        raise ValueError("Dry run and optimize can't be run together.")

    if (q_tile is not None) ^ (kv_tile is not None):
        raise ValueError(
            "Q tile and KV tile must be set together, got " f"{q_tile=}, {kv_tile=}."
        )

    if (backward_q_tile is not None) ^ (backward_kv_tile is not None):
        raise ValueError(
            "Backward Q tile and KV tile must be set together, got "
            f"{backward_q_tile=}, {backward_kv_tile=}."
        )

    q_tile_size = None if q_tile is None else math.prod(q_tile)
    kv_tile_size = None if kv_tile is None else math.prod(kv_tile)
    backward_q_tile_size = (
        None if backward_q_tile is None else math.prod(backward_q_tile)
    )
    backward_kv_tile_size = (
        None if backward_kv_tile is None else math.prod(backward_kv_tile)
    )

    if dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        raise NotImplementedError(
            f"Data type {dtype} is not supported. Choices are " "fp16, bf16, fp32."
        )

    if compile:
        allow_flex_compile(True, True)

    use_kv_parallelism_in_fused_na(True)
    set_memory_usage_preference("unrestricted")

    problem = generate_problem(
        batch_size=batch_size,
        heads=heads,
        input_size=input_size,
        dim=dim,
        dim_value=dim_value,
        window_size=window_size,
        stride=stride,
        dilation=dilation,
        dtype=torch_dtype,
        is_causal=causal,
        additional_kv_length=add_kv,
    )

    if dry_run:
        dry_run_fn(
            problem,
            backend=backend,
            fmha_backend=fmha_backend,
            backprop=backprop,
            torch_compile=compile,
            max_configs=max_configs,
        )
        return

    if optimize:
        (
            opt_backend,
            opt_fmha_backend,
            opt_q_tile,
            opt_kv_tile,
            opt_backward_q_tile,
            opt_backward_kv_tile,
            opt_q_tile_size,
            opt_kv_tile_size,
            opt_backward_q_tile_size,
            opt_backward_kv_tile_size,
            opt_schedule,
        ) = optimize_fn(
            problem=problem,
            backend=backend,
            fmha_backend=fmha_backend,
            q_tile=q_tile,
            kv_tile=kv_tile,
            backward_q_tile=backward_q_tile,
            backward_kv_tile=backward_kv_tile,
            backprop=backprop,
            schedule=schedule,
            persistent=persistent,
            torch_compile=compile,
            warmup_steps=optimize_warmup_steps,
            q_tile_size=q_tile_size,
            kv_tile_size=kv_tile_size,
            backward_q_tile_size=backward_q_tile_size,
            backward_kv_tile_size=backward_kv_tile_size,
        )

        logged_ops = do_profile(
            problem=problem,
            backend=opt_backend,
            fmha_backend=opt_fmha_backend,
            q_tile_shape=opt_q_tile,
            kv_tile_shape=opt_kv_tile,
            backward_q_tile_shape=opt_backward_q_tile,
            backward_kv_tile_shape=opt_backward_kv_tile,
            persistent=persistent,
            torch_compile=compile,
            kernel_schedule=opt_schedule,
            warmup_steps=warmup_steps,
            backprop=backprop,
            q_tile_size=opt_q_tile_size,
            kv_tile_size=opt_kv_tile_size,
            backward_q_tile_size=opt_backward_q_tile_size,
            backward_kv_tile_size=opt_backward_kv_tile_size,
        )

    else:

        logged_ops = do_profile(
            problem=problem,
            backend=backend,
            fmha_backend=fmha_backend,
            q_tile_shape=q_tile,
            kv_tile_shape=kv_tile,
            backward_q_tile_shape=backward_q_tile,
            backward_kv_tile_shape=backward_kv_tile,
            persistent=persistent,
            torch_compile=compile,
            kernel_schedule=schedule,
            warmup_steps=warmup_steps,
            backprop=backprop,
            q_tile_size=q_tile_size,
            kv_tile_size=kv_tile_size,
            backward_q_tile_size=backward_q_tile_size,
            backward_kv_tile_size=backward_kv_tile_size,
        )

    title = "Profiler results"
    headers = [
        "Framework",
        "Kernel category",
        "Arch",
        "Operation",
        "# calls",
        "Runtime",
    ]
    assert all(isinstance(x, Result) for x in logged_ops), f"{logged_ops}"
    values = [
        [
            x.framework,
            x.kernel_type,
            x.tag,
            x.op_name_formatted,
            str(x.num_calls),
            x.time_str,
        ]
        for x in logged_ops
    ]
    total = sum(logged_ops)
    assert isinstance(total, Result)
    values.append(["", "", "", "Total", "", total.time_str])
    print_table(title, headers, values, has_footer=True)


if __name__ == "__main__":
    args = get_args()
    profile(**vars(args))
