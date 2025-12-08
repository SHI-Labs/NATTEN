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

from typing import Optional, Tuple

import torch
from torch import Tensor

from natten.libnatten import (  # type: ignore[import-untyped]
    blackwell_fmha_backward as blackwell_fmha_backward_cxx,
    blackwell_fmha_forward as blackwell_fmha_forward_cxx,
    blackwell_na1d_backward as blackwell_na1d_backward_cxx,
    blackwell_na1d_forward as blackwell_na1d_forward_cxx,
    blackwell_na2d_backward as blackwell_na2d_backward_cxx,
    blackwell_na2d_forward as blackwell_na2d_forward_cxx,
    blackwell_na3d_backward as blackwell_na3d_backward_cxx,
    blackwell_na3d_forward as blackwell_na3d_forward_cxx,
    compute_delta as compute_delta_cxx,
    fmha_backward as fmha_backward_cxx,
    fmha_forward as fmha_forward_cxx,
    hopper_fmha_backward as hopper_fmha_backward_cxx,
    hopper_fmha_forward as hopper_fmha_forward_cxx,
    hopper_na1d_backward as hopper_na1d_backward_cxx,
    hopper_na1d_forward as hopper_na1d_forward_cxx,
    hopper_na2d_backward as hopper_na2d_backward_cxx,
    hopper_na2d_forward as hopper_na2d_forward_cxx,
    hopper_na3d_backward as hopper_na3d_backward_cxx,
    hopper_na3d_forward as hopper_na3d_forward_cxx,
    na1d_backward as na1d_backward_cxx,
    na1d_forward as na1d_forward_cxx,
    na2d_backward as na2d_backward_cxx,
    na2d_forward as na2d_forward_cxx,
    na3d_backward as na3d_backward_cxx,
    na3d_forward as na3d_forward_cxx,
    reference_na1d_backward as reference_na1d_backward_cxx,
    reference_na1d_forward as reference_na1d_forward_cxx,
    reference_na2d_backward as reference_na2d_backward_cxx,
    reference_na2d_forward as reference_na2d_forward_cxx,
    reference_na3d_backward as reference_na3d_backward_cxx,
    reference_na3d_forward as reference_na3d_forward_cxx,
)
from natten.utils.environment import DISABLE_TORCH_OPS


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


# https://github.com/Dao-AILab/flash-attention/blob/v2.7.4/flash_attn/flash_attn_interface.py#L65-L78
def disabled_register_op(
    name, fn=None, /, *, mutates_args, device_types=None, schema=None
):
    def wrap(func):
        return func

    if fn is None:
        return wrap
    return fn


def disabled_register_fake(op, fn=None, /, *, lib=None, _stacklevel=1):
    def wrap(func):
        return func

    if fn is None:
        return wrap
    return fn


if DISABLE_TORCH_OPS:
    register_op = disabled_register_op
    register_fake = disabled_register_fake
else:
    register_op = torch.library.custom_op
    register_fake = torch.library.register_fake

################################################################################
################################### FMHA ops ###################################
################################################################################


# blackwell_fmha_forward
@register_op("natten::blackwell_fmha_forward", mutates_args=(), device_types="cuda")
def blackwell_fmha_forward_torch_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    run_persistent_kernel: bool,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
) -> Tuple[Tensor, Tensor]:
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

    output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]

    # NOTE: always zero-init outputs when doing varlen for safety
    is_varlen = cumulative_seqlen_Q is not None
    init_fn = torch.zeros if is_varlen else torch.empty

    output = init_fn(
        output_shape, device=query.device, dtype=query.dtype
    )  # type: ignore[operator]
    logsumexp = init_fn(
        query.shape[:-1], dtype=torch.float32, device=query.device
    )  # type: ignore[operator]

    blackwell_fmha_forward_cxx(
        output,
        query,
        key,
        value,
        logsumexp,
        is_causal,
        scale,
        q_tile_size,
        kv_tile_size,
        run_persistent_kernel,
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        max_seqlen_Q,
        max_seqlen_KV,
    )

    return output, logsumexp


@register_fake("natten::blackwell_fmha_forward")
def blackwell_fmha_forward_torch_fake_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    run_persistent_kernel: bool,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
) -> Tuple[Tensor, Tensor]:
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

    output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
    output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

    logsumexp = torch.empty(query.shape[:-1], dtype=torch.float32, device=query.device)
    return output, logsumexp


# blackwell_fmha_backward
@register_op(
    "natten::blackwell_fmha_backward",
    mutates_args=(),
    device_types="cuda",
)
def blackwell_fmha_backward_torch_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    d_output: Tensor,
    logsumexp: Tensor,
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
    output, d_output, logsumexp = [
        maybe_contiguous(x) for x in (output, d_output, logsumexp)
    ]

    # NOTE: always zero-init outputs when doing varlen for safety
    is_varlen = cumulative_seqlen_Q is not None
    init_fn = torch.zeros_like if is_varlen else torch.empty_like

    d_query = init_fn(query)
    d_key = init_fn(key)
    d_value = init_fn(value)

    blackwell_fmha_backward_cxx(
        d_query,
        d_key,
        d_value,
        query,
        key,
        value,
        output,
        d_output,
        logsumexp,
        is_causal,
        scale,
        q_tile_size,
        kv_tile_size,
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        max_seqlen_Q,
        max_seqlen_KV,
    )

    return d_query, d_key, d_value


@register_fake("natten::blackwell_fmha_backward")
def blackwell_fmha_backward_torch_fake_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    d_output: Tensor,
    logsumexp: Tensor,
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
    output, d_output, logsumexp = [
        maybe_contiguous(x) for x in (output, d_output, logsumexp)
    ]

    d_query = torch.empty_like(query)
    d_key = torch.empty_like(key)
    d_value = torch.empty_like(value)
    return d_query, d_key, d_value


# hopper_fmha_forward
@register_op("natten::hopper_fmha_forward", mutates_args=(), device_types="cuda")
def hopper_fmha_forward_torch_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    kernel_schedule_int: int,
) -> Tuple[Tensor, Tensor]:
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

    output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
    output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

    logsumexp = torch.empty(query.shape[:-1], dtype=torch.float32, device=query.device)

    hopper_fmha_forward_cxx(
        output,
        query,
        key,
        value,
        logsumexp,
        scale,
        q_tile_size,
        kv_tile_size,
        kernel_schedule_int,
    )

    return output, logsumexp


@register_fake("natten::hopper_fmha_forward")
def hopper_fmha_forward_torch_fake_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    kernel_schedule_int: int,
):
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

    output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
    output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

    logsumexp = torch.empty(query.shape[:-1], dtype=torch.float32, device=query.device)

    return output, logsumexp


# hopper_fmha_backward
@register_op(
    "natten::hopper_fmha_backward",
    mutates_args=(),
    device_types="cuda",
)
def hopper_fmha_backward_torch_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    d_output: Tensor,
    logsumexp: Tensor,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
    output, d_output, logsumexp = [
        maybe_contiguous(x) for x in (output, d_output, logsumexp)
    ]

    d_query = torch.empty_like(query)
    d_key = torch.empty_like(key)
    d_value = torch.empty_like(value)

    hopper_fmha_backward_cxx(
        d_query,
        d_key,
        d_value,
        query,
        key,
        value,
        output,
        d_output,
        logsumexp,
        scale,
        q_tile_size,
        kv_tile_size,
    )

    return d_query, d_key, d_value


@register_fake("natten::hopper_fmha_backward")
def hopper_fmha_backward_torch_fake_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    d_output: Tensor,
    logsumexp: Tensor,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
):
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
    output, d_output, logsumexp = [
        maybe_contiguous(x) for x in (output, d_output, logsumexp)
    ]

    d_query = torch.empty_like(query)
    d_key = torch.empty_like(key)
    d_value = torch.empty_like(value)

    return d_query, d_key, d_value


# fmha_forward
@register_op(
    "natten::fmha_forward",
    mutates_args=(),
    device_types="cuda",
)
def fmha_forward_torch_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
) -> Tuple[Tensor, Tensor]:
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

    output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]

    # NOTE: always zero-init outputs when doing varlen for safety
    is_varlen = cumulative_seqlen_Q is not None
    init_fn = torch.zeros if is_varlen else torch.empty

    output = init_fn(
        output_shape, device=query.device, dtype=query.dtype
    )  # type: ignore[operator]
    logsumexp = init_fn(
        query.shape[:-1], dtype=torch.float32, device=query.device
    )  # type: ignore[operator]

    fmha_forward_cxx(
        output,
        query,
        key,
        value,
        logsumexp,
        is_causal,
        scale,
        q_tile_size,
        kv_tile_size,
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        max_seqlen_Q,
        max_seqlen_KV,
    )

    return output, logsumexp


@register_fake("natten::fmha_forward")
def fmha_forward_torch_fake_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
) -> Tuple[Tensor, Tensor]:
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

    output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
    output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

    logsumexp = torch.empty(query.shape[:-1], dtype=torch.float32, device=query.device)
    return output, logsumexp


# fmha_backward
@register_op(
    "natten::fmha_backward",
    mutates_args=(),
    device_types="cuda",
)
def fmha_backward_torch_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    d_output: Tensor,
    logsumexp: Tensor,
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    num_kv_splits: int,
    compute_delta_with_pt: bool,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
    output, d_output, logsumexp = [
        maybe_contiguous(x) for x in (output, d_output, logsumexp)
    ]

    # NOTE: always zero-init outputs when doing varlen for safety
    is_varlen = cumulative_seqlen_Q is not None
    init_fn = torch.zeros_like if is_varlen else torch.empty_like

    d_query = init_fn(query)
    d_key = init_fn(key)
    d_value = init_fn(value)

    fmha_backward_cxx(
        d_query,
        d_key,
        d_value,
        query,
        key,
        value,
        output,
        d_output,
        logsumexp,
        is_causal,
        scale,
        q_tile_size,
        kv_tile_size,
        num_kv_splits,
        compute_delta_with_pt,
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        max_seqlen_Q,
        max_seqlen_KV,
    )

    return d_query, d_key, d_value


@register_fake("natten::fmha_backward")
def fmha_backward_torch_fake_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    d_output: Tensor,
    logsumexp: Tensor,
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    num_kv_splits: int,
    compute_delta_with_pt: bool,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
    output, d_output, logsumexp = [
        maybe_contiguous(x) for x in (output, d_output, logsumexp)
    ]

    d_query = torch.empty_like(query)
    d_key = torch.empty_like(key)
    d_value = torch.empty_like(value)
    return d_query, d_key, d_value


################################################################################
################################### FNA ops  ###################################
################################################################################


def make_blackwell_fna_ops(na_dim):
    fwd_handle, bwd_handle = {
        1: (blackwell_na1d_forward_cxx, blackwell_na1d_backward_cxx),
        2: (blackwell_na2d_forward_cxx, blackwell_na2d_backward_cxx),
        3: (blackwell_na3d_forward_cxx, blackwell_na3d_backward_cxx),
    }[na_dim]

    # blackwell_na*d_forward
    @register_op(
        f"natten::blackwell_na{na_dim}d_forward",
        mutates_args=(),
        device_types="cuda",
    )
    def blackwell_fna_forward_torch_op(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        q_shape: list[int],
        kv_shape: list[int],
        qkv_shape: list[int],
        num_extra_kv: int,
        q_tile_shape: list[int],
        kv_tile_shape: list[int],
        run_persistent_kernel: bool,
    ) -> Tuple[Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

        output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
        output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

        logsumexp = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )

        fwd_handle(
            output,
            query,
            key,
            value,
            logsumexp,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            q_shape,
            kv_shape,
            qkv_shape,
            num_extra_kv,
            q_tile_shape,
            kv_tile_shape,
            run_persistent_kernel,
        )

        return output, logsumexp

    @register_fake(f"natten::blackwell_na{na_dim}d_forward")
    def blackwell_fna_forward_torch_fake_op(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        q_shape: list[int],
        kv_shape: list[int],
        qkv_shape: list[int],
        num_extra_kv: int,
        q_tile_shape: list[int],
        kv_tile_shape: list[int],
        run_persistent_kernel: bool,
    ) -> Tuple[Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

        output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
        output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

        logsumexp = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )

        return output, logsumexp

    # blackwell_na*d_backward
    @register_op(
        f"natten::blackwell_na{na_dim}d_backward",
        mutates_args=(),
        device_types="cuda",
    )
    def blackwell_fna_backward_torch_op(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        output: Tensor,
        d_output: Tensor,
        logsumexp: Tensor,
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        q_shape: list[int],
        kv_shape: list[int],
        qkv_shape: list[int],
        q_tile_shape: list[int],
        kv_tile_shape: list[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
        output, d_output, logsumexp = [
            maybe_contiguous(x) for x in (output, d_output, logsumexp)
        ]

        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        d_value = torch.empty_like(value)

        bwd_handle(
            d_query,
            d_key,
            d_value,
            query,
            key,
            value,
            output,
            d_output,
            logsumexp,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            q_shape,
            kv_shape,
            qkv_shape,
            q_tile_shape,
            kv_tile_shape,
        )

        return d_query, d_key, d_value

    @register_fake(f"natten::blackwell_na{na_dim}d_backward")
    def blackwell_fna_backward_torch_fake_op(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        output: Tensor,
        d_output: Tensor,
        logsumexp: Tensor,
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        q_shape: list[int],
        kv_shape: list[int],
        qkv_shape: list[int],
        q_tile_shape: list[int],
        kv_tile_shape: list[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
        output, d_output, logsumexp = [
            maybe_contiguous(x) for x in (output, d_output, logsumexp)
        ]

        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        d_value = torch.empty_like(value)

        return d_query, d_key, d_value

    return (
        blackwell_fna_forward_torch_op,
        blackwell_fna_forward_torch_fake_op,
        blackwell_fna_backward_torch_op,
        blackwell_fna_backward_torch_fake_op,
    )


def make_hopper_fna_ops(na_dim):
    fwd_handle, bwd_handle = {
        1: (hopper_na1d_forward_cxx, hopper_na1d_backward_cxx),
        2: (hopper_na2d_forward_cxx, hopper_na2d_backward_cxx),
        3: (hopper_na3d_forward_cxx, hopper_na3d_backward_cxx),
    }[na_dim]

    # hopper_na*d_forward
    @register_op(
        f"natten::hopper_na{na_dim}d_forward",
        mutates_args=(),
        device_types="cuda",
    )
    def hopper_fna_forward_torch_op(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        q_shape: list[int],
        kv_shape: list[int],
        qkv_shape: list[int],
        q_tile_shape: list[int],
        kv_tile_shape: list[int],
        kernel_schedule_int: int,
    ) -> Tuple[Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

        output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
        output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

        logsumexp = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )

        fwd_handle(
            output,
            query,
            key,
            value,
            logsumexp,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            q_shape,
            kv_shape,
            qkv_shape,
            q_tile_shape,
            kv_tile_shape,
            kernel_schedule_int,
        )

        return output, logsumexp

    @register_fake(f"natten::hopper_na{na_dim}d_forward")
    def hopper_fna_forward_torch_fake_op(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        q_shape: list[int],
        kv_shape: list[int],
        qkv_shape: list[int],
        q_tile_shape: list[int],
        kv_tile_shape: list[int],
        kernel_schedule_int: int,
    ) -> Tuple[Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

        output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
        output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

        logsumexp = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )

        return output, logsumexp

    # hopper_na*d_backward
    @register_op(
        f"natten::hopper_na{na_dim}d_backward",
        mutates_args=(),
        device_types="cuda",
    )
    def hopper_fna_backward_torch_op(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        output: Tensor,
        d_output: Tensor,
        logsumexp: Tensor,
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        q_shape: list[int],
        kv_shape: list[int],
        qkv_shape: list[int],
        q_tile_shape: list[int],
        kv_tile_shape: list[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
        output, d_output, logsumexp = [
            maybe_contiguous(x) for x in (output, d_output, logsumexp)
        ]

        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        d_value = torch.empty_like(value)

        bwd_handle(
            d_query,
            d_key,
            d_value,
            query,
            key,
            value,
            output,
            d_output,
            logsumexp,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            q_shape,
            kv_shape,
            qkv_shape,
            q_tile_shape,
            kv_tile_shape,
        )

        return d_query, d_key, d_value

    @register_fake(f"natten::hopper_na{na_dim}d_backward")
    def hopper_fna_backward_torch_fake_op(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        output: Tensor,
        d_output: Tensor,
        logsumexp: Tensor,
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        q_shape: list[int],
        kv_shape: list[int],
        qkv_shape: list[int],
        q_tile_shape: list[int],
        kv_tile_shape: list[int],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
        output, d_output, logsumexp = [
            maybe_contiguous(x) for x in (output, d_output, logsumexp)
        ]

        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        d_value = torch.empty_like(value)

        return d_query, d_key, d_value

    return (
        hopper_fna_forward_torch_op,
        hopper_fna_forward_torch_fake_op,
        hopper_fna_backward_torch_op,
        hopper_fna_backward_torch_fake_op,
    )


def make_fna_ops(na_dim):
    fwd_handle, bwd_handle = {
        1: (na1d_forward_cxx, na1d_backward_cxx),
        2: (na2d_forward_cxx, na2d_backward_cxx),
        3: (na3d_forward_cxx, na3d_backward_cxx),
    }[na_dim]

    # na*d_forward
    @register_op(
        f"natten::na{na_dim}d_forward",
        mutates_args=(),
        device_types="cuda",
    )
    def fna_forward_torch_op(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        q_tile_shape: list[int],
        kv_tile_shape: list[int],
    ) -> Tuple[Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

        output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
        output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

        logsumexp = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )

        fwd_handle(
            output,
            query,
            key,
            value,
            logsumexp,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            q_tile_shape,
            kv_tile_shape,
        )

        return output, logsumexp

    @register_fake(f"natten::na{na_dim}d_forward")
    def fna_forward_torch_fake_op(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        q_tile_shape: list[int],
        kv_tile_shape: list[int],
    ) -> Tuple[Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

        output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
        output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

        logsumexp = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )

        return output, logsumexp

    # na*d_backward
    @register_op(
        f"natten::na{na_dim}d_backward",
        mutates_args=(),
        device_types="cuda",
    )
    def fna_backward_torch_op(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        output: Tensor,
        d_output: Tensor,
        logsumexp: Tensor,
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        q_tile_shape: list[int],
        kv_tile_shape: list[int],
        num_kv_splits: list[int],
        compute_delta_with_pt: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
        output, d_output, logsumexp = [
            maybe_contiguous(x) for x in (output, d_output, logsumexp)
        ]

        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        d_value = torch.empty_like(value)

        bwd_handle(
            d_query,
            d_key,
            d_value,
            query,
            key,
            value,
            output,
            d_output,
            logsumexp,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            q_tile_shape,
            kv_tile_shape,
            num_kv_splits,
            compute_delta_with_pt,
        )

        return d_query, d_key, d_value

    @register_fake(f"natten::na{na_dim}d_backward")
    def fna_backward_torch_fake_op(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        output: Tensor,
        d_output: Tensor,
        logsumexp: Tensor,
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        q_tile_shape: list[int],
        kv_tile_shape: list[int],
        num_kv_splits: list[int],
        compute_delta_with_pt: bool,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
        output, d_output, logsumexp = [
            maybe_contiguous(x) for x in (output, d_output, logsumexp)
        ]

        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        d_value = torch.empty_like(value)

        return d_query, d_key, d_value

    return (
        fna_forward_torch_op,
        fna_forward_torch_fake_op,
        fna_backward_torch_op,
        fna_backward_torch_fake_op,
    )


def make_reference_fna_ops(na_dim):
    fwd_handle, bwd_handle = {
        1: (reference_na1d_forward_cxx, reference_na1d_backward_cxx),
        2: (reference_na2d_forward_cxx, reference_na2d_backward_cxx),
        3: (reference_na3d_forward_cxx, reference_na3d_backward_cxx),
    }[na_dim]

    # na*d_forward
    @register_op(
        f"natten::reference_na{na_dim}d_forward",
        mutates_args=(),
        device_types="cuda",
    )
    def reference_fna_forward_torch_op(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        qkv_shape: list[int],
        num_extra_kv: int,
    ) -> Tuple[Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

        output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
        output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

        logsumexp = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )

        fwd_handle(
            output,
            query,
            key,
            value,
            logsumexp,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            qkv_shape,
            num_extra_kv,
        )

        return output, logsumexp

    @register_fake(f"natten::reference_na{na_dim}d_forward")
    def reference_fna_forward_torch_fake_op(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        qkv_shape: list[int],
        num_extra_kv: int,
    ) -> Tuple[Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]

        output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
        output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

        logsumexp = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )

        return output, logsumexp

    # na*d_backward
    @register_op(
        f"natten::reference_na{na_dim}d_backward",
        mutates_args=(),
        device_types="cuda",
    )
    def reference_fna_backward_torch_op(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        output: Tensor,
        d_output: Tensor,
        logsumexp: Tensor,
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        qkv_shape: list[int],
        num_extra_kv: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
        output, d_output, logsumexp = [
            maybe_contiguous(x) for x in (output, d_output, logsumexp)
        ]

        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        d_value = torch.empty_like(value)

        bwd_handle(
            d_query,
            d_key,
            d_value,
            query,
            key,
            value,
            output,
            d_output,
            logsumexp,
            kernel_size,
            stride,
            dilation,
            is_causal,
            scale,
            qkv_shape,
            num_extra_kv,
        )

        return d_query, d_key, d_value

    @register_fake(f"natten::reference_na{na_dim}d_backward")
    def reference_fna_backward_torch_fake_op(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        output: Tensor,
        d_output: Tensor,
        logsumexp: Tensor,
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        qkv_shape: list[int],
        num_extra_kv: int,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        query, key, value = [maybe_contiguous(x) for x in (query, key, value)]
        output, d_output, logsumexp = [
            maybe_contiguous(x) for x in (output, d_output, logsumexp)
        ]

        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        d_value = torch.empty_like(value)

        return d_query, d_key, d_value

    return (
        reference_fna_forward_torch_op,
        reference_fna_forward_torch_fake_op,
        reference_fna_backward_torch_op,
        reference_fna_backward_torch_fake_op,
    )


(
    blackwell_na1d_forward_torch_op,
    blackwell_na1d_forward_torch_fake_op,
    blackwell_na1d_backward_torch_op,
    blackwell_na1d_backward_torch_fake_op,
) = make_blackwell_fna_ops(1)
(
    blackwell_na2d_forward_torch_op,
    blackwell_na2d_forward_torch_fake_op,
    blackwell_na2d_backward_torch_op,
    blackwell_na2d_backward_torch_fake_op,
) = make_blackwell_fna_ops(2)
(
    blackwell_na3d_forward_torch_op,
    blackwell_na3d_forward_torch_fake_op,
    blackwell_na3d_backward_torch_op,
    blackwell_na3d_backward_torch_fake_op,
) = make_blackwell_fna_ops(3)

(
    hopper_na1d_forward_torch_op,
    hopper_na1d_forward_torch_fake_op,
    hopper_na1d_backward_torch_op,
    hopper_na1d_backward_torch_fake_op,
) = make_hopper_fna_ops(1)
(
    hopper_na2d_forward_torch_op,
    hopper_na2d_forward_torch_fake_op,
    hopper_na2d_backward_torch_op,
    hopper_na2d_backward_torch_fake_op,
) = make_hopper_fna_ops(2)
(
    hopper_na3d_forward_torch_op,
    hopper_na3d_forward_torch_fake_op,
    hopper_na3d_backward_torch_op,
    hopper_na3d_backward_torch_fake_op,
) = make_hopper_fna_ops(3)

(
    na1d_forward_torch_op,
    na1d_forward_torch_fake_op,
    na1d_backward_torch_op,
    na1d_backward_torch_fake_op,
) = make_fna_ops(1)
(
    na2d_forward_torch_op,
    na2d_forward_torch_fake_op,
    na2d_backward_torch_op,
    na2d_backward_torch_fake_op,
) = make_fna_ops(2)
(
    na3d_forward_torch_op,
    na3d_forward_torch_fake_op,
    na3d_backward_torch_op,
    na3d_backward_torch_fake_op,
) = make_fna_ops(3)

(
    reference_na1d_forward_torch_op,
    reference_na1d_forward_torch_fake_op,
    reference_na1d_backward_torch_op,
    reference_na1d_backward_torch_fake_op,
) = make_reference_fna_ops(1)
(
    reference_na2d_forward_torch_op,
    reference_na2d_forward_torch_fake_op,
    reference_na2d_backward_torch_op,
    reference_na2d_backward_torch_fake_op,
) = make_reference_fna_ops(2)
(
    reference_na3d_forward_torch_op,
    reference_na3d_forward_torch_fake_op,
    reference_na3d_backward_torch_op,
    reference_na3d_backward_torch_fake_op,
) = make_reference_fna_ops(3)


if DISABLE_TORCH_OPS:
    # Torch wrapped handles
    blackwell_fmha_forward = blackwell_fmha_forward_torch_op
    blackwell_fmha_backward = blackwell_fmha_backward_torch_op

    hopper_fmha_forward = hopper_fmha_forward_torch_op
    hopper_fmha_backward = hopper_fmha_backward_torch_op

    fmha_forward = fmha_forward_torch_op
    fmha_backward = fmha_backward_torch_op

    blackwell_na1d_forward = blackwell_na1d_forward_torch_op
    blackwell_na1d_backward = blackwell_na1d_backward_torch_op

    blackwell_na2d_forward = blackwell_na2d_forward_torch_op
    blackwell_na2d_backward = blackwell_na2d_backward_torch_op

    blackwell_na3d_forward = blackwell_na3d_forward_torch_op
    blackwell_na3d_backward = blackwell_na3d_backward_torch_op

    hopper_na1d_forward = hopper_na1d_forward_torch_op
    hopper_na1d_backward = hopper_na1d_backward_torch_op

    hopper_na2d_forward = hopper_na2d_forward_torch_op
    hopper_na2d_backward = hopper_na2d_backward_torch_op

    hopper_na3d_forward = hopper_na3d_forward_torch_op
    hopper_na3d_backward = hopper_na3d_backward_torch_op

    na1d_forward = na1d_forward_torch_op
    na1d_backward = na1d_backward_torch_op

    na2d_forward = na2d_forward_torch_op
    na2d_backward = na2d_backward_torch_op

    na3d_forward = na3d_forward_torch_op
    na3d_backward = na3d_backward_torch_op

    reference_na1d_forward = reference_na1d_forward_torch_op
    reference_na1d_backward = reference_na1d_backward_torch_op

    reference_na2d_forward = reference_na2d_forward_torch_op
    reference_na2d_backward = reference_na2d_backward_torch_op

    reference_na3d_forward = reference_na3d_forward_torch_op
    reference_na3d_backward = reference_na3d_backward_torch_op

else:
    # Torch wrapped handles
    blackwell_fmha_forward = torch.ops.natten.blackwell_fmha_forward
    blackwell_fmha_backward = torch.ops.natten.blackwell_fmha_backward

    hopper_fmha_forward = torch.ops.natten.hopper_fmha_forward
    hopper_fmha_backward = torch.ops.natten.hopper_fmha_backward

    fmha_forward = torch.ops.natten.fmha_forward
    fmha_backward = torch.ops.natten.fmha_backward

    blackwell_na1d_forward = torch.ops.natten.blackwell_na1d_forward
    blackwell_na1d_backward = torch.ops.natten.blackwell_na1d_backward

    blackwell_na2d_forward = torch.ops.natten.blackwell_na2d_forward
    blackwell_na2d_backward = torch.ops.natten.blackwell_na2d_backward

    blackwell_na3d_forward = torch.ops.natten.blackwell_na3d_forward
    blackwell_na3d_backward = torch.ops.natten.blackwell_na3d_backward

    hopper_na1d_forward = torch.ops.natten.hopper_na1d_forward
    hopper_na1d_backward = torch.ops.natten.hopper_na1d_backward

    hopper_na2d_forward = torch.ops.natten.hopper_na2d_forward
    hopper_na2d_backward = torch.ops.natten.hopper_na2d_backward

    hopper_na3d_forward = torch.ops.natten.hopper_na3d_forward
    hopper_na3d_backward = torch.ops.natten.hopper_na3d_backward

    na1d_forward = torch.ops.natten.na1d_forward
    na1d_backward = torch.ops.natten.na1d_backward

    na2d_forward = torch.ops.natten.na2d_forward
    na2d_backward = torch.ops.natten.na2d_backward

    na3d_forward = torch.ops.natten.na3d_forward
    na3d_backward = torch.ops.natten.na3d_backward

    reference_na1d_forward = torch.ops.natten.reference_na1d_forward
    reference_na1d_backward = torch.ops.natten.reference_na1d_backward

    reference_na2d_forward = torch.ops.natten.reference_na2d_forward
    reference_na2d_backward = torch.ops.natten.reference_na2d_backward

    reference_na3d_forward = torch.ops.natten.reference_na3d_forward
    reference_na3d_backward = torch.ops.natten.reference_na3d_backward

# This is only used in unit tests, and not even auto-diffable
compute_delta = compute_delta_cxx


__all__ = [
    "blackwell_fmha_backward",
    "blackwell_fmha_forward",
    "blackwell_na1d_backward",
    "blackwell_na1d_forward",
    "blackwell_na2d_backward",
    "blackwell_na2d_forward",
    "blackwell_na3d_backward",
    "blackwell_na3d_forward",
    "compute_delta",
    "fmha_backward",
    "fmha_forward",
    "hopper_fmha_backward",
    "hopper_fmha_forward",
    "hopper_na1d_backward",
    "hopper_na1d_forward",
    "hopper_na2d_backward",
    "hopper_na2d_forward",
    "hopper_na3d_backward",
    "hopper_na3d_forward",
    "na1d_backward",
    "na1d_forward",
    "na2d_backward",
    "na2d_forward",
    "na3d_backward",
    "na3d_forward",
    "reference_na1d_backward",
    "reference_na1d_forward",
    "reference_na2d_backward",
    "reference_na2d_forward",
    "reference_na3d_backward",
    "reference_na3d_forward",
]
