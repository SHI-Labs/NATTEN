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

from typing import Optional

import torch
from torch import Tensor

from natten._libnatten.cxx_handles import (  # type: ignore[import-untyped]
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

################################################################################
################################### FMHA ops ###################################
################################################################################


# blackwell_fmha_forward
@torch.library.custom_op(
    "natten::blackwell_fmha_forward", mutates_args=("output",), device_types="cuda"
)
def blackwell_fmha_forward_torch_op(
    output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    logsumexp: Optional[Tensor],
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    run_persistent_kernel: bool,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
) -> None:
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


@torch.library.register_fake("natten::blackwell_fmha_forward")
def blackwell_fmha_forward_torch_fake_op(
    output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    logsumexp: Optional[Tensor],
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    run_persistent_kernel: bool,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
):
    pass


# blackwell_fmha_backward
@torch.library.custom_op(
    "natten::blackwell_fmha_backward",
    mutates_args=("d_query", "d_key", "d_value"),
    device_types="cuda",
)
def blackwell_fmha_backward_torch_op(
    d_query: Tensor,
    d_key: Tensor,
    d_value: Tensor,
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
) -> None:
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


@torch.library.register_fake("natten::blackwell_fmha_backward")
def blackwell_fmha_backward_torch_fake_op(
    d_query: Tensor,
    d_key: Tensor,
    d_value: Tensor,
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
):
    pass


# hopper_fmha_forward
@torch.library.custom_op(
    "natten::hopper_fmha_forward", mutates_args=("output",), device_types="cuda"
)
def hopper_fmha_forward_torch_op(
    output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    logsumexp: Optional[Tensor],
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    kernel_schedule_int: int,
) -> None:
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


@torch.library.register_fake("natten::hopper_fmha_forward")
def hopper_fmha_forward_torch_fake_op(
    output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    logsumexp: Optional[Tensor],
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    kernel_schedule_int: int,
):
    pass


# hopper_fmha_backward
@torch.library.custom_op(
    "natten::hopper_fmha_backward",
    mutates_args=("d_query", "d_key", "d_value"),
    device_types="cuda",
)
def hopper_fmha_backward_torch_op(
    d_query: Tensor,
    d_key: Tensor,
    d_value: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    d_output: Tensor,
    logsumexp: Tensor,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
) -> None:
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


@torch.library.register_fake("natten::hopper_fmha_backward")
def hopper_fmha_backward_torch_fake_op(
    d_query: Tensor,
    d_key: Tensor,
    d_value: Tensor,
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
    pass


# fmha_forward
@torch.library.custom_op(
    "natten::fmha_forward", mutates_args=("output",), device_types="cuda"
)
def fmha_forward_torch_op(
    output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    logsumexp: Optional[Tensor],
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
) -> None:
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


@torch.library.register_fake("natten::fmha_forward")
def fmha_forward_torch_fake_op(
    output: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    logsumexp: Optional[Tensor],
    is_causal: bool,
    scale: float,
    q_tile_size: int,
    kv_tile_size: int,
    cumulative_seqlen_Q: Optional[Tensor],
    cumulative_seqlen_KV: Optional[Tensor],
    max_seqlen_Q: int,
    max_seqlen_KV: int,
):
    pass


# fmha_backward
@torch.library.custom_op(
    "natten::fmha_backward",
    mutates_args=("d_query", "d_key", "d_value"),
    device_types="cuda",
)
def fmha_backward_torch_op(
    d_query: Tensor,
    d_key: Tensor,
    d_value: Tensor,
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
) -> None:
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


@torch.library.register_fake("natten::fmha_backward")
def fmha_backward_torch_fake_op(
    d_query: Tensor,
    d_key: Tensor,
    d_value: Tensor,
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
):
    pass


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
    @torch.library.custom_op(
        f"natten::blackwell_na{na_dim}d_forward",
        mutates_args=("output",),
        device_types="cuda",
    )
    def blackwell_fna_forward_torch_op(
        output: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logsumexp: Optional[Tensor],
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
    ) -> None:
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

    @torch.library.register_fake(f"natten::blackwell_na{na_dim}d_forward")
    def blackwell_fna_forward_torch_fake_op(
        output: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logsumexp: Optional[Tensor],
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
    ) -> None:
        pass

    # blackwell_na*d_backward
    @torch.library.custom_op(
        f"natten::blackwell_na{na_dim}d_backward",
        mutates_args=("d_query", "d_key", "d_value"),
        device_types="cuda",
    )
    def blackwell_fna_backward_torch_op(
        d_query: Tensor,
        d_key: Tensor,
        d_value: Tensor,
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
    ) -> None:
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

    @torch.library.register_fake(f"natten::blackwell_na{na_dim}d_backward")
    def blackwell_fna_backward_torch_fake_op(
        d_query: Tensor,
        d_key: Tensor,
        d_value: Tensor,
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
    ) -> None:
        pass

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
    @torch.library.custom_op(
        f"natten::hopper_na{na_dim}d_forward",
        mutates_args=("output",),
        device_types="cuda",
    )
    def hopper_fna_forward_torch_op(
        output: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logsumexp: Optional[Tensor],
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
    ) -> None:
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

    @torch.library.register_fake(f"natten::hopper_na{na_dim}d_forward")
    def hopper_fna_forward_torch_fake_op(
        output: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logsumexp: Optional[Tensor],
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
    ) -> None:
        pass

    # hopper_na*d_backward
    @torch.library.custom_op(
        f"natten::hopper_na{na_dim}d_backward",
        mutates_args=("d_query", "d_key", "d_value"),
        device_types="cuda",
    )
    def hopper_fna_backward_torch_op(
        d_query: Tensor,
        d_key: Tensor,
        d_value: Tensor,
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
    ) -> None:
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

    @torch.library.register_fake(f"natten::hopper_na{na_dim}d_backward")
    def hopper_fna_backward_torch_fake_op(
        d_query: Tensor,
        d_key: Tensor,
        d_value: Tensor,
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
    ) -> None:
        pass

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
    @torch.library.custom_op(
        f"natten::na{na_dim}d_forward", mutates_args=("output",), device_types="cuda"
    )
    def fna_forward_torch_op(
        output: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logsumexp: Optional[Tensor],
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        q_tile_shape: list[int],
        kv_tile_shape: list[int],
    ) -> None:
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

    @torch.library.register_fake(f"natten::na{na_dim}d_forward")
    def fna_forward_torch_fake_op(
        output: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logsumexp: Optional[Tensor],
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        q_tile_shape: list[int],
        kv_tile_shape: list[int],
    ) -> None:
        pass

    # na*d_backward
    @torch.library.custom_op(
        f"natten::na{na_dim}d_backward",
        mutates_args=("d_query", "d_key", "d_value"),
        device_types="cuda",
    )
    def fna_backward_torch_op(
        d_query: Tensor,
        d_key: Tensor,
        d_value: Tensor,
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
    ) -> None:
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

    @torch.library.register_fake(f"natten::na{na_dim}d_backward")
    def fna_backward_torch_fake_op(
        d_query: Tensor,
        d_key: Tensor,
        d_value: Tensor,
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
    ) -> None:
        pass

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
    @torch.library.custom_op(
        f"natten::reference_na{na_dim}d_forward",
        mutates_args=("output",),
        device_types="cuda",
    )
    def reference_fna_forward_torch_op(
        output: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logsumexp: Optional[Tensor],
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        qkv_shape: list[int],
        num_extra_kv: int,
    ) -> None:
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

    @torch.library.register_fake(f"natten::reference_na{na_dim}d_forward")
    def reference_fna_forward_torch_fake_op(
        output: Tensor,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logsumexp: Optional[Tensor],
        kernel_size: list[int],
        stride: list[int],
        dilation: list[int],
        is_causal: list[bool],
        scale: float,
        qkv_shape: list[int],
        num_extra_kv: int,
    ) -> None:
        pass

    # na*d_backward
    @torch.library.custom_op(
        f"natten::reference_na{na_dim}d_backward",
        mutates_args=("d_query", "d_key", "d_value"),
        device_types="cuda",
    )
    def reference_fna_backward_torch_op(
        d_query: Tensor,
        d_key: Tensor,
        d_value: Tensor,
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
    ) -> None:
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

    @torch.library.register_fake(f"natten::reference_na{na_dim}d_backward")
    def reference_fna_backward_torch_fake_op(
        d_query: Tensor,
        d_key: Tensor,
        d_value: Tensor,
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
    ) -> None:
        pass

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
