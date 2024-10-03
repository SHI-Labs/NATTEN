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
from typing import Optional, Sequence

import torch
from torch import Tensor

try:
    from natten import libnatten  # type: ignore
except ImportError:
    raise ImportError(
        "Failed to import libnatten. "
        "This could be due to an invalid/incomplete install. "
        "Please make sure you built NATTEN correctly, or refer to "
        "https://shi-labs.com/natten for more information."
    )

from ...types import (
    CausalArg1DTypeOrDed,
    CausalArg2DTypeOrDed,
    CausalArg3DTypeOrDed,
    Dimension1DTypeOrDed,
    Dimension2DTypeOrDed,
    Dimension3DTypeOrDed,
    FnaBackwardConfigType,
    FnaForwardConfigType,
)
from ...utils import (
    check_all_args,
    check_backward_tiling_config,
    check_tiling_config,
    log,
)

logger = log.get_logger(__name__)


#################################################################################################
##################################### Register forward ops. #####################################
#################################################################################################


@torch.library.custom_op(
    "natten::na1d_forward_op", mutates_args=(), device_types=("cpu", "cuda")
)
def na1d_torch_library_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    logsumexp: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
    scale: float,
    q_tiler_: Sequence[int],
    kv_tiler_: Sequence[int],
    backward_q_tiler_: Sequence[int],
    backward_kv_tiler_: Sequence[int],
    backward_kv_splits_: Sequence[int],
    backward_compute_delta_with_pt_: bool,
) -> Tensor:
    kernel_size, dilation, is_causal = check_all_args(
        1, kernel_size_, dilation_, is_causal_
    )

    assert isinstance(
        scale, float
    ), f"Expected float attention scale, got {type(scale)}."

    (q_tiler, kv_tiler) = check_tiling_config(1, (q_tiler_, kv_tiler_))

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if bias is not None:
        bias = bias.to(query.dtype).contiguous()

    output = torch.empty_like(value)
    libnatten.na1d_forward(
        output,
        query,
        key,
        value,
        bias,
        logsumexp,
        kernel_size,
        dilation,
        is_causal,
        scale,
        q_tiler,
        kv_tiler,
    )

    return output


@torch.library.custom_op(
    "natten::na2d_forward_op", mutates_args=(), device_types=("cpu", "cuda")
)
def na2d_torch_library_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    logsumexp: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
    scale: float,
    q_tiler_: Sequence[int],
    kv_tiler_: Sequence[int],
    backward_q_tiler_: Sequence[int],
    backward_kv_tiler_: Sequence[int],
    backward_kv_splits_: Sequence[int],
    backward_compute_delta_with_pt_: bool,
) -> Tensor:
    kernel_size, dilation, is_causal = check_all_args(
        2, kernel_size_, dilation_, is_causal_
    )

    assert isinstance(
        scale, float
    ), f"Expected float attention scale, got {type(scale)}."

    (q_tiler, kv_tiler) = check_tiling_config(2, (q_tiler_, kv_tiler_))

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if bias is not None:
        bias = bias.to(query.dtype).contiguous()

    output = torch.empty_like(value)
    libnatten.na2d_forward(
        output,
        query,
        key,
        value,
        bias,
        logsumexp,
        kernel_size,
        dilation,
        is_causal,
        scale,
        q_tiler,
        kv_tiler,
    )

    return output


@torch.library.custom_op(
    "natten::na3d_forward_op", mutates_args=(), device_types=("cpu", "cuda")
)
def na3d_torch_library_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    logsumexp: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
    scale: float,
    q_tiler_: Sequence[int],
    kv_tiler_: Sequence[int],
    backward_q_tiler_: Sequence[int],
    backward_kv_tiler_: Sequence[int],
    backward_kv_splits_: Sequence[int],
    backward_compute_delta_with_pt_: bool,
) -> Tensor:
    kernel_size, dilation, is_causal = check_all_args(
        3, kernel_size_, dilation_, is_causal_
    )

    assert isinstance(
        scale, float
    ), f"Expected float attention scale, got {type(scale)}."

    (q_tiler, kv_tiler) = check_tiling_config(3, (q_tiler_, kv_tiler_))

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if bias is not None:
        bias = bias.to(query.dtype).contiguous()

    output = torch.empty_like(value)
    libnatten.na3d_forward(
        output,
        query,
        key,
        value,
        bias,
        logsumexp,
        kernel_size,
        dilation,
        is_causal,
        scale,
        q_tiler,
        kv_tiler,
    )

    return output


#################################################################################################
##################################### Register backward ops. ####################################
#################################################################################################


@torch.library.custom_op(
    "natten::na1d_backward_op",
    mutates_args={"d_query", "d_key", "d_value"},
    device_types=("cpu", "cuda"),
)
def na1d_backward_op(
    d_query: Tensor,
    d_key: Tensor,
    d_value: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    d_output: Tensor,
    logsumexp: Tensor,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
    scale: float,
    q_tiler: Sequence[int],
    kv_tiler: Sequence[int],
    kv_splits: Sequence[int],
    compute_delta_with_pt: bool,
) -> None:
    libnatten.na1d_backward(
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
        dilation,
        is_causal,
        scale,
        q_tiler,
        kv_tiler,
        kv_splits,
        compute_delta_with_pt,
    )


@torch.library.custom_op(
    "natten::na2d_backward_op",
    mutates_args={"d_query", "d_key", "d_value"},
    device_types=("cpu", "cuda"),
)
def na2d_backward_op(
    d_query: Tensor,
    d_key: Tensor,
    d_value: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    d_output: Tensor,
    logsumexp: Tensor,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
    scale: float,
    q_tiler: Sequence[int],
    kv_tiler: Sequence[int],
    kv_splits: Sequence[int],
    compute_delta_with_pt: bool,
) -> None:
    libnatten.na2d_backward(
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
        dilation,
        is_causal,
        scale,
        q_tiler,
        kv_tiler,
        kv_splits,
        compute_delta_with_pt,
    )


@torch.library.custom_op(
    "natten::na3d_backward_op",
    mutates_args={"d_query", "d_key", "d_value"},
    device_types=("cpu", "cuda"),
)
def na3d_backward_op(
    d_query: Tensor,
    d_key: Tensor,
    d_value: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    output: Tensor,
    d_output: Tensor,
    logsumexp: Tensor,
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
    scale: float,
    q_tiler: Sequence[int],
    kv_tiler: Sequence[int],
    kv_splits: Sequence[int],
    compute_delta_with_pt: bool,
) -> None:
    libnatten.na3d_backward(
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
        dilation,
        is_causal,
        scale,
        q_tiler,
        kv_tiler,
        kv_splits,
        compute_delta_with_pt,
    )


#################################################################################################
################## Register "fake" ops for forward op, since it's not inplace. ##################
#################################################################################################


@na1d_torch_library_op.register_fake
def na1d_op_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    logsumexp: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
    scale: float,
    q_tiler_: Sequence[int],
    kv_tiler_: Sequence[int],
    backward_q_tiler_: Sequence[int],
    backward_kv_tiler_: Sequence[int],
    backward_kv_splits_: Sequence[int],
    backward_compute_delta_with_pt_: bool,
) -> Tensor:
    output = torch.empty_like(value)
    return output


@na2d_torch_library_op.register_fake
def na2d_op_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    logsumexp: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
    scale: float,
    q_tiler_: Sequence[int],
    kv_tiler_: Sequence[int],
    backward_q_tiler_: Sequence[int],
    backward_kv_tiler_: Sequence[int],
    backward_kv_splits_: Sequence[int],
    backward_compute_delta_with_pt_: bool,
) -> Tensor:
    output = torch.empty_like(value)
    return output


@na3d_torch_library_op.register_fake
def na3d_op_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    logsumexp: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
    scale: float,
    q_tiler_: Sequence[int],
    kv_tiler_: Sequence[int],
    backward_q_tiler_: Sequence[int],
    backward_kv_tiler_: Sequence[int],
    backward_kv_splits_: Sequence[int],
    backward_compute_delta_with_pt_: bool,
) -> Tensor:
    output = torch.empty_like(value)
    return output


#################################################################################################
#################################### Autograd context setups ####################################
#################################################################################################


def na1d_setup_context(ctx, inputs, output):
    (
        query,
        key,
        value,
        bias,
        logsumexp,
        kernel_size_,
        dilation_,
        is_causal_,
        scale,
        q_tiler_,
        kv_tiler_,
        backward_q_tiler_,
        backward_kv_tiler_,
        backward_kv_splits_,
        backward_compute_delta_with_pt_,
    ) = inputs

    (
        backward_q_tiler,
        backward_kv_tiler,
        backward_kv_splits,
        backward_compute_delta_with_pt,
    ) = check_backward_tiling_config(
        1,
        (
            backward_q_tiler_,
            backward_kv_tiler_,
            backward_kv_splits_,
            backward_compute_delta_with_pt_,
        ),
    )

    kernel_size, dilation, is_causal = check_all_args(
        1, kernel_size_, dilation_, is_causal_
    )

    assert isinstance(
        scale, float
    ), f"Expected float attention scale, got {type(scale)}."

    (q_tiler, kv_tiler) = check_tiling_config(1, (q_tiler_, kv_tiler_))

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    ctx.save_for_backward(query, key, value, logsumexp, output)
    ctx.kernel_size = kernel_size
    ctx.dilation = dilation
    ctx.is_causal = is_causal
    ctx.scale = scale
    ctx.backward_q_tiler = backward_q_tiler
    ctx.backward_kv_tiler = backward_kv_tiler
    ctx.backward_kv_splits = backward_kv_splits
    ctx.backward_compute_delta_with_pt = backward_compute_delta_with_pt
    ctx.has_bias = bias is not None


def na2d_setup_context(ctx, inputs, output):
    (
        query,
        key,
        value,
        bias,
        logsumexp,
        kernel_size_,
        dilation_,
        is_causal_,
        scale,
        q_tiler_,
        kv_tiler_,
        backward_q_tiler_,
        backward_kv_tiler_,
        backward_kv_splits_,
        backward_compute_delta_with_pt_,
    ) = inputs

    (
        backward_q_tiler,
        backward_kv_tiler,
        backward_kv_splits,
        backward_compute_delta_with_pt,
    ) = check_backward_tiling_config(
        2,
        (
            backward_q_tiler_,
            backward_kv_tiler_,
            backward_kv_splits_,
            backward_compute_delta_with_pt_,
        ),
    )

    kernel_size, dilation, is_causal = check_all_args(
        2, kernel_size_, dilation_, is_causal_
    )

    assert isinstance(
        scale, float
    ), f"Expected float attention scale, got {type(scale)}."

    (q_tiler, kv_tiler) = check_tiling_config(2, (q_tiler_, kv_tiler_))

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    ctx.save_for_backward(query, key, value, logsumexp, output)
    ctx.kernel_size = kernel_size
    ctx.dilation = dilation
    ctx.is_causal = is_causal
    ctx.scale = scale
    ctx.backward_q_tiler = backward_q_tiler
    ctx.backward_kv_tiler = backward_kv_tiler
    ctx.backward_kv_splits = backward_kv_splits
    ctx.backward_compute_delta_with_pt = backward_compute_delta_with_pt
    ctx.has_bias = bias is not None


def na3d_setup_context(ctx, inputs, output):
    (
        query,
        key,
        value,
        bias,
        logsumexp,
        kernel_size_,
        dilation_,
        is_causal_,
        scale,
        q_tiler_,
        kv_tiler_,
        backward_q_tiler_,
        backward_kv_tiler_,
        backward_kv_splits_,
        backward_compute_delta_with_pt_,
    ) = inputs

    (
        backward_q_tiler,
        backward_kv_tiler,
        backward_kv_splits,
        backward_compute_delta_with_pt,
    ) = check_backward_tiling_config(
        3,
        (
            backward_q_tiler_,
            backward_kv_tiler_,
            backward_kv_splits_,
            backward_compute_delta_with_pt_,
        ),
    )

    kernel_size, dilation, is_causal = check_all_args(
        3, kernel_size_, dilation_, is_causal_
    )

    assert isinstance(
        scale, float
    ), f"Expected float attention scale, got {type(scale)}."

    (q_tiler, kv_tiler) = check_tiling_config(3, (q_tiler_, kv_tiler_))

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    ctx.save_for_backward(query, key, value, logsumexp, output)
    ctx.kernel_size = kernel_size
    ctx.dilation = dilation
    ctx.is_causal = is_causal
    ctx.scale = scale
    ctx.backward_q_tiler = backward_q_tiler
    ctx.backward_kv_tiler = backward_kv_tiler
    ctx.backward_kv_splits = backward_kv_splits
    ctx.backward_compute_delta_with_pt = backward_compute_delta_with_pt
    ctx.has_bias = bias is not None


#################################################################################################
################################## Setup autograd backward call #################################
#################################################################################################


def na1d_backward(ctx, grad_out):
    if ctx.has_bias:
        raise NotImplementedError(
            "Fused neighborhood attention does not support training with positional biases. "
            "This feature will likely never be supported."
        )

    query, key, value, logsumexp, output = ctx.saved_tensors
    d_output = grad_out.contiguous()
    d_query = torch.empty_like(query)
    d_key = torch.empty_like(key)
    d_value = torch.empty_like(value)

    q_tile_shape, k_tile_shape, kv_splits, compute_delta_with_pt = (
        ctx.backward_q_tiler,
        ctx.backward_kv_tiler,
        ctx.backward_kv_splits,
        ctx.backward_compute_delta_with_pt,
    )

    if (
        any([kv_split > 1 for kv_split in kv_splits])
        and torch.are_deterministic_algorithms_enabled()
    ):
        new_kv_splits = tuple(1 for _ in range(len(kv_splits)))
        logger.warning(
            "You enabled PyTorch's deterministic mode, but tried to train with FNA's KV "
            "parallelism, which is non-deterministic. "
            f"Overriding {kv_splits} to {new_kv_splits}."
        )

    na1d_backward_op(
        d_query,
        d_key,
        d_value,
        query,
        key,
        value,
        output,
        d_output,
        logsumexp,
        ctx.kernel_size,
        ctx.dilation,
        ctx.is_causal,
        ctx.scale,
        q_tile_shape,
        k_tile_shape,
        kv_splits,
        compute_delta_with_pt,
    )

    return (
        d_query,
        d_key,
        d_value,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def na2d_backward(ctx, grad_out):
    if ctx.has_bias:
        raise NotImplementedError(
            "Fused neighborhood attention does not support training with positional biases. "
            "This feature will likely never be supported."
        )

    query, key, value, logsumexp, output = ctx.saved_tensors
    d_output = grad_out.contiguous()
    d_query = torch.empty_like(query)
    d_key = torch.empty_like(key)
    d_value = torch.empty_like(value)

    q_tile_shape, k_tile_shape, kv_splits, compute_delta_with_pt = (
        ctx.backward_q_tiler,
        ctx.backward_kv_tiler,
        ctx.backward_kv_splits,
        ctx.backward_compute_delta_with_pt,
    )

    if (
        any([kv_split > 1 for kv_split in kv_splits])
        and torch.are_deterministic_algorithms_enabled()
    ):
        new_kv_splits = tuple(1 for _ in range(len(kv_splits)))
        logger.warning(
            "You enabled PyTorch's deterministic mode, but tried to train with FNA's KV "
            "parallelism, which is non-deterministic. "
            f"Overriding {kv_splits} to {new_kv_splits}."
        )

    na2d_backward_op(
        d_query,
        d_key,
        d_value,
        query,
        key,
        value,
        output,
        d_output,
        logsumexp,
        ctx.kernel_size,
        ctx.dilation,
        ctx.is_causal,
        ctx.scale,
        q_tile_shape,
        k_tile_shape,
        kv_splits,
        compute_delta_with_pt,
    )

    return (
        d_query,
        d_key,
        d_value,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def na3d_backward(ctx, grad_out):
    if ctx.has_bias:
        raise NotImplementedError(
            "Fused neighborhood attention does not support training with positional biases. "
            "This feature will likely never be supported."
        )

    query, key, value, logsumexp, output = ctx.saved_tensors
    d_output = grad_out.contiguous()
    d_query = torch.empty_like(query)
    d_key = torch.empty_like(key)
    d_value = torch.empty_like(value)

    q_tile_shape, k_tile_shape, kv_splits, compute_delta_with_pt = (
        ctx.backward_q_tiler,
        ctx.backward_kv_tiler,
        ctx.backward_kv_splits,
        ctx.backward_compute_delta_with_pt,
    )

    if (
        any([kv_split > 1 for kv_split in kv_splits])
        and torch.are_deterministic_algorithms_enabled()
    ):
        new_kv_splits = tuple(1 for _ in range(len(kv_splits)))
        logger.warning(
            "You enabled PyTorch's deterministic mode, but tried to train with FNA's KV "
            "parallelism, which is non-deterministic. "
            f"Overriding {kv_splits} to {new_kv_splits}."
        )

    na3d_backward_op(
        d_query,
        d_key,
        d_value,
        query,
        key,
        value,
        output,
        d_output,
        logsumexp,
        ctx.kernel_size,
        ctx.dilation,
        ctx.is_causal,
        ctx.scale,
        q_tile_shape,
        k_tile_shape,
        kv_splits,
        compute_delta_with_pt,
    )

    return (
        d_query,
        d_key,
        d_value,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


#################################################################################################
############################## Register ops as autograd functions ###############################
#################################################################################################

na1d_torch_library_op.register_autograd(na1d_backward, setup_context=na1d_setup_context)
na2d_torch_library_op.register_autograd(na2d_backward, setup_context=na2d_setup_context)
na3d_torch_library_op.register_autograd(na3d_backward, setup_context=na3d_setup_context)


#################################################################################################
######################################## User-facing APIs #######################################
#################################################################################################


def na1d_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    kernel_size: Dimension1DTypeOrDed,
    dilation: Dimension1DTypeOrDed,
    is_causal: Optional[CausalArg1DTypeOrDed],
    scale: float,
    tiling_config_forward: FnaForwardConfigType,
    tiling_config_backward: FnaBackwardConfigType,
):
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    kernel_size_, dilation_, is_causal_ = check_all_args(
        1, kernel_size, dilation, is_causal
    )

    # TODO: logsumexp should be conditional on autograd being on
    logsumexp = torch.empty(query.shape[:-1], dtype=torch.float32, device=query.device)
    return na1d_torch_library_op(
        query,
        key,
        value,
        bias,
        logsumexp,
        kernel_size,
        dilation,
        is_causal,
        scale,
        *tiling_config_forward,
        *tiling_config_backward,
    )


def na2d_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    kernel_size: Dimension2DTypeOrDed,
    dilation: Dimension2DTypeOrDed,
    is_causal: Optional[CausalArg2DTypeOrDed],
    scale: float,
    tiling_config_forward: FnaForwardConfigType,
    tiling_config_backward: FnaBackwardConfigType,
):
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    kernel_size_, dilation_, is_causal_ = check_all_args(
        2, kernel_size, dilation, is_causal
    )

    # TODO: logsumexp should be conditional on autograd being on
    logsumexp = torch.empty(query.shape[:-1], dtype=torch.float32, device=query.device)
    return na2d_torch_library_op(
        query,
        key,
        value,
        bias,
        logsumexp,
        kernel_size_,
        dilation_,
        is_causal_,
        scale,
        *tiling_config_forward,
        *tiling_config_backward,
    )


def na3d_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    kernel_size: Dimension3DTypeOrDed,
    dilation: Dimension3DTypeOrDed,
    is_causal: Optional[CausalArg3DTypeOrDed],
    scale: float,
    tiling_config_forward: FnaForwardConfigType,
    tiling_config_backward: FnaBackwardConfigType,
):
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    kernel_size_, dilation_, is_causal_ = check_all_args(
        3, kernel_size, dilation, is_causal
    )

    # TODO: logsumexp should be conditional on autograd being on
    logsumexp = torch.empty(query.shape[:-1], dtype=torch.float32, device=query.device)
    return na3d_torch_library_op(
        query,
        key,
        value,
        bias,
        logsumexp,
        kernel_size_,
        dilation_,
        is_causal_,
        scale,
        *tiling_config_forward,
        *tiling_config_backward,
    )
