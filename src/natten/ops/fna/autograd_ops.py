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
import functools
from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch.amp import custom_bwd, custom_fwd
from torch.autograd import Function

amp_fwd = functools.partial(custom_fwd, device_type="cuda")
amp_bwd = functools.partial(custom_bwd, device_type="cuda")

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
    NoneType,
)
from ...utils import (
    check_all_args,
    check_backward_tiling_config,
    check_tiling_config,
    log,
)

logger = log.get_logger(__name__)


#################################################################################################
###################### Set up libnatten ops directly as autograd functions ######################
#################################################################################################


class FusedNeighborhoodAttention1D(Function):
    @staticmethod
    @amp_fwd
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        bias: Optional[Tensor],
        kernel_size_: Dimension1DTypeOrDed,
        dilation_: Dimension1DTypeOrDed,
        is_causal_: CausalArg1DTypeOrDed,
        scale: float,
        tiling_config_forward_: FnaForwardConfigType,
        tiling_config_backward_: FnaBackwardConfigType,
    ) -> Tensor:
        kernel_size, dilation, is_causal = check_all_args(
            1, kernel_size_, dilation_, is_causal_
        )
        assert isinstance(
            scale, float
        ), f"Expected float attention scale, got {type(scale)}."
        tiling_config_forward = check_tiling_config(1, tiling_config_forward_)
        tiling_config_backward = check_backward_tiling_config(
            1, tiling_config_backward_
        )

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
        logsumexp = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )

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
            *tiling_config_forward,
        )

        ctx.save_for_backward(query, key, value, logsumexp, output)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.tiling_config_forward = tiling_config_forward
        ctx.tiling_config_backward = tiling_config_backward
        ctx.has_bias = bias is not None

        return output

    @staticmethod
    def jvp(ctx, *grad_inputs: Any) -> Tensor:
        raise NotImplementedError(
            "Fused neighborhood attention does not support forward-mode AD yet."
        )

    @staticmethod
    @amp_bwd
    def backward(ctx, grad_out: Tensor) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
    ]:
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
            ctx.tiling_config_backward
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
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
            ctx.scale,
            q_tile_shape,
            k_tile_shape,
            kv_splits,
            compute_delta_with_pt,
        )

        return d_query, d_key, d_value, None, None, None, None, None, None, None


class FusedNeighborhoodAttention2D(Function):
    @staticmethod
    @amp_fwd
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        bias: Optional[Tensor],
        kernel_size_: Dimension2DTypeOrDed,
        dilation_: Dimension2DTypeOrDed,
        is_causal_: CausalArg2DTypeOrDed,
        scale: float,
        tiling_config_forward_: FnaForwardConfigType,
        tiling_config_backward_: FnaBackwardConfigType,
    ) -> Tensor:
        kernel_size, dilation, is_causal = check_all_args(
            2, kernel_size_, dilation_, is_causal_
        )
        assert isinstance(
            scale, float
        ), f"Expected float attention scale, got {type(scale)}."
        tiling_config_forward = check_tiling_config(2, tiling_config_forward_)
        tiling_config_backward = check_backward_tiling_config(
            2, tiling_config_backward_
        )

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
        logsumexp = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )

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
            tiling_config_forward[0],
            tiling_config_forward[1],
        )

        ctx.save_for_backward(query, key, value, logsumexp, output)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.tiling_config_forward = tiling_config_forward
        ctx.tiling_config_backward = tiling_config_backward
        ctx.has_bias = bias is not None

        return output

    @staticmethod
    def jvp(ctx, *grad_inputs: Any) -> Tensor:
        raise NotImplementedError(
            "Fused neighborhood attention does not support forward-mode AD yet."
        )

    @staticmethod
    @amp_bwd
    def backward(ctx, grad_out: Tensor) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
    ]:
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
            ctx.tiling_config_backward
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
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
            ctx.scale,
            q_tile_shape,
            k_tile_shape,
            kv_splits,
            compute_delta_with_pt,
        )

        return d_query, d_key, d_value, None, None, None, None, None, None, None


class FusedNeighborhoodAttention3D(Function):
    @staticmethod
    @amp_fwd
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        bias: Optional[Tensor],
        kernel_size_: Dimension3DTypeOrDed,
        dilation_: Dimension3DTypeOrDed,
        is_causal_: CausalArg3DTypeOrDed,
        scale: float,
        tiling_config_forward_: FnaForwardConfigType,
        tiling_config_backward_: FnaBackwardConfigType,
    ) -> Tensor:
        kernel_size, dilation, is_causal = check_all_args(
            3, kernel_size_, dilation_, is_causal_
        )
        assert isinstance(
            scale, float
        ), f"Expected float attention scale, got {type(scale)}."
        tiling_config_forward = check_tiling_config(3, tiling_config_forward_)
        tiling_config_backward = check_backward_tiling_config(
            3, tiling_config_backward_
        )

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
        logsumexp = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )

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
            *tiling_config_forward,
        )

        ctx.save_for_backward(query, key, value, logsumexp, output)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.tiling_config_forward = tiling_config_forward
        ctx.tiling_config_backward = tiling_config_backward
        ctx.has_bias = bias is not None

        return output

    @staticmethod
    def jvp(ctx, *grad_inputs: Any) -> Tensor:
        raise NotImplementedError(
            "Fused neighborhood attention does not support forward-mode AD yet."
        )

    @staticmethod
    @amp_bwd
    def backward(ctx, grad_out: Tensor) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
    ]:
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
            ctx.tiling_config_backward
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
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
            ctx.scale,
            q_tile_shape,
            k_tile_shape,
            kv_splits,
            compute_delta_with_pt,
        )

        return d_query, d_key, d_value, None, None, None, None, None, None, None


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

    return FusedNeighborhoodAttention1D.apply(
        query,
        key,
        value,
        bias,
        kernel_size_,
        dilation_,
        is_causal_,
        scale,
        tiling_config_forward,
        tiling_config_backward,
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

    return FusedNeighborhoodAttention2D.apply(
        query,
        key,
        value,
        bias,
        kernel_size_,
        dilation_,
        is_causal_,
        scale,
        tiling_config_forward,
        tiling_config_backward,
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

    return FusedNeighborhoodAttention3D.apply(
        query,
        key,
        value,
        bias,
        kernel_size_,
        dilation_,
        is_causal_,
        scale,
        tiling_config_forward,
        tiling_config_backward,
    )
