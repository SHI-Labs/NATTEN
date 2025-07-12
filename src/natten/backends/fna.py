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
import functools
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from torch.amp import custom_bwd, custom_fwd
from torch.autograd import Function

amp_fwd = functools.partial(custom_fwd, device_type="cuda")
amp_bwd = functools.partial(custom_bwd, device_type="cuda")

from .._libnatten import (
    na1d_backward,
    na1d_forward,
    na2d_backward,
    na2d_forward,
    na3d_backward,
    na3d_forward,
)
from ..types import (
    CausalArg1DTypeOrDed,
    CausalArg2DTypeOrDed,
    CausalArg3DTypeOrDed,
    CausalArgType,
    CausalArgTypeOrDed,
    CutlassFnaBackwardConfigType,
    CutlassFnaForwardConfigType,
    Dimension1DType,
    Dimension1DTypeOrDed,
    Dimension2DType,
    Dimension2DTypeOrDed,
    Dimension3DType,
    Dimension3DTypeOrDed,
    DimensionType,
    DimensionTypeOrDed,
    NoneType,
)
from ..utils import log
from ..utils.checks import check_all_args, check_args_against_input, na_tensor_checks

from .configs.checks import can_run_cutlass_fna
from .configs.cutlass import (
    check_cutlass_fna_backward_config,
    check_cutlass_fna_forward_config,
)

logger = log.get_logger(__name__)


def make_cutlass_fna_autograd_fn(na_dim):
    assert na_dim in [1, 2, 3]

    FORWARD_OPS = {
        1: na1d_forward,
        2: na2d_forward,
        3: na3d_forward,
    }

    BACKWARD_OPS = {
        1: na1d_backward,
        2: na2d_backward,
        3: na3d_backward,
    }

    class CutlassFnaGenericAutogradFn(Function):
        @staticmethod
        @amp_fwd
        def forward(
            ctx,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            kernel_size: DimensionType,
            stride: DimensionType,
            dilation: DimensionType,
            is_causal: CausalArgType,
            scale: float,
            forward_config: CutlassFnaForwardConfigType,
            backward_config: CutlassFnaBackwardConfigType,
        ) -> Tuple[Tensor, Tensor]:
            kernel_size, stride, dilation, is_causal = check_all_args(
                na_dim, kernel_size, stride, dilation, is_causal
            )

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            assert query.dim() == value.dim() == 3 + na_dim
            assert query.shape[0] == value.shape[0]
            assert query.shape[-2] == value.shape[-2]

            output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
            output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

            # TODO: logsumexp should be conditional
            logsumexp = torch.empty(
                query.shape[:-1], dtype=torch.float32, device=query.device
            )

            q_tile_shape, kv_tile_shape = forward_config
            FORWARD_OPS[na_dim](
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

            ctx.save_for_backward(query, key, value, logsumexp, output)
            ctx.kernel_size = kernel_size
            ctx.stride = stride
            ctx.dilation = dilation
            ctx.is_causal = is_causal
            ctx.scale = scale
            ctx.backward_config = backward_config

            return output, logsumexp

        @staticmethod
        @amp_bwd
        def backward(ctx, grad_out: Tensor, grad_lse: Tensor) -> Tuple[
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
            query, key, value, logsumexp, output = ctx.saved_tensors
            d_output = grad_out.contiguous()
            d_query = torch.empty_like(query)
            d_key = torch.empty_like(key)
            d_value = torch.empty_like(value)

            q_tile_shape, k_tile_shape, kv_splits, compute_delta_with_pt = (
                ctx.backward_config
            )

            num_kv_splits = kv_splits
            if (
                any([kv_split > 1 for kv_split in kv_splits])
                and torch.are_deterministic_algorithms_enabled()
            ):
                num_kv_splits = tuple(1 for _ in range(len(kv_splits)))
                logger.warning(
                    "You enabled PyTorch's deterministic mode, but tried to train with FNA's KV "
                    "parallelism, which is non-deterministic. "
                    f"Overriding {kv_splits} to {num_kv_splits}."
                )

            BACKWARD_OPS[na_dim](
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
                ctx.stride,
                ctx.dilation,
                ctx.is_causal,
                ctx.scale,
                q_tile_shape,
                k_tile_shape,
                num_kv_splits,
                compute_delta_with_pt,
            )

            return d_query, d_key, d_value, None, None, None, None, None, None, None

    return CutlassFnaGenericAutogradFn


CutlassFna1DAutogradFn = make_cutlass_fna_autograd_fn(1)
CutlassFna2DAutogradFn = make_cutlass_fna_autograd_fn(2)
CutlassFna3DAutogradFn = make_cutlass_fna_autograd_fn(3)


CutlassFNAAutogradFns = {
    1: CutlassFna1DAutogradFn,
    2: CutlassFna2DAutogradFn,
    3: CutlassFna3DAutogradFn,
}


def cutlass_fna_generic(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: DimensionTypeOrDed,
    stride: DimensionTypeOrDed = 1,
    dilation: DimensionTypeOrDed = 1,
    is_causal: Optional[CausalArgTypeOrDed] = False,
    scale: Optional[float] = None,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    backward_q_tile_shape: Optional[DimensionType] = None,
    backward_kv_tile_shape: Optional[DimensionType] = None,
    backward_kv_splits: Optional[DimensionType] = None,
    backward_use_pt_reduction: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:

    na_tensor_checks(query, key, value, must_match_head_dims=False)

    assert can_run_cutlass_fna(query, key, value, raise_error=True)

    na_dim = query.dim() - 3  # batch, heads, head_dim

    assert na_dim in [1, 2, 3]

    kernel_size, stride, dilation, is_causal = check_all_args(
        na_dim, kernel_size, stride, dilation, is_causal
    )

    check_args_against_input(
        query,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
    )

    forward_config = check_cutlass_fna_forward_config(
        input_tensor=query if value.shape[-1] <= query.shape[-1] else value,
        dilation=dilation,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
    )

    backward_config = check_cutlass_fna_backward_config(
        input_tensor=key if value.shape[-1] <= key.shape[-1] else value,
        dilation=dilation,
        q_tile_shape=backward_q_tile_shape,
        kv_tile_shape=backward_kv_tile_shape,
        kv_splits=backward_kv_splits,
        use_pt_reduction=backward_use_pt_reduction,
    )

    scale = scale or query.shape[-1] ** -0.5

    output, lse = CutlassFNAAutogradFns[na_dim].apply(
        query,
        key,
        value,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale,
        forward_config,
        backward_config,
    )

    if return_lse:
        return output, lse

    return output


def na1d_cutlass_fna(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension1DTypeOrDed,
    stride: Dimension1DTypeOrDed = 1,
    dilation: Dimension1DTypeOrDed = 1,
    is_causal: Optional[CausalArg1DTypeOrDed] = False,
    scale: Optional[float] = None,
    q_tile_shape: Optional[Dimension1DType] = None,
    kv_tile_shape: Optional[Dimension1DType] = None,
    backward_q_tile_shape: Optional[Dimension1DType] = None,
    backward_kv_tile_shape: Optional[Dimension1DType] = None,
    backward_kv_splits: Optional[Dimension1DType] = None,
    backward_use_pt_reduction: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    return cutlass_fna_generic(
        query=query,
        key=key,
        value=value,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
        backward_q_tile_shape=backward_q_tile_shape,
        backward_kv_tile_shape=backward_kv_tile_shape,
        backward_kv_splits=backward_kv_splits,
        backward_use_pt_reduction=backward_use_pt_reduction,
        return_lse=return_lse,
    )


def na2d_cutlass_fna(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension2DTypeOrDed,
    stride: Dimension2DTypeOrDed = 1,
    dilation: Dimension2DTypeOrDed = 1,
    is_causal: Optional[CausalArg2DTypeOrDed] = False,
    scale: Optional[float] = None,
    q_tile_shape: Optional[Dimension2DType] = None,
    kv_tile_shape: Optional[Dimension2DType] = None,
    backward_q_tile_shape: Optional[Dimension2DType] = None,
    backward_kv_tile_shape: Optional[Dimension2DType] = None,
    backward_kv_splits: Optional[Dimension2DType] = None,
    backward_use_pt_reduction: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    return cutlass_fna_generic(
        query=query,
        key=key,
        value=value,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
        backward_q_tile_shape=backward_q_tile_shape,
        backward_kv_tile_shape=backward_kv_tile_shape,
        backward_kv_splits=backward_kv_splits,
        backward_use_pt_reduction=backward_use_pt_reduction,
        return_lse=return_lse,
    )


def na3d_cutlass_fna(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension3DTypeOrDed,
    stride: Dimension3DTypeOrDed = 1,
    dilation: Dimension3DTypeOrDed = 1,
    is_causal: Optional[CausalArg3DTypeOrDed] = False,
    scale: Optional[float] = None,
    q_tile_shape: Optional[Dimension3DType] = None,
    kv_tile_shape: Optional[Dimension3DType] = None,
    backward_q_tile_shape: Optional[Dimension3DType] = None,
    backward_kv_tile_shape: Optional[Dimension3DType] = None,
    backward_kv_splits: Optional[Dimension3DType] = None,
    backward_use_pt_reduction: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    return cutlass_fna_generic(
        query=query,
        key=key,
        value=value,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
        backward_q_tile_shape=backward_q_tile_shape,
        backward_kv_tile_shape=backward_kv_tile_shape,
        backward_kv_splits=backward_kv_splits,
        backward_use_pt_reduction=backward_use_pt_reduction,
        return_lse=return_lse,
    )
