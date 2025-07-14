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
    reference_na1d_backward,
    reference_na1d_forward,
    reference_na2d_backward,
    reference_na2d_forward,
    reference_na3d_backward,
    reference_na3d_forward,
)
from ..types import (
    CausalArg1DTypeOrDed,
    CausalArg2DTypeOrDed,
    CausalArg3DTypeOrDed,
    CausalArgType,
    CausalArgTypeOrDed,
    Dimension1DTypeOrDed,
    Dimension2DTypeOrDed,
    Dimension3DTypeOrDed,
    DimensionType,
    DimensionTypeOrDed,
    NoneType,
)
from ..utils import log
from ..utils.checks import (
    additional_kv_tensor_checks,
    check_all_args,
    check_args_against_input,
    na_tensor_checks,
)

logger = log.get_logger(__name__)


def make_reference_fna_autograd_fn(na_dim):
    assert na_dim in [1, 2, 3]

    FORWARD_OPS = {
        1: reference_na1d_forward,
        2: reference_na2d_forward,
        3: reference_na3d_forward,
    }

    BACKWARD_OPS = {
        1: reference_na1d_backward,
        2: reference_na2d_backward,
        3: reference_na3d_backward,
    }

    class ReferenceFnaGenericAutogradFn(Function):
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
            qkv_shape: DimensionType,
            num_extra_kv: int,
        ) -> Tuple[Tensor, Tensor]:
            kernel_size, stride, dilation, is_causal = check_all_args(
                na_dim, kernel_size, stride, dilation, is_causal
            )

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()

            assert query.dim() == value.dim() == 4
            assert query.shape[0] == value.shape[0]
            assert query.shape[-2] == value.shape[-2]

            output_shape = [s for s in query.shape[:-1]] + [value.shape[-1]]
            output = torch.empty(output_shape, device=query.device, dtype=query.dtype)

            logsumexp = torch.empty(
                query.shape[:-1], dtype=torch.float32, device=query.device
            )

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
                qkv_shape,
                num_extra_kv,
            )

            ctx.save_for_backward(query, key, value, logsumexp, output)
            ctx.kernel_size = kernel_size
            ctx.stride = stride
            ctx.dilation = dilation
            ctx.is_causal = is_causal
            ctx.scale = scale
            ctx.qkv_shape = qkv_shape
            ctx.num_extra_kv = num_extra_kv

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
                ctx.qkv_shape,
                ctx.num_extra_kv,
            )

            return d_query, d_key, d_value, None, None, None, None, None, None, None

    return ReferenceFnaGenericAutogradFn


ReferenceFna1DAutogradFn = make_reference_fna_autograd_fn(1)
ReferenceFna2DAutogradFn = make_reference_fna_autograd_fn(2)
ReferenceFna3DAutogradFn = make_reference_fna_autograd_fn(3)


ReferenceFnaAutogradFns = {
    1: ReferenceFna1DAutogradFn,
    2: ReferenceFna2DAutogradFn,
    3: ReferenceFna3DAutogradFn,
}


def reference_fna_generic(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: DimensionTypeOrDed,
    stride: DimensionTypeOrDed = 1,
    dilation: DimensionTypeOrDed = 1,
    is_causal: Optional[CausalArgTypeOrDed] = False,
    scale: Optional[float] = None,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:

    na_tensor_checks(query, key, value, must_match_head_dims=False)
    additional_kv_tensor_checks(
        query,
        key,
        value,
        additional_keys,
        additional_values,
        must_match_head_dims=False,
    )

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

    scale = scale or query.shape[-1] ** -0.5

    qkv_shape = query.shape[1 : 1 + na_dim]

    query = query.flatten(1, na_dim)
    key = key.flatten(1, na_dim)
    value = value.flatten(1, na_dim)

    num_extra_kv = 0
    if additional_keys is not None and additional_values is not None:
        num_extra_kv = additional_keys.shape[1]
        key = torch.cat([key, additional_keys], dim=1)
        value = torch.cat([value, additional_values], dim=1)

    output, lse = ReferenceFnaAutogradFns[na_dim].apply(
        query,
        key,
        value,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale,
        qkv_shape,
        num_extra_kv,
    )
    output = output.reshape(
        query.shape[0], *qkv_shape, query.shape[-2], value.shape[-1]
    )
    lse = lse.reshape(query.shape[0], *qkv_shape, query.shape[-2])

    if return_lse:
        return output, lse

    return output


def na1d_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension1DTypeOrDed,
    stride: Dimension1DTypeOrDed = 1,
    dilation: Dimension1DTypeOrDed = 1,
    is_causal: Optional[CausalArg1DTypeOrDed] = False,
    scale: Optional[float] = None,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    return reference_fna_generic(
        query=query,
        key=key,
        value=value,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
        additional_keys=additional_keys,
        additional_values=additional_values,
        return_lse=return_lse,
    )


def na2d_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension2DTypeOrDed,
    stride: Dimension2DTypeOrDed = 1,
    dilation: Dimension2DTypeOrDed = 1,
    is_causal: Optional[CausalArg2DTypeOrDed] = False,
    scale: Optional[float] = None,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    return reference_fna_generic(
        query=query,
        key=key,
        value=value,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
        additional_keys=additional_keys,
        additional_values=additional_values,
        return_lse=return_lse,
    )


def na3d_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension3DTypeOrDed,
    stride: Dimension3DTypeOrDed = 1,
    dilation: Dimension3DTypeOrDed = 1,
    is_causal: Optional[CausalArg3DTypeOrDed] = False,
    scale: Optional[float] = None,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    return reference_fna_generic(
        query=query,
        key=key,
        value=value,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
        additional_keys=additional_keys,
        additional_values=additional_values,
        return_lse=return_lse,
    )
