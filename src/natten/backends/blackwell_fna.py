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
import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from torch.amp import custom_bwd, custom_fwd
from torch.autograd import Function

amp_fwd = functools.partial(custom_fwd, device_type="cuda")
amp_bwd = functools.partial(custom_bwd, device_type="cuda")

from .._libnatten import (
    blackwell_na1d_forward,
    blackwell_na2d_forward,
    blackwell_na3d_forward,
)
from ..token_permute import maybe_pad, maybe_unpad, token_permute, token_unpermute
from ..types import (
    CausalArg1DTypeOrDed,
    CausalArg2DTypeOrDed,
    CausalArg3DTypeOrDed,
    CausalArgType,
    CausalArgTypeOrDed,
    CutlassBlackwellFnaForwardConfigType,
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
from ..utils.checks import (
    additional_kv_tensor_checks,
    check_all_args,
    check_args_against_input,
    na_tensor_checks,
)

from .configs.checks import can_run_cutlass_blackwell_fna
from .configs.cutlass_blackwell import check_cutlass_blackwell_fna_forward_config


def make_cutlass_blackwell_fna_autograd_fn(na_dim):
    assert na_dim in [1, 2, 3]

    FORWARD_OPS = {
        1: blackwell_na1d_forward,
        2: blackwell_na2d_forward,
        3: blackwell_na3d_forward,
    }

    # TODO:
    # BACKWARD_OPS = {}

    class CutlassBlackwellFnaGenericAutogradFn(Function):
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
            q_shape: DimensionType,
            kv_shape: DimensionType,
            qkv_shape: DimensionType,
            num_extra_kv: int,
            tiling_config: CutlassBlackwellFnaForwardConfigType,
            run_persistent_kernel: bool,
        ) -> Tuple[Tensor, Tensor]:
            kernel_size, stride, dilation, is_causal = check_all_args(
                na_dim, kernel_size, stride, dilation, is_causal
            )

            assert isinstance(
                scale, float
            ), f"Expected float attention scale, got {type(scale)}."

            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            output = torch.empty_like(query)

            logsumexp = torch.empty(
                query.shape[:-1], dtype=torch.float32, device=query.device
            )

            q_tile_shape, kv_tile_shape = tiling_config
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
                q_shape,
                kv_shape,
                qkv_shape,
                num_extra_kv,
                q_tile_shape,
                kv_tile_shape,
                run_persistent_kernel,
            )

            ctx.save_for_backward(query, key, value, logsumexp, output)
            ctx.kernel_size = kernel_size
            ctx.stride = stride
            ctx.dilation = dilation
            ctx.is_causal = is_causal
            ctx.scale = scale
            ctx.tiling_config = tiling_config
            # ctx.tiling_config_backward = tiling_config_backward
            ctx.run_persistent_kernel = run_persistent_kernel

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
            NoneType,
            NoneType,
            NoneType,
            # NoneType,
            NoneType,
        ]:
            query, key, value, logsumexp, output = ctx.saved_tensors
            d_output = grad_out.contiguous()  # noqa: F841
            d_query = torch.empty_like(query)
            d_key = torch.empty_like(key)
            d_value = torch.empty_like(value)

            raise NotImplementedError(
                "Blackwell FNA does not support backpropagation yet."
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
            )

    return CutlassBlackwellFnaGenericAutogradFn


CutlassBlackwellFna1DAutogradFn = make_cutlass_blackwell_fna_autograd_fn(1)
CutlassBlackwellFna2DAutogradFn = make_cutlass_blackwell_fna_autograd_fn(2)
CutlassBlackwellFna3DAutogradFn = make_cutlass_blackwell_fna_autograd_fn(3)


CutlassBlackwellFNAAutogradFns = {
    1: CutlassBlackwellFna1DAutogradFn,
    2: CutlassBlackwellFna2DAutogradFn,
    3: CutlassBlackwellFna3DAutogradFn,
}


def cutlass_blackwell_fna_generic(
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
    run_persistent_kernel: bool = False,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:

    na_tensor_checks(query, key, value, must_match_head_dims=True)
    additional_kv_tensor_checks(
        query, key, value, additional_keys, additional_values, must_match_head_dims=True
    )

    assert can_run_cutlass_blackwell_fna(query, key, value, raise_error=True)

    na_dim = query.dim() - 3  # batch, heads, head_dim

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

    q_tile_shape, kv_tile_shape = check_cutlass_blackwell_fna_forward_config(
        input_tensor=query, q_tile_shape=q_tile_shape, kv_tile_shape=kv_tile_shape
    )

    scale = scale or query.shape[-1] ** -0.5

    has_additional_kv = additional_keys is not None

    # Shape before padding, token permute, and concat with extra KV (if any)
    qkv_shape = query.shape[1 : 1 + na_dim]

    query_pad, padding = maybe_pad(query, q_tile_shape, dilation=dilation)
    key, _ = maybe_pad(key, kv_tile_shape, dilation=dilation)
    value, _ = maybe_pad(value, kv_tile_shape, dilation=dilation)

    query_perm, q_shape, qR = token_permute(
        query_pad, q_tile_shape, dilation=dilation, flip_tiled_dims=True
    )
    key_perm, k_shape, kR = token_permute(
        key, kv_tile_shape, dilation=dilation, flip_tiled_dims=True
    )
    value_perm, v_shape, vR = token_permute(
        value, kv_tile_shape, dilation=dilation, flip_tiled_dims=True
    )

    assert k_shape == v_shape
    kv_shape = k_shape

    num_extra_kv = 0

    if has_additional_kv:
        assert additional_keys is not None and additional_values is not None
        assert additional_keys.shape == additional_values.shape
        num_extra_kv = additional_keys.shape[1]
        num_dilation_groups = math.prod(dilation)
        # NOTE: this is really terrible -- having an unfused additional kv with merge_attentions
        # can avoid this and the extra memory op and footprint.
        if num_dilation_groups > 1:
            assert key_perm.shape[-2] == additional_keys.shape[-2] * num_dilation_groups

            additional_keys = additional_keys.repeat(1, 1, num_dilation_groups, 1)
            additional_values = additional_values.repeat(1, 1, num_dilation_groups, 1)

            assert additional_keys.shape[-2] == key_perm.shape[-2]
            assert additional_values.shape[-2] == value_perm.shape[-2]

        key_perm = torch.cat([key_perm, additional_keys], dim=1)
        value_perm = torch.cat([value_perm, additional_values], dim=1)

    output_perm, lse_perm = CutlassBlackwellFNAAutogradFns[na_dim].apply(
        query_perm,
        key_perm,
        value_perm,
        kernel_size,
        stride,
        dilation,
        is_causal,
        scale,
        q_shape,
        kv_shape,
        qkv_shape,
        num_extra_kv,
        (q_tile_shape, kv_tile_shape),
        run_persistent_kernel,
    )

    output = maybe_unpad(
        token_unpermute(
            output_perm,
            q_tile_shape,
            q_shape,
            qR,
            dilation=dilation,
            flip_tiled_dims=True,
        ),
        padding,
    )
    if return_lse:
        lse = maybe_unpad(
            token_unpermute(
                lse_perm.unsqueeze(-1),
                q_tile_shape,
                q_shape,
                qR,
                dilation=dilation,
                flip_tiled_dims=True,
            ),
            padding,
        ).squeeze(-1)

        return output, lse

    return output


def na1d_cutlass_blackwell_fna(
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
    run_persistent_kernel: bool = False,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    return cutlass_blackwell_fna_generic(
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
        run_persistent_kernel=run_persistent_kernel,
        additional_keys=additional_keys,
        additional_values=additional_values,
    )


def na2d_cutlass_blackwell_fna(
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
    run_persistent_kernel: bool = False,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    return cutlass_blackwell_fna_generic(
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
        run_persistent_kernel=run_persistent_kernel,
        additional_keys=additional_keys,
        additional_values=additional_values,
    )


def na3d_cutlass_blackwell_fna(
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
    run_persistent_kernel: bool = False,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    return cutlass_blackwell_fna_generic(
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
        run_persistent_kernel=run_persistent_kernel,
        additional_keys=additional_keys,
        additional_values=additional_values,
    )
