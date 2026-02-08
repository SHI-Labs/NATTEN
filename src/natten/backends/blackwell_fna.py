#################################################################################################
# Copyright (c) 2022 - 2026 Ali Hassani.
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

from torch import Tensor
from torch.amp import custom_bwd, custom_fwd
from torch.autograd import Function

amp_fwd = functools.partial(custom_fwd, device_type="cuda")
amp_bwd = functools.partial(custom_bwd, device_type="cuda")

from natten._libnatten import (
    blackwell_na1d_backward,
    blackwell_na1d_forward,
    blackwell_na2d_backward,
    blackwell_na2d_forward,
    blackwell_na3d_backward,
    blackwell_na3d_forward,
)
from natten.backends.configs.checks import can_run_cutlass_blackwell_fna
from natten.backends.configs.cutlass_blackwell import (
    check_cutlass_blackwell_fna_backward_config,
    check_cutlass_blackwell_fna_forward_config,
)
from natten.token_permute import token_permute_operation, token_unpermute_operation
from natten.types import (
    CausalArg1DTypeOrDed,
    CausalArg2DTypeOrDed,
    CausalArg3DTypeOrDed,
    CausalArgType,
    CausalArgTypeOrDed,
    CutlassBlackwellFnaBackwardConfigType,
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
from natten.utils.checks import (
    check_all_args,
    check_args_against_input,
    na_tensor_checks,
)


def make_cutlass_blackwell_fna_autograd_fn(na_dim):
    assert na_dim in [1, 2, 3]

    FORWARD_OPS = {
        1: blackwell_na1d_forward,
        2: blackwell_na2d_forward,
        3: blackwell_na3d_forward,
    }

    BACKWARD_OPS = {
        1: blackwell_na1d_backward,
        2: blackwell_na2d_backward,
        3: blackwell_na3d_backward,
    }

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
            forward_config: CutlassBlackwellFnaForwardConfigType,
            backward_config: CutlassBlackwellFnaBackwardConfigType,
            run_persistent_kernel: bool,
        ) -> Tuple[Tensor, Tensor]:
            kernel_size, stride, dilation, is_causal = check_all_args(
                na_dim, kernel_size, stride, dilation, is_causal
            )

            q_tile_shape, kv_tile_shape = forward_config

            # Token permute begin
            query_perm, qkv_shape, q_shape = token_permute_operation(
                query,
                tile_shape=q_tile_shape,
                dilation=dilation,
                flip_tiled_dims=True,
            )
            key_perm, _, k_shape = token_permute_operation(
                key, tile_shape=kv_tile_shape, dilation=dilation, flip_tiled_dims=True
            )
            value_perm, _, v_shape = token_permute_operation(
                value, tile_shape=kv_tile_shape, dilation=dilation, flip_tiled_dims=True
            )

            assert k_shape == v_shape
            kv_shape = k_shape
            # Token permute end

            query_perm = query_perm.contiguous()
            key_perm = key_perm.contiguous()
            value_perm = value_perm.contiguous()

            output_perm, logsumexp_perm = FORWARD_OPS[na_dim](
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
                q_tile_shape,
                kv_tile_shape,
                run_persistent_kernel,
            )

            # Token un-permute begin
            output = token_unpermute_operation(
                output_perm,
                token_layout_shape=qkv_shape,
                tile_shape=q_tile_shape,
                dilation=dilation,
                flip_tiled_dims=True,
            )
            logsumexp = token_unpermute_operation(
                logsumexp_perm.unsqueeze(-1),
                token_layout_shape=qkv_shape,
                tile_shape=q_tile_shape,
                dilation=dilation,
                flip_tiled_dims=True,
            ).squeeze(-1)
            # Token un-permute end

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
        def backward(ctx, d_output: Tensor, d_lse: Tensor) -> Tuple[
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
        ]:
            query, key, value, logsumexp, output = ctx.saved_tensors
            kernel_size, stride, dilation, is_causal, scale = (
                ctx.kernel_size,
                ctx.stride,
                ctx.dilation,
                ctx.is_causal,
                ctx.scale,
            )

            q_tile_shape, kv_tile_shape = ctx.backward_config

            # Token permute begin

            query_perm, qkv_shape, q_shape = token_permute_operation(
                query, tile_shape=q_tile_shape, dilation=dilation, flip_tiled_dims=True
            )
            output_perm, _, o_shape = token_permute_operation(
                output, tile_shape=q_tile_shape, dilation=dilation, flip_tiled_dims=True
            )
            d_output_perm, _, d_o_shape = token_permute_operation(
                d_output,
                tile_shape=q_tile_shape,
                dilation=dilation,
                flip_tiled_dims=True,
            )
            logsumexp_perm, _, _ = token_permute_operation(
                logsumexp.unsqueeze(-1),
                tile_shape=q_tile_shape,
                dilation=dilation,
                flip_tiled_dims=True,
            )
            key_perm, _, k_shape = token_permute_operation(
                key, tile_shape=kv_tile_shape, dilation=dilation, flip_tiled_dims=True
            )
            value_perm, _, v_shape = token_permute_operation(
                value, tile_shape=kv_tile_shape, dilation=dilation, flip_tiled_dims=True
            )

            assert q_shape == o_shape == d_o_shape
            assert k_shape == v_shape
            kv_shape = k_shape
            # Token permute end

            query_perm = query_perm.contiguous()
            key_perm = key_perm.contiguous()
            value_perm = value_perm.contiguous()
            output_perm = output_perm.contiguous()
            d_output_perm = d_output_perm.contiguous()
            logsumexp_perm = logsumexp_perm.squeeze(-1)

            d_query_perm, d_key_perm, d_value_perm = BACKWARD_OPS[na_dim](
                query_perm,
                key_perm,
                value_perm,
                output_perm,
                d_output_perm,
                logsumexp_perm,
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

            # Token un-permute begin
            d_query = token_unpermute_operation(
                d_query_perm,
                token_layout_shape=qkv_shape,
                tile_shape=q_tile_shape,
                dilation=dilation,
                flip_tiled_dims=True,
            )
            d_key = token_unpermute_operation(
                d_key_perm,
                token_layout_shape=qkv_shape,
                tile_shape=kv_tile_shape,
                dilation=dilation,
                flip_tiled_dims=True,
            )
            d_value = token_unpermute_operation(
                d_value_perm,
                token_layout_shape=qkv_shape,
                tile_shape=kv_tile_shape,
                dilation=dilation,
                flip_tiled_dims=True,
            )
            # Token un-permute end

            assert d_query.shape == query.shape
            assert d_key.shape == key.shape
            assert d_value.shape == value.shape

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
    backward_q_tile_shape: Optional[DimensionType] = None,
    backward_kv_tile_shape: Optional[DimensionType] = None,
    run_persistent_kernel: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:

    na_tensor_checks(
        query, key, value, must_match_head_dims=True, supports_gqa_mqa=True
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

    forward_config = check_cutlass_blackwell_fna_forward_config(
        input_tensor=query, q_tile_shape=q_tile_shape, kv_tile_shape=kv_tile_shape
    )

    backward_config = check_cutlass_blackwell_fna_backward_config(
        input_tensor=query,
        q_tile_shape=backward_q_tile_shape,
        kv_tile_shape=backward_kv_tile_shape,
    )

    scale = scale or query.shape[-1] ** -0.5

    output, lse = CutlassBlackwellFNAAutogradFns[na_dim].apply(
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
        run_persistent_kernel,
    )

    if return_lse:
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
    backward_q_tile_shape: Optional[Dimension1DType] = None,
    backward_kv_tile_shape: Optional[Dimension1DType] = None,
    run_persistent_kernel: bool = False,
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
        backward_q_tile_shape=backward_q_tile_shape,
        backward_kv_tile_shape=backward_kv_tile_shape,
        run_persistent_kernel=run_persistent_kernel,
        return_lse=return_lse,
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
    backward_q_tile_shape: Optional[Dimension2DType] = None,
    backward_kv_tile_shape: Optional[Dimension2DType] = None,
    run_persistent_kernel: bool = False,
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
        backward_q_tile_shape=backward_q_tile_shape,
        backward_kv_tile_shape=backward_kv_tile_shape,
        run_persistent_kernel=run_persistent_kernel,
        return_lse=return_lse,
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
    backward_q_tile_shape: Optional[Dimension3DType] = None,
    backward_kv_tile_shape: Optional[Dimension3DType] = None,
    run_persistent_kernel: bool = False,
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
        backward_q_tile_shape=backward_q_tile_shape,
        backward_kv_tile_shape=backward_kv_tile_shape,
        run_persistent_kernel=run_persistent_kernel,
        return_lse=return_lse,
    )
