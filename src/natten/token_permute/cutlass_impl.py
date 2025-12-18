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
from typing import Tuple

import torch
from torch import Tensor
from torch.amp import custom_bwd, custom_fwd
from torch.autograd import Function

amp_fwd = functools.partial(custom_fwd, device_type="cuda")
amp_bwd = functools.partial(custom_bwd, device_type="cuda")

from natten._libnatten import (
    HAS_LIBNATTEN,
    token_permute_1d,
    token_permute_2d,
    token_permute_3d,
    token_permute_varlen_1d,
    token_permute_varlen_2d,
    token_permute_varlen_3d,
    token_unpermute_1d,
    token_unpermute_2d,
    token_unpermute_3d,
    token_unpermute_varlen_1d,
    token_unpermute_varlen_2d,
    token_unpermute_varlen_3d,
)
from natten.types import DimensionType, NoneType
from natten.utils import log
from natten.utils.device import get_device_cc, is_cuda

logger = log.get_logger(__name__)


def can_run_cutlass_tokperm(tensor: Tensor) -> bool:
    if not HAS_LIBNATTEN:
        logger.debug(
            "Can't use libnatten TokPerm kernels, because libnatten is not available."
        )
        return False

    if not is_cuda(tensor.device):
        logger.debug(
            "Can't use libnatten TokPerm kernels, because input is not a CUDA tensor."
        )
        return False

    is_fp8_allowed = get_device_cc(tensor.device) in [100, 103]
    if tensor.dtype not in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float16,
    ] and (
        is_fp8_allowed and tensor.dtype not in [torch.float8_e5m2, torch.float8_e4m3fn]
    ):
        logger.debug(
            f"Can't use libnatten TokPerm kernels; unexpected dtype {tensor.dtype}."
        )
        return False

    return True


PERMUTE_OPS = {1: token_permute_1d, 2: token_permute_2d, 3: token_permute_3d}
UNPERMUTE_OPS = {1: token_unpermute_1d, 2: token_unpermute_2d, 3: token_unpermute_3d}

VARLEN_PERMUTE_OPS = {
    1: token_permute_varlen_1d,
    2: token_permute_varlen_2d,
    3: token_permute_varlen_3d,
}
VARLEN_UNPERMUTE_OPS = {
    1: token_unpermute_varlen_1d,
    2: token_unpermute_varlen_2d,
    3: token_unpermute_varlen_3d,
}


def make_cutlass_token_permute_autograd_fn(na_dim):
    assert na_dim in [1, 2, 3]

    class CutlassTokenPermuteAutogradFn(Function):
        @staticmethod
        @amp_fwd
        def forward(
            ctx,
            tensor: Tensor,
            tile_shape: DimensionType,
            dilation: DimensionType,
            flip_tiled_dims: bool,
        ) -> Tensor:

            output = PERMUTE_OPS[na_dim](
                tensor,
                tile_shape=tile_shape,
                dilation=dilation,
                flip_tiled_dims=flip_tiled_dims,
            )

            ctx.tile_shape = tile_shape
            ctx.dilation = dilation
            ctx.flip_tiled_dims = flip_tiled_dims
            ctx.token_layout = tuple(x for x in tensor.shape[1 : na_dim + 1])
            assert len(ctx.token_layout) == na_dim

            return output

        @staticmethod
        @amp_bwd
        def backward(ctx, d_output: Tensor) -> Tuple[
            Tensor,
            NoneType,
            NoneType,
            NoneType,
        ]:

            d_output_unpermuted = UNPERMUTE_OPS[na_dim](
                d_output,
                token_layout_shape=ctx.token_layout,
                tile_shape=ctx.tile_shape,
                dilation=ctx.dilation,
                flip_tiled_dims=ctx.flip_tiled_dims,
            )

            return (
                d_output_unpermuted,
                None,
                None,
                None,
            )

    return CutlassTokenPermuteAutogradFn


def make_cutlass_token_unpermute_autograd_fn(na_dim):
    assert na_dim in [1, 2, 3]

    class CutlassTokenUnPermuteAutogradFn(Function):
        @staticmethod
        @amp_fwd
        def forward(
            ctx,
            tensor: Tensor,
            token_layout: DimensionType,
            tile_shape: DimensionType,
            dilation: DimensionType,
            flip_tiled_dims: bool,
        ) -> Tensor:

            output = UNPERMUTE_OPS[na_dim](
                tensor,
                token_layout_shape=token_layout,
                tile_shape=tile_shape,
                dilation=dilation,
                flip_tiled_dims=flip_tiled_dims,
            )

            ctx.tile_shape = tile_shape
            ctx.dilation = dilation
            ctx.flip_tiled_dims = flip_tiled_dims

            return output

        @staticmethod
        @amp_bwd
        def backward(ctx, d_output: Tensor) -> Tuple[
            Tensor,
            NoneType,
            NoneType,
            NoneType,
            NoneType,
        ]:

            d_output_permuted = PERMUTE_OPS[na_dim](
                d_output,
                tile_shape=ctx.tile_shape,
                dilation=ctx.dilation,
                flip_tiled_dims=ctx.flip_tiled_dims,
            )

            return (
                d_output_permuted,
                None,
                None,
                None,
                None,
            )

    return CutlassTokenUnPermuteAutogradFn


def make_cutlass_token_permute_varlen_autograd_fn(na_dim):
    assert na_dim in [1, 2, 3]

    class CutlassTokenPermuteVarlenAutogradFn(Function):
        @staticmethod
        @amp_fwd
        def forward(
            ctx,
            tensor: Tensor,
            offsets_pre_permute: Tensor,
            offsets_post_permute: Tensor,
            token_layouts_pre_permute: Tensor,
            max_seqlen: int,
            total_seqlen_post_permute: int,
            tile_shape: DimensionType,
            dilation: DimensionType,
            flip_tiled_dims: bool,
        ) -> Tensor:

            output = VARLEN_PERMUTE_OPS[na_dim](
                tensor,
                offsets_pre_permute=offsets_pre_permute,
                offsets_post_permute=offsets_post_permute,
                token_layouts_pre_permute=token_layouts_pre_permute,
                max_seqlen=max_seqlen,
                total_seqlen_post_permute=total_seqlen_post_permute,
                tile_shape=tile_shape,
                dilation=dilation,
                flip_tiled_dims=flip_tiled_dims,
            )

            ctx.save_for_backward(
                offsets_pre_permute,
                offsets_post_permute,
                token_layouts_pre_permute,
            )
            ctx.tile_shape = tile_shape
            ctx.dilation = dilation
            ctx.flip_tiled_dims = flip_tiled_dims
            ctx.total_seqlen_pre_permute = tensor.shape[1]
            ctx.max_seqlen = max_seqlen

            return output

        @staticmethod
        @amp_bwd
        def backward(ctx, d_output: Tensor) -> Tuple[
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

            offsets_pre_permute, offsets_post_permute, token_layouts_pre_permute = (
                ctx.saved_tensors
            )
            d_output_unpermuted = VARLEN_UNPERMUTE_OPS[na_dim](
                d_output,
                offsets_pre_permute=offsets_pre_permute,
                offsets_post_permute=offsets_post_permute,
                token_layouts_pre_permute=token_layouts_pre_permute,
                max_seqlen=ctx.max_seqlen,
                total_seqlen_pre_permute=ctx.total_seqlen_pre_permute,
                tile_shape=ctx.tile_shape,
                dilation=ctx.dilation,
                flip_tiled_dims=ctx.flip_tiled_dims,
            )

            return (
                d_output_unpermuted,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    return CutlassTokenPermuteVarlenAutogradFn


def make_cutlass_token_unpermute_varlen_autograd_fn(na_dim):
    assert na_dim in [1, 2, 3]

    class CutlassTokenUnPermuteVarlenAutogradFn(Function):
        @staticmethod
        @amp_fwd
        def forward(
            ctx,
            tensor: Tensor,
            offsets_pre_permute: Tensor,
            offsets_post_permute: Tensor,
            token_layouts_pre_permute: Tensor,
            max_seqlen: int,
            total_seqlen_pre_permute: int,
            tile_shape: DimensionType,
            dilation: DimensionType,
            flip_tiled_dims: bool,
        ) -> Tensor:

            output = VARLEN_UNPERMUTE_OPS[na_dim](
                tensor,
                offsets_pre_permute=offsets_pre_permute,
                offsets_post_permute=offsets_post_permute,
                token_layouts_pre_permute=token_layouts_pre_permute,
                max_seqlen=max_seqlen,
                total_seqlen_pre_permute=total_seqlen_pre_permute,
                tile_shape=tile_shape,
                dilation=dilation,
                flip_tiled_dims=flip_tiled_dims,
            )

            ctx.save_for_backward(
                offsets_pre_permute,
                offsets_post_permute,
                token_layouts_pre_permute,
            )
            ctx.tile_shape = tile_shape
            ctx.dilation = dilation
            ctx.flip_tiled_dims = flip_tiled_dims
            ctx.total_seqlen_post_permute = tensor.shape[1]
            ctx.max_seqlen = max_seqlen

            return output

        @staticmethod
        @amp_bwd
        def backward(ctx, d_output: Tensor) -> Tuple[
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

            offsets_pre_permute, offsets_post_permute, token_layouts_pre_permute = (
                ctx.saved_tensors
            )
            d_output_permuted = VARLEN_PERMUTE_OPS[na_dim](
                d_output,
                offsets_pre_permute=offsets_pre_permute,
                offsets_post_permute=offsets_post_permute,
                token_layouts_pre_permute=token_layouts_pre_permute,
                max_seqlen=ctx.max_seqlen,
                total_seqlen_post_permute=ctx.total_seqlen_post_permute,
                tile_shape=ctx.tile_shape,
                dilation=ctx.dilation,
                flip_tiled_dims=ctx.flip_tiled_dims,
            )

            return (
                d_output_permuted,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    return CutlassTokenUnPermuteVarlenAutogradFn


CutlassTokenPermute1DAutogradFn = make_cutlass_token_permute_autograd_fn(1)
CutlassTokenPermute2DAutogradFn = make_cutlass_token_permute_autograd_fn(2)
CutlassTokenPermute3DAutogradFn = make_cutlass_token_permute_autograd_fn(3)

CutlassTokenUnPermute1DAutogradFn = make_cutlass_token_unpermute_autograd_fn(1)
CutlassTokenUnPermute2DAutogradFn = make_cutlass_token_unpermute_autograd_fn(2)
CutlassTokenUnPermute3DAutogradFn = make_cutlass_token_unpermute_autograd_fn(3)

CutlassTokenPermuteVarlen1DAutogradFn = make_cutlass_token_permute_varlen_autograd_fn(1)
CutlassTokenPermuteVarlen2DAutogradFn = make_cutlass_token_permute_varlen_autograd_fn(2)
CutlassTokenPermuteVarlen3DAutogradFn = make_cutlass_token_permute_varlen_autograd_fn(3)

CutlassTokenUnPermuteVarlen1DAutogradFn = (
    make_cutlass_token_unpermute_varlen_autograd_fn(1)
)
CutlassTokenUnPermuteVarlen2DAutogradFn = (
    make_cutlass_token_unpermute_varlen_autograd_fn(2)
)
CutlassTokenUnPermuteVarlen3DAutogradFn = (
    make_cutlass_token_unpermute_varlen_autograd_fn(3)
)

CutlassTokenPermuteAutogradFns = {
    1: CutlassTokenPermute1DAutogradFn,
    2: CutlassTokenPermute2DAutogradFn,
    3: CutlassTokenPermute3DAutogradFn,
}

CutlassTokenUnPermuteAutogradFns = {
    1: CutlassTokenUnPermute1DAutogradFn,
    2: CutlassTokenUnPermute2DAutogradFn,
    3: CutlassTokenUnPermute3DAutogradFn,
}

CutlassTokenPermuteVarlenAutogradFns = {
    1: CutlassTokenPermuteVarlen1DAutogradFn,
    2: CutlassTokenPermuteVarlen2DAutogradFn,
    3: CutlassTokenPermuteVarlen3DAutogradFn,
}

CutlassTokenUnPermuteVarlenAutogradFns = {
    1: CutlassTokenUnPermuteVarlen1DAutogradFn,
    2: CutlassTokenUnPermuteVarlen2DAutogradFn,
    3: CutlassTokenUnPermuteVarlen3DAutogradFn,
}


def token_permute_cutlass(
    tensor: Tensor,
    tile_shape: DimensionType,
    dilation: DimensionType,
    flip_tiled_dims: bool,
) -> Tensor:
    if tensor.dim() not in [4, 5, 6]:
        raise ValueError(
            "Expected 4D, 5D, or 6D tensor (corresponding to NA1D, 2D, 3D), "
            f"got {tensor.dim()}D input."
        )

    na_dim = tensor.dim() - 3
    assert na_dim in [1, 2, 3]

    if len(tile_shape) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D tiler for NA{na_dim}D, " f"got {tile_shape=}."
        )

    if not can_run_cutlass_tokperm(tensor):
        raise NotImplementedError(
            "Use case is not compatible with CUTLASS Token Permute."
        )

    tensor = tensor.contiguous()
    output = CutlassTokenPermuteAutogradFns[na_dim].apply(
        tensor,
        tile_shape,
        dilation,
        flip_tiled_dims,
    )

    return output


def token_unpermute_cutlass(
    tensor: Tensor,
    token_layout_shape: DimensionType,
    tile_shape: DimensionType,
    dilation: DimensionType,
    flip_tiled_dims: bool,
) -> Tensor:
    if tensor.dim() != 4:
        raise ValueError(f"Expected flattened 4D tensor, got {tensor.dim()}D input.")

    na_dim = len(token_layout_shape)

    if len(tile_shape) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D tiler for NA{na_dim}D, " f"got {tile_shape=}."
        )

    if not can_run_cutlass_tokperm(tensor):
        raise NotImplementedError(
            "Use case is not compatible with CUTLASS Token UnPermute."
        )

    tensor = tensor.contiguous()
    output = CutlassTokenUnPermuteAutogradFns[na_dim].apply(
        tensor,
        token_layout_shape,
        tile_shape,
        dilation,
        flip_tiled_dims,
    )

    return output


def token_permute_varlen_cutlass(
    tensor: Tensor,
    metadata: dict,
    tile_shape: DimensionType,
    dilation: DimensionType,
    flip_tiled_dims: bool,
) -> Tensor:

    na_dim = len(tile_shape)
    assert na_dim in [1, 2, 3]

    if not can_run_cutlass_tokperm(tensor):
        raise NotImplementedError(
            "Use case is not compatible with CUTLASS Token Permute."
        )

    offsets_pre_permute = metadata["offsets_pre_permute"]
    offsets_post_permute = metadata["offsets_post_permute"]
    token_layouts_pre_permute = metadata["token_layouts_pre_permute"]
    total_seqlen_post_permute = metadata["total_seqlen_post_permute"]
    max_seqlen_post_permute = metadata["max_seqlen_post_permute"]

    tensor = tensor.contiguous()
    output = CutlassTokenPermuteVarlenAutogradFns[na_dim].apply(
        tensor,
        offsets_pre_permute,
        offsets_post_permute,
        token_layouts_pre_permute,
        max_seqlen_post_permute,
        total_seqlen_post_permute,
        tile_shape,
        dilation,
        flip_tiled_dims,
    )

    return output


def token_unpermute_varlen_cutlass(
    tensor: Tensor,
    metadata: dict,
    tile_shape: DimensionType,
    dilation: DimensionType,
    flip_tiled_dims: bool,
) -> Tensor:
    na_dim = len(tile_shape)
    assert na_dim in [1, 2, 3]

    if not can_run_cutlass_tokperm(tensor):
        raise NotImplementedError(
            "Use case is not compatible with CUTLASS Token UnPermute."
        )

    offsets_pre_permute = metadata["offsets_pre_permute"]
    offsets_post_permute = metadata["offsets_post_permute"]
    token_layouts_pre_permute = metadata["token_layouts_pre_permute"]
    total_seqlen_pre_permute = metadata["total_seqlen_pre_permute"]
    max_seqlen_pre_permute = metadata["max_seqlen_pre_permute"]

    tensor = tensor.contiguous()
    output = CutlassTokenUnPermuteVarlenAutogradFns[na_dim].apply(
        tensor,
        offsets_pre_permute,
        offsets_post_permute,
        token_layouts_pre_permute,
        max_seqlen_pre_permute,
        total_seqlen_pre_permute,
        tile_shape,
        dilation,
        flip_tiled_dims,
    )

    return output
