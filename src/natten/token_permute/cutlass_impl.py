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
    token_unpermute_1d,
    token_unpermute_2d,
    token_unpermute_3d,
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


CutlassTokenPermute1DAutogradFn = make_cutlass_token_permute_autograd_fn(1)
CutlassTokenPermute2DAutogradFn = make_cutlass_token_permute_autograd_fn(2)
CutlassTokenPermute3DAutogradFn = make_cutlass_token_permute_autograd_fn(3)

CutlassTokenUnPermute1DAutogradFn = make_cutlass_token_unpermute_autograd_fn(1)
CutlassTokenUnPermute2DAutogradFn = make_cutlass_token_unpermute_autograd_fn(2)
CutlassTokenUnPermute3DAutogradFn = make_cutlass_token_unpermute_autograd_fn(3)

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


def token_permute_cutlass(
    tensor: Tensor,
    tile_shape: DimensionType,
    dilation: DimensionType,
    flip_tiled_dims,
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
