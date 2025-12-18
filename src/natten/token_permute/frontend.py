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

from torch import Tensor

from natten._environment import USE_TORCH_IMPL_DEFAULT
from natten.token_permute.cutlass_impl import (
    can_run_cutlass_tokperm,
    token_permute_cutlass,
    token_permute_varlen_cutlass,
    token_unpermute_cutlass,
    token_unpermute_varlen_cutlass,
)
from natten.token_permute.torch_impl import token_permute_torch, token_unpermute_torch
from natten.token_permute.varlen import verify_fna_varlen_metadata
from natten.types import DimensionType
from natten.utils import log
from natten.utils.tuples import ceil_div_tuple, mul_tuple

logger = log.get_logger(__name__)


def token_permute_operation(
    tensor: Tensor,
    tile_shape: DimensionType,
    dilation: Optional[DimensionType] = None,
    flip_tiled_dims: bool = True,
    use_torch: bool = USE_TORCH_IMPL_DEFAULT,
) -> tuple[Tensor, DimensionType, DimensionType]:
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

    if dilation is not None and len(dilation) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D dilation for NA{na_dim}D, " f"got {dilation=}."
        )

    dilation_: DimensionType = dilation or tuple(1 for _ in range(na_dim))  # type: ignore[assignment]

    tensor = tensor.contiguous()
    batch, *token_layout_, heads, dim = tensor.shape
    token_layout: DimensionType = tuple(x for x in token_layout_)  # type: ignore[assignment]

    token_layout_post_dilation: DimensionType = mul_tuple(ceil_div_tuple(ceil_div_tuple(token_layout, tile_shape), dilation_), tile_shape)  # type: ignore[assignment]

    if not use_torch and can_run_cutlass_tokperm(tensor):
        output = token_permute_cutlass(
            tensor,
            tile_shape=tile_shape,
            dilation=dilation_,
            flip_tiled_dims=flip_tiled_dims,
        )
    else:
        output = token_permute_torch(
            tensor,
            tile_shape=tile_shape,
            dilation=dilation_,
            flip_tiled_dims=flip_tiled_dims,
        )

    return output, token_layout, token_layout_post_dilation


def token_unpermute_operation(
    tensor: Tensor,
    token_layout_shape: DimensionType,
    tile_shape: DimensionType,
    dilation: Optional[DimensionType] = None,
    flip_tiled_dims: bool = True,
    use_torch: bool = USE_TORCH_IMPL_DEFAULT,
) -> Tensor:
    if tensor.dim() != 4:
        raise ValueError(f"Expected flattened 4D tensor, got {tensor.dim()}D input.")

    na_dim = len(token_layout_shape)

    if len(tile_shape) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D tiler for NA{na_dim}D, " f"got {tile_shape=}."
        )

    if dilation is not None and len(dilation) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D dilation for NA{na_dim}D, " f"got {dilation=}."
        )

    dilation_: DimensionType = dilation or tuple(1 for _ in range(na_dim))  # type: ignore[assignment]

    tensor = tensor.contiguous()
    if not use_torch and can_run_cutlass_tokperm(tensor):
        output = token_unpermute_cutlass(
            tensor,
            token_layout_shape,
            tile_shape=tile_shape,
            dilation=dilation_,
            flip_tiled_dims=flip_tiled_dims,
        )
    else:
        output = token_unpermute_torch(
            tensor,
            token_layout_shape,
            tile_shape=tile_shape,
            dilation=dilation_,
            flip_tiled_dims=flip_tiled_dims,
        )

    return output


def token_permute_varlen_operation(
    tensor: Tensor,
    metadata: dict,
    tile_shape: DimensionType,
    dilation: Optional[DimensionType] = None,
    flip_tiled_dims: bool = True,
    use_torch: bool = USE_TORCH_IMPL_DEFAULT,
) -> Tensor:
    na_dim = len(tile_shape)
    assert na_dim in [1, 2, 3]

    dilation_: DimensionType = dilation or tuple(1 for _ in range(na_dim))  # type: ignore[assignment]

    verify_fna_varlen_metadata(
        tensor=tensor,
        metadata=metadata,
        tile_shape=tile_shape,
        dilation=dilation_,
    )

    tensor = tensor.contiguous()

    if not use_torch and can_run_cutlass_tokperm(tensor):
        output = token_permute_varlen_cutlass(
            tensor,
            metadata=metadata,
            tile_shape=tile_shape,
            dilation=dilation_,
            flip_tiled_dims=flip_tiled_dims,
        )
    else:
        raise NotImplementedError(
            "Varlen Token Permute is only implemented with CUTLASS for now."
        )

    return output


def token_unpermute_varlen_operation(
    tensor: Tensor,
    metadata: dict,
    tile_shape: DimensionType,
    dilation: Optional[DimensionType] = None,
    flip_tiled_dims: bool = True,
    use_torch: bool = USE_TORCH_IMPL_DEFAULT,
) -> Tensor:
    na_dim = len(tile_shape)
    assert na_dim in [1, 2, 3]

    dilation_: DimensionType = dilation or tuple(1 for _ in range(na_dim))  # type: ignore[assignment]

    verify_fna_varlen_metadata(
        tensor=tensor,
        metadata=metadata,
        tile_shape=tile_shape,
        dilation=dilation_,
    )

    tensor = tensor.contiguous()

    if not use_torch and can_run_cutlass_tokperm(tensor):
        output = token_unpermute_varlen_cutlass(
            tensor,
            metadata=metadata,
            tile_shape=tile_shape,
            dilation=dilation_,
            flip_tiled_dims=flip_tiled_dims,
        )
    else:
        raise NotImplementedError(
            "Varlen Token Permute is only implemented with CUTLASS for now."
        )

    return output
