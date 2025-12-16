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
import math

import torch
from torch import Tensor

from natten.types import DimensionType
from natten.utils import log
from natten.utils.environment import is_torch_compiling
from natten.utils.tuples import ceil_div_tuple, mul_tuple, sub_tuple

logger = log.get_logger(__name__)


DISABLE_PADDING_WARNING = True
TOKEN_PERMUTE_PADDING_RATIO_LIMIT_UNTIL_WARNING = 0.5


def _maybe_pad(
    tensor: Tensor, tile_shape: DimensionType, dilation: DimensionType
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

    if dilation is not None and len(dilation) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D dilation for NA{na_dim}D, " f"got {dilation=}."
        )

    token_layout = tensor.shape[1 : na_dim + 1]
    tile_shape_ = tuple(x for x in tile_shape)
    if dilation is not None:
        # NOTE: LCM?
        # tile_shape_ = tuple(math.lcm(t, d) for t, d in zip(tile_shape, dilation))
        tile_shape_ = tuple(t * d for t, d in zip(tile_shape, dilation))

    rest = tuple((x + t - 1) // t for x, t in zip(token_layout, tile_shape_))
    residual = tuple(r * t - x for x, t, r in zip(token_layout, tile_shape_, rest))

    assert all(res >= 0 for res in residual)

    if not DISABLE_PADDING_WARNING and any(
        res / sz > TOKEN_PERMUTE_PADDING_RATIO_LIMIT_UNTIL_WARNING
        for res, sz in zip(residual, token_layout)
    ):
        padded_token_layout = tuple(x + p for x, p in zip(token_layout, residual))
        logger.warning(
            "Potentially excessive padding detected in token permute: "
            f"input shape {token_layout} will be padded to {padded_token_layout} to handle "
            "token permutation, which can result in excessive memory usage, and "
            "performance implications. Consider choosing your tile shapes, input shapes "
            "(and dilation if you use it) accordingly. Refer to NATTEN docs for more info."
        )

    if any(res > 0 for res in residual):
        padding = [0, 0, 0, 0]  # head_dim_left, head_dim_right, heads_left, heads_right
        for res in reversed(residual):
            padding.append(0)  # left pad
            padding.append(res)  # right pad
        tensor_padded = torch.nn.functional.pad(tensor, padding, "constant", 0)
    else:
        tensor_padded = tensor

    return tensor_padded


def _token_permute(
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

    dilation = dilation or tuple(1 for _ in range(na_dim))

    if len(dilation) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D dilation for NA{na_dim}D, " f"got {dilation=}."
        )

    batch, *token_layout, heads, dim = tensor.shape

    if any(
        x % d != 0 or (x // d) % t != 0
        for x, t, d in zip(token_layout, tile_shape, dilation)
    ):
        raise ValueError(
            "Tensor must be divisible by static tile shape and dilation, but got "
            f"{tensor.shape=}, {tile_shape=}, {dilation=}."
        )

    num_dilation_groups = math.prod(dilation)
    token_layout_post_dilation = tuple(x // d for x, d in zip(token_layout, dilation))
    rest = tuple(x // d // t for x, t, d in zip(token_layout, tile_shape, dilation))
    logical_divide_dims = []
    for d, r, t in zip(dilation, rest, tile_shape):
        logical_divide_dims += [r, t, d]

    # Two permutations at once:
    #  1. logical divide to tiled divide
    #  2. (optionally) flip order of tiled modes (i.e. (X,Y,Z) -> (Z,Y,X)) for compatibility with
    #     CuTe's identity layout mapping.
    permutation_idxes_r = []
    permutation_idxes_t = []
    permutation_idxes_d = []
    for i in range(na_dim):
        if flip_tiled_dims:
            permutation_idxes_r += [(na_dim - i - 1) * 3 + 1]
            permutation_idxes_t += [(na_dim - i - 1) * 3 + 2]
            permutation_idxes_d += [(na_dim - i - 1) * 3 + 3]
        else:
            permutation_idxes_r += [i * 3 + 1]
            permutation_idxes_t += [i * 3 + 2]
            permutation_idxes_d += [i * 3 + 3]

    permutation_idxes = (
        [0]
        + permutation_idxes_d
        + permutation_idxes_r
        + permutation_idxes_t
        + [na_dim * 3 + 1, na_dim * 3 + 2]
    )

    # View, not copy
    tensor_tiled = tensor.view(batch, *logical_divide_dims, heads, dim)
    if not is_torch_compiling():
        assert tensor_tiled.data_ptr() == tensor.data_ptr()

    # View, not copy
    tensor_permuted = tensor_tiled.permute(*permutation_idxes)
    if not is_torch_compiling():
        assert tensor_permuted.data_ptr() == tensor_tiled.data_ptr()

    # Reshape back and copy
    tensor_flatten = tensor_permuted.reshape(
        num_dilation_groups * batch, math.prod(token_layout_post_dilation), heads, dim
    ).contiguous()
    # NOTE: token permute without dilation is a no-op for 1-D
    # assert na_dim == 1 or tensor_flatten.data_ptr() != tensor_permuted.data_ptr()
    assert tensor_flatten.is_contiguous()

    return tensor_flatten


def _token_unpermute(
    tensor: Tensor,
    token_layout: DimensionType,
    tile_shape: DimensionType,
    dilation: DimensionType,
    flip_tiled_dims: bool,
):
    if tensor.dim() != 4:
        raise ValueError(f"Expected flattened 4D tensor, got {tensor.dim()}D input.")

    na_dim = len(token_layout)
    assert na_dim in [1, 2, 3]

    if len(tile_shape) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D tiler for NA{na_dim}D, " f"got {tile_shape=}."
        )

    dilation = dilation or tuple(1 for _ in range(na_dim))

    if len(dilation) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D dilation for NA{na_dim}D, " f"got {dilation=}."
        )

    num_dilation_groups = math.prod(dilation)

    batch, seqlen, heads, dim = tensor.shape

    if batch % num_dilation_groups != 0:
        raise ValueError(
            "Expected batch size in token-permuted tensor to be divisible by "
            f"number of dilation groups {num_dilation_groups} ({dilation=}), got {batch=}."
        )

    batch_actual = batch // num_dilation_groups

    rest_shape = ceil_div_tuple(ceil_div_tuple(token_layout, tile_shape), dilation)
    token_layout_padded = mul_tuple(mul_tuple(rest_shape, tile_shape), dilation)

    # View, not copy
    rest_shape_ = reversed(rest_shape) if flip_tiled_dims else rest_shape
    tile_shape_ = reversed(tile_shape) if flip_tiled_dims else tile_shape
    dilation_ = reversed(dilation) if flip_tiled_dims else dilation
    tensor_tiled = tensor.view(
        batch_actual, *dilation_, *rest_shape_, *tile_shape_, heads, dim
    )
    if not is_torch_compiling():
        assert tensor_tiled.data_ptr() == tensor.data_ptr()

    # Undo permutation
    # batch
    permutation_idxes = [0]

    # dilation, rest, tile -> rest, tile, dilation
    for i in range(na_dim):
        if flip_tiled_dims:
            permutation_idxes += [2 * na_dim - i, 3 * na_dim - i, na_dim - i]
        else:
            permutation_idxes += [na_dim + i + 1, 2 * na_dim + i + 1, i + 1]

    # heads, head_dim
    permutation_idxes += [na_dim * 3 + 1, na_dim * 3 + 2]

    # View, not copy
    tensor_permuted = tensor_tiled.permute(*permutation_idxes)
    if not is_torch_compiling():
        assert tensor_permuted.data_ptr() == tensor_tiled.data_ptr()

    # Reshape back and copy
    out = tensor_permuted.reshape(
        batch_actual, *token_layout_padded, heads, dim
    ).contiguous()
    # NOTE: token permute without dilation is a no-op for 1-D
    # assert na_dim == 1 or out.data_ptr() != tensor_permuted.data_ptr()
    assert out.is_contiguous()

    return out


def _maybe_unpad(tensor: Tensor, padding: DimensionType):
    if tensor.dim() not in [4, 5, 6]:
        raise ValueError(
            "Expected 4D, 5D, or 6D tensor (corresponding to NA1D, 2D, 3D), "
            f"got {tensor.dim()}D input."
        )

    na_dim = tensor.dim() - 3
    assert na_dim in [1, 2, 3]

    if len(padding) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D padding shape for NA{na_dim}D, " f"got {padding=}."
        )

    token_layout = tensor.shape[1 : na_dim + 1]

    # Slice
    if any(p for p in padding):
        assert all(p >= 0 for p in padding)

        orig_lens = tuple(x - p for x, p in zip(token_layout, padding))

        # TODO: there must be a better way
        if len(orig_lens) == 1:
            x = orig_lens[0]
            return tensor[:, :x].contiguous()
        elif len(orig_lens) == 2:
            x, y = orig_lens
            return tensor[:, :x, :y].contiguous()
        elif len(orig_lens) == 3:
            x, y, z = orig_lens
            return tensor[:, :x, :y, :z].contiguous()
        else:
            raise NotImplementedError()

    return tensor


def token_permute_torch(
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

    tensor_pad = _maybe_pad(tensor, tile_shape=tile_shape, dilation=dilation)
    output = _token_permute(
        tensor_pad,
        tile_shape=tile_shape,
        dilation=dilation,
        flip_tiled_dims=flip_tiled_dims,
    )

    return output


def token_unpermute_torch(
    tensor: Tensor,
    token_layout: DimensionType,
    tile_shape: DimensionType,
    dilation: DimensionType,
    flip_tiled_dims: bool,
) -> Tensor:
    if tensor.dim() != 4:
        raise ValueError(f"Expected flattened 4D tensor, got {tensor.dim()}D input.")

    token_layout_padded = mul_tuple(
        mul_tuple(
            ceil_div_tuple(ceil_div_tuple(token_layout, tile_shape), dilation),
            dilation,
        ),
        tile_shape,
    )
    padding = sub_tuple(token_layout_padded, token_layout)

    output = _maybe_unpad(
        _token_unpermute(
            tensor,
            token_layout=token_layout,
            tile_shape=tile_shape,
            dilation=dilation,
            flip_tiled_dims=flip_tiled_dims,
        ),
        padding=padding,
    )

    return output


__all__ = [
    "token_permute_torch",
    "token_unpermute_torch",
]
