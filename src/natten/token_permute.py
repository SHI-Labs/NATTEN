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

from .utils import log

logger = log.get_logger(__name__)


DISABLE_PADDING_WARNING = True
TOKEN_PERMUTE_PADDING_RATIO_LIMIT_UNTIL_WARNING = 0.5


def maybe_pad(tensor, tile_shape, dilation=None):
    if tensor.dim() not in [4, 5, 6]:
        raise ValueError(
            "Expected 4D, 5D, or 6D tensor (corresponding to NA1D, 2D, 3D), "
            f"got {tensor.dim()}D input."
        )

    na_dim = tensor.dim() - 3
    assert na_dim in [1, 2, 3]

    if len(tile_shape) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D tiler for NA{na_dim}, " f"got {tile_shape=}."
        )

    if dilation is not None and len(dilation) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D dilation for NA{na_dim}, " f"got {dilation=}."
        )

    attn_dims = tensor.shape[1 : na_dim + 1]
    tile_shape_ = tuple(x for x in tile_shape)
    if dilation is not None:
        # NOTE: LCM?
        # tile_shape_ = tuple(math.lcm(t, d) for t, d in zip(tile_shape, dilation))
        tile_shape_ = tuple(t * d for t, d in zip(tile_shape, dilation))

    rest = tuple((x + t - 1) // t for x, t in zip(attn_dims, tile_shape_))
    residual = tuple(r * t - x for x, t, r in zip(attn_dims, tile_shape_, rest))

    assert all(res >= 0 for res in residual)

    if not DISABLE_PADDING_WARNING and any(
        res / sz > TOKEN_PERMUTE_PADDING_RATIO_LIMIT_UNTIL_WARNING
        for res, sz in zip(residual, attn_dims)
    ):
        padded_attn_dims = tuple(x + p for x, p in zip(attn_dims, residual))
        logger.warning(
            "Potentially excessive padding detected in token permute: "
            f"input shape {attn_dims} will be padded to {padded_attn_dims} to handle "
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

    return tensor_padded, residual


def token_permute(tensor, tile_shape, dilation=None, flip_tiled_dims: bool = True):
    if tensor.dim() not in [4, 5, 6]:
        raise ValueError(
            "Expected 4D, 5D, or 6D tensor (corresponding to NA1D, 2D, 3D), "
            f"got {tensor.dim()}D input."
        )

    na_dim = tensor.dim() - 3
    assert na_dim in [1, 2, 3]

    if len(tile_shape) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D tiler for NA{na_dim}, " f"got {tile_shape=}."
        )

    dilation = dilation or tuple(1 for _ in range(na_dim))

    if len(dilation) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D dilation for NA{na_dim}, " f"got {dilation=}."
        )

    batch, *attn_dims, heads, dim = tensor.shape

    if any(
        x % d != 0 or (x // d) % t != 0
        for x, t, d in zip(attn_dims, tile_shape, dilation)
    ):
        raise ValueError(
            "Tensor must be divisible by static tile shape and dilation, but got "
            f"{tensor.shape=}, {tile_shape=}, {dilation=}."
        )

    num_dilation_groups = math.prod(dilation)
    attn_dims_post_dilation = tuple(x // d for x, d in zip(attn_dims, dilation))
    rest = tuple(x // d // t for x, t, d in zip(attn_dims, tile_shape, dilation))
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
        + permutation_idxes_r
        + permutation_idxes_t
        + permutation_idxes_d
        + [na_dim * 3 + 1, na_dim * 3 + 2]
    )

    # View, not copy
    tensor_tiled = tensor.view(batch, *logical_divide_dims, heads, dim)
    assert tensor_tiled.data_ptr() == tensor.data_ptr()

    # View, not copy
    tensor_permuted = tensor_tiled.permute(*permutation_idxes)
    assert tensor_permuted.data_ptr() == tensor_tiled.data_ptr()

    # Reshape back and copy
    tensor_flatten = tensor_permuted.reshape(
        batch, math.prod(attn_dims_post_dilation), num_dilation_groups * heads, dim
    ).contiguous()
    # NOTE: token permute without dilation is a no-op for 1-D
    # assert na_dim == 1 or tensor_flatten.data_ptr() != tensor_permuted.data_ptr()
    assert tensor_flatten.is_contiguous()

    return tensor_flatten, attn_dims_post_dilation, rest


def token_unpermute(
    tensor,
    tile_shape,
    orig_shape,
    rest_shape,
    dilation=None,
    flip_tiled_dims: bool = True,
):
    if tensor.dim() != 4:
        raise ValueError(f"Expected flattened 4D tensor, got {tensor.dim()}D input.")

    na_dim = len(orig_shape)
    assert na_dim in [1, 2, 3]

    if len(tile_shape) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D tiler for NA{na_dim}, " f"got {tile_shape=}."
        )

    if len(rest_shape) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D rest shape for NA{na_dim}, " f"got {rest_shape=}."
        )

    dilation = dilation or tuple(1 for _ in range(na_dim))

    if len(dilation) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D dilation for NA{na_dim}, " f"got {dilation=}."
        )

    num_dilation_groups = math.prod(dilation)

    batch, seqlen, heads, dim = tensor.shape

    if heads % num_dilation_groups != 0:
        raise ValueError(
            "Expected number of heads in token-permuted tensor to be divisible by "
            f"number of dilation groups {num_dilation_groups} ({dilation=}), got {heads=}."
        )

    heads_actual = heads // num_dilation_groups

    orig_shape_pre_dilation = tuple(x * d for x, d in zip(orig_shape, dilation))

    # View, not copy
    rest_shape_ = reversed(rest_shape) if flip_tiled_dims else rest_shape
    tile_shape_ = reversed(tile_shape) if flip_tiled_dims else tile_shape
    dilation_ = reversed(dilation) if flip_tiled_dims else dilation
    tensor_tiled = tensor.view(
        batch, *rest_shape_, *tile_shape_, *dilation_, heads_actual, dim
    )
    assert tensor_tiled.data_ptr() == tensor.data_ptr()

    # Undo permutation
    permutation_idxes = [0]
    for i in range(na_dim):
        if flip_tiled_dims:
            permutation_idxes += [na_dim - i, 2 * na_dim - i, 3 * na_dim - i]
        else:
            permutation_idxes += [i + 1, na_dim + i + 1, 2 * na_dim + i + 1]

    permutation_idxes += [na_dim * 3 + 1, na_dim * 3 + 2]

    # View, not copy
    tensor_permuted = tensor_tiled.permute(*permutation_idxes)
    assert tensor_permuted.data_ptr() == tensor_tiled.data_ptr()

    # Reshape back and copy
    out = tensor_permuted.reshape(
        batch, *orig_shape_pre_dilation, heads_actual, dim
    ).contiguous()
    # NOTE: token permute without dilation is a no-op for 1-D
    # assert na_dim == 1 or out.data_ptr() != tensor_permuted.data_ptr()
    assert out.is_contiguous()

    return out


def maybe_unpad(tensor, padding):
    if tensor.dim() not in [4, 5, 6]:
        raise ValueError(
            "Expected 4D, 5D, or 6D tensor (corresponding to NA1D, 2D, 3D), "
            f"got {tensor.dim()}D input."
        )

    na_dim = tensor.dim() - 3
    assert na_dim in [1, 2, 3]

    if len(padding) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D padding shape for NA{na_dim}, " f"got {padding=}."
        )

    attn_dims = tensor.shape[1 : na_dim + 1]

    # Slice
    if any(p for p in padding):
        assert all(p >= 0 for p in padding)

        orig_lens = tuple(x - p for x, p in zip(attn_dims, padding))

        # TODO: there must be a beter way
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
