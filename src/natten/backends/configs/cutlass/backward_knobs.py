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

# Backward pass "knobs" for CUTLASS FNA/FMHA:
#   - kv_splits: number of KV splits for parallelism
#   - use_pt_reduction: whether to use PyTorch for delta computation
#
# These are independent of tile shape selection and are validated/defaulted
# in the torch ops (torch_wrappers.py), not in the config selection logic.

import itertools
import math
from typing import Optional

import torch
from torch import Tensor

from natten.context import (
    is_kv_parallelism_in_fused_na_enabled,
    is_memory_usage_strict,
    is_memory_usage_unrestricted,
)
from natten.types import DimensionType
from natten.utils.checks import check_dilation_arg, check_input_size_arg
from natten.utils.tuples import ceil_div_int, ceil_div_tuple


def _get_max_grid_size_allowed() -> int:
    if is_memory_usage_unrestricted():
        return 65535
    if is_memory_usage_strict():
        return 1024

    return 4096


def get_min_splits(na_dim: int) -> DimensionType:
    assert na_dim in [1, 2, 3]
    return tuple(1 for _ in range(na_dim))  # type: ignore


def get_max_splits(
    input_shape: DimensionType, dilation: DimensionType, kv_tile_shape: DimensionType
) -> DimensionType:
    extent_per_dilation_group = ceil_div_tuple(input_shape, dilation)
    return tuple(
        ceil_div_int(x, t) for x, t in zip(extent_per_dilation_group, kv_tile_shape)
    )  # type: ignore


def _reduce_max_kv_splits(
    na_dim: int,
    kv_splits: DimensionType,
    max_splits: int,
) -> DimensionType:
    assert isinstance(kv_splits, tuple)
    assert na_dim in [1, 2, 3]

    if na_dim == 1:
        assert len(kv_splits) == 1
        return (min(kv_splits[0], max_splits),)

    if na_dim == 2:
        assert len(kv_splits) == 2
        splits_x = max(min(max_splits // 2, kv_splits[0]), 1)
        splits_y = max(min(max_splits // splits_x, kv_splits[1]), 1)
        assert (
            0 < splits_x * splits_y <= max_splits
        ), f"{splits_x=} * {splits_y=} does not fall in range [0, {max_splits}]"
        return (splits_x, splits_y)

    if na_dim == 3:
        assert len(kv_splits) == 3
        splits_x = max(min(max_splits // 3, kv_splits[0]), 1)
        splits_y = max(min(max_splits // splits_x, kv_splits[1]), 1)
        splits_z = max(min(max_splits // (splits_x * splits_y), kv_splits[2]), 1)
        assert (
            0 < splits_x * splits_y * splits_z <= max_splits
        ), f"{splits_x=} * {splits_y=} * {splits_z=} does not fall in range [0, {max_splits}]"
        return (splits_x, splits_y, splits_z)

    raise NotImplementedError()


def _get_possible_kv_splits(
    min_splits: DimensionType,
    max_splits: DimensionType,
):
    assert 0 < len(min_splits) == len(max_splits) < 4
    na_dim = len(max_splits)
    if na_dim == 1:
        return itertools.product(
            range(min_splits[0], max_splits[0] + 1),
        )
    if na_dim == 2:
        assert len(min_splits) == len(max_splits) == 2
        return itertools.product(
            range(min_splits[0], max_splits[0] + 1),
            range(min_splits[1], max_splits[1] + 1),
        )
    if na_dim == 3:
        assert len(min_splits) == len(max_splits) == 3
        return itertools.product(
            range(min_splits[0], max_splits[0] + 1),
            range(min_splits[1], max_splits[1] + 1),
            range(min_splits[2], max_splits[2] + 1),
        )

    raise NotImplementedError()


def get_default_kv_splits_backward(
    input_tensor: Tensor,
    kv_tile_shape: DimensionType,
    dilation: Optional[DimensionType] = None,
    max_seqlen: Optional[DimensionType] = None,
) -> DimensionType:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim
    dilation = check_dilation_arg(na_dim, dilation)
    input_shape: DimensionType = tuple(int(x) for x in input_tensor.shape[1 : na_dim + 1])  # type: ignore
    if max_seqlen is not None:
        input_shape = check_input_size_arg(na_dim, max_seqlen)

    assert na_dim in [1, 2, 3]
    if na_dim == 1:
        kv_splits: DimensionType = (1,)
    elif na_dim == 2:
        kv_splits = (1, 1)

    elif na_dim == 3:
        kv_splits = (1, 1, 1)

    if (
        is_kv_parallelism_in_fused_na_enabled()
        and not torch.are_deterministic_algorithms_enabled()
    ):
        kv_splits = get_max_splits(
            input_shape, dilation=dilation, kv_tile_shape=kv_tile_shape
        )
        total_kv_splits = math.prod(kv_splits)

        batch_size = input_tensor.shape[0]
        num_heads = input_tensor.shape[-2]
        num_dilation_splits = math.prod(dilation)
        max_kv_splits_allowed = max(
            1,
            _get_max_grid_size_allowed()
            // (batch_size * num_heads * num_dilation_splits),
        )

        if total_kv_splits > max_kv_splits_allowed:
            kv_splits = _reduce_max_kv_splits(
                na_dim=na_dim, kv_splits=kv_splits, max_splits=max_kv_splits_allowed
            )

    return kv_splits


def check_fmha_kv_splits(
    kv_splits: Optional[int],
    input_tensor: Tensor,
    kv_tile_size: int,
    max_seqlen: Optional[int] = None,
) -> int:
    if kv_splits is not None and isinstance(kv_splits, int):
        seqlen_kv = input_tensor.shape[1] if max_seqlen is None else max_seqlen
        num_kv_tiles = (seqlen_kv + kv_tile_size - 1) // kv_tile_size
        assert num_kv_tiles > 0
        return min(num_kv_tiles, kv_splits)

    if kv_splits is None:
        max_seqlen_tuple = None if max_seqlen is None else (max_seqlen,)
        default_kv_splits: DimensionType = get_default_kv_splits_backward(
            input_tensor=input_tensor,
            kv_tile_shape=(kv_tile_size,),
            max_seqlen=max_seqlen_tuple,
        )
        assert len(default_kv_splits) == 1
        return default_kv_splits[0]

    raise ValueError(f"Invalid type {type(kv_splits)} for kv_splits.")


def check_fna_kv_splits(
    kv_splits: Optional[DimensionType],
    input_tensor: Tensor,
    kv_tile_shape: DimensionType,
    dilation: Optional[DimensionType] = None,
) -> DimensionType:
    if kv_splits is not None and isinstance(kv_splits, tuple):
        na_dim = input_tensor.dim() - 3
        dilation = check_dilation_arg(na_dim, dilation)
        input_shape: DimensionType = tuple(int(x) for x in input_tensor.shape[1 : na_dim + 1])  # type: ignore
        max_kv_splits = get_max_splits(
            input_shape, dilation=dilation, kv_tile_shape=kv_tile_shape
        )
        return tuple(min(s, m) for s, m in zip(kv_splits, max_kv_splits))  # type: ignore

    if kv_splits is None:
        return get_default_kv_splits_backward(
            input_tensor=input_tensor,
            kv_tile_shape=kv_tile_shape,
            dilation=dilation,
        )

    raise ValueError(f"Invalid type {type(kv_splits)} for kv_splits.")
