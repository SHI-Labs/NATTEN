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

import itertools
import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from ....context import (
    is_kv_parallelism_in_fused_na_enabled,
    is_memory_usage_strict,
    is_memory_usage_unrestricted,
)
from ....types import (
    CutlassFmhaBackwardConfigType,
    CutlassFmhaForwardConfigType,
    CutlassFnaBackwardConfigType,
    CutlassFnaForwardConfigType,
    DimensionType,
)
from ....utils.checks import check_dilation_arg, check_tile_shape
from ....utils.device import get_device_cc, is_cuda

from ....utils.tuples import ceil_div_int, ceil_div_tuple

# FNA/FMHA forward supports 64x64 and 32x128 GEMM configs in all
# use cases. Some architectures (SM80 and SM90 )have more shared
# memory so they can handle 64x128 GEMMs.

from .fna_backward_128x128 import _FNA_BACKWARD_128x128_TILE_SIZES
from .fna_backward_128x64 import _FNA_BACKWARD_128x64_TILE_SIZES
from .fna_backward_64x64 import _FNA_BACKWARD_64x64_TILE_SIZES

# FNA/FMHA backward supports 64x64 GEMM configs in all
# use cases. Some architectures have more shared memory
# so they can handle 128x64 or 128x128 GEMMs, but that
# is also dependent on the GEMM K.

from .fna_forward_32x128 import _FNA_FORWARD_32x128_TILE_SIZES
from .fna_forward_64x128 import _FNA_FORWARD_64x128_TILE_SIZES
from .fna_forward_64x64 import _FNA_FORWARD_64x64_TILE_SIZES


def _get_default_tile_shapes_forward(
    na_dim: int,
) -> CutlassFnaForwardConfigType:
    assert na_dim in [1, 2, 3]

    if na_dim == 1:
        return ((64,), (64,))
    if na_dim == 2:
        return ((8, 8), (8, 8))
    if na_dim == 3:
        return ((4, 4, 4), (4, 4, 4))

    raise NotImplementedError()


def get_all_tile_shapes_forward(
    input_tensor: Tensor,
) -> List[CutlassFnaForwardConfigType]:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim
    device = input_tensor.device

    if not is_cuda(device):
        return []

    # SM80 and SM90 have more shared memory than SM86 and SM89
    # and their tensor core GEMMs can therefore target larger
    # tile shapes.
    # SM80 and 86 have been tested, but I don't have an SM89.
    # However, I suspect SM89 is to SM90 what SM86 was to SM80
    # in terms of shared memory (and only that).
    # Better to disable the larger tile configs for SM89 as well
    # as 86 until we can test it.
    if get_device_cc(device) in [86, 89]:
        return (
            _FNA_FORWARD_32x128_TILE_SIZES[na_dim]
            + _FNA_FORWARD_64x64_TILE_SIZES[na_dim]
        )

    return (
        _FNA_FORWARD_32x128_TILE_SIZES[na_dim]
        + _FNA_FORWARD_64x64_TILE_SIZES[na_dim]
        + _FNA_FORWARD_64x128_TILE_SIZES[na_dim]
    )


# For FMHA
def get_all_tile_sizes_forward(
    input_tensor: Tensor,
) -> List[CutlassFmhaForwardConfigType]:
    if input_tensor.dim() != 4:
        raise ValueError("Only 4-D tensors are supported in FMHA.")

    tile_shapes = get_all_tile_shapes_forward(input_tensor)
    assert all(len(q_t) == len(kv_t) == 1 for q_t, kv_t in tile_shapes)

    tile_sizes = [(q_t[0], kv_t[0]) for q_t, kv_t in tile_shapes]

    return tile_sizes


def get_default_forward_config(
    input_tensor: Tensor, dilation: Optional[DimensionType] = None
) -> CutlassFnaForwardConfigType:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim
    return _get_default_tile_shapes_forward(na_dim)


get_all_forward_configs = get_all_tile_shapes_forward
get_all_fmha_forward_configs = get_all_tile_sizes_forward


def check_cutlass_fna_forward_config(
    input_tensor: Tensor,
    dilation: Optional[DimensionType] = None,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
) -> CutlassFnaForwardConfigType:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim

    if (q_tile_shape is None) ^ (kv_tile_shape is None):
        raise ValueError(
            "Please specify both q_tile_shape and kv_tile_shape, or neither one. "
            f"Got {q_tile_shape=}, {kv_tile_shape=}."
        )

    if q_tile_shape is None and kv_tile_shape is None:
        return get_default_forward_config(input_tensor=input_tensor, dilation=dilation)

    q_tile_shape = check_tile_shape(q_tile_shape)
    kv_tile_shape = check_tile_shape(kv_tile_shape)

    tile_shapes = get_all_tile_shapes_forward(input_tensor=input_tensor)

    for q_t, kv_t in tile_shapes:
        if q_t == q_tile_shape and kv_t == kv_tile_shape:
            return (q_t, kv_t)  # type: ignore

    # Fail and make suggestions
    device_cc = get_device_cc(input_tensor.device)
    MAX_EXAMPLES = 3
    examples = ""
    for i, (q_t, kv_t) in enumerate(tile_shapes):
        examples += f"\n  q_tile_shape={q_t}, kv_tile_shape={kv_t}"
        if i > MAX_EXAMPLES:
            break

    raise ValueError(
        f"Invalid configuration for CUTLASS FNA-{na_dim}D. "
        f"Q tile shape {q_tile_shape} and KV tile shape {kv_tile_shape} "
        f"are not among the {len(tile_shapes)} configurations implementable "
        f"with CUTLASS 2.X FNA for SM{device_cc}. "
        "Try selecting a combination from: \n"
        "  natten.get_configs_for_cutlass_fna(q, k, v)"
        "\n"
        "Here's a few examples of available combinations for your use case:\n"
        f"{examples}"
    )


def check_cutlass_fmha_forward_config(
    input_tensor: Tensor,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
) -> CutlassFmhaForwardConfigType:
    assert input_tensor.dim() == 4

    if (q_tile_size is None) ^ (kv_tile_size is None):
        raise ValueError(
            "Please specify both q_tile_size and kv_tile_size, or neither one. "
            f"Got {q_tile_size=}, {kv_tile_size=}."
        )

    if q_tile_size is None and kv_tile_size is None:
        q_tile_shape, kv_tile_shape = get_default_forward_config(
            input_tensor=input_tensor
        )
        assert len(q_tile_shape) == len(kv_tile_shape) == 1
        return (q_tile_shape[0], kv_tile_shape[0])

    tile_sizes = get_all_tile_sizes_forward(input_tensor=input_tensor)

    for q_t, kv_t in tile_sizes:
        if q_t == q_tile_size and kv_t == kv_tile_size:
            return (q_t, kv_t)

    # Fail and make suggestions
    device_cc = get_device_cc(input_tensor.device)
    MAX_EXAMPLES = 3
    examples = ""
    for i, (q_t, kv_t) in enumerate(tile_sizes):
        examples += f"\n  q_tile_size={q_t}, kv_tile_size={kv_t}"
        if i > MAX_EXAMPLES:
            break

    raise ValueError(
        f"Invalid configuration for CUTLASS FMHA. "
        f"Q tile size {q_tile_size} and KV tile size {kv_tile_size} "
        f"are not among the {len(tile_sizes)} configurations implementable "
        f"with CUTLASS 2.X FMHA for SM{device_cc}. "
        "Try selecting a combination from: \n"
        "  natten.get_configs_for_cutlass_fmha(q, k, v)"
        "\n"
        "Here's a few examples of available combinations for your use case:\n"
        f"{examples}"
    )


###### Backward


def _get_default_tile_shapes_backward(
    na_dim: int,
) -> Tuple[DimensionType, DimensionType]:
    assert na_dim in [1, 2, 3]

    if na_dim == 1:
        return ((64,), (64,))
    if na_dim == 2:
        return ((8, 8), (8, 8))
    if na_dim == 3:
        return ((4, 4, 4), (4, 4, 4))

    raise NotImplementedError()


def get_all_tile_shapes_backward(
    input_tensor: Tensor,
) -> List[CutlassFnaForwardConfigType]:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim
    device = input_tensor.device
    dtype = input_tensor.dtype
    dim_per_head = input_tensor.shape[-1]

    if not is_cuda(device):
        return []

    compute_cap = get_device_cc(device)

    assert dtype in [torch.float32, torch.float16, torch.bfloat16]

    if dtype == torch.float32 and compute_cap not in [80, 90]:
        return _FNA_BACKWARD_64x64_TILE_SIZES[na_dim]

    elif dtype == torch.float32:
        return (
            _FNA_BACKWARD_64x64_TILE_SIZES[na_dim]
            + _FNA_BACKWARD_128x64_TILE_SIZES[na_dim]
        )

    if compute_cap == 70:
        return (
            _FNA_BACKWARD_64x64_TILE_SIZES[na_dim]
            + _FNA_BACKWARD_128x64_TILE_SIZES[na_dim]
        )

    if compute_cap in [80, 90] and dim_per_head <= 128:
        return (
            _FNA_BACKWARD_64x64_TILE_SIZES[na_dim]
            + _FNA_BACKWARD_128x128_TILE_SIZES[na_dim]
        )
    elif compute_cap in [80, 90]:
        return (
            _FNA_BACKWARD_64x64_TILE_SIZES[na_dim]
            + _FNA_BACKWARD_128x64_TILE_SIZES[na_dim]
        )

    return _FNA_BACKWARD_64x64_TILE_SIZES[na_dim]


# For FMHA
def get_all_tile_sizes_backward(
    input_tensor: Tensor,
) -> List[CutlassFmhaForwardConfigType]:
    if input_tensor.dim() != 4:
        raise ValueError("Only 4-D tensors are supported in FMHA.")

    tile_shapes = get_all_tile_shapes_backward(input_tensor)
    assert all(len(q_t) == len(kv_t) == 1 for q_t, kv_t in tile_shapes)

    tile_sizes = [(q_t[0], kv_t[0]) for q_t, kv_t in tile_shapes]

    return tile_sizes


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
) -> DimensionType:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim
    dilation = check_dilation_arg(na_dim, dilation)
    input_shape: DimensionType = tuple(int(x) for x in input_tensor.shape[1 : na_dim + 1])  # type: ignore

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


def get_default_backward_config(
    input_tensor: Tensor, dilation: Optional[DimensionType] = None
) -> CutlassFnaBackwardConfigType:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim
    dilation = check_dilation_arg(na_dim, dilation)

    q_tile_shape, kv_tile_shape = _get_default_tile_shapes_backward(na_dim)
    kv_splits = get_default_kv_splits_backward(
        input_tensor=input_tensor, kv_tile_shape=kv_tile_shape, dilation=dilation
    )
    use_pt_reduction = False
    return (q_tile_shape, kv_tile_shape, kv_splits, use_pt_reduction)  # type: ignore


# WARNING: use with caution
# There's quite a lot of configurations possible, especially if we don't restrict
# KV parallelism.
def get_all_backward_configs(
    input_tensor: Tensor, dilation: Optional[DimensionType] = None
) -> List[CutlassFnaBackwardConfigType]:

    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim
    dilation = check_dilation_arg(na_dim, dilation)

    input_shape: DimensionType = tuple(x for x in input_tensor.shape[1 : na_dim + 1])  # type: ignore
    assert len(input_shape) == na_dim and all(isinstance(x, int) for x in input_shape)
    batch_size = input_tensor.shape[0]
    num_heads = input_tensor.shape[-2]

    possible_tile_shapes = get_all_tile_shapes_backward(input_tensor)

    possible_configs = []
    assert len(input_shape) == len(dilation) == na_dim
    num_dilation_splits = math.prod(dilation)
    max_kv_splits_allowed = (
        1
        if not is_kv_parallelism_in_fused_na_enabled()
        else max(
            1,
            _get_max_grid_size_allowed()
            // (batch_size * num_heads * num_dilation_splits),
        )
    )
    for query_tile_shape, kv_tile_shape in possible_tile_shapes:
        min_kv_splits = get_min_splits(na_dim)
        max_kv_splits = get_max_splits(
            input_shape, dilation=dilation, kv_tile_shape=kv_tile_shape
        )
        max_kv_splits_total = math.prod(max_kv_splits)
        if max_kv_splits_total > max_kv_splits_allowed:
            max_kv_splits = _reduce_max_kv_splits(
                na_dim=na_dim, kv_splits=max_kv_splits, max_splits=max_kv_splits_allowed
            )

        # Potential duplicates
        if math.prod(max_kv_splits) > 1:
            for kv_splits in _get_possible_kv_splits(min_kv_splits, max_kv_splits):
                for use_pt_reduction in [False, True]:
                    possible_configs.append(
                        (
                            query_tile_shape,
                            kv_tile_shape,
                            kv_splits,
                            use_pt_reduction,
                        )
                    )
        else:
            # min_kv_splits == max_kv_splits
            for use_pt_reduction in [False, True]:
                possible_configs.append(
                    (query_tile_shape, kv_tile_shape, min_kv_splits, use_pt_reduction)
                )

    return possible_configs  # type: ignore


def get_all_fmha_backward_configs(
    input_tensor: Tensor, dilation: Optional[DimensionType] = None
) -> List[CutlassFmhaBackwardConfigType]:
    if input_tensor.dim() != 4:
        raise ValueError("Only 4-D tensors are supported in FMHA.")

    if dilation is not None:
        raise ValueError(f"FMHA does not support dilation, got {dilation=}.")

    backward_configs = get_all_backward_configs(input_tensor)
    assert all(
        len(q_t) == len(kv_t) == len(kv_s) == 1
        for q_t, kv_t, kv_s, _ in backward_configs
    )

    fmha_backward_configs = [
        (q_t[0], kv_t[0], kv_s[0], use_pt_red)
        for q_t, kv_t, kv_s, use_pt_red in backward_configs
    ]

    return fmha_backward_configs


def check_cutlass_fna_backward_config(
    input_tensor: Tensor,
    dilation: Optional[DimensionType] = None,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    kv_splits: Optional[DimensionType] = None,
    use_pt_reduction: bool = False,
) -> CutlassFnaBackwardConfigType:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim
    dilation = check_dilation_arg(na_dim, dilation)

    if (q_tile_shape is None) ^ (kv_tile_shape is None):
        raise ValueError(
            "Please specify both q_tile_shape and kv_tile_shape, or neither one. "
            f"Got {q_tile_shape=}, {kv_tile_shape=}."
        )

    if q_tile_shape is None and kv_tile_shape is None:
        q_tile_shape, kv_tile_shape, _, _ = get_default_backward_config(
            input_tensor=input_tensor, dilation=dilation
        )

    q_tile_shape = check_tile_shape(q_tile_shape)
    kv_tile_shape = check_tile_shape(kv_tile_shape)

    tile_shapes = get_all_tile_shapes_backward(input_tensor)

    for q_t, kv_t in tile_shapes:
        if q_t == q_tile_shape and kv_t == kv_tile_shape:
            kv_splits = kv_splits or get_default_kv_splits_backward(
                input_tensor=input_tensor,
                kv_tile_shape=kv_tile_shape,
                dilation=dilation,
            )
            return (q_t, kv_t, kv_splits, use_pt_reduction)  # type: ignore

    # Fail and make suggestions
    device_cc = get_device_cc(input_tensor.device)
    MAX_EXAMPLES = 3
    examples = ""
    for i, (q_t, kv_t) in enumerate(tile_shapes):
        examples += f"\n  q_tile_shape={q_t}, kv_tile_shape={kv_t}"
        if i > MAX_EXAMPLES:
            break

    raise ValueError(
        f"Invalid configuration for CUTLASS FNA-{na_dim}D. "
        f"Q tile shape {q_tile_shape} and KV tile shape {kv_tile_shape} "
        f"are not among the {len(tile_shapes)} configurations implementable "
        f"with CUTLASS 2.X FNA for SM{device_cc}, with input tensor shape "
        f"{input_tensor.shape} and {dilation=}. Try selecting a combination from: \n"
        "  natten.get_bwd_configs_for_cutlass_fna(q, k, v)"
        "\n"
        "Here's a few examples of available combinations for your use case:\n"
        f"{examples}"
    )


def check_fmha_kv_splits(
    kv_splits: Optional[int], input_tensor: Tensor, kv_tile_size: int
) -> int:
    if kv_splits is not None and isinstance(kv_splits, int):
        return kv_splits

    if kv_splits is None:
        default_kv_splits: DimensionType = get_default_kv_splits_backward(
            input_tensor=input_tensor, kv_tile_shape=(kv_tile_size,)
        )
        assert len(default_kv_splits) == 1
        return default_kv_splits[0]

    raise ValueError(f"Invalid type {type(kv_splits)} for kv_splits.")


def check_cutlass_fmha_backward_config(
    input_tensor: Tensor,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
    kv_splits: Optional[int] = None,
    use_pt_reduction: bool = False,
) -> CutlassFmhaBackwardConfigType:
    assert input_tensor.dim() == 4

    if (q_tile_size is None) ^ (kv_tile_size is None):
        raise ValueError(
            "Please specify both q_tile_size and kv_tile_size, or neither one. "
            f"Got {q_tile_size=}, {kv_tile_size=}."
        )

    if q_tile_size is None and kv_tile_size is None:
        q_tile_shape, kv_tile_shape, _, _ = get_default_backward_config(
            input_tensor=input_tensor
        )
        assert len(q_tile_shape) == len(kv_tile_shape) == 1
        q_tile_size, kv_tile_size = q_tile_shape[0], kv_tile_shape[0]

    tile_sizes = get_all_tile_sizes_backward(input_tensor)

    for q_t, kv_t in tile_sizes:
        if q_t == q_tile_size and kv_t == kv_tile_size:
            kvs = check_fmha_kv_splits(
                kv_splits, input_tensor=input_tensor, kv_tile_size=kv_t
            )
            return (q_t, kv_t, kvs, use_pt_reduction)

    # Fail and make suggestions
    device_cc = get_device_cc(input_tensor.device)
    MAX_EXAMPLES = 3
    examples = ""
    for i, (q_t, kv_t) in enumerate(tile_sizes):
        examples += f"\n  q_tile_size={q_t}, kv_tile_size={kv_t}"
        if i > MAX_EXAMPLES:
            break

    raise ValueError(
        f"Invalid configuration for CUTLASS FMHA. "
        f"Q tile size {q_tile_size} and KV tile size {kv_tile_size} "
        f"are not among the {len(tile_sizes)} configurations implementable "
        f"with CUTLASS 2.X FNA for SM{device_cc}, with input tensor shape "
        f"{input_tensor.shape}. Try selecting a combination from: \n"
        "  natten.get_bwd_configs_for_cutlass_fmha(q, k, v)"
        "\n"
        "Here's a few examples of available combinations for your use case:\n"
        f"{examples}"
    )
