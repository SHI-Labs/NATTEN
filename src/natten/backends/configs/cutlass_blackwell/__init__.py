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

from typing import List, Optional

import torch  # noqa: F401
from torch import Tensor

from ....types import (
    CutlassBlackwellFmhaBackwardConfigType,
    CutlassBlackwellFmhaForwardConfigType,
    CutlassBlackwellFnaBackwardConfigType,
    CutlassBlackwellFnaForwardConfigType,
    DimensionType,
)
from ....utils.checks import check_tile_shape
from ....utils.device import get_device_cc


# The current CUTLASS FMHA forward kernel can only do Q tile size 256, KV tile size 128.
# This limits 1D tile shapes to just the one, but for 2-D and 3-D we can have many more shapes,
# only some of which we compile. Adding new ones requires adding them to autogen, regenerating
# the instantiations, and recompiling libnatten. Unlike CUTLASS 2.X FNA, multi-dim tile shapes are
# static in Blackwell FNA, and not dynamic.

BLACKWELL_FORWARD_TILE_SHAPES = {
    1: [
        ((256,), (128,)),
    ],
    2: [
        ((16, 16), (16, 8)),
        ((16, 16), (8, 16)),
        ((8, 32), (8, 16)),
        ((8, 32), (4, 32)),
    ],
    3: [
        ((8, 4, 8), (4, 4, 8)),
        ((8, 4, 8), (2, 8, 8)),
        ((2, 8, 16), (4, 4, 8)),
        ((2, 8, 16), (2, 8, 8)),
        ((4, 4, 16), (2, 4, 16)),
        ((2, 16, 8), (2, 8, 8)),
        ((4, 8, 8), (2, 8, 8)),
    ],
}

BLACKWELL_BACKWARD_TILE_SHAPES = {
    1: [
        ((128,), (128,)),
    ],
    2: [
        ((16, 8), (16, 8)),
        ((16, 8), (8, 16)),
        ((8, 16), (16, 8)),
        ((8, 16), (8, 16)),
    ],
    3: [
        ((4, 4, 8), (4, 4, 8)),
        ((4, 4, 8), (2, 8, 8)),
        ((1, 8, 16), (4, 4, 8)),
        ((2, 8, 8), (4, 4, 8)),
        ((1, 8, 16), (2, 8, 8)),
        ((2, 4, 16), (2, 4, 16)),
        ((4, 2, 16), (2, 4, 16)),
        ((4, 4, 8), (2, 4, 16)),
        ((2, 8, 8), (2, 8, 8)),
    ],
}


def _get_default_tile_shapes_forward(
    na_dim: int,
) -> CutlassBlackwellFnaForwardConfigType:
    assert na_dim in [1, 2, 3]

    if na_dim == 1:
        return ((256,), (128,))
    if na_dim == 2:
        return ((16, 16), (16, 8))
    if na_dim == 3:
        return ((8, 4, 8), (4, 4, 8))

    raise NotImplementedError()


def get_all_forward_configs(
    input_tensor: Tensor,
) -> List[CutlassBlackwellFnaForwardConfigType]:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim

    device_cc = get_device_cc(input_tensor.device)
    if device_cc != 100:
        return []

    return BLACKWELL_FORWARD_TILE_SHAPES[na_dim]  # type: ignore


def get_all_backward_configs(
    input_tensor: Tensor,
) -> List[CutlassBlackwellFnaBackwardConfigType]:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim

    device_cc = get_device_cc(input_tensor.device)
    if device_cc != 100:
        return []

    return BLACKWELL_BACKWARD_TILE_SHAPES[na_dim]  # type: ignore


# For FMHA
def get_all_fmha_forward_configs(
    input_tensor: Tensor,
) -> List[CutlassBlackwellFmhaForwardConfigType]:
    if input_tensor.dim() != 4:
        raise ValueError("Only 4-D tensors are supported in FMHA.")

    tile_shapes = get_all_forward_configs(input_tensor)
    assert all(len(q_t) == len(kv_t) == 1 for q_t, kv_t in tile_shapes)

    tile_sizes = [(q_t[0], kv_t[0]) for q_t, kv_t in tile_shapes]

    return tile_sizes


def get_all_fmha_backward_configs(
    input_tensor: Tensor,
) -> List[CutlassBlackwellFmhaBackwardConfigType]:
    if input_tensor.dim() != 4:
        raise ValueError("Only 4-D tensors are supported in FMHA.")

    tile_shapes = get_all_backward_configs(input_tensor)
    assert all(len(q_t) == len(kv_t) == 1 for q_t, kv_t in tile_shapes)

    tile_sizes = [(q_t[0], kv_t[0]) for q_t, kv_t in tile_shapes]

    return tile_sizes


def get_default_forward_tile_shapes(
    input_tensor: Tensor,
) -> CutlassBlackwellFnaForwardConfigType:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim

    return _get_default_tile_shapes_forward(na_dim)


def get_default_forward_tile_sizes(
    input_tensor: Tensor,
) -> CutlassBlackwellFmhaForwardConfigType:
    if input_tensor.dim() != 4:
        raise ValueError("Only 4-D tensors are supported in FMHA.")

    q_t, kv_t = get_default_forward_tile_shapes(input_tensor)
    assert len(q_t) == len(kv_t) == 1

    return (q_t[0], kv_t[0])


def get_default_backward_tile_shapes(
    input_tensor: Tensor,
) -> CutlassBlackwellFnaBackwardConfigType:
    all_configs = get_all_backward_configs(input_tensor)

    if len(all_configs) < 1:
        device_cc = get_device_cc(input_tensor.device)
        raise ValueError(
            "No configs exist for this use case; Blackwell FMHA/FNA does not support it: "
            f"{input_tensor.shape=}, {input_tensor.dtype=}, {device_cc=}."
        )

    return all_configs[0]


def get_default_backward_tile_sizes(
    input_tensor: Tensor,
) -> CutlassBlackwellFmhaBackwardConfigType:
    if input_tensor.dim() != 4:
        raise ValueError("Only 4-D tensors are supported in FMHA.")

    q_t, kv_t = get_default_backward_tile_shapes(input_tensor)
    assert len(q_t) == len(kv_t) == 1

    return (q_t[0], kv_t[0])


def check_cutlass_blackwell_fna_forward_config(
    input_tensor: Tensor,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
) -> CutlassBlackwellFnaForwardConfigType:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim

    if (q_tile_shape is None) ^ (kv_tile_shape is None):
        raise ValueError(
            "Please specify both q_tile_shape and kv_tile_shape, or neither one. "
            f"Got {q_tile_shape=}, {kv_tile_shape=}."
        )

    if q_tile_shape is None and kv_tile_shape is None:
        return get_default_forward_tile_shapes(input_tensor=input_tensor)

    q_tile_shape = check_tile_shape(q_tile_shape)
    kv_tile_shape = check_tile_shape(kv_tile_shape)

    tile_shapes = get_all_forward_configs(input_tensor=input_tensor)

    for q_t, kv_t in tile_shapes:
        if q_t == q_tile_shape and kv_t == kv_tile_shape:
            return (q_t, kv_t)  # type: ignore

    # Fail and make suggestions
    MAX_EXAMPLES = 3
    examples = ""
    for i, (q_t, kv_t) in enumerate(tile_shapes):
        examples += f"\n  q_tile_shape={q_t}, kv_tile_shape={kv_t}"
        if i > MAX_EXAMPLES:
            break

    raise ValueError(
        f"Invalid configuration for CUTLASS Blackwell FNA-{na_dim}D. "
        f"Q tile shape {q_tile_shape} and KV tile shape {kv_tile_shape} "
        f"are not among the {len(tile_shapes)} configurations implementable "
        f"with CUTLASS Blackwell FNA. "
        "Try selecting a combination from: \n"
        "  natten.get_configs_for_blackwell_fna(q, k, v)"
        "\n"
        "Here's a few examples of available combinations for your use case:\n"
        f"{examples}"
    )


def check_cutlass_blackwell_fmha_forward_config(
    input_tensor: Tensor,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
) -> CutlassBlackwellFmhaForwardConfigType:
    assert input_tensor.dim() == 4

    if (q_tile_size is None) ^ (kv_tile_size is None):
        raise ValueError(
            "Please specify both q_tile_size and kv_tile_size, or neither one. "
            f"Got {q_tile_size=}, {kv_tile_size=}."
        )

    if q_tile_size is None and kv_tile_size is None:
        q_tile_size, kv_tile_size = get_default_forward_tile_sizes(
            input_tensor=input_tensor
        )
        return (q_tile_size, kv_tile_size)

    tile_sizes = get_all_fmha_forward_configs(input_tensor=input_tensor)

    for q_t, kv_t in tile_sizes:
        if q_t == q_tile_size and kv_t == kv_tile_size:
            return (q_t, kv_t)

    # Fail and make suggestions
    MAX_EXAMPLES = 3
    examples = ""
    for i, (q_t, kv_t) in enumerate(tile_sizes):
        examples += f"\n  q_tile_size={q_t}, kv_tile_size={kv_t}"
        if i > MAX_EXAMPLES:
            break

    raise ValueError(
        f"Invalid configuration for CUTLASS Blackwell FMHA. "
        f"Q tile size {q_tile_size} and KV tile size {kv_tile_size} "
        f"are not among the {len(tile_sizes)} configurations implementable "
        f"with CUTLASS Blackwell FMHA. "
        "Try selecting a combination from: \n"
        "  natten.get_configs_for_blackwell_fmha(q, k, v)"
        "\n"
        "Here's a few examples of available combinations for your use case:\n"
        f"{examples}"
    )


def check_cutlass_blackwell_fna_backward_config(
    input_tensor: Tensor,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
) -> CutlassBlackwellFnaBackwardConfigType:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim

    if (q_tile_shape is None) ^ (kv_tile_shape is None):
        raise ValueError(
            "Please specify both q_tile_shape and kv_tile_shape, or neither one. "
            f"Got {q_tile_shape=}, {kv_tile_shape=}."
        )

    if q_tile_shape is None and kv_tile_shape is None:
        return get_default_backward_tile_shapes(input_tensor=input_tensor)

    q_tile_shape = check_tile_shape(q_tile_shape)
    kv_tile_shape = check_tile_shape(kv_tile_shape)

    tile_shapes = get_all_backward_configs(input_tensor=input_tensor)

    for q_t, kv_t in tile_shapes:
        if q_t == q_tile_shape and kv_t == kv_tile_shape:
            return (q_t, kv_t)  # type: ignore

    # Fail and make suggestions
    MAX_EXAMPLES = 3
    examples = ""
    for i, (q_t, kv_t) in enumerate(tile_shapes):
        examples += f"\n  q_tile_shape={q_t}, kv_tile_shape={kv_t}"
        if i > MAX_EXAMPLES:
            break

    raise ValueError(
        f"Invalid configuration for CUTLASS Blackwell FNA-{na_dim}D Backward. "
        f"Q tile shape {q_tile_shape} and KV tile shape {kv_tile_shape} "
        f"are not among the {len(tile_shapes)} configurations implementable "
        f"with CUTLASS Blackwell FNA Backward. "
        "Try selecting a combination from: \n"
        "  natten.get_bwd_configs_for_blackwell_fna(q, k, v)"
        "\n"
        "Here's a few examples of available combinations for your use case:\n"
        f"{examples}"
    )


def check_cutlass_blackwell_fmha_backward_config(
    input_tensor: Tensor,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
) -> CutlassBlackwellFmhaBackwardConfigType:
    assert input_tensor.dim() == 4

    if (q_tile_size is None) ^ (kv_tile_size is None):
        raise ValueError(
            "Please specify both q_tile_size and kv_tile_size, or neither one. "
            f"Got {q_tile_size=}, {kv_tile_size=}."
        )

    if q_tile_size is None and kv_tile_size is None:
        q_tile_size, kv_tile_size = get_default_backward_tile_sizes(
            input_tensor=input_tensor
        )
        return (q_tile_size, kv_tile_size)

    tile_sizes = get_all_fmha_backward_configs(input_tensor=input_tensor)

    for q_t, kv_t in tile_sizes:
        if q_t == q_tile_size and kv_t == kv_tile_size:
            return (q_t, kv_t)

    # Fail and make suggestions
    MAX_EXAMPLES = 3
    examples = ""
    for i, (q_t, kv_t) in enumerate(tile_sizes):
        examples += f"\n  q_tile_size={q_t}, kv_tile_size={kv_t}"
        if i > MAX_EXAMPLES:
            break

    raise ValueError(
        f"Invalid configuration for CUTLASS Blackwell FMHA Backward. "
        f"Q tile size {q_tile_size} and KV tile size {kv_tile_size} "
        f"are not among the {len(tile_sizes)} configurations implementable "
        f"with CUTLASS Blackwell FMHA Backward. "
        "Try selecting a combination from: \n"
        "  natten.get_bwd_configs_for_blackwell_fmha(q, k, v)"
        "\n"
        "Here's a few examples of available combinations for your use case:\n"
        f"{examples}"
    )
