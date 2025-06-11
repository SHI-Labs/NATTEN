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

from ....types import DimensionType, FlexFmhaForwardConfigType, FlexFnaForwardConfigType
from ....utils.checks import check_tile_shape
from ....utils.device import get_device_cc


# TODO: add more tile sizes/shapes
# TODO: add backprop tile sizes/shapes
# Only doing 64 x 64 for now, since it's the one that successfully compiles across devices and
# use cases without running into compile errors (i.e. shmem over-subscription)
# Once Flex with compilation actually starts working as expected and is out of prototype, we can
# add in more tile sizes/shapes and condition them on arch / use case, like we do for CUTLASS FNA.

FLEX_FORWARD_TILE_SHAPES = {
    1: [
        # ((128, ), (128, )),
        ((64,), (64,)),
    ],
    2: [
        # ((8, 16), (8, 16)),
        ((8, 8), (8, 8)),
        ((4, 16), (4, 16)),
        ((4, 16), (8, 8)),
    ],
    3: [
        # ((4, 4, 8), (4, 4, 8)),
        ((4, 4, 4), (4, 4, 4)),
        ((2, 4, 8), (2, 4, 8)),
        ((2, 4, 8), (4, 4, 4)),
    ],
}


def _get_default_tile_shapes_forward(
    na_dim: int,
) -> FlexFnaForwardConfigType:
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
) -> List[FlexFnaForwardConfigType]:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim

    return FLEX_FORWARD_TILE_SHAPES[na_dim]  # type: ignore


# For FMHA
def get_all_tile_sizes_forward(input_tensor: Tensor) -> List[FlexFmhaForwardConfigType]:
    if input_tensor.dim() != 4:
        raise ValueError("Only 4-D tensors are supported in FMHA.")

    tile_shapes = get_all_tile_shapes_forward(input_tensor)
    assert all(len(q_t) == len(kv_t) == 1 for q_t, kv_t in tile_shapes)

    tile_sizes = [(q_t[0], kv_t[0]) for q_t, kv_t in tile_shapes]

    return tile_sizes


def get_default_forward_tile_shapes(input_tensor: Tensor) -> FlexFnaForwardConfigType:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim

    return _get_default_tile_shapes_forward(na_dim)


def get_default_forward_tile_sizes(input_tensor: Tensor) -> FlexFmhaForwardConfigType:
    if input_tensor.dim() != 4:
        raise ValueError("Only 4-D tensors are supported in FMHA.")

    q_t, kv_t = get_default_forward_tile_shapes(input_tensor)
    assert len(q_t) == len(kv_t) == 1

    return (q_t[0], kv_t[0])


def check_flex_fna_forward_config(
    input_tensor: Tensor,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
) -> FlexFnaForwardConfigType:
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
        f"Invalid configuration for Flex FNA-{na_dim}D. "
        f"Q tile shape {q_tile_shape} and KV tile shape {kv_tile_shape} "
        f"are not among the {len(tile_shapes)} configurations implementable "
        f"with Flex FNA for SM{device_cc}. "
        "Try selecting a combination from: \n"
        "  natten.get_configs_for_flex_fna(q, k, v)"
        "\n"
        "Here's a few examples of available combinations for your use case:\n"
        f"{examples}"
    )


def check_flex_fmha_forward_config(
    input_tensor: Tensor,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
) -> FlexFmhaForwardConfigType:
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
        f"Invalid configuration for Flex FMHA. "
        f"Q tile size {q_tile_size} and KV tile size {kv_tile_size} "
        f"are not among the {len(tile_sizes)} configurations implementable "
        f"with Flex FMHA for SM{device_cc}. "
        "Try selecting a combination from: \n"
        "  natten.get_configs_for_flex_fmha(q, k, v)\n"
        "\n"
        "Here's a few examples of available combinations for your use case:\n"
        f"{examples}"
    )
