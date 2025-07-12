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


from typing import List

from ...utils import log

logger = log.get_logger(__name__)

import torch  # noqa: F401
from torch import Tensor

from ...types import (
    CutlassBlackwellFmhaBackwardConfigType,
    CutlassBlackwellFmhaForwardConfigType,
    CutlassBlackwellFnaBackwardConfigType,
    CutlassBlackwellFnaForwardConfigType,
    CutlassFmhaForwardConfigType,
    CutlassFnaForwardConfigType,
    CutlassHopperFmhaBackwardConfigType,
    CutlassHopperFmhaForwardConfigType,
    CutlassHopperFnaBackwardConfigType,
    CutlassHopperFnaForwardConfigType,
    FlexFmhaForwardConfigType,
    FlexFnaForwardConfigType,
)

from .checks import (
    can_run_cutlass_blackwell_fmha,
    can_run_cutlass_blackwell_fna,
    can_run_cutlass_fmha,
    can_run_cutlass_fna,
    can_run_cutlass_hopper_fmha,
    can_run_cutlass_hopper_fna,
    can_run_flex_attention,
)

from .cutlass import (
    get_all_tile_shapes_backward as get_all_cutlass_fna_backward_configs,
    get_all_tile_shapes_forward as get_all_cutlass_fna_forward_configs,
    get_all_tile_sizes_backward as get_all_cutlass_fmha_backward_configs,
    get_all_tile_sizes_forward as get_all_cutlass_fmha_forward_configs,
)

from .cutlass_blackwell import (
    get_all_backward_configs as get_all_blackwell_fna_backward_configs,
    get_all_fmha_backward_configs as get_all_blackwell_fmha_backward_configs,
    get_all_fmha_forward_configs as get_all_blackwell_fmha_forward_configs,
    get_all_forward_configs as get_all_blackwell_fna_forward_configs,
)

from .cutlass_hopper import (
    get_all_backward_configs as get_all_hopper_fna_backward_configs,
    get_all_fmha_backward_configs as get_all_hopper_fmha_backward_configs,
    get_all_fmha_forward_configs as get_all_hopper_fmha_forward_configs,
    get_all_forward_configs as get_all_hopper_fna_forward_configs,
)

from .flex import (
    get_all_tile_shapes_forward as get_all_flex_fna_forward_configs,
    get_all_tile_sizes_forward as get_all_flex_fmha_forward_configs,
)


### CUTLASS Blackwell kernels


def get_configs_for_cutlass_blackwell_fmha(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> List[CutlassBlackwellFmhaForwardConfigType]:
    """Returns Blackwell FMHA configurations compatible with input tensors, if any.

    Checks first if a CUDA tensor, and on a Blackwell datacenter GPU (SM100; compute capability
    10.0), and if so, returns *forward pass* configurations compatible with the tensor dtype and
    head dim.

    Each configuration for this operation is a tuple of two integers: `(q_tile_size,
    kv_tile_size)`. These are arguments to [natten.attention][natten.attention].

    Args:
        query: Query tensor matching the shape, dtype, and device of your use case.
        key:   Key tensor matching the shape, dtype, and device of your use case.
        value: Value tensor matching the shape, dtype, and device of your use case.

    Returns:
        (List[Tuple[int, int]]): List of tuples of two integers corresponding to query and KV tile
            sizes.
    """

    if not can_run_cutlass_blackwell_fmha(
        query=query, key=key, value=value, raise_error=False
    ):
        return []

    return get_all_blackwell_fmha_forward_configs(input_tensor=query)


def get_bwd_configs_for_cutlass_blackwell_fmha(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> List[CutlassBlackwellFmhaBackwardConfigType]:
    """Returns Blackwell FMHA backward pass configurations compatible with input tensors, if any.

    Checks first if a CUDA tensor, and on a Blackwell datacenter GPU (SM100; compute capability
    10.0), and if so, returns *backward pass* configurations compatible with the tensor dtype and
    head dim.

    Each configuration for this operation is a tuple of two integers: `(q_tile_size,
    kv_tile_size)`. These are arguments to [natten.attention][natten.attention].

    Args:
        query: Query tensor matching the shape, dtype, and device of your use case.
        key:   Key tensor matching the shape, dtype, and device of your use case.
        value: Value tensor matching the shape, dtype, and device of your use case.

    Returns:
        (List[Tuple[int, int]]): List of tuples of two integers corresponding to query and KV tile
            sizes.
    """

    if not can_run_cutlass_blackwell_fmha(
        query=query, key=key, value=value, raise_error=False
    ):
        return []

    return get_all_blackwell_fmha_backward_configs(input_tensor=query)


def get_configs_for_cutlass_blackwell_fna(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> List[CutlassBlackwellFnaForwardConfigType]:
    """Returns Blackwell FNA configurations compatible with input tensors, if any.

    Checks first if a CUDA tensor, and on a Blackwell datacenter GPU (SM100; compute capability
    10.0), and if so, returns *forward pass* configurations compatible with the tensor dtype and
    head dim, and according to the rank of the token layout (1D/2D/3D).

    Each configuration for this operation is a tuple of two integer tuples: `(q_tile_shape,
    kv_tile_shape)`. These are arguments to [natten.na1d][natten.na1d], [natten.na2d][natten.na2d],
    and [natten.na3d][natten.na3d].

    Args:
        query: Query tensor matching the shape, dtype, and device of your use case.
        key:   Key tensor matching the shape, dtype, and device of your use case.
        value: Value tensor matching the shape, dtype, and device of your use case.

    Returns:
        (List[Tuple[tuple, tuple]]): List of tuples of two integer tuples corresponding to query
            and KV tile *shapes*.
    """
    if not can_run_cutlass_blackwell_fna(
        query=query, key=key, value=value, raise_error=False
    ):
        return []

    return get_all_blackwell_fna_forward_configs(input_tensor=query)


def get_bwd_configs_for_cutlass_blackwell_fna(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> List[CutlassBlackwellFnaBackwardConfigType]:
    """Returns Blackwell FNA backward pass configurations compatible with input tensors, if any.

    Checks first if a CUDA tensor, and on a Blackwell datacenter GPU (SM100; compute capability
    10.0), and if so, returns *backward pass* configurations compatible with the tensor dtype and
    head dim, and according to the rank of the token layout (1D/2D/3D).

    Each configuration for this operation is a tuple of two integer tuples: `(q_tile_shape,
    kv_tile_shape)`. These are arguments to [natten.na1d][natten.na1d], [natten.na2d][natten.na2d],
    and [natten.na3d][natten.na3d].

    Args:
        query: Query tensor matching the shape, dtype, and device of your use case.
        key:   Key tensor matching the shape, dtype, and device of your use case.
        value: Value tensor matching the shape, dtype, and device of your use case.

    Returns:
        (List[Tuple[tuple, tuple]]): List of tuples of two integer tuples corresponding to query
            and KV tile *shapes*.
    """
    if not can_run_cutlass_blackwell_fna(
        query=query, key=key, value=value, raise_error=False
    ):
        return []

    return get_all_blackwell_fna_backward_configs(input_tensor=query)


### CUTLASS Hopper kernels


def get_configs_for_cutlass_hopper_fmha(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> List[CutlassHopperFmhaForwardConfigType]:
    """Returns Hopper FMHA configurations compatible with input tensors, if any.

    Checks first if a CUDA tensor, and on a Hopper GPU (SM90; compute capability 9.0), and if so,
    returns *forward pass* configurations compatible with the tensor dtype and head dim.

    Each configuration for this operation is a tuple of one integer tuple, and another integer:
    `((q_tile_size, kv_tile_size), kernel_schedule)`. These are arguments to
    [natten.attention][natten.attention].
    `kernel_schedule` is specific to Hopper FNA/FMHA only.

    Args:
        query: Query tensor matching the shape, dtype, and device of your use case.
        key:   Key tensor matching the shape, dtype, and device of your use case.
        value: Value tensor matching the shape, dtype, and device of your use case.

    Returns:
        (List[Tuple[Tuple[int, int], KernelSchedule]]): List of tuples of one tuple of two integers
            corresponding to query and KV tile sizes, and a kernel schedule enum type.
    """
    if not can_run_cutlass_hopper_fmha(
        query=query, key=key, value=value, raise_error=False
    ):
        return []

    return get_all_hopper_fmha_forward_configs(input_tensor=query)


def get_bwd_configs_for_cutlass_hopper_fmha(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> List[CutlassHopperFmhaBackwardConfigType]:
    """Returns Hopper FMHA backward pass configurations compatible with input tensors, if any.

    Checks first if a CUDA tensor, and on a Hopper GPU (SM90; compute capability 9.0), and if so,
    returns *backward pass* configurations compatible with the tensor dtype and head dim.

    Each configuration for this operation is an integer tuple:
    `((backward_q_tile_size, backward_kv_tile_size), kernel_schedule)`. These are arguments to
    [natten.attention][natten.attention].

    Note that unlike forward pass, kernel schedule is not part of the configuration. All backward
    pass kernels are persistent warp-specialized.
    See CUTLASS's [example 88](https://github.com/NVIDIA/cutlass/tree/main/examples/88_hopper_fmha)
    for more.

    Args:
        query: Query tensor matching the shape, dtype, and device of your use case.
        key:   Key tensor matching the shape, dtype, and device of your use case.
        value: Value tensor matching the shape, dtype, and device of your use case.

    Returns:
        (List[Tuple[int, int]]): List of integer tuples corresponding to query and KV tile sizes.
    """
    if not can_run_cutlass_hopper_fmha(
        query=query, key=key, value=value, raise_error=False
    ):
        return []

    return get_all_hopper_fmha_backward_configs(input_tensor=query)


def get_configs_for_cutlass_hopper_fna(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> List[CutlassHopperFnaForwardConfigType]:
    """Returns Hopper FNA configurations compatible with input tensors, if any.

    Checks first if a CUDA tensor, and on a Hopper GPU (SM90; compute capability 9.0), and if so,
    returns *forward pass* configurations compatible with the tensor dtype and head dim.

    Each configuration for this operation is a tuple of one tuple, and another integer:
    `((q_tile_shape, kv_tile_shape), kernel_schedule)`. These are arguments to
    [natten.na1d][natten.na1d], [natten.na2d][natten.na2d], and [natten.na3d][natten.na3d].

    `kernel_schedule` is specific to Hopper FNA/FMHA only.

    Args:
        query: Query tensor matching the shape, dtype, and device of your use case.
        key:   Key tensor matching the shape, dtype, and device of your use case.
        value: Value tensor matching the shape, dtype, and device of your use case.

    Returns:
        (List[Tuple[Tuple[tuple, tuple], KernelSchedule]]): List of tuples of one tuple of two
            shape tuples, corresponding to query and KV tile *shapes*, and a kernel schedule enum
            type.
    """
    if not can_run_cutlass_hopper_fna(
        query=query, key=key, value=value, raise_error=False
    ):
        return []

    return get_all_hopper_fna_forward_configs(input_tensor=query)


def get_bwd_configs_for_cutlass_hopper_fna(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> List[CutlassHopperFnaBackwardConfigType]:
    """Returns Hopper FNA backward pass configurations compatible with input tensors, if any.

    Checks first if a CUDA tensor, and on a Hopper GPU (SM90; compute capability 9.0), and if so,
    returns *backward pass* configurations compatible with the tensor dtype and head dim.

    Each configuration for this operation is a tuple of two tuples:
    `(q_tile_shape, kv_tile_shape)`. These are arguments to [natten.na1d][natten.na1d],
    [natten.na2d][natten.na2d], and [natten.na3d][natten.na3d].

    Note that unlike forward pass, kernel schedule is not part of the configuration. All backward
    pass kernels are persistent warp-specialized.
    See CUTLASS's [example 88](https://github.com/NVIDIA/cutlass/tree/main/examples/88_hopper_fmha)
    for more.

    Args:
        query: Query tensor matching the shape, dtype, and device of your use case.
        key:   Key tensor matching the shape, dtype, and device of your use case.
        value: Value tensor matching the shape, dtype, and device of your use case.

    Returns:
        (List[Tuple[tuple, tuple]]): List of tuples of two shape tuples, corresponding to query and
        KV tile *shapes*.
    """
    if not can_run_cutlass_hopper_fna(
        query=query, key=key, value=value, raise_error=False
    ):
        return []

    return get_all_hopper_fna_backward_configs(input_tensor=query)


### CUTLASS 2.X kernels


def get_configs_for_cutlass_fmha(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> List[CutlassFmhaForwardConfigType]:
    """Returns CUTLASS FMHA configurations compatible with input tensors, if any.

    Checks first if a CUDA tensor, and on a device with compute capability >= 5.0, and if so,
    returns *forward pass* configurations compatible with the specific compute capability, tensor
    dtype and head dim.

    Each configuration for this operation is a tuple of two integers: `(q_tile_size,
    kv_tile_size)`. These are arguments to [natten.attention][natten.attention].

    Args:
        query: Query tensor matching the shape, dtype, and device of your use case.
        key:   Key tensor matching the shape, dtype, and device of your use case.
        value: Value tensor matching the shape, dtype, and device of your use case.

    Returns:
        (List[Tuple[int, int]]): List of tuples of two integers corresponding to query and KV tile
            sizes.
    """
    if not can_run_cutlass_fmha(query=query, key=key, value=value, raise_error=False):
        return []

    return get_all_cutlass_fmha_forward_configs(input_tensor=query)


def get_bwd_configs_for_cutlass_fmha(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> List[CutlassFmhaForwardConfigType]:
    """Returns CUTLASS FMHA backward pass configurations compatible with input tensors, if any.

    Checks first if a CUDA tensor, and on a device with compute capability >= 5.0, and if so,
    returns *backward pass* configurations compatible with the specific compute capability, tensor
    dtype and head dim.

    Each configuration for this operation is a tuple of two integers: `(backward_q_tile_size,
    backward_kv_tile_size)`. These are arguments to [natten.attention][natten.attention].

    Args:
        query: Query tensor matching the shape, dtype, and device of your use case.
        key:   Key tensor matching the shape, dtype, and device of your use case.
        value: Value tensor matching the shape, dtype, and device of your use case.

    Returns:
        (List[Tuple[int, int]]): List of tuples of two integers corresponding to query and KV tile
            sizes in the *backward pass*.
    """
    if not can_run_cutlass_fmha(query=query, key=key, value=value, raise_error=False):
        return []

    return get_all_cutlass_fmha_backward_configs(
        input_tensor=key if key.shape[-1] >= value.shape[-1] else value
    )


def get_configs_for_cutlass_fna(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> List[CutlassFnaForwardConfigType]:
    """Returns CUTLASS FNA configurations compatible with input tensors, if any.

    Checks first if a CUDA tensor, and on a device with compute capability >= 5.0, and if so,
    returns *forward pass* configurations compatible with the specific compute capability, tensor
    dtype and head dim, and according to the rank of the token layout (1D/2D/3D).

    Each configuration for this operation is a tuple of two integer tuples: `(q_tile_shape,
    kv_tile_shape)`. These are arguments to [natten.na1d][natten.na1d], [natten.na2d][natten.na2d],
    and [natten.na3d][natten.na3d].

    Args:
        query: Query tensor matching the shape, dtype, and device of your use case.
        key:   Key tensor matching the shape, dtype, and device of your use case.
        value: Value tensor matching the shape, dtype, and device of your use case.

    Returns:
        (List[Tuple[tuple, tuple]]): List of tuples of two integer tuples corresponding to query
            and KV tile *shapes*.
    """
    if not can_run_cutlass_fna(query=query, key=key, value=value, raise_error=False):
        return []

    return get_all_cutlass_fna_forward_configs(input_tensor=query)


def get_bwd_configs_for_cutlass_fna(
    query: Tensor,
    key: Tensor,
    value: Tensor,
) -> List[CutlassFnaForwardConfigType]:
    """Returns CUTLASS FNA backward pass configurations compatible with input tensors, if any.

    Checks first if a CUDA tensor, and on a device with compute capability >= 5.0, and if so,
    returns *backward pass* configurations compatible with the specific compute capability, tensor
    dtype and head dim.

    Each configuration for this operation is a tuple of two integers: `(backward_q_tile_shape,
    backward_kv_tile_shape)`. These are arguments to [natten.na1d][natten.na1d],
    [natten.na2d][natten.na2d], and [natten.na3d][natten.na3d].

    Args:
        query: Query tensor matching the shape, dtype, and device of your use case.
        key:   Key tensor matching the shape, dtype, and device of your use case.
        value: Value tensor matching the shape, dtype, and device of your use case.

    Returns:
        (List[Tuple[tuple, tuple]]): List of tuples of two integer tuples corresponding to query
            and KV tile *shapes* in the *backward pass*.
    """
    if not can_run_cutlass_fna(query=query, key=key, value=value, raise_error=False):
        return []

    return get_all_cutlass_fna_backward_configs(
        input_tensor=key if key.shape[-1] >= value.shape[-1] else value
    )


### Flex


def get_configs_for_flex_fmha(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    torch_compile: bool = False,
) -> List[FlexFmhaForwardConfigType]:
    """Returns Flex FMHA configurations compatible with input tensors, if any.

    Each configuration for this operation is a tuple of two integers: `(q_tile_size,
    kv_tile_size)`. These are arguments to [natten.attention][natten.attention].
    Not specifying these arguments while backend is Flex will default to `q_tile_size = 64` and
    `kv_tile_size = 64`.

    Args:
        query: Query tensor matching the shape, dtype, and device of your use case.
        key:   Key tensor matching the shape, dtype, and device of your use case.
        value: Value tensor matching the shape, dtype, and device of your use case.
        torch_compile: Whether or not you intend to use compiled block mask and flex attention kernel.

    Returns:
        (List[Tuple[int, int]]): List of tuples of two integers corresponding to query and KV tile
            sizes.
    """
    if not can_run_flex_attention(
        query=query,
        key=key,
        value=value,
        torch_compile=torch_compile,
        raise_error=False,
    ):
        return []

    return get_all_flex_fmha_forward_configs(input_tensor=query)


def get_configs_for_flex_fna(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    torch_compile: bool = False,
) -> List[FlexFnaForwardConfigType]:
    """Returns Flex FNA configurations compatible with input tensors, if any.

    Each configuration for this operation is a tuple of two integer tuples: `(q_tile_shape,
    kv_tile_shape)`. These are arguments to [natten.na1d][natten.na1d], [natten.na2d][natten.na2d],
    and [natten.na3d][natten.na3d].
    Not specifying these arguments while backend is Flex will default to single-dimensional tiling,
    and will not use our Token Permutation approach. By explicitly specifying tile shapes, you will
    automatically use our Token Permutation approach, which saves you the most compute.

    Args:
        query: Query tensor matching the shape, dtype, and device of your use case.
        key:   Key tensor matching the shape, dtype, and device of your use case.
        value: Value tensor matching the shape, dtype, and device of your use case.
        torch_compile: Whether or not you intend to use compiled block mask and flex attention kernel.

    Returns:
        (List[Tuple[tuple, tuple]]): List of tuples of two integer tuples corresponding to query
            and KV tile *shapes*.
    """
    if not can_run_flex_attention(
        query=query,
        key=key,
        value=value,
        torch_compile=torch_compile,
        raise_error=False,
    ):
        return []

    return get_all_flex_fna_forward_configs(input_tensor=query)
