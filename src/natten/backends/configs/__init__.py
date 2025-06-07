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
    CutlassBlackwellFmhaForwardConfigType,
    CutlassBlackwellFnaForwardConfigType,
    CutlassFmhaForwardConfigType,
    CutlassFnaForwardConfigType,
    CutlassHopperFmhaForwardConfigType,
    CutlassHopperFnaForwardConfigType,
    FlexFmhaForwardConfigType,
    FlexFnaForwardConfigType,
)

from .cutlass import (
    get_all_tile_shapes_backward as get_all_cutlass_fna_backward_configs,
    get_all_tile_shapes_forward as get_all_cutlass_fna_forward_configs,
    get_all_tile_sizes_backward as get_all_cutlass_fmha_backward_configs,
    get_all_tile_sizes_forward as get_all_cutlass_fmha_forward_configs,
)

from .cutlass_blackwell import (
    get_all_fmha_forward_configs as get_all_blackwell_fmha_forward_configs,
    get_all_forward_configs as get_all_blackwell_fna_forward_configs,
)

from .cutlass_hopper import (
    get_all_fmha_forward_configs as get_all_hopper_fmha_forward_configs,
    get_all_forward_configs as get_all_hopper_fna_forward_configs,
)

from .flex import (
    get_all_tile_shapes_forward as get_all_flex_fna_forward_configs,
    get_all_tile_sizes_forward as get_all_flex_fmha_forward_configs,
)


### CUTLASS Blackwell kernels


def get_configs_for_cutlass_blackwell_fmha(
    input_tensor: Tensor,
) -> List[CutlassBlackwellFmhaForwardConfigType]:
    """Returns Blackwell FMHA configurations compatible with input tensor, if any.

    Checks first if a CUDA tensor, and on a Blackwell datacenter GPU (SM100; compute capability
    10.0), and if so, returns *forward pass* configurations compatible with the tensor dtype and
    head dim.

    Each configuration for this operation is a tuple of two integers: `(q_tile_size,
    kv_tile_size)`. These are arguments to [natten.attention][natten.attention].

    Args:
        input_tensor: Input torch tensor. Either Q, K, or V. Since NATTEN does not support GQA/MQA,
            or V with a different head dim, it doesn't make a difference which one is passed in.

    Returns:
        (List[Tuple[int, int]]): List of tuples of two integers corresponding to query and KV tile
            sizes.
    """
    return get_all_blackwell_fmha_forward_configs(input_tensor)


def get_configs_for_cutlass_blackwell_fna(
    input_tensor: Tensor,
) -> List[CutlassBlackwellFnaForwardConfigType]:
    """Returns Blackwell FNA configurations compatible with input tensor, if any.

    Checks first if a CUDA tensor, and on a Blackwell datacenter GPU (SM100; compute capability
    10.0), and if so, returns *forward pass* configurations compatible with the tensor dtype and
    head dim, and according to the rank of the token layout (1D/2D/3D).

    Each configuration for this operation is a tuple of two integer tuples: `(q_tile_shape,
    kv_tile_shape)`. These are arguments to [natten.na1d][natten.na1d], [natten.na2d][natten.na2d],
    and [natten.na3d][natten.na3d].

    Args:
        input_tensor: Input torch tensor. Either Q, K, or V. Since NATTEN does not support GQA/MQA,
            or V with a different head dim, and NA operations require matching token layouts, Q, K,
            and V are guaranteed to have the same shape, therefore it doesn't make a difference
            which one is passed in.

    Returns:
        (List[Tuple[tuple, tuple]]): List of tuples of two integer tuples corresponding to query
            and KV tile *shapes*.
    """
    return get_all_blackwell_fna_forward_configs(input_tensor)


### CUTLASS Hopper kernels


def get_configs_for_cutlass_hopper_fmha(
    input_tensor: Tensor,
) -> List[CutlassHopperFmhaForwardConfigType]:
    """Returns Hopper FMHA configurations compatible with input tensor, if any.

    Checks first if a CUDA tensor, and on a Hopper datacenter GPU (SM90; compute capability
    9.0), and if so, returns *forward pass* configurations compatible with the tensor dtype and
    head dim.

    Each configuration for this operation is a tuple of one integer tuple, and another integer:
    `((q_tile_size, kv_tile_size), kernel_schedule)`. These are arguments to
    [natten.attention][natten.attention].
    `kernel_schedule` is specific to Hopper FNA/FMHA only.

    Args:
        input_tensor: Input torch tensor. Either Q, K, or V. Since NATTEN does not support GQA/MQA,
            or V with a different head dim, it doesn't make a difference which one is passed in.

    Returns:
        (List[Tuple[Tuple[int, int], KernelSchedule]]): List of tuples of one tuple of two integers
            corresponding to query and KV tile sizes, and a kernel schedule enum type.
    """
    return get_all_hopper_fmha_forward_configs(input_tensor)


def get_configs_for_cutlass_hopper_fna(
    input_tensor: Tensor,
) -> List[CutlassHopperFnaForwardConfigType]:
    """Returns Hopper FNA configurations compatible with input tensor, if any.

    Checks first if a CUDA tensor, and on a Hopper datacenter GPU (SM90; compute capability
    9.0), and if so, returns *forward pass* configurations compatible with the tensor dtype and
    head dim.

    Each configuration for this operation is a tuple of one tuple, and another integer:
    `((q_tile_shape, kv_tile_shape), kernel_schedule)`. These are arguments to
    [natten.attention][natten.attention].
    `kernel_schedule` is specific to Hopper FNA/FMHA only.

    Args:
        input_tensor: Input torch tensor. Either Q, K, or V. Since NATTEN does not support GQA/MQA,
            or V with a different head dim, and NA operations require matching token layouts, Q, K,
            and V are guaranteed to have the same shape, therefore it doesn't make a difference
            which one is passed in.

    Returns:
        (List[Tuple[Tuple[tuple, tuple], KernelSchedule]]): List of tuples of one tuple of two
            shape tuples, corresponding to query and KV tile *shapes*, and a kernel schedule enum
            type.
    """
    return get_all_hopper_fna_forward_configs(input_tensor)


### CUTLASS 2.X kernels


def get_configs_for_cutlass_fmha(
    input_tensor: Tensor,
) -> List[CutlassFmhaForwardConfigType]:
    """Returns CUTLASS FMHA configurations compatible with input tensor, if any.

    Checks first if a CUDA tensor, and on a device with compute capability >= 5.0, and if so,
    returns *forward pass* configurations compatible with the specific compute capability, tensor
    dtype and head dim.

    Each configuration for this operation is a tuple of two integers: `(q_tile_size,
    kv_tile_size)`. These are arguments to [natten.attention][natten.attention].

    Args:
        input_tensor: Input torch tensor. Either Q, K, or V. Since NATTEN does not support GQA/MQA,
            or V with a different head dim, it doesn't make a difference which one is passed in.

    Returns:
        (List[Tuple[int, int]]): List of tuples of two integers corresponding to query and KV tile
            sizes.
    """
    return get_all_cutlass_fmha_forward_configs(input_tensor)


def get_bwd_configs_for_cutlass_fmha(
    input_tensor: Tensor,
) -> List[CutlassFmhaForwardConfigType]:
    """Returns CUTLASS FMHA backward pass configurations compatible with input tensor, if any.

    Checks first if a CUDA tensor, and on a device with compute capability >= 5.0, and if so,
    returns *backward pass* configurations compatible with the specific compute capability, tensor
    dtype and head dim.

    Each configuration for this operation is a tuple of two integers: `(backward_q_tile_size,
    backward_kv_tile_size)`. These are arguments to [natten.attention][natten.attention].

    Args:
        input_tensor: Input torch tensor. Either Q, K, or V. Since NATTEN does not support GQA/MQA,
            or V with a different head dim, it doesn't make a difference which one is passed in.

    Returns:
        (List[Tuple[int, int]]): List of tuples of two integers corresponding to query and KV tile
            sizes in the *backward pass*.
    """
    return get_all_cutlass_fmha_backward_configs(input_tensor)


def get_configs_for_cutlass_fna(
    input_tensor: Tensor,
) -> List[CutlassFnaForwardConfigType]:
    """Returns CUTLASS FNA configurations compatible with input tensor, if any.

    Checks first if a CUDA tensor, and on a device with compute capability >= 5.0, and if so,
    returns *forward pass* configurations compatible with the specific compute capability, tensor
    dtype and head dim, and according to the rank of the token layout (1D/2D/3D).

    Each configuration for this operation is a tuple of two integer tuples: `(q_tile_shape,
    kv_tile_shape)`. These are arguments to [natten.na1d][natten.na1d], [natten.na2d][natten.na2d],
    and [natten.na3d][natten.na3d].

    Args:
        input_tensor: Input torch tensor. Either Q, K, or V. Since NATTEN does not support GQA/MQA,
            or V with a different head dim, and NA operations require matching token layouts, Q, K,
            and V are guaranteed to have the same shape, therefore it doesn't make a difference
            which one is passed in.

    Returns:
        (List[Tuple[tuple, tuple]]): List of tuples of two integer tuples corresponding to query
            and KV tile *shapes*.
    """
    return get_all_cutlass_fna_forward_configs(input_tensor)


def get_bwd_configs_for_cutlass_fna(
    input_tensor: Tensor,
) -> List[CutlassFnaForwardConfigType]:
    """Returns CUTLASS FNA backward pass configurations compatible with input tensor, if any.

    Checks first if a CUDA tensor, and on a device with compute capability >= 5.0, and if so,
    returns *backward pass* configurations compatible with the specific compute capability, tensor
    dtype and head dim.

    Each configuration for this operation is a tuple of two integers: `(backward_q_tile_shape,
    backward_kv_tile_shape)`. These are arguments to [natten.na1d][natten.na1d],
    [natten.na2d][natten.na2d], and [natten.na3d][natten.na3d].

    Args:
        input_tensor: Input torch tensor. Either Q, K, or V. Since NATTEN does not support GQA/MQA,
            or V with a different head dim, and NA operations require matching token layouts, Q, K,
            and V are guaranteed to have the same shape, therefore it doesn't make a difference
            which one is passed in.

    Returns:
        (List[Tuple[tuple, tuple]]): List of tuples of two integer tuples corresponding to query
            and KV tile *shapes* in the *backward pass*.
    """
    return get_all_cutlass_fna_backward_configs(input_tensor)


### Flex


def get_configs_for_flex_fmha(
    input_tensor: Tensor,
) -> List[FlexFmhaForwardConfigType]:
    """Returns Flex FMHA configurations compatible with input tensor, if any.

    Each configuration for this operation is a tuple of two integers: `(q_tile_size,
    kv_tile_size)`. These are arguments to [natten.attention][natten.attention].
    Not specifying these arguments while backend is Flex will default to `q_tile_size = 64` and
    `kv_tile_size = 64`.

    Args:
        input_tensor: Input torch tensor. Either Q, K, or V. Since NATTEN does not support GQA/MQA,
            or V with a different head dim, it doesn't make a difference which one is passed in.

    Returns:
        (List[Tuple[int, int]]): List of tuples of two integers corresponding to query and KV tile
            sizes.
    """
    return get_all_flex_fmha_forward_configs(input_tensor)


def get_configs_for_flex_fna(
    input_tensor: Tensor,
) -> List[FlexFnaForwardConfigType]:
    """Returns Flex FNA configurations compatible with input tensor, if any.

    Each configuration for this operation is a tuple of two integer tuples: `(q_tile_shape,
    kv_tile_shape)`. These are arguments to [natten.na1d][natten.na1d], [natten.na2d][natten.na2d],
    and [natten.na3d][natten.na3d].
    Not specifying these arguments while backend is Flex will default to single-dimensional tiling,
    and will not use our Token Permutation approach. By explicitly specifying tile shapes, you will
    automatically use our Token Permutation approach, which saves you the most compute.

    Args:
        input_tensor: Input torch tensor. Either Q, K, or V. Since NATTEN does not support GQA/MQA,
            or V with a different head dim, and NA operations require matching token layouts, Q, K,
            and V are guaranteed to have the same shape, therefore it doesn't make a difference
            which one is passed in.

    Returns:
        (List[Tuple[tuple, tuple]]): List of tuples of two integer tuples corresponding to query
            and KV tile *shapes*.
    """
    return get_all_flex_fna_forward_configs(input_tensor)
