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
from collections.abc import Mapping
from functools import partial
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from natten.types import (
    CausalArgType,
    CausalArgTypeOrDed,
    DimensionType,
    DimensionTypeOrDed,
)
from natten.utils.checks import (
    check_all_args,
    check_args_against_token_layout,
    check_dilation_arg,
)
from natten.utils.environment import is_torch_compiling
from natten.utils.tuples import ceil_div_tuple, idx2crd, mul_tuple


def _get_dilated_token_layouts(
    token_layout: DimensionType, dilation: DimensionType, flip_tiled_dims: bool
):
    """
    Tiles token_layout into dilation groups, and returns a list of the token layout within each
    dilation group. Number of dilation groups is always size(dilation).
    If dilation evenly divides token_layout, then all groups have the same token layout shape.
    Otherwise, each dimension not evenly dividing can vary by 1.
    """
    if math.prod(dilation) == 1:
        return [token_layout]

    if flip_tiled_dims:
        dilation_ = tuple(x for x in reversed(dilation))
        token_layout_ = tuple(x for x in reversed(token_layout))
    else:
        dilation_ = dilation
        token_layout_ = token_layout

    num_dilation_groups = math.prod(dilation)
    token_layout_list = []
    for dig in range(num_dilation_groups):
        dig_crd = idx2crd(dig, dilation_)

        # Fixup
        dilation_group_padding = tuple(
            1 - ((dg + (d - (x % d))) // d)
            for dg, d, x in zip(dig_crd, dilation_, token_layout_)
        )
        token_layout_dig = tuple(
            (x // d) + p
            for p, d, x in zip(dilation_group_padding, dilation_, token_layout_)
        )

        if flip_tiled_dims:
            token_layout_dig = tuple(x for x in reversed(token_layout_dig))

        token_layout_list.append(token_layout_dig)

    return token_layout_list


VariableDimensionType = Optional[List[DimensionType]]


def _verify_variable_dimension_type(
    batch_size: int, parameter_list: Optional[list], parameter_name: str
):
    if parameter_list is None:
        return

    if not isinstance(parameter_list, list):
        raise ValueError(
            f"{parameter_name} must be a list, got {type(parameter_list)} ({parameter_list})."
        )

    if len(parameter_list) != batch_size:
        raise ValueError(
            f"{parameter_name} must contain {batch_size} elements, got {len(parameter_list)} ({parameter_list})."
        )


def _verify_variable_parameters(
    na_dim: int,
    token_layout_list: List[DimensionType],
    kernel_size: DimensionTypeOrDed,
    stride: DimensionTypeOrDed,
    dilation: DimensionTypeOrDed,
    is_causal: Optional[CausalArgTypeOrDed],
    # var-param
    kernel_size_list: VariableDimensionType,
    stride_list: VariableDimensionType,
    dilation_list: VariableDimensionType,
    device: torch.device,
) -> Tuple[
    Tuple[DimensionType, DimensionType, DimensionType, CausalArgType],
    Tuple[VariableDimensionType, VariableDimensionType, VariableDimensionType],
    Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]],
]:
    batch_size = len(token_layout_list)

    is_varparam = (
        kernel_size_list is not None
        or stride_list is not None
        or dilation_list is not None
    )
    has_variable_kernel_sizes = kernel_size_list is not None
    has_variable_strides = stride_list is not None
    has_variable_dilations = dilation_list is not None

    if is_varparam:
        # Dummy static params, so that end-users can just specify Nones for them
        kernel_size = (
            tuple(2 for _ in range(na_dim))  # type: ignore[assignment]
            if has_variable_kernel_sizes
            else kernel_size
        )
        stride = 1 if has_variable_strides else stride
        dilation = 1 if has_variable_dilations else dilation

    kernel_size, stride, dilation, is_causal = check_all_args(
        na_dim, kernel_size, stride, dilation, is_causal
    )

    static_params = (kernel_size, stride, dilation, is_causal)
    if not is_varparam:
        for b in range(batch_size):
            check_args_against_token_layout(
                token_layout=token_layout_list[b],
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
            )

        return static_params, (None, None, None), (None, None, None)

    _verify_variable_dimension_type(
        batch_size=batch_size,
        parameter_list=kernel_size_list,
        parameter_name="kernel_size_list",
    )
    _verify_variable_dimension_type(
        batch_size=batch_size, parameter_list=stride_list, parameter_name="stride_list"
    )
    _verify_variable_dimension_type(
        batch_size=batch_size,
        parameter_list=dilation_list,
        parameter_name="dilation_list",
    )

    kernel_size_list_out: list | None = [] if kernel_size_list is not None else None
    stride_list_out: list | None = [] if stride_list is not None else None
    dilation_list_out: list | None = [] if dilation_list is not None else None

    for b in range(batch_size):
        kernel_size_ = kernel_size
        stride_ = stride
        dilation_ = dilation

        if kernel_size_list is not None:
            kernel_size_ = kernel_size_list[b]

        if stride_list is not None:
            stride_ = stride_list[b]

        if dilation_list is not None:
            dilation_ = dilation_list[b]

        kernel_size_, stride_, dilation_, _ = check_all_args(
            na_dim, kernel_size_, stride_, dilation_, is_causal
        )

        check_args_against_token_layout(
            token_layout=token_layout_list[b],
            kernel_size=kernel_size_,
            stride=stride_,
            dilation=dilation_,
            is_causal=is_causal,
        )

        if kernel_size_list is not None:
            assert kernel_size_list_out is not None
            kernel_size_list_out.append(kernel_size_)

        if stride_list is not None:
            assert stride_list_out is not None
            stride_list_out.append(stride_)

        if dilation_list is not None:
            assert dilation_list_out is not None
            dilation_list_out.append(dilation_)

    if kernel_size_list_out is not None and all(
        k == kernel_size_list_out[0] for k in kernel_size_list_out
    ):
        kernel_size = kernel_size_list_out[0]
        kernel_size_list_out = None

    if stride_list_out is not None and all(
        k == stride_list_out[0] for k in stride_list_out
    ):
        stride = stride_list_out[0]
        stride_list_out = None

    if dilation_list_out is not None and all(
        k == dilation_list_out[0] for k in dilation_list_out
    ):
        dilation = dilation_list_out[0]
        dilation_list_out = None

    def _tensor_from_optional_list(parameter_list: Optional[list]) -> Optional[Tensor]:
        if parameter_list is None:
            return None

        tensor = torch.tensor(parameter_list, device=device, dtype=torch.int32)
        assert tensor.shape[0] == batch_size
        assert tensor.shape[1] == na_dim

        return tensor

    # Construct device tensors
    kernel_sizes = _tensor_from_optional_list(kernel_size_list_out)
    strides = _tensor_from_optional_list(stride_list_out)
    dilations = _tensor_from_optional_list(dilation_list_out)

    return (
        (kernel_size, stride, dilation, is_causal),
        (kernel_size_list_out, stride_list_out, dilation_list_out),
        (kernel_sizes, strides, dilations),
    )  # type: ignore[return-value]


def generate_fna_varlen_metadata(
    token_layout_list: List[DimensionType],
    q_tile_shape: DimensionType,
    kv_tile_shape: DimensionType,
    backward_q_tile_shape: Optional[DimensionType],
    backward_kv_tile_shape: Optional[DimensionType],
    device: torch.device,
    flip_tiled_dims: bool,
    kernel_size: DimensionTypeOrDed,
    stride: DimensionTypeOrDed = 1,
    dilation: DimensionTypeOrDed = 1,
    is_causal: Optional[CausalArgTypeOrDed] = False,
    # var-param
    kernel_size_list: VariableDimensionType = None,
    stride_list: VariableDimensionType = None,
    dilation_list: VariableDimensionType = None,
) -> dict:
    """
    Takes list of token layouts and produces metadata required for performing variable-length
    Token Permutation (TokPerm) and Fused Neighborhood Attention (FNA).

    The default layout for variable-length attention is the "sequence-packed" layout, meaning there
    is no batch dimension in the tensors (though for consistency we still force the existence of a
    batch dimension but with size 1), and the differently-shaped/sized sequences are just
    concatenated together along the sequence.

    Variable length neighborhood attention follows the same layout, and therefore expects
    sequence-packed inputs.
    The key difference is that instead of a list/tensor of sequence _lengths_, we expect a list of
    token layouts, or shapes that would correspond to individual sequences that are currently
    packed.

    For example, if we're packing a 16x16 and a 32x32 token layout into a single tensor with 1280
    total tokens (16x16 = 256, 32x32 = 1024), we should supply [(16, 16), (32, 32)] as the
    token_layout_list.
    NOTE: We should NEVER mix tokens with different ranked layouts together.
    For example, token_layout_list = [(16, 16), (16, 32, 32)] is invalid!

    Parameters:
        token_layout_list (list[tuple]): list of token layouts that describe the various independent
            sets of tokens / sequences in QKV. All elements must be integer tuples of size 1, 2, or
            3, and match each other in size as well.

        q_tile_shape (tuple): query tile shape in the forward pass.

        kv_tile_shape (tuple): key/value tile shape in the forward pass.

        backward_q_tile_shape (tuple): query tile shape in the backward pass.

        backward_kv_tile_shape (tuple): key/value tile shape in the backward pass.

        device (torch.device): Target torch device for performing Token Permute and FNA.

        flip_tiled_dims (bool): flip_tiled_dims argument from TokPerm.

        kernel_size (tuple): kernel / window size must be provided for verification.

        stride (Optional[tuple]): stride parameter, if used, must be provided for verification.

        dilation (Optional[tuple]): dilation parameter, if used, must be provided for verification.

        is_causal (Optional[tuple]): is_causal parameter, if used, must be provided for verification.

        kernel_size_list (Optional[list[tuple]]): (VarParam) List of kernel size parameters, in
            case different sets of tokens have varying kernel sizes.

        stride_list (Optional[list[tuple]]): (VarParam) List of stride parameters, in
            case different sets of tokens have varying stride values.

        dilation_list (Optional[list[tuple]]): (VarParam) List of dilation parameters, in
            case different sets of tokens have varying dilation values.

    Outputs:
        metadata (dict): Metadata required by token_{permute,unpermute}_varlen_operation and the
            varlen FNA operations.
    """

    # Varlen initializers always break graphs -- they need to be done ahead of time
    if is_torch_compiling():
        raise RuntimeError(
            "Running 'generate_fna_varlen_metadata' in a torch-compiled region is disallowed as it "
            "results in graph breaks. Please consider calling ahead of time and pass "
            "the generated metadata directly instead of a token_layout_list."
        )

    na_dim = len(q_tile_shape)
    batch_size = len(token_layout_list)

    params_static, params_list, params_tensor = _verify_variable_parameters(
        na_dim=na_dim,
        token_layout_list=token_layout_list,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        kernel_size_list=kernel_size_list,
        stride_list=stride_list,
        dilation_list=dilation_list,
        device=device,
    )
    kernel_size, stride, dilation, is_causal = params_static
    kernel_size_list, stride_list, dilation_list = params_list
    kernel_sizes, strides, dilations = params_tensor

    gen_metadata = partial(
        generate_tokperm_varlen_metadata,
        token_layout_list=token_layout_list,
        device=device,
        flip_tiled_dims=flip_tiled_dims,
        dilation=dilation,
        dilation_list=dilation_list,
    )

    metadata_q = gen_metadata(tile_shape=q_tile_shape)
    metadata_kv = gen_metadata(tile_shape=kv_tile_shape)
    metadata_q_bwd, metadata_kv_bwd = None, None
    if backward_q_tile_shape is not None:
        metadata_q_bwd = gen_metadata(tile_shape=backward_q_tile_shape)
    if backward_kv_tile_shape is not None:
        metadata_kv_bwd = gen_metadata(tile_shape=backward_kv_tile_shape)

    return {
        "na_dim": na_dim,
        "batch_size": batch_size,
        "q_tile_shape": q_tile_shape,
        "kv_tile_shape": kv_tile_shape,
        "q_tile_shape_bwd": backward_q_tile_shape,
        "kv_tile_shape_bwd": backward_kv_tile_shape,
        "metadata_q": metadata_q,
        "metadata_kv": metadata_kv,
        "metadata_q_bwd": metadata_q_bwd,
        "metadata_kv_bwd": metadata_kv_bwd,
        # static params
        "kernel_size": kernel_size,
        "stride": stride,
        "dilation": dilation,
        "is_causal": is_causal,
        # var-param
        "kernel_sizes": kernel_sizes,
        "strides": strides,
        "dilations": dilations,
    }


def generate_tokperm_varlen_metadata(
    token_layout_list: List[DimensionType],
    tile_shape: DimensionType,
    device: torch.device,
    flip_tiled_dims: bool,
    # indicating dilation is always required!
    dilation: DimensionTypeOrDed,
    # variable-parameter
    dilation_list: VariableDimensionType,
) -> dict:

    if not isinstance(tile_shape, tuple):
        raise ValueError(f"tile_shape must be a tuple, got {type(tile_shape)=}.")

    if any(not isinstance(x, int) for x in tile_shape):
        raise ValueError("tile_shape must be an integer tuple.")

    na_dim = len(tile_shape)
    has_variable_dilations = dilation_list is not None

    if na_dim not in [1, 2, 3]:
        raise ValueError(
            f"tile_shape must be a tuple of size 1, 2, or 3, got {na_dim}."
        )

    if not isinstance(token_layout_list, list):
        raise ValueError(
            f"token_layout_list must be a list, got {type(token_layout_list)=}."
        )

    if len(token_layout_list) < 1:
        raise ValueError("token_layout_list cannot be an empty list.")

    if any(not isinstance(x, tuple) for x in token_layout_list):
        raise ValueError("token_layout_list must be a list of tuples.")

    if any(len(x) != na_dim for x in token_layout_list):
        raise ValueError(
            f"Tuples in token_layout_list must be all be size {na_dim} given a {na_dim}-D problem and tile shape."
        )

    if any(any(not isinstance(y, int) for y in x) for x in token_layout_list):
        raise ValueError("Tuples in token_layout_list must be integer tuples.")

    batch = len(token_layout_list)

    if has_variable_dilations:
        if not isinstance(dilation_list, list):
            raise ValueError(
                f"dilation_list must be a list, got, {type(dilation_list)=}."
            )

        if len(dilation_list) != batch:
            raise ValueError(
                "dilation_list must be the same size as token_layout_list, got "
                f"{len(dilation_list)=}, {len(token_layout_list)=}."
            )

    # Dilation is always implemented as a non-contiguous extra tiling on top of the multi-dim
    # tiling, where we break up a single token layout into "dilation groups".
    # For example, this 4x4 token layout under a 2x2 dilation is tiled as follows:
    #
    #                 dilation   dilation   dilation   dilation
    #                 group 0     group 1    group 2    group 3
    #   a b c d
    #   e f g h  -->    a c        b d        e g        f h
    #   i j k l         i k        j l        m o        n p
    #   m n o p
    #
    # So we now have 4 smaller non-dilated problems, and we move the new size 4 dimension
    # to batch, and the FNA kernel does not need to concern itself with anything dilation-related.
    # The exception is when dilation doesn't evenly divide the input, where we'd need to pad, but
    # the kernel also needs to fix up the token layout for getting a correct fine-grained mask.
    dilation = check_dilation_arg(na_dim, dilation)
    num_dilation_groups = math.prod(dilation)
    batch_kernel = batch * num_dilation_groups

    # Verification, and exact calculation of variable num_dilation_groups
    dilation_list_ = []
    if has_variable_dilations:
        batch_kernel = 0
        assert dilation_list is not None
        for d in dilation_list:
            d = check_dilation_arg(na_dim, d)
            dilation_list_.append(d)
            batch_kernel += math.prod(d)

    # We create triplets of offsets: original, tokperm, and kernel.
    # Original always corresponds to the "correct" batch size and sequence lengths,
    # which means no padding, and no dilation tiling.
    # TokPerm corresponds to offsets after TokPerm (after padding), but with the same batch size as
    # the original.
    # Kernel corresponds to the batch size AFTER the dilation tiling in TokPerm, where batch is
    # multiplied by number of dilation groups.

    offset_list_original = [0]  # size: batch + 1
    offset_list_tokperm = [0]  # size: batch + 1
    offset_list_kernel = [0]  # size: batch_kernel + 1

    # Mapping from batch_kernel (post-dilation batch index) to batch (pre-dilation, or "original"
    # batch index). Helps the FNA kernel find the correct token layout shape and, if var-param, num
    # dilation groups.
    batch_map_list: Optional[List[List[int]]] = (
        None if not has_variable_dilations else []
    )  # size: batch_kernel

    # Total sequence length is also tracked pre- and post- permute
    # This is used ONLY for memory planning / allocation.
    total_seqlen_pre_permute = sum([math.prod(x) for x in token_layout_list])
    total_seqlen_post_permute = 0

    # Max sequence length is used both in the TokPerm kernel and the FNA kernel for
    # scheduling/launch.
    max_seqlen_original = 0
    max_seqlen_tokperm = 0
    max_seqlen_kernel = 0

    assert not has_variable_dilations or len(dilation_list_) == batch
    for i, token_layout in enumerate(token_layout_list):
        # Identical to counterparts in standard varlen attention
        offset_list_original.append(offset_list_original[-1] + math.prod(token_layout))
        max_seqlen_original = max(max_seqlen_original, math.prod(token_layout))

        dilation_ = dilation if not has_variable_dilations else dilation_list_[i]

        token_layout_padded = mul_tuple(
            mul_tuple(
                ceil_div_tuple(ceil_div_tuple(token_layout, tile_shape), dilation_),
                dilation_,
            ),
            tile_shape,
        )
        max_seqlen_tokperm = max(max_seqlen_tokperm, math.prod(token_layout_padded))

        seqlen_per_dilation_group_padded = math.prod(
            ceil_div_tuple(token_layout_padded, dilation_)
        )

        # FNA kernels need the maximum over all dilation groups (post-dilation-tiling)
        max_seqlen_kernel = max(max_seqlen_kernel, seqlen_per_dilation_group_padded)

        num_dilation_groups = math.prod(dilation_)
        seqlen_instance = math.prod(token_layout_padded)

        offset_list_tokperm.append(offset_list_tokperm[-1] + seqlen_instance)
        total_seqlen_post_permute += seqlen_instance

        for j in range(num_dilation_groups):
            if has_variable_dilations:
                assert batch_map_list is not None
                batch_map_list.append([i, j])

            offset_list_kernel.append(
                offset_list_kernel[-1] + seqlen_per_dilation_group_padded
            )

    # Convert all lists into tensors
    token_layouts = torch.tensor(token_layout_list, device=device, dtype=torch.int32)
    offsets_original = torch.tensor(
        offset_list_original, device=device, dtype=torch.int32
    )
    offsets_tokperm = torch.tensor(
        offset_list_tokperm, device=device, dtype=torch.int32
    )
    offsets_kernel = torch.tensor(offset_list_kernel, device=device, dtype=torch.int32)
    batch_map = (
        None
        if not has_variable_dilations
        else torch.tensor(batch_map_list, device=device, dtype=torch.int32)
    )

    assert offsets_original.dim() == 1
    assert offsets_tokperm.dim() == 1
    assert offsets_kernel.dim() == 1
    assert offsets_original.shape[0] == batch + 1
    assert offsets_tokperm.shape[0] == batch + 1
    assert offsets_kernel.shape[0] == batch_kernel + 1

    assert token_layouts.dim() == 2
    assert token_layouts.shape[0] == batch
    assert token_layouts.shape[1] == na_dim

    if batch_map is not None:
        assert batch_map.dim() == 2
        assert batch_map.shape[0] == batch_kernel
        assert batch_map.shape[1] == 2

    return {
        "na_dim": na_dim,
        "batch": batch,
        "batch_kernel": batch_kernel,
        "has_variable_dilations": has_variable_dilations,
        "token_layouts": token_layouts,
        "batch_map": batch_map,
        "offsets_original": offsets_original,
        "offsets_tokperm": offsets_tokperm,
        "offsets_kernel": offsets_kernel,
        "total_seqlen_pre_permute": total_seqlen_pre_permute,
        "total_seqlen_post_permute": total_seqlen_post_permute,
        "max_seqlen_original": max_seqlen_original,
        "max_seqlen_tokperm": max_seqlen_tokperm,
        "max_seqlen_kernel": max_seqlen_kernel,
    }


def _verify_parameter_tensor(
    na_dim: int,
    batch_size: int,
    tensor: Optional[Tensor],
    input_tensor: Tensor,
    tensor_name: str,
):
    if tensor is None:
        return

    if not isinstance(tensor, Tensor):
        raise ValueError(f"'{tensor_name}' must be a tensor, got {type(tensor)=}.")

    if tensor.dtype != torch.int32:
        raise ValueError(
            f"'{tensor_name}' must be an int32 tensor, got {tensor.dtype=}."
        )

    if tensor.device != input_tensor.device:
        raise ValueError(
            f"'{tensor_name}' must be on the same device as the input tensor, "
            f"got {tensor.device=} != {input_tensor.device=}."
        )

    if tensor.dim() != 2:
        raise ValueError(f"'{tensor_name}' must be a 2-D tensor, got {tensor.dim()=}.")

    if tensor.shape[0] != batch_size:
        raise ValueError(
            f"{tensor_name}.shape[0] must be {batch_size=}, got {tensor.shape[0]=}."
        )

    if tensor.shape[1] != na_dim:
        raise ValueError(
            f"{tensor_name}.shape[1] for NA{na_dim}D must be {na_dim}, got {tensor.shape[1]=}."
        )


def verify_fna_varlen_metadata(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    metadata: dict,
    requires_grad: bool,
):
    if metadata is None:
        raise ValueError("Varlen metadata must be specified.")

    if not isinstance(metadata, Mapping):
        raise ValueError(
            f"Varlen metadata must be a mapping type (i.e. dict), got {type(metadata)=}."
        )

    expected_keys = [
        "na_dim",
        "batch_size",
        "q_tile_shape",
        "kv_tile_shape",
        "q_tile_shape_bwd",
        "kv_tile_shape_bwd",
        "metadata_q",
        "metadata_kv",
        "metadata_q_bwd",
        "metadata_kv_bwd",
        # static params
        "kernel_size",
        "stride",
        "dilation",
        "is_causal",
        # var-param
        "kernel_sizes",
        "strides",
        "dilations",
    ]

    if any(k not in metadata for k in expected_keys):
        raise ValueError(
            f"Unexpected varlen metadata format. Expected to find keys {expected_keys}, but got "
            f"{metadata.keys()=}."
        )

    # verify parameters
    na_dim = metadata["na_dim"]
    batch_size = metadata["batch_size"]
    _verify_parameter_tensor(
        na_dim=na_dim,
        batch_size=batch_size,
        tensor=metadata["kernel_sizes"],
        input_tensor=query,
        tensor_name="kernel_sizes",
    )
    _verify_parameter_tensor(
        na_dim=na_dim,
        batch_size=batch_size,
        tensor=metadata["strides"],
        input_tensor=query,
        tensor_name="strides",
    )
    _verify_parameter_tensor(
        na_dim=na_dim,
        batch_size=batch_size,
        tensor=metadata["dilations"],
        input_tensor=query,
        tensor_name="dilations",
    )

    verify_tokperm_varlen_metadata(
        tensor=query,
        metadata=metadata["metadata_q"],
    )
    verify_tokperm_varlen_metadata(
        tensor=key,
        metadata=metadata["metadata_kv"],
    )
    verify_tokperm_varlen_metadata(
        tensor=value,
        metadata=metadata["metadata_kv"],
    )

    if requires_grad:
        verify_tokperm_varlen_metadata(
            tensor=query,
            metadata=metadata["metadata_q_bwd"],
        )
        verify_tokperm_varlen_metadata(
            tensor=key,
            metadata=metadata["metadata_kv_bwd"],
        )
        verify_tokperm_varlen_metadata(
            tensor=value,
            metadata=metadata["metadata_kv_bwd"],
        )


def verify_tokperm_varlen_metadata(
    tensor: Tensor,
    metadata: dict,
):
    if tensor.dim() != 4:
        raise ValueError("Varlen Token Permute expects 4D tensor inputs.")

    if tensor.shape[0] != 1:
        raise ValueError(
            f"Varlen Token Permute expects sequence-packed tensors (batch=1), got {tensor.shape[0]=}."
        )

    if not isinstance(metadata, Mapping):
        raise ValueError(
            f"Varlen TokPerm metadata must be a mapping type (i.e. dict), got {type(metadata)=}."
        )

    expected_keys = [
        "na_dim",
        "batch",
        "batch_kernel",
        "has_variable_dilations",
        "token_layouts",
        "batch_map",
        "offsets_original",
        "offsets_tokperm",
        "offsets_kernel",
        "total_seqlen_pre_permute",
        "total_seqlen_post_permute",
        "max_seqlen_original",
        "max_seqlen_tokperm",
        "max_seqlen_kernel",
    ]

    if any(x not in metadata for x in expected_keys):
        raise ValueError(
            f"Expected to find keys {expected_keys} in varlen metadata, got {metadata.keys()}."
        )

    na_dim = metadata["na_dim"]
    batch = metadata["batch"]
    batch_kernel = metadata["batch_kernel"]
    has_variable_dilations = metadata["has_variable_dilations"]

    offsets_original = metadata["offsets_original"]
    offsets_tokperm = metadata["offsets_tokperm"]
    offsets_kernel = metadata["offsets_kernel"]
    token_layouts = metadata["token_layouts"]
    # Optional; only specified when using variable dilations:
    batch_map = metadata["batch_map"]

    total_seqlen_pre_permute = metadata["total_seqlen_pre_permute"]
    total_seqlen_post_permute = metadata["total_seqlen_post_permute"]
    max_seqlen_original = metadata["max_seqlen_original"]
    max_seqlen_tokperm = metadata["max_seqlen_tokperm"]
    max_seqlen_kernel = metadata["max_seqlen_kernel"]

    assert all(isinstance(x, int) for x in [na_dim, batch, batch_kernel])
    assert isinstance(has_variable_dilations, bool)
    assert all(
        isinstance(x, Tensor)
        for x in [offsets_original, offsets_tokperm, offsets_kernel, token_layouts]
    )

    if (
        offsets_original.dim() != 1
        or offsets_tokperm.dim() != 1
        or offsets_kernel.dim() != 1
    ):
        raise ValueError(
            "Offset tensors (offsets_original, offsets_tokperm, offsets_kernel) must all be 1-D "
            f"tensors, got {offsets_original.dim()=}, {offsets_tokperm.dim()=}, {offsets_kernel.dim()=}."
        )

    if (
        offsets_original.device != tensor.device
        or offsets_tokperm.device != tensor.device
        or offsets_kernel.device != tensor.device
    ):
        raise ValueError(
            "Offset tensors (offsets_original, offsets_tokperm, offsets_kernel) must be on the same "
            f"device as the input tensor, got {offsets_original.device=}, {offsets_tokperm.device=}, "
            f"{offsets_kernel.device=}, {tensor.device=}."
        )

    if offsets_original.shape[0] != batch + 1:
        raise ValueError(
            "offsets_original must be of size batch_size + 1, got "
            f"{offsets_original.shape[0]=} != {batch=}."
        )

    if offsets_tokperm.shape[0] != batch + 1:
        raise ValueError(
            "offsets_tokperm must be of size batch_size + 1, got "
            f"{offsets_tokperm.shape[0]=} != {batch=}."
        )

    if offsets_kernel.shape[0] != batch_kernel + 1:
        raise ValueError(
            "offsets_kernel must be of size batch_size * num_dilation_groups + 1, got "
            f"{offsets_kernel.shape[0]=} != {batch_kernel=}."
        )

    if (
        offsets_original.dtype != torch.int32
        or offsets_tokperm.dtype != torch.int32
        or offsets_kernel.dtype != torch.int32
    ):
        raise ValueError(
            "Offset tensors (offsets_original, offsets_tokperm, offsets_kernel) must all be be int32 "
            f"tensors, got {offsets_original.dtype=}, {offsets_tokperm.dtype=}, {offsets_kernel.dtype=}."
        )

    if token_layouts.dim() != 2:
        raise ValueError(
            f"token_layouts must be a 2-D tensor, got {token_layouts.dim()=}."
        )

    if token_layouts.dtype != torch.int32:
        raise ValueError(
            f"token_layouts must be an int32 tensor, got {token_layouts.dtype=}."
        )

    if token_layouts.shape[0] != batch:
        raise ValueError(
            f"token_layouts.shape[0] must be {batch=}, got {token_layouts.shape[0]=}."
        )

    if token_layouts.shape[1] != na_dim:
        raise ValueError(
            f"token_layouts.shape[1] for NA{na_dim}D must be {na_dim}, got {token_layouts.shape[1]=}."
        )

    if token_layouts.device != tensor.device:
        raise ValueError(
            f"token_layouts must be on the same device as the input tensor, got {tensor.device=}, "
            f"{token_layouts.device=}."
        )

    if not isinstance(total_seqlen_pre_permute, int) or total_seqlen_pre_permute < 1:
        raise ValueError(
            f"total_seqlen_pre_permute must be a positive integer, got {total_seqlen_pre_permute}."
        )

    if not isinstance(total_seqlen_post_permute, int) or total_seqlen_post_permute < 1:
        raise ValueError(
            f"total_seqlen_post_permute must be a positive integer, got {total_seqlen_post_permute}."
        )

    if not isinstance(max_seqlen_original, int) or max_seqlen_original < 1:
        raise ValueError(
            f"max_seqlen_original must be a positive integer, got {max_seqlen_original}."
        )

    if not isinstance(max_seqlen_tokperm, int) or max_seqlen_tokperm < 1:
        raise ValueError(
            f"max_seqlen_tokperm must be a positive integer, got {max_seqlen_tokperm}."
        )

    if not isinstance(max_seqlen_kernel, int) or max_seqlen_kernel < 1:
        raise ValueError(
            f"max_seqlen_kernel must be a positive integer, got {max_seqlen_kernel}."
        )

    if has_variable_dilations:
        if not isinstance(batch_map, Tensor):
            raise ValueError(
                f"Use case has variable dilations, therefore batch_map must be a Tensor, got {batch_map=}."
            )

        if batch_map.dim() != 2:
            raise ValueError(f"batch_map must be a 2-D tensor, got {batch_map.dim()=}.")

        if batch_map.dtype != torch.int32:
            raise ValueError(
                f"batch_map must be an int32 tensor, got {batch_map.dtype=}."
            )

        if batch_map.shape[0] != batch_kernel:
            raise ValueError(
                f"batch_map.shape[0] must be {batch_kernel=}, got {batch_map.shape[0]=}."
            )

        if batch_map.shape[1] != 2:
            raise ValueError(
                f"batch_map.shape[1] must be 2, got {batch_map.shape[1]=}."
            )

        if batch_map.device != tensor.device:
            raise ValueError(
                f"batch_map must be on the same device as the input tensor, got {tensor.device=}, "
                f"{batch_map.device=}."
            )


__all__ = [
    "generate_tokperm_varlen_metadata",
    "generate_fna_varlen_metadata",
    "verify_fna_varlen_metadata",
    "verify_tokperm_varlen_metadata",
]
