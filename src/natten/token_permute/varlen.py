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
from functools import partial
from typing import List, Optional

import torch
from torch import Tensor

from natten.types import CausalArgTypeOrDed, DimensionType, DimensionTypeOrDed
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

    kernel_size, stride, dilation, is_causal = check_all_args(
        na_dim, kernel_size, stride, dilation, is_causal
    )

    gen_metadata = partial(
        generate_tokperm_varlen_metadata,
        token_layout_list=token_layout_list,
        device=device,
        flip_tiled_dims=flip_tiled_dims,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
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
        "kernel_size": kernel_size,
        "stride": stride,
        "dilation": dilation,
        "is_causal": is_causal,
        "q_tile_shape": q_tile_shape,
        "kv_tile_shape": kv_tile_shape,
        "q_tile_shape_bwd": backward_q_tile_shape,
        "kv_tile_shape_bwd": backward_kv_tile_shape,
        "metadata_q": metadata_q,
        "metadata_kv": metadata_kv,
        "metadata_q_bwd": metadata_q_bwd,
        "metadata_kv_bwd": metadata_kv_bwd,
    }


def generate_tokperm_varlen_metadata(
    token_layout_list: List[DimensionType],
    tile_shape: DimensionType,
    device: torch.device,
    flip_tiled_dims: bool,
    # indicating dilation is always required!
    dilation: DimensionTypeOrDed,
    # if we're only interested in running/verifying tokperm, there's no need to verify kernel_size
    # stride, and is_causal.
    kernel_size: Optional[DimensionTypeOrDed] = None,
    stride: DimensionTypeOrDed = 1,
    is_causal: CausalArgTypeOrDed = False,
) -> dict:

    if not isinstance(tile_shape, tuple):
        raise ValueError(f"tile_shape must be a tuple, got {type(tile_shape)=}.")

    if any(not isinstance(x, int) for x in tile_shape):
        raise ValueError("tile_shape must be an integer tuple.")

    na_dim = len(tile_shape)

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

    # kernel_size is optional in this internal API so we can test tokperm
    # users will still be forced to specify kernel_size for validation
    should_verify = kernel_size is not None
    if should_verify:
        kernel_size, stride, dilation, is_causal = check_all_args(
            na_dim, kernel_size, stride, dilation, is_causal
        )
    else:
        dilation = check_dilation_arg(na_dim, dilation)

    batch = len(token_layout_list)

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
    num_dilation_groups = math.prod(dilation)

    # We create two pairs of offsets and extents: "pre-permute" and "post-permute".
    # Pre-permute always corresponds to the "correct" batch size and sequence lengths,
    # which means no padding, and no dilation tiling.
    # Post-permute corresponds to the batch size AFTER Token Permutation, where batch is
    # multiplied by number of dilation groups.
    #
    # Standard varlen attention only needs to be provided with offsets, because the difference
    # in offsets is exactly the extent (sequence length) for each unique sequence in the batch,
    # but here we have to provide the attention kernel with the post-permute offsets to get the
    # correct tokens, and subtracting them will include padding, which gives us incorrect offsets.
    # Therefore we need to have "pre-permute" and "post-permute" token_layouts (extents) as well.
    #
    # Varlen TokPerm kernel needs:
    #  * pre-permute offsets: slice out the tokens belonging to the current batch
    #
    #  * post-permute offsets: slice out the region where the permuted tokens will be written, with
    #      guaranteed padding space available.
    #
    #  * pre-permute token layouts: compute the token permuted layout (R T D) based on it on the fly
    # This kernel does NOT need the post-permute token layouts.
    #
    # Varlen FNA kernel needs:
    #  * post-permute offsets: find the tokens belonging to the current permuted batch
    #
    #  * post-permute token layouts: find the EXACT token layout for the current batch for
    #      fine-grained masking.

    offset_list_pre_permute = [0 for _ in range(batch + 1)]  # size: batch + 1
    offset_list_post_permute = [
        0 for _ in range(batch * num_dilation_groups + 1)
    ]  # size: batch * num_dilation_groups + 1

    token_layout_list_post_permute = []  # size: batch * num_dilation_groups

    # Total sequence length is also tracked pre- and post- permute
    # This is used ONLY for memory planning / allocation.
    total_seqlen_pre_permute = sum([math.prod(x) for x in token_layout_list])
    total_seqlen_post_permute = 0

    # Max sequence length is used both in the TokPerm kernel and the FNA kernel for
    # scheduling/launch.
    max_seqlen_pre_permute = 0
    max_seqlen_post_permute = 0
    max_seqlen_kernel = 0

    for i, token_layout in enumerate(token_layout_list):
        if should_verify:
            # Verifies parameters are compatible with each token layout
            check_args_against_token_layout(
                token_layout=token_layout,
                kernel_size=kernel_size,  # type: ignore[arg-type]
                stride=stride,  # type: ignore[arg-type]
                dilation=dilation,  # type: ignore[arg-type]
                is_causal=is_causal,  # type: ignore[arg-type]
            )

        # Identical to counterparts in standard varlen attention
        offset_list_pre_permute[i + 1] = offset_list_pre_permute[i] + math.prod(
            token_layout
        )
        max_seqlen_pre_permute = max(max_seqlen_pre_permute, math.prod(token_layout))

        token_layouts_post_dilation = _get_dilated_token_layouts(
            token_layout,
            dilation,
            flip_tiled_dims=flip_tiled_dims,
        )

        token_layout_padded = mul_tuple(
            mul_tuple(
                ceil_div_tuple(ceil_div_tuple(token_layout, tile_shape), dilation),
                dilation,
            ),
            tile_shape,
        )
        max_seqlen_post_permute = max(
            max_seqlen_post_permute, math.prod(token_layout_padded)
        )

        seqlen_per_dilation_group_padded = math.prod(
            ceil_div_tuple(token_layout_padded, dilation)
        )

        # FNA kernels need the maximum over all dilation groups (post-dilation-tiling)
        max_seqlen_kernel = max(max_seqlen_kernel, seqlen_per_dilation_group_padded)

        for j, token_layout_dilation_group in enumerate(token_layouts_post_dilation):

            offset = i * num_dilation_groups + j
            offset_list_post_permute[offset + 1] = (
                offset_list_post_permute[offset] + seqlen_per_dilation_group_padded
            )

            total_seqlen_post_permute += seqlen_per_dilation_group_padded

            token_layout_list_post_permute.append(token_layout_dilation_group)

    # Convert all lists into tensors
    token_layouts_pre_permute = torch.tensor(
        token_layout_list, device=device, dtype=torch.int32
    )
    token_layouts_post_permute = torch.tensor(
        token_layout_list_post_permute, device=device, dtype=torch.int32
    )
    offsets_pre_permute = torch.tensor(
        offset_list_pre_permute, device=device, dtype=torch.int32
    )
    offsets_post_permute = torch.tensor(
        offset_list_post_permute, device=device, dtype=torch.int32
    )

    assert offsets_pre_permute.dim() == 1
    assert offsets_post_permute.dim() == 1
    assert offsets_pre_permute.shape[0] == batch + 1
    assert offsets_post_permute.shape[0] == batch * num_dilation_groups + 1

    assert token_layouts_pre_permute.dim() == 2
    assert token_layouts_pre_permute.shape[0] == batch
    assert token_layouts_pre_permute.shape[1] == na_dim

    assert token_layouts_post_permute.dim() == 2
    assert token_layouts_post_permute.shape[0] == batch * num_dilation_groups
    assert token_layouts_post_permute.shape[1] == na_dim

    return {
        "token_layouts_pre_permute": token_layouts_pre_permute,
        "token_layouts_post_permute": token_layouts_post_permute,
        "offsets_pre_permute": offsets_pre_permute,
        "offsets_post_permute": offsets_post_permute,
        "total_seqlen_pre_permute": total_seqlen_pre_permute,
        "total_seqlen_post_permute": total_seqlen_post_permute,
        "max_seqlen_pre_permute": max_seqlen_pre_permute,
        "max_seqlen_post_permute": max_seqlen_post_permute,
        "max_seqlen_kernel": max_seqlen_kernel,
    }


def verify_fna_varlen_metadata(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    metadata: dict,
    is_backward: bool,
):
    if not is_backward:
        verify_tokperm_varlen_metadata(
            tensor=query,
            metadata=metadata["metadata_q"],
            tile_shape=metadata["q_tile_shape"],
            dilation=metadata["dilation"],
        )
        verify_tokperm_varlen_metadata(
            tensor=key,
            metadata=metadata["metadata_kv"],
            tile_shape=metadata["kv_tile_shape"],
            dilation=metadata["dilation"],
        )
        verify_tokperm_varlen_metadata(
            tensor=value,
            metadata=metadata["metadata_kv"],
            tile_shape=metadata["kv_tile_shape"],
            dilation=metadata["dilation"],
        )
    else:
        verify_tokperm_varlen_metadata(
            tensor=query,
            metadata=metadata["metadata_q_bwd"],
            tile_shape=metadata["q_tile_shape_bwd"],
            dilation=metadata["dilation"],
        )
        verify_tokperm_varlen_metadata(
            tensor=key,
            metadata=metadata["metadata_kv_bwd"],
            tile_shape=metadata["kv_tile_shape_bwd"],
            dilation=metadata["dilation"],
        )
        verify_tokperm_varlen_metadata(
            tensor=value,
            metadata=metadata["metadata_kv_bwd"],
            tile_shape=metadata["kv_tile_shape_bwd"],
            dilation=metadata["dilation"],
        )


def verify_tokperm_varlen_metadata(
    tensor: Tensor,
    metadata: dict,
    tile_shape: DimensionType,
    dilation: DimensionType,
):
    na_dim = len(tile_shape)
    assert na_dim in [1, 2, 3]
    num_dilation_groups = math.prod(dilation)

    if tensor.dim() != 4:
        raise ValueError("Varlen Token Permute expects 4D tensor inputs.")

    if tensor.shape[0] != 1:
        raise ValueError(
            f"Varlen Token Permute expects sequence-packed tensors (batch=1), got {tensor.shape[0]=}."
        )

    if len(dilation) != na_dim:
        raise ValueError(
            f"Expected {na_dim}D dilation for NA{na_dim}D, " f"got {dilation=}."
        )

    if not isinstance(metadata, dict):
        raise ValueError(
            f"Varlen TokPerm metadata must be a dict0, got {type(metadata)=}."
        )

    expected_keys = [
        "token_layouts_pre_permute",
        "token_layouts_post_permute",
        "offsets_pre_permute",
        "offsets_post_permute",
        "total_seqlen_pre_permute",
        "total_seqlen_post_permute",
        "max_seqlen_pre_permute",
        "max_seqlen_post_permute",
        "max_seqlen_kernel",
    ]

    if any(x not in metadata for x in expected_keys):
        raise ValueError(
            f"Expected to find keys {expected_keys} in varlen metadata, got {metadata.keys()}."
        )

    offsets_pre_permute = metadata["offsets_pre_permute"]
    offsets_post_permute = metadata["offsets_post_permute"]

    batch_size = offsets_pre_permute.shape[0] - 1

    if batch_size <= 0:
        raise ValueError(
            "Batch size must be >= 1, got "
            f"{batch_size=} ({offsets_pre_permute.shape[0]=} - 1)."
        )

    if offsets_pre_permute.dim() != 1 or offsets_post_permute.dim() != 1:
        raise ValueError(
            "offsets_pre_permute and offsets_post_permute must be 1-D tensors, got "
            f"{offsets_pre_permute.dim()=}, {offsets_post_permute.dim()=}."
        )

    if (
        offsets_pre_permute.device != tensor.device
        or offsets_post_permute.device != tensor.device
    ):
        raise ValueError(
            "offsets_pre_permute and offsets_post_permute must be on the same device as the input tensor, got "
            f"{offsets_pre_permute.device=}, {offsets_post_permute.device=}, {tensor.device=}."
        )

    if offsets_post_permute.shape[0] != batch_size * num_dilation_groups + 1:
        raise ValueError(
            "offsets_post_permute must be of size batch_size * num_dilation_groups + 1, got "
            f"{offsets_post_permute.shape[0]=} with {batch_size=}, {num_dilation_groups=}."
        )

    if (
        offsets_pre_permute.dtype != torch.int32
        or offsets_post_permute.dtype != torch.int32
    ):
        raise ValueError(
            "offsets_pre_permute and offsets_post_permute must be both be int32 tensors, got "
            f"{offsets_pre_permute.dtype=}, {offsets_post_permute.dtype=}."
        )

    token_layouts_pre_permute = metadata["token_layouts_pre_permute"]
    token_layouts_post_permute = metadata["token_layouts_post_permute"]

    if token_layouts_pre_permute.dim() != 2 or token_layouts_post_permute.dim() != 2:
        raise ValueError(
            "token_layouts_pre_permute and token_layouts_post_permute must both be a 2-D tensors, got "
            f"{token_layouts_pre_permute.dim()=}, {token_layouts_post_permute.dim()=}."
        )

    if (
        token_layouts_pre_permute.dtype != torch.int32
        or token_layouts_post_permute.dtype != torch.int32
    ):
        raise ValueError(
            f"token_layouts_pre_permute and token_layouts_post_permute must both be int32 tensors, got "
            f"{token_layouts_pre_permute.dtype=}, {token_layouts_post_permute.dtype=}."
        )

    if token_layouts_pre_permute.shape[0] != batch_size:
        raise ValueError(
            f"token_layouts_pre_permute.shape[0] be {batch_size=}, got {token_layouts_pre_permute.shape[0]=}."
        )

    if token_layouts_pre_permute.shape[1] != na_dim:
        raise ValueError(
            f"token_layouts_pre_permute.shape[1] for NA{na_dim}D must be {na_dim}, got {token_layouts_pre_permute.shape[1]=}."
        )

    if token_layouts_post_permute.shape[0] != batch_size * num_dilation_groups:
        raise ValueError(
            f"token_layouts_post_permute.shape[0] be {batch_size * num_dilation_groups=}, got {token_layouts_post_permute.shape[0]=}."
        )

    if token_layouts_post_permute.shape[1] != na_dim:
        raise ValueError(
            f"token_layouts_post_permute.shape[1] for NA{na_dim}D must be {na_dim}, got {token_layouts_post_permute.shape[1]=}."
        )

    total_seqlen_pre_permute = metadata["total_seqlen_pre_permute"]
    total_seqlen_post_permute = metadata["total_seqlen_post_permute"]

    if not isinstance(total_seqlen_pre_permute, int) or total_seqlen_pre_permute < 1:
        raise ValueError(
            f"total_seqlen_pre_permute must be a positive integer, got {total_seqlen_pre_permute}."
        )

    if not isinstance(total_seqlen_post_permute, int) or total_seqlen_post_permute < 1:
        raise ValueError(
            f"total_seqlen_post_permute must be a positive integer, got {total_seqlen_post_permute}."
        )

    max_seqlen_pre_permute = metadata["max_seqlen_pre_permute"]
    max_seqlen_post_permute = metadata["max_seqlen_post_permute"]
    max_seqlen_kernel = metadata["max_seqlen_kernel"]

    if not isinstance(max_seqlen_pre_permute, int) or max_seqlen_pre_permute < 1:
        raise ValueError(
            f"max_seqlen_pre_permute must be a positive integer, got {max_seqlen_pre_permute}."
        )

    if not isinstance(max_seqlen_post_permute, int) or max_seqlen_post_permute < 1:
        raise ValueError(
            f"max_seqlen_post_permute must be a positive integer, got {max_seqlen_post_permute}."
        )

    if not isinstance(max_seqlen_kernel, int) or max_seqlen_kernel < 1:
        raise ValueError(
            f"max_seqlen_kernel must be a positive integer, got {max_seqlen_kernel}."
        )


def get_na_dim(
    varlen_metadata: Optional[dict] = None,
    token_layout_list: Optional[list] = None,
) -> int:
    if varlen_metadata is not None:
        if not isinstance(varlen_metadata, dict) or "na_dim" not in varlen_metadata:
            raise ValueError(f"Invalid varlen metadata object {varlen_metadata=}.")

        return varlen_metadata["na_dim"]

    if token_layout_list is not None:
        if (
            not isinstance(token_layout_list, list)
            or len(token_layout_list) < 1
            or not all(isinstance(x, tuple) for x in token_layout_list)
        ):
            raise ValueError(
                f"token_layout_list must be a non-empty list of tuples, got {token_layout_list=}."
            )

        na_dim = len(token_layout_list[0])

        if na_dim not in [1, 2, 3]:
            raise ValueError(
                f"Token layouts can only be 1, 2, or 3 dimensions, got {na_dim} ({token_layout_list=})."
            )

        if any(
            len(x) != na_dim or any(not isinstance(y, int) for y in x)
            for x in token_layout_list
        ):
            raise ValueError(
                f"Elements in token_layout_list must be integer tuples of size {na_dim}, got {token_layout_list=}."
            )

        return na_dim

    raise ValueError(
        "Variable-length/size operations require either varlen_metadata or token_layout_list "
        "to be passed, but both are None."
    )


__all__ = [
    "generate_tokperm_varlen_metadata",
    "generate_fna_varlen_metadata",
    "verify_fna_varlen_metadata",
    "verify_tokperm_varlen_metadata",
]
