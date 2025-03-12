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

import functools
import math
from typing import Dict, Optional, Tuple

import torch
from torch import BoolTensor, IntTensor, Tensor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from .ops import additional_sdpa, merge_attentions
from .types import (
    CausalArg1DTypeOrDed,
    CausalArg2DTypeOrDed,
    CausalArg3DTypeOrDed,
    CausalArgType,
    Dimension1DTypeOrDed,
    Dimension2DTypeOrDed,
    Dimension3DTypeOrDed,
    DimensionType,
)
from .utils import check_all_args, check_input_size_arg


def can_run_flex_attention(input_shape) -> bool:
    batch_size, *seq_dims, num_heads, head_dim = input_shape
    seq_length = math.prod(seq_dims)
    return seq_length % 128 == 0 and math.log2(head_dim).is_integer()


def get_flex_attention_compiled():
    return torch.compile(flex_attention, dynamic=False)


def get_na_flex_mask(
    na_dim: int,
    input_size: DimensionType,
    kernel_size: DimensionType,
    dilation: DimensionType,
    is_causal: CausalArgType,
):

    def index_to_coord_1d(idx: IntTensor) -> Tuple[IntTensor]:
        assert len(input_size) == 1
        return (idx,)

    def index_to_coord_2d(idx: IntTensor) -> Tuple[IntTensor, IntTensor]:
        assert len(input_size) == 2
        return (idx // input_size[1], idx % input_size[1])  # type: ignore

    def index_to_coord_3d(idx: IntTensor) -> Tuple[IntTensor, IntTensor, IntTensor]:
        assert len(input_size) == 3
        return (
            idx // input_size[2] // input_size[1],  # type: ignore
            (idx // input_size[2]) % input_size[1],  # type: ignore
            idx % input_size[2],  # type: ignore
        )

    index_to_coord = {
        1: index_to_coord_1d,
        2: index_to_coord_2d,
        3: index_to_coord_3d,
    }[na_dim]

    def na_mask_mod(
        b: IntTensor, h: IntTensor, q_idx: IntTensor, kv_idx: IntTensor
    ) -> BoolTensor:
        q_coord = index_to_coord(q_idx)
        kv_coord = index_to_coord(kv_idx)

        masks = []
        for i in range(na_dim):
            kernel_times_dilation = kernel_size[i] * dilation[i]
            if is_causal[i]:
                mask = (
                    (q_coord[i] - kv_coord[i] >= 0)
                    & (q_coord[i] - kv_coord[i] < kernel_times_dilation)
                    & ((q_coord[i] % dilation[i]) == (kv_coord[i] % dilation[i]))
                )
            else:
                window_size_left = (kernel_times_dilation) // 2
                window_size_right = (kernel_times_dilation) // 2 + (
                    (kernel_times_dilation) % 2 - 1
                )
                kernel_center_x = q_coord[i].clamp(
                    window_size_left, input_size[i] - 1 - window_size_right
                )
                w0 = kernel_center_x - kv_coord[i]
                w1 = kv_coord[i] - kernel_center_x
                mask = (
                    ((0 <= w0) & (w0 <= window_size_left))
                    | ((0 <= w1) & (w1 <= window_size_right))
                ) & ((q_coord[i] % dilation[i]) == (kv_coord[i] % dilation[i]))

            masks.append(mask)

        return functools.reduce(lambda x, y: x & y, masks)  # type: ignore

    seq_length = math.prod(input_size)
    return create_block_mask(
        na_mask_mod, B=None, H=None, Q_LEN=seq_length, KV_LEN=seq_length, _compile=True  # type: ignore
    )


def flex_na1d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension1DTypeOrDed,
    dilation: Dimension1DTypeOrDed = 1,
    is_causal: Optional[CausalArg1DTypeOrDed] = False,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    xformers_kwargs: Optional[Dict] = None,
) -> Tensor:

    kernel_size_, dilation_, is_causal_ = check_all_args(
        1, kernel_size, dilation, is_causal
    )

    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        raise ValueError(
            "flex_na1d expects query, key, and value to be 4-dimensional tensors, "
            f"got {query.shape=}, {key.shape=}, {value.shape=}."
        )

    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            "flex_na1d expects query, key, and value to have the same shape, "
            f"got {query.shape=}, {key.shape=}, {value.shape=}."
        )

    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            "flex_na1d expects query, key, and value to have the same dtype, "
            f"got {query.dtype=}, {key.dtype=}, {value.dtype=}."
        )

    batch_size, *num_tokens_tuple_, num_heads, head_dim = query.shape
    num_tokens_tuple = check_input_size_arg(1, num_tokens_tuple_)
    seqlen = math.prod(num_tokens_tuple)

    if not can_run_flex_attention(query.shape):
        raise ValueError(
            f"FlexAttention backend only allows sequence lengths "
            f"divisible by 128, and head dims that are powers "
            f"of 2. Got {seqlen=} and {head_dim=}."
        )

    query_ = query.transpose(1, 2)
    key_ = key.transpose(1, 2)
    value_ = value.transpose(1, 2)

    na_mask = get_na_flex_mask(1, num_tokens_tuple, kernel_size_, dilation_, is_causal_)
    flex_attention_compiled = get_flex_attention_compiled()
    out_, lse_ = flex_attention_compiled(
        query_, key_, value_, block_mask=na_mask, return_lse=True
    )

    out = out_.transpose(1, 2)
    lse = lse_.transpose(1, 2)

    if additional_keys is not None and additional_values is not None:
        if additional_keys is None or additional_values is None:
            raise ValueError(
                "Both `additional_keys` and `additional_values` must be "
                "either Tensors or NoneTypes."
            )

        scale = query.shape[-1] ** -0.5
        additional_output, additional_lse = additional_sdpa(
            query,
            additional_keys,
            additional_values,
            scale=scale,
            attn_kwargs=xformers_kwargs,
        )

        return merge_attentions(out, additional_output, lse, additional_lse)

    return out


def flex_na2d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension2DTypeOrDed,
    dilation: Dimension2DTypeOrDed = 1,
    is_causal: Optional[CausalArg2DTypeOrDed] = False,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    xformers_kwargs: Optional[Dict] = None,
) -> Tensor:

    kernel_size_, dilation_, is_causal_ = check_all_args(
        2, kernel_size, dilation, is_causal
    )

    if query.dim() != 5 or key.dim() != 5 or value.dim() != 5:
        raise ValueError(
            "flex_na2d expects query, key, and value to be 5-dimensional tensors, "
            f"got {query.shape=}, {key.shape=}, {value.shape=}."
        )

    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            "flex_na2d expects query, key, and value to have the same shape, "
            f"got {query.shape=}, {key.shape=}, {value.shape=}."
        )

    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            "flex_na2d expects query, key, and value to have the same dtype, "
            f"got {query.dtype=}, {key.dtype=}, {value.dtype=}."
        )

    batch_size, *num_tokens_tuple_, num_heads, head_dim = query.shape
    num_tokens_tuple = check_input_size_arg(2, num_tokens_tuple_)
    seqlen = math.prod(num_tokens_tuple)

    if not can_run_flex_attention(query.shape):
        raise ValueError(
            f"FlexAttention backend only allows sequence lengths "
            f"divisible by 128, and head dims that are powers "
            f"of 2. Got {seqlen=} ({num_tokens_tuple=}) and {head_dim=}."
        )

    query_ = query.view(batch_size, seqlen, num_heads, head_dim).transpose(1, 2)
    key_ = key.view(batch_size, seqlen, num_heads, head_dim).transpose(1, 2)
    value_ = value.view(batch_size, seqlen, num_heads, head_dim).transpose(1, 2)

    na_mask = get_na_flex_mask(2, num_tokens_tuple, kernel_size_, dilation_, is_causal_)
    flex_attention_compiled = get_flex_attention_compiled()
    out_, lse_ = flex_attention_compiled(
        query_, key_, value_, block_mask=na_mask, return_lse=True
    )

    out = out_.transpose(1, 2).view(batch_size, *num_tokens_tuple, num_heads, head_dim)
    lse = lse_.transpose(1, 2).view(batch_size, *num_tokens_tuple, num_heads)

    if additional_keys is not None and additional_values is not None:
        if additional_keys is None or additional_values is None:
            raise ValueError(
                "Both `additional_keys` and `additional_values` must be "
                "either Tensors or NoneTypes."
            )

        scale = query.shape[-1] ** -0.5
        additional_output, additional_lse = additional_sdpa(
            query,
            additional_keys,
            additional_values,
            scale=scale,
            attn_kwargs=xformers_kwargs,
        )

        return merge_attentions(out, additional_output, lse, additional_lse)

    return out


def flex_na3d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension3DTypeOrDed,
    dilation: Dimension3DTypeOrDed = 1,
    is_causal: Optional[CausalArg3DTypeOrDed] = False,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    xformers_kwargs: Optional[Dict] = None,
) -> Tensor:

    kernel_size_, dilation_, is_causal_ = check_all_args(
        3, kernel_size, dilation, is_causal
    )

    if query.dim() != 6 or key.dim() != 6 or value.dim() != 6:
        raise ValueError(
            "flex_na3d expects query, key, and value to be 6-dimensional tensors, "
            f"got {query.shape=}, {key.shape=}, {value.shape=}."
        )

    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            "flex_na3d expects query, key, and value to have the same shape, "
            f"got {query.shape=}, {key.shape=}, {value.shape=}."
        )

    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            "flex_na3d expects query, key, and value to have the same dtype, "
            f"got {query.dtype=}, {key.dtype=}, {value.dtype=}."
        )

    batch_size, *num_tokens_tuple_, num_heads, head_dim = query.shape
    num_tokens_tuple = check_input_size_arg(3, num_tokens_tuple_)
    seqlen = math.prod(num_tokens_tuple)

    if not can_run_flex_attention(query.shape):
        raise ValueError(
            f"FlexAttention backend only allows sequence lengths "
            f"divisible by 128, and head dims that are powers "
            f"of 2. Got {seqlen=} ({num_tokens_tuple=}) and {head_dim=}."
        )

    query_ = query.view(batch_size, seqlen, num_heads, head_dim).transpose(1, 2)
    key_ = key.view(batch_size, seqlen, num_heads, head_dim).transpose(1, 2)
    value_ = value.view(batch_size, seqlen, num_heads, head_dim).transpose(1, 2)

    na_mask = get_na_flex_mask(3, num_tokens_tuple, kernel_size_, dilation_, is_causal_)
    flex_attention_compiled = get_flex_attention_compiled()
    out_, lse_ = flex_attention_compiled(
        query_, key_, value_, block_mask=na_mask, return_lse=True
    )

    out = out_.transpose(1, 2).view(batch_size, *num_tokens_tuple, num_heads, head_dim)
    lse = lse_.transpose(1, 2).view(batch_size, *num_tokens_tuple, num_heads)

    if additional_keys is not None and additional_values is not None:
        if additional_keys is None or additional_values is None:
            raise ValueError(
                "Both `additional_keys` and `additional_values` must be "
                "either Tensors or NoneTypes."
            )

        scale = query.shape[-1] ** -0.5
        additional_output, additional_lse = additional_sdpa(
            query,
            additional_keys,
            additional_values,
            scale=scale,
            attn_kwargs=xformers_kwargs,
        )

        return merge_attentions(out, additional_output, lse, additional_lse)

    return out
