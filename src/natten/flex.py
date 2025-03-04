#################################################################################################
# Copyright (c) 2022-2024 Ali Hassani.
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
from typing import Optional, Tuple

import torch
from torch import BoolTensor, IntTensor, Tensor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

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
from .utils import check_all_args


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
                kernel_center_x = q_coord[i].clamp(
                    (kernel_times_dilation - 1) // 2,
                    (input_size[i] - 1) - (kernel_times_dilation - 1) // 2,
                )
                mask = (
                    (kernel_center_x - kv_coord[i]).abs() <= kernel_times_dilation // 2
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
) -> torch.Tensor:

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

    batch_size, seqlen, num_heads, head_dim = query.shape
    input_size = (seqlen,)

    query_ = query.transpose(1, 2)
    key_ = key.transpose(1, 2)
    value_ = value.transpose(1, 2)

    na_mask = get_na_flex_mask(1, input_size, kernel_size_, dilation_, is_causal_)
    flex_attention_compiled = get_flex_attention_compiled()
    out_ = flex_attention_compiled(query_, key_, value_, block_mask=na_mask)

    out = out_.transpose(1, 2)

    return out


def flex_na2d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension2DTypeOrDed,
    dilation: Dimension2DTypeOrDed = 1,
    is_causal: Optional[CausalArg2DTypeOrDed] = False,
) -> torch.Tensor:

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

    batch_size, seqlen_1, seqlen_2, num_heads, head_dim = query.shape
    seq_length = seqlen_1 * seqlen_2
    input_size = (seqlen_1, seqlen_2)

    query_ = query.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    key_ = key.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    value_ = value.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)

    na_mask = get_na_flex_mask(2, input_size, kernel_size_, dilation_, is_causal_)
    flex_attention_compiled = get_flex_attention_compiled()
    out_ = flex_attention_compiled(query_, key_, value_, block_mask=na_mask)

    out = out_.transpose(1, 2).view(batch_size, seqlen_1, seqlen_2, num_heads, head_dim)

    return out


def flex_na3d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension3DTypeOrDed,
    dilation: Dimension3DTypeOrDed = 1,
    is_causal: Optional[CausalArg3DTypeOrDed] = False,
) -> torch.Tensor:

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

    batch_size, seqlen_0, seqlen_1, seqlen_2, num_heads, head_dim = query.shape
    seq_length = seqlen_0 * seqlen_1 * seqlen_2
    input_size = (seqlen_0, seqlen_1, seqlen_2)

    query_ = query.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    key_ = key.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    value_ = value.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)

    na_mask = get_na_flex_mask(3, input_size, kernel_size_, dilation_, is_causal_)
    flex_attention_compiled = get_flex_attention_compiled()
    out_ = flex_attention_compiled(query_, key_, value_, block_mask=na_mask)

    out = out_.transpose(1, 2).view(
        batch_size, seqlen_0, seqlen_1, seqlen_2, num_heads, head_dim
    )

    return out
