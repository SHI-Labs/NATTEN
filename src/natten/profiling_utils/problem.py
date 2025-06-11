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
from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor

from natten.types import CausalArgType, DimensionType


class Problem:
    def __init__(
        self,
        na_dim: int,
        batch_size: int,
        heads: int,
        dim: int,
        input_size: DimensionType,
        window_size: DimensionType,
        stride: DimensionType,
        dilation: DimensionType,
        dtype: torch.dtype,
        is_causal: CausalArgType,
        additional_kv_length: Optional[int] = None,
        dim_value: Optional[int] = None,
    ):
        self.na_dim = na_dim
        self.batch_size = batch_size
        self.heads = heads
        self.dim = dim
        self.dim_value = dim_value or dim
        assert (
            len(window_size)
            == len(stride)
            == len(dilation)
            == len(input_size)
            == self.na_dim
        )
        self.input_size = input_size
        self.window_size = window_size
        self.stride = stride
        self.dilation = dilation
        self.dtype = dtype
        self.is_causal = is_causal or [False for _ in range(na_dim)]
        self.has_additional_kv = (
            additional_kv_length is not None and additional_kv_length > 0
        )
        self.additional_kv_length = additional_kv_length

    @property
    def is_self_attn(self) -> bool:
        return not any(c for c in self.is_causal) and all(
            x == w for x, w in zip(self.input_size, self.window_size)
        )

    def get_q_tensor_shape(self, heads_last: bool, flatten: bool = False) -> List:
        token_layout_shape = (
            [x for x in self.input_size]
            if not flatten
            else [math.prod(self.input_size)]
        )
        if not heads_last:
            return [self.batch_size, self.heads] + token_layout_shape + [self.dim]
        return [self.batch_size] + token_layout_shape + [self.heads, self.dim]

    def get_k_tensor_shape(self, heads_last: bool, flatten: bool = False) -> List:
        return self.get_q_tensor_shape(heads_last=heads_last, flatten=flatten)

    def get_v_tensor_shape(self, heads_last: bool, flatten: bool = False) -> List:
        token_layout_shape = (
            [x for x in self.input_size]
            if not flatten
            else [math.prod(self.input_size)]
        )
        if not heads_last:
            return [self.batch_size, self.heads] + token_layout_shape + [self.dim_value]
        return [self.batch_size] + token_layout_shape + [self.heads, self.dim_value]

    def get_o_tensor_shape(self, heads_last: bool, flatten: bool = False) -> List:
        return self.get_v_tensor_shape(heads_last=heads_last, flatten=flatten)

    def get_add_k_tensor_shape(self, heads_last: bool) -> List:
        assert self.additional_kv_length is not None
        if not heads_last:
            return [self.batch_size, self.heads, self.additional_kv_length, self.dim]
        return [self.batch_size, self.additional_kv_length, self.heads, self.dim]

    def get_add_v_tensor_shape(self, heads_last: bool) -> List:
        assert self.additional_kv_length is not None
        if not heads_last:
            return [
                self.batch_size,
                self.heads,
                self.additional_kv_length,
                self.dim_value,
            ]
        return [self.batch_size, self.additional_kv_length, self.heads, self.dim_value]

    def make_qkv_tensors(
        self, device, requires_grad: bool, heads_last: bool, flatten: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor]:
        q = torch.randn(
            self.get_q_tensor_shape(heads_last=heads_last, flatten=flatten),
            device=device,
            dtype=self.dtype,
            requires_grad=requires_grad,
        )
        k = torch.randn(
            self.get_k_tensor_shape(heads_last=heads_last, flatten=flatten),
            device=device,
            dtype=self.dtype,
            requires_grad=requires_grad,
        )
        v = torch.randn(
            self.get_v_tensor_shape(heads_last=heads_last, flatten=flatten),
            device=device,
            dtype=self.dtype,
            requires_grad=requires_grad,
        )

        return q, k, v

    def make_qkvo_tensors(
        self, device, requires_grad: bool, heads_last: bool, flatten: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        q, k, v = self.make_qkv_tensors(
            device=device,
            requires_grad=requires_grad,
            heads_last=heads_last,
            flatten=flatten,
        )
        o = torch.randn(
            self.get_o_tensor_shape(heads_last=heads_last, flatten=flatten),
            device=device,
            dtype=self.dtype,
            requires_grad=requires_grad,
        )
        return q, k, v, o

    def make_additional_kv_tensors(
        self, device, requires_grad: bool, heads_last: bool
    ) -> Tuple[Tensor, Tensor]:
        add_k = torch.randn(
            self.get_add_k_tensor_shape(heads_last=heads_last),
            device=device,
            dtype=self.dtype,
            requires_grad=requires_grad,
        )
        add_v = torch.randn(
            self.get_add_v_tensor_shape(heads_last=heads_last),
            device=device,
            dtype=self.dtype,
            requires_grad=requires_grad,
        )

        return add_k, add_v

    def __str__(self):
        return (
            "Problem("
            + f"batch_size={self.batch_size}, "
            + f"heads={self.heads}, "
            + f"input_size={self.input_size}, "
            + f"window_size={self.window_size}, "
            + f"stride={self.stride}, "
            + f"dilation={self.dilation}, "
            + f"is_causal={self.is_causal}, "
            + f"additional_kv_length={self.additional_kv_length}, "
            + f"dtype={self.dtype})"
        )

    def __repr__(self):
        return self.__str__()


def generate_problem(
    batch_size: int,
    heads: int,
    input_size: DimensionType,
    dim: int,
    window_size: DimensionType,
    stride: DimensionType,
    dilation: DimensionType,
    dtype: Any,
    is_causal: CausalArgType,
    additional_kv_length: Optional[int] = None,
    dim_value: Optional[int] = None,
) -> Problem:
    na_dim = len(input_size)
    assert len(window_size) == na_dim
    assert len(stride) == na_dim
    assert len(dilation) == na_dim
    return Problem(
        na_dim=na_dim,
        batch_size=batch_size,
        heads=heads,
        dim=dim,
        dim_value=dim_value,
        input_size=input_size,
        window_size=window_size,
        stride=stride,
        dilation=dilation,
        dtype=dtype,
        is_causal=is_causal,
        additional_kv_length=additional_kv_length,
    )
