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

import copy
import math
from typing import Any, List, Optional, Union

# NOTE: switch to | when < 3.10 support is dropped
CausalType = Optional[Union[bool, tuple]]


class Problem:
    def __init__(
        self,
        na_dim: int,
        batch_size: int,
        heads: int,
        dim: int,
        spatial_size: List[int],
        kernel_size: List[int],
        dilation: List[int],
        dtype: Any,
        has_bias: bool,
        is_causal: CausalType = None,
    ):
        self.na_dim = na_dim
        self.batch_size = batch_size
        self.heads = heads
        self.dim = dim
        assert len(kernel_size) == len(dilation) == len(spatial_size) == self.na_dim
        self.spatial_size = spatial_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dtype = dtype
        self.has_bias = has_bias
        self.is_causal = is_causal or [False for _ in range(na_dim)]

    def get_tensor_shape(self, fused_layout: bool) -> List:
        if not fused_layout:
            return (
                [self.batch_size, self.heads]
                + copy.deepcopy(self.spatial_size)
                + [self.dim]
            )
        return (
            [self.batch_size]
            + copy.deepcopy(self.spatial_size)
            + [self.heads, self.dim]
        )

    def get_flattened_tensor_shape(self, fused_layout: bool) -> List:
        if not fused_layout:
            return (
                [self.batch_size, self.heads]
                + [math.prod(self.spatial_size)]
                + [self.dim]
            )
        return (
            [self.batch_size] + [math.prod(self.spatial_size)] + [self.heads, self.dim]
        )

    def get_attn_tensor_shape(self, fused_layout: bool) -> List:
        if not fused_layout:
            return (
                [self.batch_size, self.heads]
                + copy.deepcopy(self.spatial_size)
                + [math.prod(self.kernel_size)]
            )
        return (
            [self.batch_size]
            + copy.deepcopy(self.spatial_size)
            + [self.heads]
            + [math.prod(self.kernel_size)]
        )

    def get_bias_shape(self) -> List:
        return [self.heads] + [k * 2 - 1 for k in self.kernel_size]


def generate_1d_problem(
    batch_size: int,
    heads: int,
    length: int,
    dim: int,
    kernel_size: int,
    dilation: int,
    dtype: Any,
    has_bias: bool,
    is_causal: CausalType = None,
) -> Problem:
    return Problem(
        na_dim=1,
        batch_size=batch_size,
        heads=heads,
        dim=dim,
        spatial_size=[length],
        kernel_size=[kernel_size],
        dilation=[dilation],
        dtype=dtype,
        has_bias=has_bias,
        is_causal=is_causal,
    )


def generate_2d_problem(
    batch_size: int,
    heads: int,
    height: int,
    width: int,
    dim: int,
    kernel_size: int,
    dilation: int,
    dtype: Any,
    has_bias: bool,
    is_causal: CausalType = None,
) -> Problem:
    return Problem(
        na_dim=2,
        batch_size=batch_size,
        heads=heads,
        dim=dim,
        spatial_size=[height, width],
        kernel_size=[kernel_size, kernel_size],
        dilation=[dilation, dilation],
        dtype=dtype,
        has_bias=has_bias,
        is_causal=is_causal,
    )


def generate_3d_problem(
    batch_size: int,
    heads: int,
    depth: int,
    height: int,
    width: int,
    dim: int,
    kernel_size: int,
    dilation: int,
    dtype: Any,
    has_bias: bool,
    is_causal: CausalType = None,
) -> Problem:
    return Problem(
        na_dim=3,
        batch_size=batch_size,
        heads=heads,
        dim=dim,
        spatial_size=[depth, height, width],
        kernel_size=[kernel_size, kernel_size, kernel_size],
        dilation=[dilation, dilation, dilation],
        dtype=dtype,
        has_bias=has_bias,
        is_causal=is_causal,
    )
