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
from numbers import Number
from typing import Any, List

from fvcore.nn import FlopCountAnalysis  # type: ignore
from fvcore.nn.jit_handles import get_shape  # type: ignore


def qk_1d_rpb_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Counts flops for the 1D QK operation.
    """
    assert (
        len(inputs) >= 2
    ), f"Expected at least 2 inputs (query, key), got {len(inputs)}"
    assert len(outputs) == 1, f"Expected 1 output (attn), got {len(outputs)}"
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]
    assert (
        len(input_shapes[0]) == 4
    ), f"Query must be a 4-dim tensor, got {len(input_shapes[0])}"
    assert (
        len(input_shapes[1]) == 4
    ), f"Key must be a 4-dim tensor, got {len(input_shapes[1])}"
    assert (
        len(output_shapes[0]) == 4
    ), f"Output must be a 4-dim tensor, got {len(output_shapes[0])}"
    assert (
        input_shapes[0] == input_shapes[1]
    ), f"Query and Key shapes did not match! Q: {input_shapes[0]}, K: {input_shapes[1]}"
    batch_size, heads, length, dim = input_shapes[0]
    batch_size, heads, length, kernel_size = output_shapes[0]

    flops = batch_size * heads * length * dim * kernel_size
    flops += batch_size * heads * length * kernel_size
    return flops


def av_1d_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Counts flops for the 1D AV operation.
    """
    assert len(inputs) == 2, f"Expected 2 inputs (attn and value), got {len(inputs)}"
    assert len(outputs) == 1, f"Expected 1 output (out), got {len(outputs)}"
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]
    assert (
        len(input_shapes[0]) == 4
    ), f"Attn must be a 4-dim tensor, got {len(input_shapes[0])}"
    assert (
        len(input_shapes[1]) == 4
    ), f"Value must be a 4-dim tensor, got {len(input_shapes[1])}"
    assert (
        len(output_shapes[0]) == 4
    ), f"Output must be a 4-dim tensor, got {len(output_shapes[0])}"
    assert output_shapes[0] == input_shapes[1], (
        f"Out and Value shapes did not match! O: {output_shapes[0]}, V:"
        f" {input_shapes[1]}"
    )
    batch_size, heads, length, kernel_size = input_shapes[0]
    batch_size, heads, length, dim = output_shapes[0]
    flops = batch_size * heads * length * dim * kernel_size
    return flops


def qk_2d_rpb_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Counts flops for the 2D QK operation.
    """
    assert (
        len(inputs) >= 2
    ), f"Expected at least 2 inputs (query, key), got {len(inputs)}"
    assert len(outputs) == 1, f"Expected 1 output (attn), got {len(outputs)}"
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]
    assert (
        len(input_shapes[0]) == 5
    ), f"Query must be a 5-dim tensor, got {len(input_shapes[0])}"
    assert (
        len(input_shapes[1]) == 5
    ), f"Key must be a 5-dim tensor, got {len(input_shapes[1])}"
    assert (
        len(output_shapes[0]) == 5
    ), f"Output must be a 5-dim tensor, got {len(output_shapes[0])}"
    assert (
        input_shapes[0] == input_shapes[1]
    ), f"Query and Key shapes did not match! Q: {input_shapes[0]}, K: {input_shapes[1]}"
    batch_size, heads, height, width, dim = input_shapes[0]
    batch_size, heads, height, width, kernel_size_sq = output_shapes[0]

    flops = batch_size * heads * height * width * dim * kernel_size_sq
    flops += batch_size * heads * height * width * kernel_size_sq
    return flops


def av_2d_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Counts flops for the 2D AV operation.
    """
    assert len(inputs) == 2, f"Expected 2 inputs (attn and value), got {len(inputs)}"
    assert len(outputs) == 1, f"Expected 1 output (out), got {len(outputs)}"
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]
    assert (
        len(input_shapes[0]) == 5
    ), f"Attn must be a 5-dim tensor, got {len(input_shapes[0])}"
    assert (
        len(input_shapes[1]) == 5
    ), f"Value must be a 5-dim tensor, got {len(input_shapes[1])}"
    assert (
        len(output_shapes[0]) == 5
    ), f"Output must be a 5-dim tensor, got {len(output_shapes[0])}"
    assert output_shapes[0] == input_shapes[1], (
        f"Out and Value shapes did not match! O: {output_shapes[0]}, V:"
        f" {input_shapes[1]}"
    )
    batch_size, heads, height, width, kernel_size_sq = input_shapes[0]
    batch_size, heads, height, width, dim = output_shapes[0]
    flops = batch_size * heads * height * width * dim * kernel_size_sq
    return flops


def qk_3d_rpb_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Counts flops for the 3D QK operation.
    """
    assert (
        len(inputs) >= 2
    ), f"Expected at least 2 inputs (query, key), got {len(inputs)}"
    assert len(outputs) == 1, f"Expected 1 output (attn), got {len(outputs)}"
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]
    assert (
        len(input_shapes[0]) == 6
    ), f"Query must be a 6-dim tensor, got {len(input_shapes[0])}"
    assert (
        len(input_shapes[1]) == 6
    ), f"Key must be a 6-dim tensor, got {len(input_shapes[1])}"
    assert (
        len(output_shapes[0]) == 6
    ), f"Output must be a 6-dim tensor, got {len(output_shapes[0])}"
    assert (
        input_shapes[0] == input_shapes[1]
    ), f"Query and Key shapes did not match! Q: {input_shapes[0]}, K: {input_shapes[1]}"
    batch_size, heads, depth, height, width, dim = input_shapes[0]
    batch_size, heads, depth, height, width, kernel_size_cu = output_shapes[0]

    flops = batch_size * heads * depth * height * width * dim * kernel_size_cu
    flops += batch_size * heads * depth * height * width * kernel_size_cu
    return flops


def av_3d_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Counts flops for the 3D AV operation.
    """
    assert len(inputs) == 2, f"Expected 2 inputs (attn and value), got {len(inputs)}"
    assert len(outputs) == 1, f"Expected 1 output (out), got {len(outputs)}"
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]
    assert (
        len(input_shapes[0]) == 6
    ), f"Attn must be a 6-dim tensor, got {len(input_shapes[0])}"
    assert (
        len(input_shapes[1]) == 6
    ), f"Value must be a 6-dim tensor, got {len(input_shapes[1])}"
    assert (
        len(output_shapes[0]) == 6
    ), f"Output must be a 6-dim tensor, got {len(output_shapes[0])}"
    assert output_shapes[0] == input_shapes[1], (
        f"Out and Value shapes did not match! O: {output_shapes[0]}, V:"
        f" {input_shapes[1]}"
    )
    batch_size, heads, depth, height, width, kernel_size_cu = input_shapes[0]
    batch_size, heads, depth, height, width, dim = output_shapes[0]
    flops = batch_size * heads * depth * height * width * dim * kernel_size_cu
    return flops


def add_natten_handle(flop_ctr):
    return flop_ctr.set_op_handle(
        **{
            "prim::PythonOp.NATTEN1DQKRPBFunction": qk_1d_rpb_flop,
            "prim::PythonOp.NATTEN1DAVFunction": av_1d_flop,
            "prim::PythonOp.NATTEN2DQKRPBFunction": qk_2d_rpb_flop,
            "prim::PythonOp.NATTEN2DAVFunction": av_2d_flop,
            "prim::PythonOp.NATTEN3DQKRPBFunction": qk_3d_rpb_flop,
            "prim::PythonOp.NATTEN3DAVFunction": av_3d_flop,
            "prim::PythonOp.NeighborhoodAttention1DQKAutogradFunction": qk_1d_rpb_flop,
            "prim::PythonOp.NeighborhoodAttention1DAVAutogradFunction": av_1d_flop,
            "prim::PythonOp.NeighborhoodAttention2DQKAutogradFunction": qk_2d_rpb_flop,
            "prim::PythonOp.NeighborhoodAttention2DAVAutogradFunction": av_2d_flop,
            "prim::PythonOp.NeighborhoodAttention3DQKAutogradFunction": qk_3d_rpb_flop,
            "prim::PythonOp.NeighborhoodAttention3DAVAutogradFunction": av_3d_flop,
        }
    )


def get_flops(model, input, disable_warnings=False):
    flop_ctr = FlopCountAnalysis(model, input)
    flop_ctr = add_natten_handle(flop_ctr)
    if disable_warnings:
        flop_ctr = flop_ctr.unsupported_ops_warnings(False)
    return flop_ctr.total()
