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

#
# NATTEN's FLOP/MAC utilities and helpers.
#
# NATTEN supports two FLOP counters: torch.utils.flop_counter, and fvcore.
# NATTEN implements its own counters, and provides APIs for torch.utils.flop_counter
# and fvcore.
#
# NOTE: fvcore computes MACs, not FLOPs. GEMM flops in fvcore are MNK not 2MNK.
#

import math
from typing import Any, List, Sequence, Tuple

import torch

from .utils import check_all_args, get_num_na_weights


def _bmm_flops(b: int, m: int, n: int, k: int) -> int:
    return 2 * b * m * n * k


def _bmm_macs(b: int, m: int, n: int, k: int) -> int:
    return b * m * n * k


def _na_qk_flops(
    batch_size: int,
    heads: int,
    num_tokens: int,
    dim: int,
    num_attn_weights: int,
    return_macs: bool = False,
) -> int:
    b = batch_size * heads
    m = num_tokens
    n = num_attn_weights
    k = dim
    return _bmm_macs(b, m, n, k) if return_macs else _bmm_flops(b, m, n, k)


def _na_av_flops(
    batch_size: int,
    heads: int,
    num_tokens: int,
    dim: int,
    num_attn_weights: int,
    return_macs: bool = False,
) -> int:
    b = batch_size * heads
    m = num_tokens
    n = dim
    k = num_attn_weights
    return _bmm_macs(b, m, n, k) if return_macs else _bmm_flops(b, m, n, k)


# "Heads last" layout -- primarily used in fused impls.
def _get_parameters_from_inputs_BLHD(
    input_shape: torch.Size | Sequence[int],
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
) -> Tuple[int, int, int, int, int]:  # batch, heads, num_tokens, dim, num_attn_weights
    assert len(input_shape) in [4, 5, 6], (
        "Expected QKV shapes to be either 4D, 5D, or 6D (for 1D, 2D and 3D NA respectively), got "
        + f"{input_shape}."
    )
    na_dim = len(input_shape) - 3

    kernel_size_, dilation_, is_causal_ = check_all_args(
        na_dim, kernel_size, dilation, is_causal
    )

    batch_size, heads, dim = (
        input_shape[0],
        input_shape[-2],
        input_shape[-1],
    )

    spatial_extent = input_shape[1 : na_dim + 1]
    assert len(spatial_extent) == len(kernel_size_) == na_dim, (
        f"Expected both spatial extent and kernel size to be {na_dim} tuples, got "
        + f"{len(spatial_extent)=}, {len(kernel_size_)=}."
    )

    num_tokens = math.prod(spatial_extent)
    num_attn_weights = get_num_na_weights(kernel_size_)

    return (batch_size, heads, num_tokens, dim, num_attn_weights)


# "Heads first" layout -- primarily used in BMM impls.
def _get_parameters_from_inputs_BHLD(
    input_shape: torch.Size | Sequence[int],
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
) -> Tuple[int, int, int, int, int]:  # batch, heads, num_tokens, dim, num_attn_weights
    assert len(input_shape) in [4, 5, 6], (
        "Expected QKV shapes to be either 4D, 5D, or 6D (for 1D, 2D and 3D NA respectively), got "
        + f"{input_shape}."
    )
    na_dim = len(input_shape) - 3

    kernel_size_, dilation_, is_causal_ = check_all_args(
        na_dim, kernel_size, dilation, is_causal
    )

    batch_size, heads, dim = (
        input_shape[0],
        input_shape[1],
        input_shape[-1],
    )

    spatial_extent = input_shape[2 : na_dim + 2]
    assert len(spatial_extent) == len(kernel_size_) == na_dim, (
        f"Expected both spatial extent and kernel size to be {na_dim} tuples, got "
        + f"{len(spatial_extent)=}, {len(kernel_size_)=}."
    )

    num_tokens = math.prod(spatial_extent)
    num_attn_weights = get_num_na_weights(kernel_size_)

    return (batch_size, heads, num_tokens, dim, num_attn_weights)


def _count_na_flops_generic(
    input_shape: torch.Size | Sequence[int],
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
    is_heads_last: bool,
    return_macs: bool = False,
) -> Tuple[int, int]:
    (batch_size, heads, num_tokens, dim, num_attn_weights) = (
        _get_parameters_from_inputs_BLHD
        if is_heads_last
        else _get_parameters_from_inputs_BHLD
    )(
        input_shape=input_shape,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
    )

    # TODO: causal masking does not affect FLOPs/MACs, when it should.
    # But if we go down that rabbit hole, we should also consider counting FLOPs for
    # non-TC operations, but separately, and report both in a reasonable manner.
    # Non-TC ops and TC ops shouldn't be combined, because they are not expected to have
    # at all similar implications. But that's a rant for a different day.
    qk_flops = _na_qk_flops(
        batch_size=batch_size,
        heads=heads,
        num_tokens=num_tokens,
        dim=dim,
        num_attn_weights=num_attn_weights,
        return_macs=return_macs,
    )
    av_flops = _na_av_flops(
        batch_size=batch_size,
        heads=heads,
        num_tokens=num_tokens,
        dim=dim,
        num_attn_weights=num_attn_weights,
        return_macs=return_macs,
    )

    return qk_flops, av_flops


def fna_flop_count(
    q_shape: torch.Size | Sequence[int],
    k_shape: torch.Size | Sequence[int],
    v_shape: torch.Size | Sequence[int],
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
    is_heads_last: bool,
    return_macs: bool = False,
) -> int:
    assert (
        q_shape == k_shape == v_shape
    ), f"Expected identical QKV shapes, got {q_shape=}, {k_shape=}, {v_shape=}."

    qk_flops, av_flops = _count_na_flops_generic(
        input_shape=q_shape,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
        is_heads_last=is_heads_last,
        return_macs=return_macs,
    )

    # NOTE: attention bias and softmax are ignored.
    return qk_flops + av_flops


def na_qk_flop_count(
    q_shape: torch.Size | Sequence[int],
    k_shape: torch.Size | Sequence[int],
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
    is_heads_last: bool,
    return_macs: bool = False,
) -> int:
    assert (
        q_shape == k_shape
    ), f"Expected identical Q and K shapes, got {q_shape=}, {k_shape=}."

    qk_flops, _ = _count_na_flops_generic(
        input_shape=q_shape,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
        is_heads_last=is_heads_last,
        return_macs=return_macs,
    )

    # NOTE: attention bias and softmax are ignored.
    return qk_flops


def na_av_flop_count(
    a_shape: torch.Size | Sequence[int],
    v_shape: torch.Size | Sequence[int],
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
    is_heads_last: bool,
    return_macs: bool = False,
) -> int:

    _, av_flops = _count_na_flops_generic(
        input_shape=v_shape,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
        is_heads_last=is_heads_last,
        return_macs=return_macs,
    )
    return av_flops


#################################################################################################
########################################## FVCore addons ########################################
#################################################################################################


try:
    from fvcore.nn import FlopCountAnalysis  # type: ignore
    from fvcore.nn.jit_handles import get_shape  # type: ignore

    has_fvcore = True

except ImportError:
    has_fvcore = False


if has_fvcore:

    def _fvcore_fna_mac_count(inputs: List[Any], outputs: List[Any]) -> int:
        assert (
            len(inputs) >= 3
        ), f"Expected at least 3 inputs (query, key, value), got {len(inputs)}"
        input_shapes = [get_shape(v) for v in inputs]

        # Weird fvcore / jit bug
        assert len(outputs) in [
            1,
            2,
        ], f"Expected exactly 1 or 2 outputs (tuple), got {len(outputs)}"
        if len(outputs) == 1:
            outputs_ = outputs[0].uses()[0].user.outputs()
            output_shapes = [get_shape(v) for v in outputs_]
            assert (
                len(output_shapes) == 2
            ), f"Expected exactly 2 outputs (attention output, and LSE), got {len(output_shapes)}"
        else:
            assert (
                len(outputs) == 2
            ), f"Expected exactly 2 outputs (attention output, and LSE), got {len(outputs)}"
            output_shapes = [get_shape(v) for v in outputs]

        assert len(input_shapes[0]) in [
            4,
            5,
            6,
        ], f"Input tensors must be of rank 4, 5, or 6, got {len(input_shapes[0])}"

        assert len(input_shapes[1]) == len(input_shapes[1]) == len(input_shapes[2]), (
            f"All input tensors must be of the same rank, got {len(input_shapes[0])}, "
            + f"{len(input_shapes[1])} and {len(input_shapes[2])}"
        )

        assert len(output_shapes[0]) == len(
            input_shapes[0]
        ), f"Output tensor must match the rank of input tensors, got {len(output_shapes[0])} != {len(input_shapes[0])}"
        assert len(output_shapes[1]) == len(
            output_shapes[0][:-1]
        ), f"Mismatch between output and LSE shape, got {len(output_shapes[0])} != {len(output_shapes[1])}"

        assert (
            input_shapes[0] == input_shapes[1] == input_shapes[2] == output_shapes[0]
        ), (
            "Query, key, value, and output must match in shape, got q.shape="
            + f"{input_shapes[0]}, k.shape={input_shapes[1]}, v.shape={input_shapes[1]}."
        )

        # NOTE: really hacky way to extract non-tensor args, but gets the job done.
        # The jit trace only picks up tensor operands in inputs and outputs, but
        # it's impossible to compute FLOPs without knowing kernel size.
        assert hasattr(inputs[0], "uses") and callable(inputs[0].uses)
        _uses = inputs[0].uses()
        assert (
            hasattr(_uses, "__len__")
            and len(_uses) == 1
            and hasattr(_uses[0], "user")
            and hasattr(_uses[0].user, "scalar_args")
            and callable(_uses[0].user.scalar_args)
        )
        scalar_args = _uses[0].user.scalar_args()
        assert hasattr(scalar_args, "__len__") and len(scalar_args) in [
            6,
            7,
        ], f"Expected 6 or 7 non-tensor args to the op, got {len(scalar_args)}. {scalar_args=}"

        # Kick off rpb=None
        if len(scalar_args) == 7:
            assert scalar_args[0] is None
            scalar_args = scalar_args[1:]

        (
            kernel_size,
            dilation,
            is_causal,
            attn_scale,
            tiling_config,
            tiling_config_backward,
        ) = scalar_args
        # TODO: it's very easy to hit this assertion. We must make sure
        # arguments like kernel size are checked before calling the autograd function,
        # not inside it.
        assert isinstance(
            kernel_size, tuple
        ), f"Expected kernel_size to be a tuple, got {type(kernel_size)=}."

        assert len(kernel_size) + 3 == len(input_shapes[0]), (
            "Tensor rank must be equal to len(kernel_size) + 3 = "
            + f"{len(kernel_size) + 3}, got {len(input_shapes[0])}"
        )

        q_shape = input_shapes[0]
        k_shape = input_shapes[1]
        v_shape = input_shapes[2]

        return fna_flop_count(
            q_shape=q_shape,
            k_shape=k_shape,
            v_shape=v_shape,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            is_heads_last=True,
            return_macs=True,  # FVCore reports MACs, not FLOPs.
        )

    def _fvcore_na_qk_mac_count(inputs: List[Any], outputs: List[Any]) -> int:
        assert (
            len(inputs) >= 2
        ), f"Expected at least 2 inputs (query, key), got {len(inputs)}"
        assert len(outputs) == 1, f"Expected 1 output (attn), got {len(outputs)}"
        input_shapes = [get_shape(v) for v in inputs]

        # NOTE: really hacky way to extract non-tensor args, but gets the job done.
        # The jit trace only picks up tensor operands in inputs and outputs, but
        # it's impossible to compute FLOPs without knowing kernel size.
        assert hasattr(inputs[0], "uses") and callable(inputs[0].uses)
        _uses = inputs[0].uses()
        assert (
            hasattr(_uses, "__len__")
            and len(_uses) == 1
            and hasattr(_uses[0], "user")
            and hasattr(_uses[0].user, "scalar_args")
            and callable(_uses[0].user.scalar_args)
        )
        scalar_args = _uses[0].user.scalar_args()
        assert hasattr(scalar_args, "__len__") and len(scalar_args) in [
            3,
            4,
            5,
        ], f"Expected 3, 4 or 5 non-tensor args to the op, got {len(scalar_args)}. {scalar_args=}"

        # Kick off optional bias, additional keys
        if len(scalar_args) > 3:
            num_nones = len(scalar_args) - 3
            assert num_nones in [1, 2]
            for i in range(num_nones):
                assert scalar_args[i] is None
            scalar_args = scalar_args[num_nones:]

        (
            kernel_size,
            dilation,
            is_causal,
        ) = scalar_args
        # TODO: it's very easy to hit this assertion. We must make sure
        # arguments like kernel size are checked before calling the autograd function,
        # not inside it.
        assert isinstance(
            kernel_size, tuple
        ), f"Expected kernel_size to be a tuple, got {type(kernel_size)=}."

        assert len(kernel_size) + 3 == len(input_shapes[0]), (
            "Tensor rank must be equal to len(kernel_size) + 3 = "
            + f"{len(kernel_size) + 3}, got {len(input_shapes[0])}"
        )

        q_shape = input_shapes[0]
        k_shape = input_shapes[1]

        return na_qk_flop_count(
            q_shape=q_shape,
            k_shape=k_shape,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            is_heads_last=True,
            return_macs=True,  # FVCore reports MACs, not FLOPs.
        )

    def _fvcore_na_av_mac_count(inputs: List[Any], outputs: List[Any]) -> int:
        assert (
            len(inputs) == 2
        ), f"Expected 2 inputs (attn and value), got {len(inputs)}"
        assert len(outputs) == 1, f"Expected 1 output (out), got {len(outputs)}"
        input_shapes = [get_shape(v) for v in inputs]

        # NOTE: really hacky way to extract non-tensor args, but gets the job done.
        # The jit trace only picks up tensor operands in inputs and outputs, but
        # it's impossible to compute FLOPs without knowing kernel size.
        assert hasattr(inputs[0], "uses") and callable(inputs[0].uses)
        _uses = inputs[0].uses()
        assert (
            hasattr(_uses, "__len__")
            and len(_uses) == 1
            and hasattr(_uses[0], "user")
            and hasattr(_uses[0].user, "scalar_args")
            and callable(_uses[0].user.scalar_args)
        )
        scalar_args = _uses[0].user.scalar_args()
        assert hasattr(scalar_args, "__len__") and len(scalar_args) in [
            3,
            4,
        ], f"Expected 3 or 4 non-tensor args to the op, got {len(scalar_args)}. {scalar_args=}"

        # Kick off optional additional values
        if len(scalar_args) == 4:
            assert scalar_args[0] is None
            scalar_args = scalar_args[1:]

        (
            kernel_size,
            dilation,
            is_causal,
        ) = scalar_args
        # TODO: it's very easy to hit this assertion. We must make sure
        # arguments like kernel size are checked before calling the autograd function,
        # not inside it.
        assert isinstance(
            kernel_size, tuple
        ), f"Expected kernel_size to be a tuple, got {type(kernel_size)=}."

        assert len(kernel_size) + 3 == len(input_shapes[0]), (
            "Tensor rank must be equal to len(kernel_size) + 3 = "
            + f"{len(kernel_size) + 3}, got {len(input_shapes[0])}"
        )

        a_shape = input_shapes[0]
        v_shape = input_shapes[1]

        return na_av_flop_count(
            a_shape=a_shape,
            v_shape=v_shape,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            is_heads_last=True,
            return_macs=True,  # FVCore reports MACs, not FLOPs.
        )

    def add_natten_handle(flop_ctr):
        return flop_ctr.set_op_handle(
            **{
                # Legacy
                "prim::PythonOp.NATTEN1DQKRPBFunction": _fvcore_na_qk_mac_count,
                "prim::PythonOp.NATTEN1DAVFunction": _fvcore_na_av_mac_count,
                "prim::PythonOp.NATTEN2DQKRPBFunction": _fvcore_na_qk_mac_count,
                "prim::PythonOp.NATTEN2DAVFunction": _fvcore_na_av_mac_count,
                "prim::PythonOp.NATTEN3DQKRPBFunction": _fvcore_na_qk_mac_count,
                "prim::PythonOp.NATTEN3DAVFunction": _fvcore_na_av_mac_count,
                # Unfused
                "prim::PythonOp.NeighborhoodAttention1DQKAutogradFunction": _fvcore_na_qk_mac_count,
                "prim::PythonOp.NeighborhoodAttention1DAVAutogradFunction": _fvcore_na_av_mac_count,
                "prim::PythonOp.NeighborhoodAttention2DQKAutogradFunction": _fvcore_na_qk_mac_count,
                "prim::PythonOp.NeighborhoodAttention2DAVAutogradFunction": _fvcore_na_av_mac_count,
                "prim::PythonOp.NeighborhoodAttention3DQKAutogradFunction": _fvcore_na_qk_mac_count,
                "prim::PythonOp.NeighborhoodAttention3DAVAutogradFunction": _fvcore_na_av_mac_count,
                # Fused ops
                "prim::PythonOp.FusedNeighborhoodAttention1D": _fvcore_fna_mac_count,
                "prim::PythonOp.FusedNeighborhoodAttention2D": _fvcore_fna_mac_count,
                "prim::PythonOp.FusedNeighborhoodAttention3D": _fvcore_fna_mac_count,
            }
        )

    def get_flops(model, input, disable_warnings=False):
        flop_ctr = FlopCountAnalysis(model, input)
        flop_ctr = add_natten_handle(flop_ctr)
        if disable_warnings:
            flop_ctr = flop_ctr.unsupported_ops_warnings(False)
        return flop_ctr.total()
