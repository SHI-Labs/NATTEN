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
import math
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple

import torch
from torch import Size, Tensor
from torch._ops import OpOverloadPacket
from torch.utils.flop_counter import bmm_flop

try:
    from natten import libnatten  # type: ignore
except ImportError:
    raise ImportError(
        "Failed to import libnatten. "
        "This could be due to an invalid/incomplete install. "
        "Please make sure you built NATTEN correctly, or refer to "
        "https://shi-labs.com/natten for more information."
    )

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
if torch_ver < [2, 4]:
    raise NotImplementedError(
        "NATTEN's experimental interface is written for torch >= 2.4, "
        f"your torch version: {torch.__version__}."
    )

from .autotuner.fna_forward import get_default_tiling_config_for_fna_forward
from .ops import additional_sdpa, merge_attentions

from .types import (
    CausalArg1DTypeOrDed,
    CausalArg2DTypeOrDed,
    CausalArg3DTypeOrDed,
    Dimension1DTypeOrDed,
    Dimension2DTypeOrDed,
    Dimension3DTypeOrDed,
)

from .utils import check_all_args, check_tiling_config, get_num_na_weights, log

logger = log.get_logger(__name__)


#################################################################################################
##################################### Register forward ops. #####################################
#################################################################################################


@torch.library.custom_op(
    "natten::na1d_forward_op", mutates_args=(), device_types=("cpu", "cuda")
)
def na1d_torch_library_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
    scale: float,
    q_tiler_: Sequence[int],
    kv_tiler_: Sequence[int],
) -> Tuple[Tensor, Tensor]:
    kernel_size, dilation, is_causal = check_all_args(
        1, kernel_size_, dilation_, is_causal_
    )

    assert isinstance(
        scale, float
    ), f"Expected float attention scale, got {type(scale)}."

    (q_tiler, kv_tiler) = check_tiling_config(1, (q_tiler_, kv_tiler_))

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    # NOTE: this is what needs a PyTorch fix; ops should not have to call .contiguous()
    # and instead indicate they require contiguous operands.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if bias is not None:
        bias = bias.to(query.dtype).contiguous()
    output = torch.empty_like(value)

    # TODO: logsumexp should be conditional
    logsumexp = torch.empty(query.shape[:-1], dtype=torch.float32, device=query.device)

    libnatten.na1d_forward(
        output,
        query,
        key,
        value,
        bias,
        logsumexp,
        kernel_size,
        dilation,
        is_causal,
        scale,
        q_tiler,
        kv_tiler,
    )

    return output, logsumexp


@torch.library.custom_op(
    "natten::na2d_forward_op", mutates_args=(), device_types=("cpu", "cuda")
)
def na2d_torch_library_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
    scale: float,
    q_tiler_: Sequence[int],
    kv_tiler_: Sequence[int],
) -> Tuple[Tensor, Tensor]:
    kernel_size, dilation, is_causal = check_all_args(
        2, kernel_size_, dilation_, is_causal_
    )

    assert isinstance(
        scale, float
    ), f"Expected float attention scale, got {type(scale)}."

    (q_tiler, kv_tiler) = check_tiling_config(2, (q_tiler_, kv_tiler_))

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    # NOTE: this is what needs a PyTorch fix; ops should not have to call .contiguous()
    # and instead indicate they require contiguous operands.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if bias is not None:
        bias = bias.to(query.dtype).contiguous()
    output = torch.empty_like(value)

    # TODO: logsumexp should be conditional
    logsumexp = torch.empty(query.shape[:-1], dtype=torch.float32, device=query.device)

    libnatten.na2d_forward(
        output,
        query,
        key,
        value,
        bias,
        logsumexp,
        kernel_size,
        dilation,
        is_causal,
        scale,
        q_tiler,
        kv_tiler,
    )

    return output, logsumexp


@torch.library.custom_op(
    "natten::na3d_forward_op", mutates_args=(), device_types=("cpu", "cuda")
)
def na3d_torch_library_op(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
    scale: float,
    q_tiler_: Sequence[int],
    kv_tiler_: Sequence[int],
) -> Tuple[Tensor, Tensor]:
    kernel_size, dilation, is_causal = check_all_args(
        3, kernel_size_, dilation_, is_causal_
    )

    assert isinstance(
        scale, float
    ), f"Expected float attention scale, got {type(scale)}."

    (q_tiler, kv_tiler) = check_tiling_config(3, (q_tiler_, kv_tiler_))

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    # NOTE: this is what needs a PyTorch fix; ops should not have to call .contiguous()
    # and instead indicate they require contiguous operands.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    if bias is not None:
        bias = bias.to(query.dtype).contiguous()
    output = torch.empty_like(value)

    # TODO: logsumexp should be conditional
    logsumexp = torch.empty(query.shape[:-1], dtype=torch.float32, device=query.device)

    libnatten.na3d_forward(
        output,
        query,
        key,
        value,
        bias,
        logsumexp,
        kernel_size,
        dilation,
        is_causal,
        scale,
        q_tiler,
        kv_tiler,
    )

    return output, logsumexp


#################################################################################################
################## Register "fake" ops for forward op, since it's not inplace. ##################
#################################################################################################


@na1d_torch_library_op.register_fake
def na1d_op_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
    scale: float,
    q_tiler_: Sequence[int],
    kv_tiler_: Sequence[int],
) -> Tuple[Tensor, Tensor]:
    output = torch.empty_like(value)
    # TODO: logsumexp should be conditional
    logsumexp = torch.empty(query.shape[:-1], dtype=torch.float32, device=query.device)
    return output, logsumexp


@na2d_torch_library_op.register_fake
def na2d_op_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
    scale: float,
    q_tiler_: Sequence[int],
    kv_tiler_: Sequence[int],
) -> Tuple[Tensor, Tensor]:
    output = torch.empty_like(value)
    # TODO: logsumexp should be conditional
    logsumexp = torch.empty(query.shape[:-1], dtype=torch.float32, device=query.device)
    return output, logsumexp


@na3d_torch_library_op.register_fake
def na3d_op_fake(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    bias: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
    scale: float,
    q_tiler_: Sequence[int],
    kv_tiler_: Sequence[int],
) -> Tuple[Tensor, Tensor]:
    output = torch.empty_like(value)
    # TODO: logsumexp should be conditional
    logsumexp = torch.empty(query.shape[:-1], dtype=torch.float32, device=query.device)
    return output, logsumexp


#################################################################################################
########################################### FLOP Counts #########################################
#################################################################################################


class FlopCountFn(Protocol):
    @staticmethod
    def __call__(*args, **kwargs) -> int: ...


def na1d_flop_count(
    q: Size,
    k: Size,
    v: Size,
    bias: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
    scale: float,
    q_tiler_: Sequence[int],
    kv_tiler_: Sequence[int],
    out_shape: Any,
):
    kernel_size, dilation, is_causal = check_all_args(
        1, kernel_size_, dilation_, is_causal_
    )

    assert q == k == v

    batch_size, heads, dim = (
        q[0],
        q[-2],
        q[-1],
    )

    spatial_extent = q[1 : len(kernel_size) + 1]
    assert len(spatial_extent) == len(kernel_size)

    num_tokens = math.prod(spatial_extent)
    num_attn_weights = get_num_na_weights(kernel_size)

    total_flops = 0

    total_flops += bmm_flop(
        (batch_size * heads, num_tokens, dim), (batch_size * heads, dim, num_tokens)
    )

    total_flops += bmm_flop(
        (batch_size * heads, num_tokens, num_attn_weights),
        (batch_size * heads, num_attn_weights, dim),
    )

    return total_flops


def na2d_flop_count(
    q: Size,
    k: Size,
    v: Size,
    bias: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
    scale: float,
    q_tiler_: Sequence[int],
    kv_tiler_: Sequence[int],
    out_shape: Any,
):
    kernel_size, dilation, is_causal = check_all_args(
        2, kernel_size_, dilation_, is_causal_
    )

    assert q == k == v

    batch_size, heads, dim = (
        q[0],
        q[-2],
        q[-1],
    )

    spatial_extent = q[1 : len(kernel_size) + 1]
    assert len(spatial_extent) == len(kernel_size)

    num_tokens = math.prod(spatial_extent)
    num_attn_weights = get_num_na_weights(kernel_size)

    total_flops = 0

    total_flops += bmm_flop(
        (batch_size * heads, num_tokens, dim), (batch_size * heads, dim, num_tokens)
    )

    total_flops += bmm_flop(
        (batch_size * heads, num_tokens, num_attn_weights),
        (batch_size * heads, num_attn_weights, dim),
    )

    return total_flops


def na3d_flop_count(
    q: Size,
    k: Size,
    v: Size,
    bias: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
    scale: float,
    q_tiler_: Sequence[int],
    kv_tiler_: Sequence[int],
    out_shape: Any,
):
    kernel_size, dilation, is_causal = check_all_args(
        3, kernel_size_, dilation_, is_causal_
    )

    assert q == k == v

    batch_size, heads, dim = (
        q[0],
        q[-2],
        q[-1],
    )

    spatial_extent = q[1 : len(kernel_size) + 1]
    assert len(spatial_extent) == len(kernel_size)

    num_tokens = math.prod(spatial_extent)
    num_attn_weights = get_num_na_weights(kernel_size)

    total_flops = 0

    total_flops += bmm_flop(
        (batch_size * heads, num_tokens, dim), (batch_size * heads, dim, num_tokens)
    )

    total_flops += bmm_flop(
        (batch_size * heads, num_tokens, num_attn_weights),
        (batch_size * heads, num_attn_weights, dim),
    )

    return total_flops


custom_mapping: dict[OpOverloadPacket, FlopCountFn] = {
    torch.ops.natten.na1d_forward_op: na1d_flop_count,
    torch.ops.natten.na2d_forward_op: na2d_flop_count,
    torch.ops.natten.na3d_forward_op: na3d_flop_count,
}


#################################################################################################
######################################## User-facing APIs #######################################
#################################################################################################


def na1d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension1DTypeOrDed,
    dilation: Dimension1DTypeOrDed = 1,
    is_causal: Optional[CausalArg1DTypeOrDed] = False,
    rpb: Optional[Tensor] = None,
    scale: Optional[float] = None,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    xformers_kwargs: Optional[Dict] = None,
) -> Tensor:
    if query.is_nested or key.is_nested or value.is_nested:
        raise NotImplementedError(
            "Fused neighborhood attention does not support nested tensors yet."
        )

    kernel_size_, dilation_, is_causal_ = check_all_args(
        1, kernel_size, dilation, is_causal
    )
    # TODO: autotuner and torch compile aren't supported together
    tiling_config_forward = get_default_tiling_config_for_fna_forward(
        1, input_tensor=query, dilation=dilation_
    )

    scale = scale or query.shape[-1] ** -0.5

    output, lse = torch.ops.natten.na1d_forward_op(
        query,
        key,
        value,
        rpb,
        kernel_size_,
        dilation_,
        is_causal_,
        scale,
        *tiling_config_forward,
    )

    if additional_keys is not None and additional_values is not None:
        if additional_keys is None or additional_values is None:
            raise ValueError(
                "Both `additional_keys` and `additional_values` must be "
                "either Tensors or NoneTypes."
            )

        additional_output, additional_lse = additional_sdpa(
            query,
            additional_keys,
            additional_values,
            scale=scale,
            attn_kwargs=xformers_kwargs,
        )

        return merge_attentions(output, additional_output, lse, additional_lse)

    return output


def na2d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension2DTypeOrDed,
    dilation: Dimension2DTypeOrDed = 1,
    is_causal: Optional[CausalArg2DTypeOrDed] = False,
    rpb: Optional[Tensor] = None,
    scale: Optional[float] = None,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    xformers_kwargs: Optional[Dict] = None,
) -> Tensor:
    if query.is_nested or key.is_nested or value.is_nested:
        raise NotImplementedError(
            "Fused neighborhood attention does not support nested tensors yet."
        )

    kernel_size_, dilation_, is_causal_ = check_all_args(
        2, kernel_size, dilation, is_causal
    )
    # TODO: autotuner and torch compile aren't supported together
    tiling_config_forward = get_default_tiling_config_for_fna_forward(
        2, input_tensor=query, dilation=dilation_
    )

    scale = scale or query.shape[-1] ** -0.5

    output, lse = torch.ops.natten.na2d_forward_op(
        query,
        key,
        value,
        rpb,
        kernel_size_,
        dilation_,
        is_causal_,
        scale,
        *tiling_config_forward,
    )

    if additional_keys is not None and additional_values is not None:
        if additional_keys is None or additional_values is None:
            raise ValueError(
                "Both `additional_keys` and `additional_values` must be "
                "either Tensors or NoneTypes."
            )

        additional_output, additional_lse = additional_sdpa(
            query,
            additional_keys,
            additional_values,
            scale=scale,
            attn_kwargs=xformers_kwargs,
        )

        return merge_attentions(output, additional_output, lse, additional_lse)

    return output


def na3d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension3DTypeOrDed,
    dilation: Dimension3DTypeOrDed = 1,
    is_causal: Optional[CausalArg3DTypeOrDed] = False,
    rpb: Optional[Tensor] = None,
    scale: Optional[float] = None,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    xformers_kwargs: Optional[Dict] = None,
) -> Tensor:
    if query.is_nested or key.is_nested or value.is_nested:
        raise NotImplementedError(
            "Fused neighborhood attention does not support nested tensors yet."
        )

    kernel_size_, dilation_, is_causal_ = check_all_args(
        3, kernel_size, dilation, is_causal
    )
    # TODO: autotuner and torch compile aren't supported together
    tiling_config_forward = get_default_tiling_config_for_fna_forward(
        3, input_tensor=query, dilation=dilation_
    )

    scale = scale or query.shape[-1] ** -0.5

    output, lse = torch.ops.natten.na3d_forward_op(
        query,
        key,
        value,
        rpb,
        kernel_size_,
        dilation_,
        is_causal_,
        scale,
        *tiling_config_forward,
    )

    if additional_keys is not None and additional_values is not None:
        if additional_keys is None or additional_values is None:
            raise ValueError(
                "Both `additional_keys` and `additional_values` must be "
                "either Tensors or NoneTypes."
            )

        additional_output, additional_lse = additional_sdpa(
            query,
            additional_keys,
            additional_values,
            scale=scale,
            attn_kwargs=xformers_kwargs,
        )

        return merge_attentions(output, additional_output, lse, additional_lse)

    return output
