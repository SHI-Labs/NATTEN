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
from typing import List, Optional, Sequence

import torch
from torch import Tensor

try:
    from natten import libnatten  # type: ignore
except ImportError:
    raise ImportError(
        "Failed to import libnatten. "
        "This could be due to an invalid/incomplete install. "
        "Please make sure you built NATTEN correctly, or refer to "
        "https://shi-labs.com/natten for more information."
    )

from ...types import (
    CausalArg1DTypeOrDed,
    CausalArg2DTypeOrDed,
    CausalArg3DTypeOrDed,
    Dimension1DTypeOrDed,
    Dimension2DTypeOrDed,
    Dimension3DTypeOrDed,
)
from ...utils import (
    check_additional_keys,
    check_additional_values,
    check_all_args,
    get_num_na_weights,
    log,
    make_attn_tensor_from_input,
)
from .torch_native_ops import (
    av_cross_backward,
    av_cross_forward,
    qk_cross_backward,
    qk_cross_forward,
)

logger = log.get_logger(__name__)


#################################################################################################
##################################### Register forward ops. #####################################
#################################################################################################


@torch.library.custom_op(
    "natten::na1d_qk_forward_op", mutates_args=(), device_types=("cpu", "cuda")
)
def na1d_qk_torch_library_op(
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    additional_keys: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> Tensor:
    kernel_size, dilation, is_causal = check_all_args(
        1, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    query = query.contiguous()
    key = key.contiguous()
    n_additional_tokens = check_additional_keys(query, additional_keys)
    if bias is not None:
        bias = bias.to(key.dtype)
    attn = make_attn_tensor_from_input(query, num_na_weights + n_additional_tokens)
    attn_na, attn_add = attn.split(
        [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
    )

    if n_additional_tokens:
        assert additional_keys is not None and attn_add.numel() > 0
        qk_cross_forward(query, additional_keys, attn_add)

    libnatten.na1d_qk_forward(
        attn_na, query, key, bias, kernel_size, dilation, is_causal
    )

    return attn


@torch.library.custom_op(
    "natten::na1d_av_forward_op", mutates_args=(), device_types=("cpu", "cuda")
)
def na1d_av_torch_library_op(
    attn: Tensor,
    value: Tensor,
    additional_values: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> Tensor:
    kernel_size, dilation, is_causal = check_all_args(
        1, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)
    attn = attn.to(value.dtype)

    value = value.contiguous()
    out = torch.empty_like(value)
    out_add = None
    n_additional_tokens = check_additional_values(
        attn, additional_values, value, num_na_weights
    )
    attn_na, attn_add = attn.split(
        [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
    )

    if n_additional_tokens:
        assert additional_values is not None and attn_add.numel() > 0
        out_add = torch.empty_like(out)
        av_cross_forward(attn_add, additional_values, out_add)

    libnatten.na1d_av_forward(out, attn_na, value, kernel_size, dilation, is_causal)

    if out_add is not None:
        out += out_add

    return out


@torch.library.custom_op(
    "natten::na2d_qk_forward_op", mutates_args=(), device_types=("cpu", "cuda")
)
def na2d_qk_torch_library_op(
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    additional_keys: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> Tensor:
    kernel_size, dilation, is_causal = check_all_args(
        2, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    query = query.contiguous()
    key = key.contiguous()
    n_additional_tokens = check_additional_keys(query, additional_keys)
    if bias is not None:
        bias = bias.to(key.dtype)
    attn = make_attn_tensor_from_input(query, num_na_weights + n_additional_tokens)
    attn_na, attn_add = attn.split(
        [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
    )

    if n_additional_tokens:
        assert additional_keys is not None and attn_add.numel() > 0
        qk_cross_forward(query, additional_keys, attn_add)

    libnatten.na2d_qk_forward(
        attn_na, query, key, bias, kernel_size, dilation, is_causal
    )

    return attn


@torch.library.custom_op(
    "natten::na2d_av_forward_op", mutates_args=(), device_types=("cpu", "cuda")
)
def na2d_av_torch_library_op(
    attn: Tensor,
    value: Tensor,
    additional_values: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> Tensor:
    kernel_size, dilation, is_causal = check_all_args(
        2, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)
    attn = attn.to(value.dtype)

    value = value.contiguous()
    out = torch.empty_like(value)
    out_add = None
    n_additional_tokens = check_additional_values(
        attn, additional_values, value, num_na_weights
    )
    attn_na, attn_add = attn.split(
        [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
    )

    if n_additional_tokens:
        assert additional_values is not None and attn_add.numel() > 0
        out_add = torch.empty_like(out)
        av_cross_forward(attn_add, additional_values, out_add)

    libnatten.na2d_av_forward(out, attn_na, value, kernel_size, dilation, is_causal)

    if out_add is not None:
        out += out_add

    return out


@torch.library.custom_op(
    "natten::na3d_qk_forward_op", mutates_args=(), device_types=("cpu", "cuda")
)
def na3d_qk_torch_library_op(
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    additional_keys: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> Tensor:
    kernel_size, dilation, is_causal = check_all_args(
        3, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    query = query.contiguous()
    key = key.contiguous()
    n_additional_tokens = check_additional_keys(query, additional_keys)
    if bias is not None:
        bias = bias.to(key.dtype)
    attn = make_attn_tensor_from_input(query, num_na_weights + n_additional_tokens)
    attn_na, attn_add = attn.split(
        [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
    )

    if n_additional_tokens:
        assert additional_keys is not None and attn_add.numel() > 0
        qk_cross_forward(query, additional_keys, attn_add)

    libnatten.na3d_qk_forward(
        attn_na, query, key, bias, kernel_size, dilation, is_causal
    )

    return attn


@torch.library.custom_op(
    "natten::na3d_av_forward_op", mutates_args=(), device_types=("cpu", "cuda")
)
def na3d_av_torch_library_op(
    attn: Tensor,
    value: Tensor,
    additional_values: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> Tensor:
    kernel_size, dilation, is_causal = check_all_args(
        3, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)
    attn = attn.to(value.dtype)

    value = value.contiguous()
    out = torch.empty_like(value)
    out_add = None
    n_additional_tokens = check_additional_values(
        attn, additional_values, value, num_na_weights
    )
    attn_na, attn_add = attn.split(
        [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
    )

    if n_additional_tokens:
        assert additional_values is not None and attn_add.numel() > 0
        out_add = torch.empty_like(out)
        av_cross_forward(attn_add, additional_values, out_add)

    libnatten.na3d_av_forward(out, attn_na, value, kernel_size, dilation, is_causal)

    if out_add is not None:
        out += out_add

    return out


#################################################################################################
##################################### Register backward ops. ####################################
#################################################################################################


@torch.library.custom_op(
    "natten::na1d_qk_backward_op",
    mutates_args=(),
    device_types=("cpu", "cuda"),
)
def na1d_qk_backward_op(
    d_attn: Tensor,
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    additional_keys: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> List[Tensor]:
    kernel_size, dilation, is_causal = check_all_args(
        1, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)

    d_query = torch.empty_like(query)
    d_key = torch.empty_like(key)
    # d_bias has to be zero filled
    d_bias = None if bias is None else torch.zeros_like(bias)
    d_additional_keys = None
    n_additional_tokens = check_additional_keys(query, additional_keys)
    d_query_add_key = None
    d_attn_na, d_attn_add = d_attn.split(
        [num_na_weights, d_attn.shape[-1] - num_na_weights], dim=-1
    )

    if n_additional_tokens:
        assert additional_keys is not None and d_attn_add.numel() > 0
        d_query_add_key, d_additional_keys = torch.empty_like(
            d_query
        ), torch.empty_like(additional_keys)
        qk_cross_backward(
            query, d_attn_add, additional_keys, d_query_add_key, d_additional_keys
        )

    if d_bias is not None and torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            "You enabled PyTorch's deterministic mode, but training neighborhood attention "
            "with bias is only implemented with a non-deterministic kernel. "
            "Please consider either disabling attention bias, or torch's deterministic mode."
        )

    libnatten.na1d_qk_backward(
        d_query,
        d_key,
        d_bias,
        d_attn_na,
        query,
        key,
        kernel_size,
        dilation,
        is_causal,
    )

    if d_query_add_key is not None:
        d_query += d_query_add_key

    outputs = [d_query, d_key]

    if d_bias is not None:
        outputs.append(d_bias)

    if d_additional_keys is not None:
        outputs.append(d_additional_keys)

    return outputs


@torch.library.custom_op(
    "natten::na1d_av_backward_op",
    mutates_args=(),
    device_types=("cpu", "cuda"),
)
def na1d_av_backward_op(
    d_out: Tensor,
    attn: Tensor,
    value: Tensor,
    additional_values: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> List[Tensor]:
    kernel_size, dilation, is_causal = check_all_args(
        1, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)

    d_attn = torch.empty_like(attn)
    d_value = torch.empty_like(value)
    d_additional_values = None
    n_additional_tokens = check_additional_values(
        attn, additional_values, value, num_na_weights
    )
    d_attn_na, d_attn_add = d_attn.split(
        [num_na_weights, d_attn.shape[-1] - num_na_weights], dim=-1
    )
    attn_na, attn_add = attn.split(
        [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
    )

    if n_additional_tokens:
        assert additional_values is not None
        assert d_attn_add.numel() > 0 and attn_add.numel() > 0
        d_additional_values = torch.empty_like(additional_values)
        av_cross_backward(
            d_out, additional_values, attn_add, d_attn_add, d_additional_values
        )

    libnatten.na1d_av_backward(
        d_attn_na,
        d_value,
        d_out,
        attn_na,
        value,
        kernel_size,
        dilation,
        is_causal,
    )

    outputs = [d_attn, d_value]

    if d_additional_values is not None:
        outputs.append(d_additional_values)

    return outputs


@torch.library.custom_op(
    "natten::na2d_qk_backward_op",
    mutates_args=(),
    device_types=("cpu", "cuda"),
)
def na2d_qk_backward_op(
    d_attn: Tensor,
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    additional_keys: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> List[Tensor]:
    kernel_size, dilation, is_causal = check_all_args(
        2, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)

    d_query = torch.empty_like(query)
    d_key = torch.empty_like(key)
    # d_bias has to be zero filled
    d_bias = None if bias is None else torch.zeros_like(bias)
    d_additional_keys = None
    n_additional_tokens = check_additional_keys(query, additional_keys)
    d_query_add_key = None
    d_attn_na, d_attn_add = d_attn.split(
        [num_na_weights, d_attn.shape[-1] - num_na_weights], dim=-1
    )

    if n_additional_tokens:
        assert additional_keys is not None and d_attn_add.numel() > 0
        d_query_add_key, d_additional_keys = torch.empty_like(
            d_query
        ), torch.empty_like(additional_keys)
        qk_cross_backward(
            query, d_attn_add, additional_keys, d_query_add_key, d_additional_keys
        )

    if d_bias is not None and torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            "You enabled PyTorch's deterministic mode, but training neighborhood attention "
            "with bias is only implemented with a non-deterministic kernel. "
            "Please consider either disabling attention bias, or torch's deterministic mode."
        )

    libnatten.na2d_qk_backward(
        d_query,
        d_key,
        d_bias,
        d_attn_na,
        query,
        key,
        kernel_size,
        dilation,
        is_causal,
    )

    if d_query_add_key is not None:
        d_query += d_query_add_key

    outputs = [d_query, d_key]

    if d_bias is not None:
        outputs.append(d_bias)

    if d_additional_keys is not None:
        outputs.append(d_additional_keys)

    return outputs


@torch.library.custom_op(
    "natten::na2d_av_backward_op",
    mutates_args=(),
    device_types=("cpu", "cuda"),
)
def na2d_av_backward_op(
    d_out: Tensor,
    attn: Tensor,
    value: Tensor,
    additional_values: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> List[Tensor]:
    kernel_size, dilation, is_causal = check_all_args(
        2, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)

    d_attn = torch.empty_like(attn)
    d_value = torch.empty_like(value)
    d_additional_values = None
    n_additional_tokens = check_additional_values(
        attn, additional_values, value, num_na_weights
    )
    d_attn_na, d_attn_add = d_attn.split(
        [num_na_weights, d_attn.shape[-1] - num_na_weights], dim=-1
    )
    attn_na, attn_add = attn.split(
        [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
    )

    if n_additional_tokens:
        assert additional_values is not None
        assert d_attn_add.numel() > 0 and attn_add.numel() > 0
        d_additional_values = torch.empty_like(additional_values)
        av_cross_backward(
            d_out, additional_values, attn_add, d_attn_add, d_additional_values
        )

    libnatten.na2d_av_backward(
        d_attn_na,
        d_value,
        d_out,
        attn_na,
        value,
        kernel_size,
        dilation,
        is_causal,
    )

    outputs = [d_attn, d_value]

    if d_additional_values is not None:
        outputs.append(d_additional_values)

    return outputs


@torch.library.custom_op(
    "natten::na3d_qk_backward_op",
    mutates_args=(),
    device_types=("cpu", "cuda"),
)
def na3d_qk_backward_op(
    d_attn: Tensor,
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    additional_keys: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> List[Tensor]:
    kernel_size, dilation, is_causal = check_all_args(
        3, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)

    d_query = torch.empty_like(query)
    d_key = torch.empty_like(key)
    # d_bias has to be zero filled
    d_bias = None if bias is None else torch.zeros_like(bias)
    d_additional_keys = None
    n_additional_tokens = check_additional_keys(query, additional_keys)
    d_query_add_key = None
    d_attn_na, d_attn_add = d_attn.split(
        [num_na_weights, d_attn.shape[-1] - num_na_weights], dim=-1
    )

    if n_additional_tokens:
        assert additional_keys is not None and d_attn_add.numel() > 0
        d_query_add_key, d_additional_keys = torch.empty_like(
            d_query
        ), torch.empty_like(additional_keys)
        qk_cross_backward(
            query, d_attn_add, additional_keys, d_query_add_key, d_additional_keys
        )

    if d_bias is not None and torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            "You enabled PyTorch's deterministic mode, but training neighborhood attention "
            "with bias is only implemented with a non-deterministic kernel. "
            "Please consider either disabling attention bias, or torch's deterministic mode."
        )

    libnatten.na3d_qk_backward(
        d_query,
        d_key,
        d_bias,
        d_attn_na,
        query,
        key,
        kernel_size,
        dilation,
        is_causal,
    )

    if d_query_add_key is not None:
        d_query += d_query_add_key

    outputs = [d_query, d_key]

    if d_bias is not None:
        outputs.append(d_bias)

    if d_additional_keys is not None:
        outputs.append(d_additional_keys)

    return outputs


@torch.library.custom_op(
    "natten::na3d_av_backward_op",
    mutates_args=(),
    device_types=("cpu", "cuda"),
)
def na3d_av_backward_op(
    d_out: Tensor,
    attn: Tensor,
    value: Tensor,
    additional_values: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> List[Tensor]:
    kernel_size, dilation, is_causal = check_all_args(
        3, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)

    d_attn = torch.empty_like(attn)
    d_value = torch.empty_like(value)
    d_additional_values = None
    n_additional_tokens = check_additional_values(
        attn, additional_values, value, num_na_weights
    )
    d_attn_na, d_attn_add = d_attn.split(
        [num_na_weights, d_attn.shape[-1] - num_na_weights], dim=-1
    )
    attn_na, attn_add = attn.split(
        [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
    )

    if n_additional_tokens:
        assert additional_values is not None
        assert d_attn_add.numel() > 0 and attn_add.numel() > 0
        d_additional_values = torch.empty_like(additional_values)
        av_cross_backward(
            d_out, additional_values, attn_add, d_attn_add, d_additional_values
        )

    libnatten.na3d_av_backward(
        d_attn_na,
        d_value,
        d_out,
        attn_na,
        value,
        kernel_size,
        dilation,
        is_causal,
    )

    outputs = [d_attn, d_value]

    if d_additional_values is not None:
        outputs.append(d_additional_values)

    return outputs


#################################################################################################
################## Register "fake" ops for forward op, since it's not inplace. ##################
#################################################################################################


@na1d_qk_torch_library_op.register_fake
def na1d_qk_torch_library_op_fake(
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    additional_keys: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> Tensor:
    kernel_size, dilation, is_causal = check_all_args(
        1, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    n_additional_tokens = check_additional_keys(query, additional_keys)
    attn = make_attn_tensor_from_input(query, num_na_weights + n_additional_tokens)

    return attn


@na1d_av_torch_library_op.register_fake
def na1d_av_torch_library_op_fake(
    attn: Tensor,
    value: Tensor,
    additional_values: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> Tensor:
    out = torch.empty_like(value)
    return out


@na1d_qk_backward_op.register_fake
def na1d_qk_backward_op_fake(
    d_attn: Tensor,
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    additional_keys: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> List[Tensor]:
    d_query = torch.empty_like(query)
    d_key = torch.empty_like(key)
    # d_bias has to be zero filled
    d_bias = None if bias is None else torch.zeros_like(bias)
    d_additional_keys = None
    n_additional_tokens = check_additional_keys(query, additional_keys)

    if n_additional_tokens:
        assert additional_keys is not None
        d_additional_keys = torch.empty_like(additional_keys)

    if d_bias is not None and torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            "You enabled PyTorch's deterministic mode, but training neighborhood attention "
            "with bias is only implemented with a non-deterministic kernel. "
            "Please consider either disabling attention bias, or torch's deterministic mode."
        )

    outputs = [d_query, d_key]

    if d_bias is not None:
        outputs.append(d_bias)

    if d_additional_keys is not None:
        outputs.append(d_additional_keys)

    return outputs


@na1d_av_backward_op.register_fake
def na1d_av_backward_op_fake(
    d_out: Tensor,
    attn: Tensor,
    value: Tensor,
    additional_values: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> List[Tensor]:
    kernel_size, dilation, is_causal = check_all_args(
        2, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)

    d_attn = torch.empty_like(attn)
    d_value = torch.empty_like(value)
    d_additional_values = None
    n_additional_tokens = check_additional_values(
        attn, additional_values, value, num_na_weights
    )

    if n_additional_tokens:
        assert additional_values is not None
        d_additional_values = torch.empty_like(additional_values)

    outputs = [d_attn, d_value]

    if d_additional_values is not None:
        outputs.append(d_additional_values)

    return outputs


@na2d_qk_torch_library_op.register_fake
def na2d_qk_torch_library_op_fake(
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    additional_keys: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> Tensor:
    kernel_size, dilation, is_causal = check_all_args(
        2, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    n_additional_tokens = check_additional_keys(query, additional_keys)
    attn = make_attn_tensor_from_input(query, num_na_weights + n_additional_tokens)

    return attn


@na2d_av_torch_library_op.register_fake
def na2d_av_torch_library_op_fake(
    attn: Tensor,
    value: Tensor,
    additional_values: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> Tensor:

    out = torch.empty_like(value)
    return out


@na2d_qk_backward_op.register_fake
def na2d_qk_backward_op_fake(
    d_attn: Tensor,
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    additional_keys: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> List[Tensor]:

    d_query = torch.empty_like(query)
    d_key = torch.empty_like(key)
    # d_bias has to be zero filled
    d_bias = None if bias is None else torch.zeros_like(bias)
    d_additional_keys = None
    n_additional_tokens = check_additional_keys(query, additional_keys)

    if n_additional_tokens:
        assert additional_keys is not None
        d_additional_keys = torch.empty_like(additional_keys)

    if d_bias is not None and torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            "You enabled PyTorch's deterministic mode, but training neighborhood attention "
            "with bias is only implemented with a non-deterministic kernel. "
            "Please consider either disabling attention bias, or torch's deterministic mode."
        )

    outputs = [d_query, d_key]

    if d_bias is not None:
        outputs.append(d_bias)

    if d_additional_keys is not None:
        outputs.append(d_additional_keys)

    return outputs


@na2d_av_backward_op.register_fake
def na2d_av_backward_op_fake(
    d_out: Tensor,
    attn: Tensor,
    value: Tensor,
    additional_values: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> List[Tensor]:
    kernel_size, dilation, is_causal = check_all_args(
        2, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)

    d_attn = torch.empty_like(attn)
    d_value = torch.empty_like(value)
    d_additional_values = None
    n_additional_tokens = check_additional_values(
        attn, additional_values, value, num_na_weights
    )

    if n_additional_tokens:
        assert additional_values is not None
        d_additional_values = torch.empty_like(additional_values)

    outputs = [d_attn, d_value]

    if d_additional_values is not None:
        outputs.append(d_additional_values)

    return outputs


@na3d_qk_torch_library_op.register_fake
def na3d_qk_torch_library_op_fake(
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    additional_keys: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> Tensor:
    kernel_size, dilation, is_causal = check_all_args(
        3, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    n_additional_tokens = check_additional_keys(query, additional_keys)
    attn = make_attn_tensor_from_input(query, num_na_weights + n_additional_tokens)

    return attn


@na3d_av_torch_library_op.register_fake
def na3d_av_torch_library_op_fake(
    attn: Tensor,
    value: Tensor,
    additional_values: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> Tensor:
    out = torch.empty_like(value)
    return out


@na3d_qk_backward_op.register_fake
def na3d_qk_backward_op_fake(
    d_attn: Tensor,
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    additional_keys: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> List[Tensor]:

    d_query = torch.empty_like(query)
    d_key = torch.empty_like(key)
    # d_bias has to be zero filled
    d_bias = None if bias is None else torch.zeros_like(bias)
    d_additional_keys = None
    n_additional_tokens = check_additional_keys(query, additional_keys)

    if n_additional_tokens:
        assert additional_keys is not None
        d_additional_keys = torch.empty_like(additional_keys)

    if d_bias is not None and torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            "You enabled PyTorch's deterministic mode, but training neighborhood attention "
            "with bias is only implemented with a non-deterministic kernel. "
            "Please consider either disabling attention bias, or torch's deterministic mode."
        )

    outputs = [d_query, d_key]

    if d_bias is not None:
        outputs.append(d_bias)

    if d_additional_keys is not None:
        outputs.append(d_additional_keys)

    return outputs


@na3d_av_backward_op.register_fake
def na3d_av_backward_op_fake(
    d_out: Tensor,
    attn: Tensor,
    value: Tensor,
    additional_values: Optional[Tensor],
    kernel_size_: Sequence[int],
    dilation_: Sequence[int],
    is_causal_: Sequence[bool],
) -> List[Tensor]:
    kernel_size, dilation, is_causal = check_all_args(
        3, kernel_size_, dilation_, is_causal_
    )
    num_na_weights = get_num_na_weights(kernel_size)

    d_attn = torch.empty_like(attn)
    d_value = torch.empty_like(value)
    d_additional_values = None
    n_additional_tokens = check_additional_values(
        attn, additional_values, value, num_na_weights
    )

    if n_additional_tokens:
        assert additional_values is not None
        d_additional_values = torch.empty_like(additional_values)

    outputs = [d_attn, d_value]

    if d_additional_values is not None:
        outputs.append(d_additional_values)

    return outputs


#################################################################################################
#################################### Autograd context setups ####################################
#################################################################################################


def na1d_qk_setup_context(ctx, inputs, output):
    (
        query,
        key,
        bias,
        additional_keys,
        kernel_size_,
        dilation_,
        is_causal_,
    ) = inputs

    kernel_size, dilation, is_causal = check_all_args(
        1, kernel_size_, dilation_, is_causal_
    )

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    ctx.save_for_backward(query, key, bias, additional_keys)
    ctx.kernel_size = kernel_size
    ctx.dilation = dilation
    ctx.is_causal = is_causal


def na1d_av_setup_context(ctx, inputs, output):
    (
        attn,
        value,
        additional_values,
        kernel_size_,
        dilation_,
        is_causal_,
    ) = inputs

    kernel_size, dilation, is_causal = check_all_args(
        1, kernel_size_, dilation_, is_causal_
    )

    ctx.save_for_backward(attn, value, additional_values)
    ctx.kernel_size = kernel_size
    ctx.dilation = dilation
    ctx.is_causal = is_causal


def na2d_qk_setup_context(ctx, inputs, output):
    (
        query,
        key,
        bias,
        additional_keys,
        kernel_size_,
        dilation_,
        is_causal_,
    ) = inputs

    kernel_size, dilation, is_causal = check_all_args(
        2, kernel_size_, dilation_, is_causal_
    )

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    ctx.save_for_backward(query, key, bias, additional_keys)
    ctx.kernel_size = kernel_size
    ctx.dilation = dilation
    ctx.is_causal = is_causal


def na2d_av_setup_context(ctx, inputs, output):
    (
        attn,
        value,
        additional_values,
        kernel_size_,
        dilation_,
        is_causal_,
    ) = inputs

    kernel_size, dilation, is_causal = check_all_args(
        2, kernel_size_, dilation_, is_causal_
    )

    ctx.save_for_backward(attn, value, additional_values)
    ctx.kernel_size = kernel_size
    ctx.dilation = dilation
    ctx.is_causal = is_causal


def na3d_qk_setup_context(ctx, inputs, output):
    (
        query,
        key,
        bias,
        additional_keys,
        kernel_size_,
        dilation_,
        is_causal_,
    ) = inputs

    kernel_size, dilation, is_causal = check_all_args(
        3, kernel_size_, dilation_, is_causal_
    )

    if any(is_causal) and bias is not None:
        raise NotImplementedError(
            "Positional biases for causal neighborhood attention is not yet implemented."
        )

    ctx.save_for_backward(query, key, bias, additional_keys)
    ctx.kernel_size = kernel_size
    ctx.dilation = dilation
    ctx.is_causal = is_causal


def na3d_av_setup_context(ctx, inputs, output):
    (
        attn,
        value,
        additional_values,
        kernel_size_,
        dilation_,
        is_causal_,
    ) = inputs

    kernel_size, dilation, is_causal = check_all_args(
        3, kernel_size_, dilation_, is_causal_
    )

    ctx.save_for_backward(attn, value, additional_values)
    ctx.kernel_size = kernel_size
    ctx.dilation = dilation
    ctx.is_causal = is_causal


#################################################################################################
################################## Setup autograd backward call #################################
#################################################################################################


def na1d_qk_backward(ctx, grad_out):
    query, key, bias, additional_keys = ctx.saved_tensors
    d_attn = grad_out.contiguous()

    outputs = na1d_qk_backward_op(
        d_attn,
        query,
        key,
        bias,
        additional_keys,
        ctx.kernel_size,
        ctx.dilation,
        ctx.is_causal,
    )
    assert len(outputs) >= 2
    d_query, d_key = outputs[0], outputs[1]
    d_bias = None
    d_additional_keys = None

    if bias is not None and additional_keys is None:
        assert len(outputs) == 3
        d_bias = outputs[2]

    elif bias is None and additional_keys is not None:
        assert len(outputs) == 3
        d_additional_keys = outputs[2]

    elif bias is not None and additional_keys is not None:
        assert len(outputs) == 4
        d_bias = outputs[2]
        d_additional_keys = outputs[3]

    return d_query, d_key, d_bias, d_additional_keys, None, None, None


def na1d_av_backward(ctx, grad_out):
    attn, value, additional_values = ctx.saved_tensors
    d_output = grad_out.contiguous()

    outputs = na1d_av_backward_op(
        d_output,
        attn,
        value,
        additional_values,
        ctx.kernel_size,
        ctx.dilation,
        ctx.is_causal,
    )
    assert len(outputs) >= 2
    d_attn, d_value = outputs[0], outputs[1]
    d_additional_values = None

    if additional_values is not None:
        assert len(outputs) == 3
        d_additional_values = outputs[2]

    return d_attn, d_value, d_additional_values, None, None, None


def na2d_qk_backward(ctx, grad_out):
    query, key, bias, additional_keys = ctx.saved_tensors
    d_attn = grad_out.contiguous()

    outputs = na2d_qk_backward_op(
        d_attn,
        query,
        key,
        bias,
        additional_keys,
        ctx.kernel_size,
        ctx.dilation,
        ctx.is_causal,
    )
    assert len(outputs) >= 2
    d_query, d_key = outputs[0], outputs[1]
    d_bias = None
    d_additional_keys = None

    if bias is not None and additional_keys is None:
        assert len(outputs) == 3
        d_bias = outputs[2]

    elif bias is None and additional_keys is not None:
        assert len(outputs) == 3
        d_additional_keys = outputs[2]

    elif bias is not None and additional_keys is not None:
        assert len(outputs) == 4
        d_bias = outputs[2]
        d_additional_keys = outputs[3]

    return d_query, d_key, d_bias, d_additional_keys, None, None, None


def na2d_av_backward(ctx, grad_out):
    attn, value, additional_values = ctx.saved_tensors
    d_output = grad_out.contiguous()

    outputs = na2d_av_backward_op(
        d_output,
        attn,
        value,
        additional_values,
        ctx.kernel_size,
        ctx.dilation,
        ctx.is_causal,
    )
    assert len(outputs) >= 2
    d_attn, d_value = outputs[0], outputs[1]
    d_additional_values = None

    if additional_values is not None:
        assert len(outputs) == 3
        d_additional_values = outputs[2]

    return d_attn, d_value, d_additional_values, None, None, None


def na3d_qk_backward(ctx, grad_out):
    query, key, bias, additional_keys = ctx.saved_tensors
    d_attn = grad_out.contiguous()

    outputs = na3d_qk_backward_op(
        d_attn,
        query,
        key,
        bias,
        additional_keys,
        ctx.kernel_size,
        ctx.dilation,
        ctx.is_causal,
    )
    assert len(outputs) >= 2
    d_query, d_key = outputs[0], outputs[1]
    d_bias = None
    d_additional_keys = None

    if bias is not None and additional_keys is None:
        assert len(outputs) == 3
        d_bias = outputs[2]

    elif bias is None and additional_keys is not None:
        assert len(outputs) == 3
        d_additional_keys = outputs[2]

    elif bias is not None and additional_keys is not None:
        assert len(outputs) == 4
        d_bias = outputs[2]
        d_additional_keys = outputs[3]

    return d_query, d_key, d_bias, d_additional_keys, None, None, None


def na3d_av_backward(ctx, grad_out):
    attn, value, additional_values = ctx.saved_tensors
    d_output = grad_out.contiguous()

    outputs = na3d_av_backward_op(
        d_output,
        attn,
        value,
        additional_values,
        ctx.kernel_size,
        ctx.dilation,
        ctx.is_causal,
    )
    assert len(outputs) >= 2
    d_attn, d_value = outputs[0], outputs[1]
    d_additional_values = None

    if additional_values is not None:
        assert len(outputs) == 3
        d_additional_values = outputs[2]

    return d_attn, d_value, d_additional_values, None, None, None


#################################################################################################
############################## Register ops as autograd functions ###############################
#################################################################################################

na1d_qk_torch_library_op.register_autograd(
    na1d_qk_backward, setup_context=na1d_qk_setup_context
)
na1d_av_torch_library_op.register_autograd(
    na1d_av_backward, setup_context=na1d_av_setup_context
)

na2d_qk_torch_library_op.register_autograd(
    na2d_qk_backward, setup_context=na2d_qk_setup_context
)
na2d_av_torch_library_op.register_autograd(
    na2d_av_backward, setup_context=na2d_av_setup_context
)

na3d_qk_torch_library_op.register_autograd(
    na3d_qk_backward, setup_context=na3d_qk_setup_context
)
na3d_av_torch_library_op.register_autograd(
    na3d_av_backward, setup_context=na3d_av_setup_context
)


#################################################################################################
######################################## User-facing APIs #######################################
#################################################################################################


def na1d_qk_op(
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    additional_keys: Optional[Tensor],
    kernel_size: Dimension1DTypeOrDed,
    dilation: Dimension1DTypeOrDed,
    is_causal: Optional[CausalArg1DTypeOrDed] = False,
):
    query = query.contiguous()
    key = key.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    kernel_size_, dilation_, is_causal_ = check_all_args(
        1, kernel_size, dilation, is_causal
    )

    return na1d_qk_torch_library_op(
        query,
        key,
        bias,
        additional_keys,
        kernel_size_,
        dilation_,
        is_causal_,
    )


def na1d_av_op(
    attn: Tensor,
    value: Tensor,
    additional_values: Optional[Tensor],
    kernel_size: Dimension1DTypeOrDed,
    dilation: Dimension1DTypeOrDed,
    is_causal: Optional[CausalArg1DTypeOrDed] = False,
):
    value = value.contiguous()
    kernel_size_, dilation_, is_causal_ = check_all_args(
        1, kernel_size, dilation, is_causal
    )

    return na1d_av_torch_library_op(
        attn,
        value,
        additional_values,
        kernel_size_,
        dilation_,
        is_causal_,
    )


def na2d_qk_op(
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    additional_keys: Optional[Tensor],
    kernel_size: Dimension2DTypeOrDed,
    dilation: Dimension2DTypeOrDed,
    is_causal: Optional[CausalArg2DTypeOrDed] = False,
):
    query = query.contiguous()
    key = key.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    kernel_size_, dilation_, is_causal_ = check_all_args(
        2, kernel_size, dilation, is_causal
    )

    return na2d_qk_torch_library_op(
        query,
        key,
        bias,
        additional_keys,
        kernel_size_,
        dilation_,
        is_causal_,
    )


def na2d_av_op(
    attn: Tensor,
    value: Tensor,
    additional_values: Optional[Tensor],
    kernel_size: Dimension2DTypeOrDed,
    dilation: Dimension2DTypeOrDed,
    is_causal: Optional[CausalArg2DTypeOrDed] = False,
):
    value = value.contiguous()
    kernel_size_, dilation_, is_causal_ = check_all_args(
        2, kernel_size, dilation, is_causal
    )

    return na2d_av_torch_library_op(
        attn,
        value,
        additional_values,
        kernel_size_,
        dilation_,
        is_causal_,
    )


def na3d_qk_op(
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    additional_keys: Optional[Tensor],
    kernel_size: Dimension3DTypeOrDed,
    dilation: Dimension3DTypeOrDed,
    is_causal: Optional[CausalArg3DTypeOrDed] = False,
):
    query = query.contiguous()
    key = key.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    kernel_size_, dilation_, is_causal_ = check_all_args(
        3, kernel_size, dilation, is_causal
    )

    return na3d_qk_torch_library_op(
        query,
        key,
        bias,
        additional_keys,
        kernel_size_,
        dilation_,
        is_causal_,
    )


def na3d_av_op(
    attn: Tensor,
    value: Tensor,
    additional_values: Optional[Tensor],
    kernel_size: Dimension3DTypeOrDed,
    dilation: Dimension3DTypeOrDed,
    is_causal: Optional[CausalArg3DTypeOrDed] = False,
):
    value = value.contiguous()
    kernel_size_, dilation_, is_causal_ = check_all_args(
        3, kernel_size, dilation, is_causal
    )

    return na3d_av_torch_library_op(
        attn,
        value,
        additional_values,
        kernel_size_,
        dilation_,
        is_causal_,
    )
