#################################################################################################
# Copyright (c) 2023 Ali Hassani.
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
from typing import Optional

import torch
from torch import Tensor

try:
    from natten import libnatten  # type: ignore
except ImportError:
    raise ImportError(
        "Failed to import NATTEN's CPP backend. "
        "This could be due to an invalid/incomplete install. "
        "Please uninstall NATTEN (pip uninstall natten) and re-install with the"
        " correct torch build: shi-labs.com/natten ."
    )

from .utils import make_attn_tensor_from_input


def na1d_qk_nested(
    query: Tensor, key: Tensor, rpb: Optional[Tensor], kernel_size: int, dilation: int
) -> Tensor:
    if not query.is_nested or not key.is_nested:
        raise ValueError("Expected all inputs to be nested.")
    if not query.is_leaf or not key.is_leaf:
        raise ValueError("Only one level of nested tensors is supported at the moment.")
    if query.requires_grad or key.requires_grad:
        raise ValueError("Autograd is not supported for nested tensors.")
    if rpb is not None and rpb.is_nested:
        raise ValueError("Positional biases cannot be nested.")

    if rpb is not None:
        rpb = rpb.contiguous().to(key.dtype)

    if query.size(0) != key.size(0):
        raise ValueError("Got nested inputs, but they don't match in size.")

    attn = torch.nested.nested_tensor(
        [make_attn_tensor_from_input(q, kernel_size) for q in query]
    )
    for q, k, a in zip(query, key, attn):
        libnatten.na1d_qk_forward(a, q, k, rpb, kernel_size, dilation)

    return attn


def na1d_av_nested(attn: Tensor, value: Tensor, kernel_size: int, dilation: int):
    if not attn.is_nested or not value.is_nested:
        raise ValueError("Expected all inputs to be nested.")
    if not attn.is_leaf or not value.is_leaf:
        raise ValueError("Only one level of nested tensors is supported at the moment.")
    if attn.requires_grad or value.requires_grad:
        raise ValueError("Autograd is not supported for nested tensors.")

    if attn.size(0) != value.size(0):
        raise ValueError("Got nested inputs, but they don't match in size.")

    out = torch.empty_like(value)
    for a, v, o in zip(attn, value, out):
        libnatten.na1d_av_forward(o, a, v, kernel_size, dilation)

    return out


def na2d_qk_nested(
    query: Tensor, key: Tensor, rpb: Optional[Tensor], kernel_size: int, dilation: int
) -> Tensor:
    if not query.is_nested or not key.is_nested:
        raise ValueError("Expected all inputs to be nested.")
    if not query.is_leaf or not key.is_leaf:
        raise ValueError("Only one level of nested tensors is supported at the moment.")
    if query.requires_grad or key.requires_grad:
        raise ValueError("Autograd is not supported for nested tensors.")
    if rpb is not None and rpb.is_nested:
        raise ValueError("Positional biases cannot be nested.")

    if rpb is not None:
        rpb = rpb.contiguous().to(key.dtype)

    if query.size(0) != key.size(0):
        raise ValueError("Got nested inputs, but they don't match in size.")

    attn = torch.nested.nested_tensor(
        [make_attn_tensor_from_input(q, kernel_size**2) for q in query]
    )
    for q, k, a in zip(query, key, attn):
        libnatten.na2d_qk_forward(a, q, k, rpb, kernel_size, dilation)

    return attn


def na2d_av_nested(attn: Tensor, value: Tensor, kernel_size: int, dilation: int):
    if not attn.is_nested or not value.is_nested:
        raise ValueError("Expected all inputs to be nested.")
    if not attn.is_leaf or not value.is_leaf:
        raise ValueError("Only one level of nested tensors is supported at the moment.")
    if attn.requires_grad or value.requires_grad:
        raise ValueError("Autograd is not supported for nested tensors.")

    if attn.size(0) != value.size(0):
        raise ValueError("Got nested inputs, but they don't match in size.")

    out = torch.empty_like(value)
    for a, v, o in zip(attn, value, out):
        libnatten.na2d_av_forward(o, a, v, kernel_size, dilation)

    return out


def na3d_qk_nested(
    query: Tensor,
    key: Tensor,
    rpb: Optional[Tensor],
    kernel_size_d: int,
    kernel_size: int,
    dilation_d: int,
    dilation: int,
) -> Tensor:
    if not query.is_nested or not key.is_nested:
        raise ValueError("Expected all inputs to be nested.")
    if not query.is_leaf or not key.is_leaf:
        raise ValueError("Only one level of nested tensors is supported at the moment.")
    if query.requires_grad or key.requires_grad:
        raise ValueError("Autograd is not supported for nested tensors.")
    if rpb is not None and rpb.is_nested:
        raise ValueError("Positional biases cannot be nested.")

    if rpb is not None:
        rpb = rpb.contiguous().to(key.dtype)

    if query.size(0) != key.size(0):
        raise ValueError("Got nested inputs, but they don't match in size.")

    attn = torch.nested.nested_tensor(
        [
            make_attn_tensor_from_input(q, kernel_size * kernel_size * kernel_size_d)
            for q in query
        ]
    )
    for q, k, a in zip(query, key, attn):
        libnatten.na3d_qk_forward(
            a, q, k, rpb, kernel_size, dilation, kernel_size_d, dilation_d
        )

    return attn


def na3d_av_nested(
    attn: Tensor,
    value: Tensor,
    kernel_size_d: int,
    kernel_size: int,
    dilation_d: int,
    dilation: int,
):
    if not attn.is_nested or not value.is_nested:
        raise ValueError("Expected all inputs to be nested.")
    if not attn.is_leaf or not value.is_leaf:
        raise ValueError("Only one level of nested tensors is supported at the moment.")
    if attn.requires_grad or value.requires_grad:
        raise ValueError("Autograd is not supported for nested tensors.")

    if attn.size(0) != value.size(0):
        raise ValueError("Got nested inputs, but they don't match in size.")

    out = torch.empty_like(value)
    for a, v, o in zip(attn, value, out):
        libnatten.na3d_av_forward(
            o, a, v, kernel_size, dilation, kernel_size_d, dilation_d
        )

    return out
