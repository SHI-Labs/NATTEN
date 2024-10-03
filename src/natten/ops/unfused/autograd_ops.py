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
from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch.amp import custom_bwd, custom_fwd
from torch.autograd import Function

amp_fwd = functools.partial(custom_fwd, device_type="cuda")
amp_bwd = functools.partial(custom_bwd, device_type="cuda")

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
    NoneType,
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
###################### Set up libnatten ops directly as autograd functions ######################
#################################################################################################


class NeighborhoodAttention1DQKAutogradFunction(Function):
    @staticmethod
    @amp_fwd
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        bias: Optional[Tensor],
        additional_keys: Optional[Tensor],
        kernel_size_: Dimension1DTypeOrDed,
        dilation_: Dimension1DTypeOrDed,
        is_causal_: CausalArg1DTypeOrDed,
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
        ctx.save_for_backward(query, key, bias, additional_keys)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.is_causal = is_causal

        return attn

    @staticmethod
    def jvp(ctx, *grad_inputs: Any) -> Tensor:
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the QK operation is:
        qk(query.tangent, key.primal) + qk(query.primal, key.tangent)
        """
        if any(ctx.is_causal):
            raise ValueError(
                "Causal neighborhood attention doesn't support forward mode "
                "auto-diff yet."
            )
        num_na_weights = get_num_na_weights(ctx.kernel_size)

        assert len(grad_inputs) == 7
        query_t: Tensor = grad_inputs[0]
        key_t: Tensor = grad_inputs[1]
        bias: Optional[Tensor] = grad_inputs[2]
        additional_key_t: Optional[Tensor] = grad_inputs[3]
        n_additional_tokens = check_additional_keys(query_t, additional_key_t)

        if bias is not None:
            raise ValueError(
                "Positional biases are currently not supported "
                "in forward mode autodiff."
            )

        query_p, key_p, _, additional_key_p = ctx.to_save

        if (additional_key_t is not None and additional_key_p is None) or (
            additional_key_t is None and additional_key_p is not None
        ):
            raise ValueError(
                "Expected either both additional_key_t and additional_key_p, or neither."
            )

        query_t = query_t.contiguous()
        key_t = key_t.contiguous()
        attn_0 = make_attn_tensor_from_input(
            query_t, num_na_weights + n_additional_tokens
        )
        attn_1 = torch.empty_like(attn_0)
        attn_na_0, attn_add_0 = attn_0.split(
            [num_na_weights, attn_0.shape[-1] - num_na_weights], dim=-1
        )
        attn_na_1, attn_add_1 = attn_1.split(
            [num_na_weights, attn_1.shape[-1] - num_na_weights], dim=-1
        )

        if n_additional_tokens:
            assert additional_key_p is not None and additional_key_t is not None
            assert attn_add_0.numel() > 0 and attn_add_1.numel() > 0
            qk_cross_forward(query_t, additional_key_p, attn_add_0)
            qk_cross_forward(query_p, additional_key_t, attn_add_1)

        libnatten.na1d_qk_forward(
            attn_na_0,
            query_t,
            key_p,
            None,
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )
        libnatten.na1d_qk_forward(
            attn_na_1,
            query_p,
            key_t,
            None,
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )

        return attn_0 + attn_1

    @staticmethod
    @amp_bwd
    def backward(
        ctx, grad_out: Tensor
    ) -> Tuple[
        Tensor, Tensor, Optional[Tensor], Optional[Tensor], NoneType, NoneType, NoneType
    ]:
        num_na_weights = get_num_na_weights(ctx.kernel_size)

        query, key, bias, additional_keys = ctx.saved_tensors
        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        # d_bias has to be zero filled
        d_bias = None if bias is None else torch.zeros_like(bias)
        d_additional_keys = None
        n_additional_tokens = check_additional_keys(query, additional_keys)
        d_query_add_key = None
        d_attn_na, d_attn_add = grad_out.split(
            [num_na_weights, grad_out.shape[-1] - num_na_weights], dim=-1
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
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )

        if d_query_add_key is not None:
            d_query += d_query_add_key

        return d_query, d_key, d_bias, d_additional_keys, None, None, None


class NeighborhoodAttention1DAVAutogradFunction(Function):
    @staticmethod
    @amp_fwd
    def forward(
        ctx,
        attn: Tensor,
        value: Tensor,
        additional_values: Optional[Tensor],
        kernel_size_: Dimension1DTypeOrDed,
        dilation_: Dimension1DTypeOrDed,
        is_causal_: CausalArg1DTypeOrDed,
    ):
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

        ctx.save_for_backward(attn, value, additional_values)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.is_causal = is_causal

        if out_add is not None:
            out += out_add

        return out

    @staticmethod
    def jvp(ctx, *grad_inputs: Any) -> Tensor:
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the AV operation is:
        av(attn.tangent, value.primal) + av(attn.primal, value.tangent)
        """
        if any(ctx.is_causal):
            raise ValueError(
                "Causal neighborhood attention doesn't support forward mode "
                "auto-diff yet."
            )
        num_na_weights = get_num_na_weights(ctx.kernel_size)

        assert len(grad_inputs) == 6
        attn_t: Tensor = grad_inputs[0]
        value_t: Tensor = grad_inputs[1]
        additional_value_t: Optional[Tensor] = grad_inputs[2]

        attn_p, value_p, additional_value_p = ctx.to_save

        if (additional_value_t is not None and additional_value_p is None) or (
            additional_value_t is None and additional_value_p is not None
        ):
            raise ValueError(
                "Expected either both additional_value_t and additional_value_p, or neither."
            )

        attn_t = attn_t.to(value_t.dtype)
        attn_t = attn_t.contiguous()
        value_t = value_t.contiguous()
        out_0 = torch.empty_like(value_p)
        out_1 = torch.empty_like(out_0)
        attn_na_t, attn_add_t = attn_t.split(
            [num_na_weights, attn_t.shape[-1] - num_na_weights], dim=-1
        )
        attn_na_p, attn_add_p = attn_p.split(
            [num_na_weights, attn_p.shape[-1] - num_na_weights], dim=-1
        )

        out_0_add, out_1_add = None, None
        n_additional_tokens = check_additional_values(
            attn_t, additional_value_t, value_t, num_na_weights
        )
        if n_additional_tokens:
            assert additional_value_p is not None and additional_value_t is not None
            assert attn_add_p.numel() > 0 and attn_add_t.numel() > 0
            out_0_add, out_1_add = torch.empty_like(out_0), torch.empty_like(out_1)
            av_cross_forward(attn_add_t, additional_value_p, out_0_add)
            av_cross_forward(attn_add_p, additional_value_t, out_1_add)

        libnatten.na1d_av_forward(
            out_0,
            attn_na_t,
            value_p,
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )
        libnatten.na1d_av_forward(
            out_1,
            attn_na_p,
            value_t,
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )

        if out_0_add is not None and out_1_add is not None:
            out_0 += out_0_add
            out_1 += out_1_add
        else:
            assert out_0_add is None and out_1_add is None

        return out_0 + out_1

    @staticmethod
    @amp_bwd
    def backward(
        ctx, grad_out: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], NoneType, NoneType, NoneType]:
        num_na_weights = get_num_na_weights(ctx.kernel_size)

        attn, value, additional_values = ctx.saved_tensors
        d_out = grad_out.contiguous()
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
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )

        return d_attn, d_value, d_additional_values, None, None, None


class NeighborhoodAttention2DQKAutogradFunction(Function):
    @staticmethod
    @amp_fwd
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        bias: Optional[Tensor],
        additional_keys: Optional[Tensor],
        kernel_size_: Dimension2DTypeOrDed,
        dilation_: Dimension2DTypeOrDed,
        is_causal_: CausalArg2DTypeOrDed,
    ):
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
        ctx.save_for_backward(query, key, bias, additional_keys)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.is_causal = is_causal

        return attn

    @staticmethod
    def jvp(ctx, *grad_inputs: Any) -> Tensor:
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the QK operation is:
        qk(query.tangent, key.primal) + qk(query.primal, key.tangent)
        """
        if any(ctx.is_causal):
            raise ValueError(
                "Causal neighborhood attention doesn't support forward mode "
                "auto-diff yet."
            )
        num_na_weights = get_num_na_weights(ctx.kernel_size)

        assert len(grad_inputs) == 7
        query_t: Tensor = grad_inputs[0]
        key_t: Tensor = grad_inputs[1]
        bias: Optional[Tensor] = grad_inputs[2]
        additional_key_t: Optional[Tensor] = grad_inputs[3]
        n_additional_tokens = check_additional_keys(query_t, additional_key_t)

        if bias is not None:
            raise ValueError(
                "Positional biases are currently not supported "
                "in forward mode autodiff."
            )

        query_p, key_p, _, additional_key_p = ctx.to_save

        if (additional_key_t is not None and additional_key_p is None) or (
            additional_key_t is None and additional_key_p is not None
        ):
            raise ValueError(
                "Expected either both additional_key_t and additional_key_p, or neither."
            )

        query_t = query_t.contiguous()
        key_t = key_t.contiguous()
        attn_0 = make_attn_tensor_from_input(
            query_t, num_na_weights + n_additional_tokens
        )
        attn_1 = torch.empty_like(attn_0)
        attn_na_0, attn_add_0 = attn_0.split(
            [num_na_weights, attn_0.shape[-1] - num_na_weights], dim=-1
        )
        attn_na_1, attn_add_1 = attn_1.split(
            [num_na_weights, attn_1.shape[-1] - num_na_weights], dim=-1
        )

        if n_additional_tokens:
            assert additional_key_p is not None and additional_key_t is not None
            assert attn_add_0.numel() > 0 and attn_add_1.numel() > 0
            qk_cross_forward(query_t, additional_key_p, attn_add_0)
            qk_cross_forward(query_p, additional_key_t, attn_add_1)

        libnatten.na2d_qk_forward(
            attn_na_0,
            query_t,
            key_p,
            None,
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )
        libnatten.na2d_qk_forward(
            attn_na_1,
            query_p,
            key_t,
            None,
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )
        return attn_0 + attn_1

    @staticmethod
    @amp_bwd
    def backward(
        ctx, grad_out: Tensor
    ) -> Tuple[
        Tensor, Tensor, Optional[Tensor], Optional[Tensor], NoneType, NoneType, NoneType
    ]:
        num_na_weights = get_num_na_weights(ctx.kernel_size)

        query, key, bias, additional_keys = ctx.saved_tensors
        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        # d_bias has to be zero filled
        d_bias = None if bias is None else torch.zeros_like(bias)
        d_additional_keys = None
        n_additional_tokens = check_additional_keys(query, additional_keys)
        d_query_add_key = None
        d_attn_na, d_attn_add = grad_out.split(
            [num_na_weights, grad_out.shape[-1] - num_na_weights], dim=-1
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
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )

        if d_query_add_key is not None:
            d_query += d_query_add_key

        return d_query, d_key, d_bias, d_additional_keys, None, None, None


class NeighborhoodAttention2DAVAutogradFunction(Function):
    @staticmethod
    @amp_fwd
    def forward(
        ctx,
        attn: Tensor,
        value: Tensor,
        additional_values: Optional[Tensor],
        kernel_size_: Dimension2DTypeOrDed,
        dilation_: Dimension2DTypeOrDed,
        is_causal_: CausalArg2DTypeOrDed,
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

        ctx.save_for_backward(attn, value, additional_values)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.is_causal = is_causal

        if out_add is not None:
            out += out_add

        return out

    @staticmethod
    def jvp(ctx, *grad_inputs: Any) -> Tensor:
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the AV operation is:
        av(attn.tangent, value.primal) + av(attn.primal, value.tangent)
        """
        if any(ctx.is_causal):
            raise ValueError(
                "Causal neighborhood attention doesn't support forward mode "
                "auto-diff yet."
            )
        num_na_weights = get_num_na_weights(ctx.kernel_size)

        assert len(grad_inputs) == 6
        attn_t: Tensor = grad_inputs[0]
        value_t: Tensor = grad_inputs[1]
        additional_value_t: Optional[Tensor] = grad_inputs[2]

        attn_p, value_p, additional_value_p = ctx.to_save

        if (additional_value_t is not None and additional_value_p is None) or (
            additional_value_t is None and additional_value_p is not None
        ):
            raise ValueError(
                "Expected either both additional_value_t and additional_value_p, or neither."
            )

        attn_t = attn_t.to(value_t.dtype)
        attn_t = attn_t.contiguous()
        value_t = value_t.contiguous()
        out_0 = torch.empty_like(value_p)
        out_1 = torch.empty_like(out_0)
        attn_na_t, attn_add_t = attn_t.split(
            [num_na_weights, attn_t.shape[-1] - num_na_weights], dim=-1
        )
        attn_na_p, attn_add_p = attn_p.split(
            [num_na_weights, attn_p.shape[-1] - num_na_weights], dim=-1
        )

        out_0_add, out_1_add = None, None
        n_additional_tokens = check_additional_values(
            attn_t, additional_value_t, value_t, num_na_weights
        )
        if n_additional_tokens:
            assert additional_value_p is not None and additional_value_t is not None
            assert attn_add_p.numel() > 0 and attn_add_t.numel() > 0
            out_0_add, out_1_add = torch.empty_like(out_0), torch.empty_like(out_1)
            av_cross_forward(attn_add_t, additional_value_p, out_0_add)
            av_cross_forward(attn_add_p, additional_value_t, out_1_add)

        libnatten.na2d_av_forward(
            out_0,
            attn_na_t,
            value_p,
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )
        libnatten.na2d_av_forward(
            out_1,
            attn_na_p,
            value_t,
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )

        if out_0_add is not None and out_1_add is not None:
            out_0 += out_0_add
            out_1 += out_1_add
        else:
            assert out_0_add is None and out_1_add is None

        return out_0 + out_1

    @staticmethod
    @amp_bwd
    def backward(
        ctx, grad_out: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], NoneType, NoneType, NoneType]:
        num_na_weights = get_num_na_weights(ctx.kernel_size)

        attn, value, additional_values = ctx.saved_tensors
        d_out = grad_out.contiguous()
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
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )

        return d_attn, d_value, d_additional_values, None, None, None


class NeighborhoodAttention3DQKAutogradFunction(Function):
    @staticmethod
    @amp_fwd
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        bias: Optional[Tensor],
        additional_keys: Optional[Tensor],
        kernel_size_: Dimension3DTypeOrDed,
        dilation_: Dimension3DTypeOrDed,
        is_causal_: CausalArg3DTypeOrDed,
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
            attn_na,
            query,
            key,
            bias,
            kernel_size,
            dilation,
            is_causal,
        )

        ctx.save_for_backward(query, key, bias, additional_keys)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.is_causal = is_causal

        return attn

    @staticmethod
    def jvp(ctx, *grad_inputs: Any) -> Tensor:
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the QK operation is:
        qk(query.tangent, key.primal) + qk(query.primal, key.tangent)
        """
        if any(ctx.is_causal):
            raise ValueError(
                "Causal neighborhood attention doesn't support forward mode "
                "auto-diff yet."
            )
        num_na_weights = get_num_na_weights(ctx.kernel_size)

        assert len(grad_inputs) == 7
        query_t: Tensor = grad_inputs[0]
        key_t: Tensor = grad_inputs[1]
        bias: Optional[Tensor] = grad_inputs[2]
        additional_key_t: Optional[Tensor] = grad_inputs[3]
        n_additional_tokens = check_additional_keys(query_t, additional_key_t)

        if bias is not None:
            raise ValueError(
                "Positional biases are currently not supported "
                "in forward mode autodiff."
            )
        query_p, key_p, _, additional_key_p = ctx.to_save

        if (additional_key_t is not None and additional_key_p is None) or (
            additional_key_t is None and additional_key_p is not None
        ):
            raise ValueError(
                "Expected either both additional_key_t and additional_key_p, or neither."
            )

        query_t = query_t.contiguous()
        key_t = key_t.contiguous()
        attn_0 = make_attn_tensor_from_input(
            query_t, num_na_weights + n_additional_tokens
        )
        attn_1 = torch.empty_like(attn_0)
        attn_na_0, attn_add_0 = attn_0.split(
            [num_na_weights, attn_0.shape[-1] - num_na_weights], dim=-1
        )
        attn_na_1, attn_add_1 = attn_1.split(
            [num_na_weights, attn_1.shape[-1] - num_na_weights], dim=-1
        )

        if n_additional_tokens:
            assert additional_key_p is not None and additional_key_t is not None
            assert attn_add_0.numel() > 0 and attn_add_1.numel() > 0
            qk_cross_forward(query_t, additional_key_p, attn_add_0)
            qk_cross_forward(query_p, additional_key_t, attn_add_1)

        libnatten.na3d_qk_forward(
            attn_na_0,
            query_t,
            key_p,
            None,
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )
        libnatten.na3d_qk_forward(
            attn_na_1,
            query_p,
            key_t,
            None,
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )
        return attn_0 + attn_1

    @staticmethod
    @amp_bwd
    def backward(ctx, grad_out: Tensor) -> Tuple[
        Tensor,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        NoneType,
        NoneType,
        NoneType,
    ]:
        num_na_weights = get_num_na_weights(ctx.kernel_size)

        query, key, bias, additional_keys = ctx.saved_tensors
        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        # d_bias has to be zero filled
        d_bias = None if bias is None else torch.zeros_like(bias)
        d_additional_keys = None
        n_additional_tokens = check_additional_keys(query, additional_keys)
        d_query_add_key = None
        d_attn_na, d_attn_add = grad_out.split(
            [num_na_weights, grad_out.shape[-1] - num_na_weights], dim=-1
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
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )

        if d_query_add_key is not None:
            d_query += d_query_add_key

        return d_query, d_key, d_bias, d_additional_keys, None, None, None


class NeighborhoodAttention3DAVAutogradFunction(Function):
    @staticmethod
    @amp_fwd
    def forward(
        ctx,
        attn: Tensor,
        value: Tensor,
        additional_values: Optional[Tensor],
        kernel_size_: Dimension3DTypeOrDed,
        dilation_: Dimension3DTypeOrDed,
        is_causal_: CausalArg3DTypeOrDed,
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

        libnatten.na3d_av_forward(
            out,
            attn_na,
            value,
            kernel_size,
            dilation,
            is_causal,
        )

        ctx.save_for_backward(attn, value, additional_values)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.is_causal = is_causal

        if out_add is not None:
            out += out_add

        return out

    @staticmethod
    def jvp(ctx, *grad_inputs: Any) -> Tensor:
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the AV operation is:
        av(attn.tangent, value.primal) + av(attn.primal, value.tangent)
        """
        if any(ctx.is_causal):
            raise ValueError(
                "Causal neighborhood attention doesn't support forward mode "
                "auto-diff yet."
            )
        num_na_weights = get_num_na_weights(ctx.kernel_size)

        assert len(grad_inputs) == 6
        attn_t: Tensor = grad_inputs[0]
        value_t: Tensor = grad_inputs[1]
        additional_value_t: Optional[Tensor] = grad_inputs[2]

        attn_p, value_p, additional_value_p = ctx.to_save

        if (additional_value_t is not None and additional_value_p is None) or (
            additional_value_t is None and additional_value_p is not None
        ):
            raise ValueError(
                "Expected either both additional_value_t and additional_value_p, or neither."
            )

        attn_t = attn_t.to(value_t.dtype)
        attn_t = attn_t.contiguous()
        value_t = value_t.contiguous()
        out_0 = torch.empty_like(value_p)
        out_1 = torch.empty_like(out_0)
        attn_na_t, attn_add_t = attn_t.split(
            [num_na_weights, attn_t.shape[-1] - num_na_weights], dim=-1
        )
        attn_na_p, attn_add_p = attn_p.split(
            [num_na_weights, attn_p.shape[-1] - num_na_weights], dim=-1
        )

        out_0_add, out_1_add = None, None
        n_additional_tokens = check_additional_values(
            attn_t, additional_value_t, value_t, num_na_weights
        )
        if n_additional_tokens:
            assert additional_value_p is not None and additional_value_t is not None
            attn_add_t = attn_t[:, :, :, :, :, num_na_weights:]
            attn_add_p = attn_p[:, :, :, :, :, num_na_weights:]
            out_0_add, out_1_add = torch.empty_like(out_0), torch.empty_like(out_1)
            av_cross_forward(attn_add_t, additional_value_p, out_0_add)
            av_cross_forward(attn_add_p, additional_value_t, out_1_add)

        libnatten.na3d_av_forward(
            out_0,
            attn_na_t,
            value_p,
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )
        libnatten.na3d_av_forward(
            out_1,
            attn_na_p,
            value_t,
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )

        if out_0_add is not None and out_1_add is not None:
            out_0 += out_0_add
            out_1 += out_1_add
        else:
            assert out_0_add is None and out_1_add is None

        return out_0 + out_1

    @staticmethod
    @amp_bwd
    def backward(
        ctx, grad_out: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], NoneType, NoneType, NoneType]:
        num_na_weights = get_num_na_weights(ctx.kernel_size)
        attn, value, additional_values = ctx.saved_tensors

        d_out = grad_out.contiguous()
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
            ctx.kernel_size,
            ctx.dilation,
            ctx.is_causal,
        )

        return d_attn, d_value, d_additional_values, None, None, None


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

    return NeighborhoodAttention1DQKAutogradFunction.apply(
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

    return NeighborhoodAttention1DAVAutogradFunction.apply(
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

    return NeighborhoodAttention2DQKAutogradFunction.apply(
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

    return NeighborhoodAttention2DAVAutogradFunction.apply(
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

    return NeighborhoodAttention3DQKAutogradFunction.apply(
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

    return NeighborhoodAttention3DAVAutogradFunction.apply(
        attn,
        value,
        additional_values,
        kernel_size_,
        dilation_,
        is_causal_,
    )
