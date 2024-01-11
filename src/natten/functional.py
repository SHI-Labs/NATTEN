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
from typing import Any, Optional, Tuple

import torch
from torch import Tensor
from torch.autograd import Function
from torch.cuda import _device_t
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    from natten import libnatten  # type: ignore
except ImportError:
    raise ImportError(
        "Failed to import NATTEN's CPP backend. "
        "This could be due to an invalid/incomplete install. "
        "Please uninstall NATTEN (pip uninstall natten) and re-install with the"
        " correct torch build: shi-labs.com/natten ."
    )

from .nested import (
    na1d_av_nested,
    na1d_qk_nested,
    na2d_av_nested,
    na2d_qk_nested,
    na3d_av_nested,
    na3d_qk_nested,
)
from .ops import (
    av_cross_backward,
    av_cross_forward,
    qk_cross_backward,
    qk_cross_forward,
)
from .utils.tensor import (
    check_additional_keys,
    check_additional_values,
    make_attn_tensor_from_input,
)
from .utils.typing import NoneType


def get_device_cc(device_index: Optional[_device_t] = None) -> int:
    major, minor = torch.cuda.get_device_capability(device_index)
    return major * 10 + minor


def has_cuda() -> bool:
    return torch.cuda.is_available() and libnatten.has_cuda()


def has_half(device_index: Optional[_device_t] = None) -> bool:
    return has_cuda() and get_device_cc(device_index) >= 60


def has_bfloat(device_index: Optional[_device_t] = None) -> bool:
    return has_cuda() and get_device_cc(device_index) >= 80


def has_gemm(device_index: Optional[_device_t] = None) -> bool:
    return has_cuda() and get_device_cc(device_index) >= 70 and libnatten.has_gemm()


def has_tf32_gemm(device_index: Optional[_device_t] = None) -> bool:
    return has_cuda() and get_device_cc(device_index) >= 80 and libnatten.has_gemm()


has_fp32_gemm = has_tf32_gemm


def has_fp64_gemm(device_index: Optional[_device_t] = None) -> bool:
    return has_cuda() and get_device_cc(device_index) >= 80 and libnatten.has_gemm()


def enable_tf32() -> bool:
    if has_tf32_gemm():
        libnatten.set_gemm_tf32(True)
        return libnatten.get_gemm_tf32()
    return False


def disable_tf32() -> bool:
    if has_tf32_gemm():
        libnatten.set_gemm_tf32(False)
        return libnatten.get_gemm_tf32()
    return False


def enable_tiled_na():
    libnatten.set_tiled_na(True)


def disable_tiled_na():
    libnatten.set_tiled_na(False)


def enable_gemm_na():
    libnatten.set_gemm_na(True)


def disable_gemm_na():
    libnatten.set_gemm_na(False)


class NeighborhoodAttention1DQKAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        bias: Optional[Tensor],
        additional_key: Optional[Tensor],
        kernel_size: int,
        dilation: int,
    ) -> Tensor:
        num_na_weights = kernel_size
        query = query.contiguous()
        key = key.contiguous()
        n_additional_tokens = check_additional_keys(query, additional_key)
        if bias is not None:
            bias = bias.to(key.dtype)
        attn = make_attn_tensor_from_input(query, num_na_weights + n_additional_tokens)
        attn_na, attn_add = attn.split(
            [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
        )

        if n_additional_tokens:
            assert additional_key is not None and attn_add.numel() > 0
            qk_cross_forward(query, additional_key, attn_add)

        libnatten.na1d_qk_forward(attn_na, query, key, bias, kernel_size, dilation)
        ctx.save_for_backward(query, key, bias, additional_key)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation

        return attn

    @staticmethod
    def jvp(ctx, *grad_inputs: Any) -> Tensor:
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the QK operation is:
        qk(query.tangent, key.primal) + qk(query.primal, key.tangent)
        """
        num_na_weights = ctx.kernel_size
        assert len(grad_inputs) == 6
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
            attn_na_0, query_t, key_p, None, ctx.kernel_size, ctx.dilation
        )
        libnatten.na1d_qk_forward(
            attn_na_1, query_p, key_t, None, ctx.kernel_size, ctx.dilation
        )

        return attn_0 + attn_1

    @staticmethod
    @custom_bwd
    def backward(
        ctx, grad_out: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], NoneType, NoneType]:
        num_na_weights = ctx.kernel_size
        query, key, bias, additional_key = ctx.saved_tensors
        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        # d_bias has to be zero filled
        d_bias = None if bias is None else torch.zeros_like(bias)
        d_additional_key = None
        n_additional_tokens = check_additional_keys(query, additional_key)
        d_query_add_key = None
        d_attn_na, d_attn_add = grad_out.split(
            [num_na_weights, grad_out.shape[-1] - num_na_weights], dim=-1
        )

        if n_additional_tokens:
            assert additional_key is not None and d_attn_add.numel() > 0
            d_query_add_key, d_additional_key = torch.empty_like(
                d_query
            ), torch.empty_like(additional_key)
            qk_cross_backward(
                query, d_attn_add, additional_key, d_query_add_key, d_additional_key
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
        )

        if d_query_add_key is not None:
            d_query += d_query_add_key

        return d_query, d_key, d_bias, d_additional_key, None, None


class NeighborhoodAttention1DAVAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        attn: Tensor,
        value: Tensor,
        additional_value: Optional[Tensor],
        kernel_size: int,
        dilation: int,
    ):
        num_na_weights = kernel_size
        value = value.contiguous()
        out = torch.empty_like(value)
        out_add = None
        n_additional_tokens = check_additional_values(
            attn, additional_value, value, num_na_weights
        )
        attn_na, attn_add = attn.split(
            [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
        )

        if n_additional_tokens:
            assert additional_value is not None and attn_add.numel() > 0
            out_add = torch.empty_like(out)
            av_cross_forward(attn_add, additional_value, out_add)

        libnatten.na1d_av_forward(out, attn_na, value, kernel_size, dilation)

        ctx.save_for_backward(attn, value, additional_value)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation

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
        num_na_weights = ctx.kernel_size
        assert len(grad_inputs) == 5
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
            out_0, attn_na_t, value_p, ctx.kernel_size, ctx.dilation
        )
        libnatten.na1d_av_forward(
            out_1, attn_na_p, value_t, ctx.kernel_size, ctx.dilation
        )

        if out_0_add is not None and out_1_add is not None:
            out_0 += out_0_add
            out_1 += out_1_add
        else:
            assert out_0_add is None and out_1_add is None

        return out_0 + out_1

    @staticmethod
    @custom_bwd
    def backward(
        ctx, grad_out: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], NoneType, NoneType]:
        num_na_weights = ctx.kernel_size
        attn, value, additional_value = ctx.saved_tensors
        d_out = grad_out.contiguous()
        d_attn = torch.empty_like(attn)
        d_value = torch.empty_like(value)
        d_additional_value = None
        n_additional_tokens = check_additional_values(
            attn, additional_value, value, num_na_weights
        )
        d_attn_na, d_attn_add = d_attn.split(
            [num_na_weights, d_attn.shape[-1] - num_na_weights], dim=-1
        )
        attn_na, attn_add = attn.split(
            [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
        )

        if n_additional_tokens:
            assert additional_value is not None
            assert d_attn_add.numel() > 0 and attn_add.numel() > 0
            d_additional_value = torch.empty_like(additional_value)
            av_cross_backward(
                d_out, additional_value, attn_add, d_attn_add, d_additional_value
            )

        libnatten.na1d_av_backward(
            d_attn_na,
            d_value,
            d_out,
            attn_na,
            value,
            ctx.kernel_size,
            ctx.dilation,
        )

        return d_attn, d_value, d_additional_value, None, None


class NeighborhoodAttention2DQKAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        bias: Optional[Tensor],
        additional_key: Optional[Tensor],
        kernel_size: int,
        dilation: int,
    ):
        num_na_weights = kernel_size**2
        query = query.contiguous()
        key = key.contiguous()
        n_additional_tokens = check_additional_keys(query, additional_key)
        if bias is not None:
            bias = bias.to(key.dtype)
        attn = make_attn_tensor_from_input(query, num_na_weights + n_additional_tokens)
        attn_na, attn_add = attn.split(
            [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
        )

        if n_additional_tokens:
            assert additional_key is not None and attn_add.numel() > 0
            qk_cross_forward(query, additional_key, attn_add)

        libnatten.na2d_qk_forward(attn_na, query, key, bias, kernel_size, dilation)
        ctx.save_for_backward(query, key, bias, additional_key)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation

        return attn

    @staticmethod
    def jvp(ctx, *grad_inputs: Any) -> Tensor:
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the QK operation is:
        qk(query.tangent, key.primal) + qk(query.primal, key.tangent)
        """
        num_na_weights = ctx.kernel_size**2
        assert len(grad_inputs) == 6
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
            attn_na_0, query_t, key_p, None, ctx.kernel_size, ctx.dilation
        )
        libnatten.na2d_qk_forward(
            attn_na_1, query_p, key_t, None, ctx.kernel_size, ctx.dilation
        )
        return attn_0 + attn_1

    @staticmethod
    @custom_bwd
    def backward(
        ctx, grad_out: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor], NoneType, NoneType]:
        num_na_weights = ctx.kernel_size**2
        query, key, bias, additional_key = ctx.saved_tensors
        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        # d_bias has to be zero filled
        d_bias = None if bias is None else torch.zeros_like(bias)
        d_additional_key = None
        n_additional_tokens = check_additional_keys(query, additional_key)
        d_query_add_key = None
        d_attn_na, d_attn_add = grad_out.split(
            [num_na_weights, grad_out.shape[-1] - num_na_weights], dim=-1
        )

        if n_additional_tokens:
            assert additional_key is not None and d_attn_add.numel() > 0
            d_query_add_key, d_additional_key = torch.empty_like(
                d_query
            ), torch.empty_like(additional_key)
            qk_cross_backward(
                query, d_attn_add, additional_key, d_query_add_key, d_additional_key
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
        )

        if d_query_add_key is not None:
            d_query += d_query_add_key

        return d_query, d_key, d_bias, d_additional_key, None, None


class NeighborhoodAttention2DAVAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        attn: Tensor,
        value: Tensor,
        additional_value: Optional[Tensor],
        kernel_size: int,
        dilation: int,
    ) -> Tensor:
        num_na_weights = kernel_size**2
        value = value.contiguous()
        out = torch.empty_like(value)
        out_add = None
        n_additional_tokens = check_additional_values(
            attn, additional_value, value, num_na_weights
        )
        attn_na, attn_add = attn.split(
            [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
        )

        if n_additional_tokens:
            assert additional_value is not None and attn_add.numel() > 0
            out_add = torch.empty_like(out)
            av_cross_forward(attn_add, additional_value, out_add)

        libnatten.na2d_av_forward(out, attn_na, value, kernel_size, dilation)

        ctx.save_for_backward(attn, value, additional_value)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation

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
        num_na_weights = ctx.kernel_size**2
        assert len(grad_inputs) == 5
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
            out_0, attn_na_t, value_p, ctx.kernel_size, ctx.dilation
        )
        libnatten.na2d_av_forward(
            out_1, attn_na_p, value_t, ctx.kernel_size, ctx.dilation
        )

        if out_0_add is not None and out_1_add is not None:
            out_0 += out_0_add
            out_1 += out_1_add
        else:
            assert out_0_add is None and out_1_add is None

        return out_0 + out_1

    @staticmethod
    @custom_bwd
    def backward(
        ctx, grad_out: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], NoneType, NoneType]:
        num_na_weights = ctx.kernel_size**2
        attn, value, additional_value = ctx.saved_tensors
        d_out = grad_out.contiguous()
        d_attn = torch.empty_like(attn)
        d_value = torch.empty_like(value)
        d_additional_value = None
        n_additional_tokens = check_additional_values(
            attn, additional_value, value, num_na_weights
        )
        d_attn_na, d_attn_add = d_attn.split(
            [num_na_weights, d_attn.shape[-1] - num_na_weights], dim=-1
        )
        attn_na, attn_add = attn.split(
            [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
        )

        if n_additional_tokens:
            assert additional_value is not None
            assert d_attn_add.numel() > 0 and attn_add.numel() > 0
            d_additional_value = torch.empty_like(additional_value)
            av_cross_backward(
                d_out, additional_value, attn_add, d_attn_add, d_additional_value
            )

        libnatten.na2d_av_backward(
            d_attn_na,
            d_value,
            d_out,
            attn_na,
            value,
            ctx.kernel_size,
            ctx.dilation,
        )

        return d_attn, d_value, d_additional_value, None, None


class NeighborhoodAttention3DQKAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        bias: Optional[Tensor],
        additional_key: Optional[Tensor],
        kernel_size_d: int,
        kernel_size: int,
        dilation_d: int,
        dilation: int,
    ) -> Tensor:
        num_na_weights = kernel_size_d * kernel_size * kernel_size
        query = query.contiguous()
        key = key.contiguous()
        n_additional_tokens = check_additional_keys(query, additional_key)
        if bias is not None:
            bias = bias.to(key.dtype)
        attn = make_attn_tensor_from_input(query, num_na_weights + n_additional_tokens)
        attn_na, attn_add = attn.split(
            [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
        )

        if n_additional_tokens:
            assert additional_key is not None and attn_add.numel() > 0
            qk_cross_forward(query, additional_key, attn_add)

        libnatten.na3d_qk_forward(
            attn_na, query, key, bias, kernel_size, dilation, kernel_size_d, dilation_d
        )

        ctx.save_for_backward(query, key, bias, additional_key)
        ctx.kernel_size_d = kernel_size_d
        ctx.kernel_size = kernel_size
        ctx.dilation_d = dilation_d
        ctx.dilation = dilation

        return attn

    @staticmethod
    def jvp(ctx, *grad_inputs: Any) -> Tensor:
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the QK operation is:
        qk(query.tangent, key.primal) + qk(query.primal, key.tangent)
        """
        num_na_weights = ctx.kernel_size_d * ctx.kernel_size * ctx.kernel_size
        assert len(grad_inputs) == 8
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
            ctx.kernel_size_d,
            ctx.dilation_d,
        )
        libnatten.na3d_qk_forward(
            attn_na_1,
            query_p,
            key_t,
            None,
            ctx.kernel_size,
            ctx.dilation,
            ctx.kernel_size_d,
            ctx.dilation_d,
        )
        return attn_0 + attn_1

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out: Tensor) -> Tuple[
        Tensor,
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        NoneType,
        NoneType,
        NoneType,
        NoneType,
    ]:
        num_na_weights = ctx.kernel_size_d * ctx.kernel_size * ctx.kernel_size
        query, key, bias, additional_key = ctx.saved_tensors
        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        # d_bias has to be zero filled
        d_bias = None if bias is None else torch.zeros_like(bias)
        d_additional_key = None
        n_additional_tokens = check_additional_keys(query, additional_key)
        d_query_add_key = None
        d_attn_na, d_attn_add = grad_out.split(
            [num_na_weights, grad_out.shape[-1] - num_na_weights], dim=-1
        )

        if n_additional_tokens:
            assert additional_key is not None and d_attn_add.numel() > 0
            d_query_add_key, d_additional_key = torch.empty_like(
                d_query
            ), torch.empty_like(additional_key)
            qk_cross_backward(
                query, d_attn_add, additional_key, d_query_add_key, d_additional_key
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
            ctx.kernel_size_d,
            ctx.dilation_d,
        )

        if d_query_add_key is not None:
            d_query += d_query_add_key

        return d_query, d_key, d_bias, d_additional_key, None, None, None, None


class NeighborhoodAttention3DAVAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        attn: Tensor,
        value: Tensor,
        additional_value: Optional[Tensor],
        kernel_size_d: int,
        kernel_size: int,
        dilation_d: int,
        dilation: int,
    ) -> Tensor:
        num_na_weights = kernel_size_d * kernel_size * kernel_size
        value = value.contiguous()
        out = torch.empty_like(value)
        out_add = None
        n_additional_tokens = check_additional_values(
            attn, additional_value, value, num_na_weights
        )
        attn_na, attn_add = attn.split(
            [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
        )

        if n_additional_tokens:
            assert additional_value is not None and attn_add.numel() > 0
            out_add = torch.empty_like(out)
            av_cross_forward(attn_add, additional_value, out_add)

        libnatten.na3d_av_forward(
            out, attn_na, value, kernel_size, dilation, kernel_size_d, dilation_d
        )

        ctx.save_for_backward(attn, value, additional_value)
        ctx.kernel_size_d = kernel_size_d
        ctx.kernel_size = kernel_size
        ctx.dilation_d = dilation_d
        ctx.dilation = dilation

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
        num_na_weights = ctx.kernel_size_d * ctx.kernel_size * ctx.kernel_size
        assert len(grad_inputs) == 7
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
            attn_add_t = attn_t[
                :, :, :, :, :, ctx.kernel_size * ctx.kernel_size * ctx.kernel_size_d :
            ]
            attn_add_p = attn_p[
                :, :, :, :, :, ctx.kernel_size * ctx.kernel_size * ctx.kernel_size_d :
            ]
            out_0_add, out_1_add = torch.empty_like(out_0), torch.empty_like(out_1)
            av_cross_forward(attn_add_t, additional_value_p, out_0_add)
            av_cross_forward(attn_add_p, additional_value_t, out_1_add)

        libnatten.na3d_av_forward(
            out_0,
            attn_na_t,
            value_p,
            ctx.kernel_size,
            ctx.dilation,
            ctx.kernel_size_d,
            ctx.dilation_d,
        )
        libnatten.na3d_av_forward(
            out_1,
            attn_na_p,
            value_t,
            ctx.kernel_size,
            ctx.dilation,
            ctx.kernel_size_d,
            ctx.dilation_d,
        )

        if out_0_add is not None and out_1_add is not None:
            out_0 += out_0_add
            out_1 += out_1_add
        else:
            assert out_0_add is None and out_1_add is None

        return out_0 + out_1

    @staticmethod
    @custom_bwd
    def backward(
        ctx, grad_out: Tensor
    ) -> Tuple[
        Tensor, Tensor, Optional[Tensor], NoneType, NoneType, NoneType, NoneType
    ]:
        num_na_weights = ctx.kernel_size_d * ctx.kernel_size * ctx.kernel_size
        attn, value, additional_value = ctx.saved_tensors
        d_out = grad_out.contiguous()
        d_attn = torch.empty_like(attn)
        d_value = torch.empty_like(value)
        d_additional_value = None
        n_additional_tokens = check_additional_values(
            attn, additional_value, value, num_na_weights
        )
        d_attn_na, d_attn_add = d_attn.split(
            [num_na_weights, d_attn.shape[-1] - num_na_weights], dim=-1
        )
        attn_na, attn_add = attn.split(
            [num_na_weights, attn.shape[-1] - num_na_weights], dim=-1
        )

        if n_additional_tokens:
            assert additional_value is not None
            assert d_attn_add.numel() > 0 and attn_add.numel() > 0
            d_additional_value = torch.empty_like(additional_value)
            av_cross_backward(
                d_out, additional_value, attn_add, d_attn_add, d_additional_value
            )

        libnatten.na3d_av_backward(
            d_attn_na,
            d_value,
            d_out,
            attn_na,
            value,
            ctx.kernel_size,
            ctx.dilation,
            ctx.kernel_size_d,
            ctx.dilation_d,
        )

        return d_attn, d_value, d_additional_value, None, None, None, None


def na1d_qk(
    query: Tensor,
    key: Tensor,
    kernel_size: int,
    dilation: int,
    additional_keys: Optional[Tensor] = None,
) -> Tensor:
    if query.is_nested or key.is_nested:
        return na1d_qk_nested(
            query, key, None, kernel_size, dilation, additional_keys=additional_keys
        )
    return NeighborhoodAttention1DQKAutogradFunction.apply(
        query, key, None, additional_keys, kernel_size, dilation
    )


def na1d_qk_with_bias(
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    kernel_size: int,
    dilation: int,
    additional_keys: Optional[Tensor] = None,
) -> Tensor:
    if query.is_nested or key.is_nested:
        return na1d_qk_nested(
            query, key, bias, kernel_size, dilation, additional_keys=additional_keys
        )
    return NeighborhoodAttention1DQKAutogradFunction.apply(
        query, key, bias, additional_keys, kernel_size, dilation
    )


def na1d_av(
    attn: Tensor,
    value: Tensor,
    kernel_size: int,
    dilation: int,
    additional_values: Optional[Tensor] = None,
) -> Tensor:
    if attn.is_nested or value.is_nested:
        return na1d_av_nested(
            attn, value, kernel_size, dilation, additional_values=additional_values
        )
    return NeighborhoodAttention1DAVAutogradFunction.apply(
        attn, value, additional_values, kernel_size, dilation
    )


def na2d_qk(
    query: Tensor,
    key: Tensor,
    kernel_size: int,
    dilation: int,
    additional_keys: Optional[Tensor] = None,
) -> Tensor:
    if query.is_nested or key.is_nested:
        return na2d_qk_nested(
            query, key, None, kernel_size, dilation, additional_keys=additional_keys
        )
    return NeighborhoodAttention2DQKAutogradFunction.apply(
        query, key, None, additional_keys, kernel_size, dilation
    )


def na2d_qk_with_bias(
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    kernel_size: int,
    dilation: int,
    additional_keys: Optional[Tensor] = None,
) -> Tensor:
    if query.is_nested or key.is_nested:
        return na2d_qk_nested(
            query, key, bias, kernel_size, dilation, additional_keys=additional_keys
        )
    return NeighborhoodAttention2DQKAutogradFunction.apply(
        query, key, bias, additional_keys, kernel_size, dilation
    )


def na2d_av(
    attn: Tensor,
    value: Tensor,
    kernel_size: int,
    dilation: int,
    additional_values: Optional[Tensor] = None,
) -> Tensor:
    if attn.is_nested or value.is_nested:
        return na2d_av_nested(
            attn, value, kernel_size, dilation, additional_values=additional_values
        )
    return NeighborhoodAttention2DAVAutogradFunction.apply(
        attn, value, additional_values, kernel_size, dilation
    )


def na3d_qk_with_bias(
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    kernel_size_d: int,
    kernel_size: int,
    dilation_d: int,
    dilation: int,
    additional_keys: Optional[Tensor] = None,
) -> Tensor:
    if query.is_nested or key.is_nested:
        return na3d_qk_nested(
            query,
            key,
            bias,
            kernel_size_d,
            kernel_size,
            dilation_d,
            dilation,
            additional_keys=additional_keys,
        )
    return NeighborhoodAttention3DQKAutogradFunction.apply(
        query,
        key,
        bias,
        additional_keys,
        kernel_size_d,
        kernel_size,
        dilation_d,
        dilation,
    )


def na3d_qk(
    query: Tensor,
    key: Tensor,
    kernel_size_d: int,
    kernel_size: int,
    dilation_d: int,
    dilation: int,
    additional_keys: Optional[Tensor] = None,
) -> Tensor:
    if query.is_nested or key.is_nested:
        return na3d_qk_nested(
            query,
            key,
            None,
            kernel_size_d,
            kernel_size,
            dilation_d,
            dilation,
            additional_keys=additional_keys,
        )
    return NeighborhoodAttention3DQKAutogradFunction.apply(
        query,
        key,
        None,
        additional_keys,
        kernel_size_d,
        kernel_size,
        dilation_d,
        dilation,
    )


def na3d_av(
    attn: Tensor,
    value: Tensor,
    kernel_size_d: int,
    kernel_size: int,
    dilation_d: int,
    dilation: int,
    additional_values: Optional[Tensor] = None,
) -> Tensor:
    if attn.is_nested or value.is_nested:
        return na3d_av_nested(
            attn,
            value,
            kernel_size_d,
            kernel_size,
            dilation_d,
            dilation,
            additional_values=additional_values,
        )
    return NeighborhoodAttention3DAVAutogradFunction.apply(
        attn, value, additional_values, kernel_size_d, kernel_size, dilation_d, dilation
    )


#################################################################################################
# Soon to be deprecated functions
#################################################################################################


def natten1dqkrpb(
    query: Tensor, key: Tensor, rpb: Optional[Tensor], kernel_size: int, dilation: int
) -> Tensor:
    return na1d_qk_with_bias(query, key, rpb, kernel_size, dilation)


def natten2dqkrpb(
    query: Tensor, key: Tensor, rpb: Optional[Tensor], kernel_size: int, dilation: int
) -> Tensor:
    return na2d_qk_with_bias(query, key, rpb, kernel_size, dilation)


def natten3dqkrpb(
    query: Tensor,
    key: Tensor,
    rpb: Optional[Tensor],
    kernel_size_d: int,
    kernel_size: int,
    dilation_d: int,
    dilation: int,
) -> Tensor:
    return na3d_qk_with_bias(
        query, key, rpb, kernel_size_d, kernel_size, dilation_d, dilation
    )


natten1dqk = na1d_qk
natten1dav = na1d_av
natten2dqk = na2d_qk
natten2dav = na2d_av
natten3dqk = na3d_qk
natten3dav = na3d_av
