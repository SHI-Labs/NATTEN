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
from .utils.tensor import make_attn_tensor_from_input
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
        kernel_size: int,
        dilation: int,
    ) -> Tensor:
        query = query.contiguous()
        key = key.contiguous()
        if bias is not None:
            bias = bias.to(key.dtype)
        attn = make_attn_tensor_from_input(query, kernel_size)
        libnatten.na1d_qk_forward(attn, query, key, bias, kernel_size, dilation)
        ctx.save_for_backward(query, key, bias)
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
        assert len(grad_inputs) == 5
        query_t: Tensor = grad_inputs[0]
        key_t: Tensor = grad_inputs[1]
        bias: Optional[Tensor] = grad_inputs[2]

        if bias is not None:
            raise ValueError(
                "Positional biases are currently not supported "
                "in forward mode autodiff."
            )

        query_p, key_p, _ = ctx.to_save
        query_t = query_t.contiguous()
        key_t = key_t.contiguous()
        attn_0 = make_attn_tensor_from_input(query_t, ctx.kernel_size)
        attn_1 = torch.empty_like(attn_0)
        libnatten.na1d_qk_forward(
            attn_0, query_t, key_p, None, ctx.kernel_size, ctx.dilation
        )
        libnatten.na1d_qk_forward(
            attn_1, query_p, key_t, None, ctx.kernel_size, ctx.dilation
        )
        return attn_0 + attn_1

    @staticmethod
    @custom_bwd
    def backward(
        ctx, grad_out: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], NoneType, NoneType]:
        query, key, bias = ctx.saved_tensors
        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        # dbias has to be zero filled
        d_bias = None if bias is None else torch.zeros_like(bias)
        libnatten.na1d_qk_backward(
            d_query,
            d_key,
            d_bias,
            grad_out.contiguous(),
            query,
            key,
            ctx.kernel_size,
            ctx.dilation,
        )
        return d_query, d_key, d_bias, None, None


class NeighborhoodAttention1DAVAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, attn, value, kernel_size, dilation):
        attn = attn.contiguous()
        value = value.contiguous()
        out = torch.empty_like(value)
        libnatten.na1d_av_forward(out, attn, value, kernel_size, dilation)
        ctx.save_for_backward(attn, value)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        return out

    @staticmethod
    def jvp(ctx, *grad_inputs: Any) -> Tensor:
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the AV operation is:
        av(attn.tangent, value.primal) + av(attn.primal, value.tangent)
        """
        assert len(grad_inputs) == 4
        attn_t: Tensor = grad_inputs[0]
        value_t: Tensor = grad_inputs[1]

        attn_p, value_p = ctx.to_save
        attn_t = attn_t.contiguous()
        value_t = value_t.contiguous()
        out_0 = torch.empty_like(value_p)
        out_1 = torch.empty_like(out_0)
        libnatten.na1d_av_forward(out_0, attn_t, value_p, ctx.kernel_size, ctx.dilation)
        libnatten.na1d_av_forward(out_1, attn_p, value_t, ctx.kernel_size, ctx.dilation)
        return out_0 + out_1

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out: Tensor) -> Tuple[Tensor, Tensor, NoneType, NoneType]:
        attn, value = ctx.saved_tensors
        d_attn = torch.empty_like(attn)
        d_value = torch.empty_like(value)
        libnatten.na1d_av_backward(
            d_attn,
            d_value,
            grad_out.contiguous(),
            attn,
            value,
            ctx.kernel_size,
            ctx.dilation,
        )
        return d_attn, d_value, None, None


class NeighborhoodAttention2DQKAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        bias: Optional[Tensor],
        kernel_size: int,
        dilation: int,
    ):
        query = query.contiguous()
        key = key.contiguous()
        if bias is not None:
            bias = bias.to(key.dtype)
        attn = make_attn_tensor_from_input(query, kernel_size**2)
        libnatten.na2d_qk_forward(attn, query, key, bias, kernel_size, dilation)
        ctx.save_for_backward(query, key, bias)
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
        assert len(grad_inputs) == 5
        query_t: Tensor = grad_inputs[0]
        key_t: Tensor = grad_inputs[1]
        bias: Optional[Tensor] = grad_inputs[2]

        if bias is not None:
            raise ValueError(
                "Positional biases are currently not supported "
                "in forward mode autodiff."
            )
        query_p, key_p, _ = ctx.to_save
        query_t = query_t.contiguous()
        key_t = key_t.contiguous()
        attn_0 = make_attn_tensor_from_input(query_t, ctx.kernel_size**2)
        attn_1 = torch.empty_like(attn_0)
        libnatten.na2d_qk_forward(
            attn_0, query_t, key_p, None, ctx.kernel_size, ctx.dilation
        )
        libnatten.na2d_qk_forward(
            attn_1, query_p, key_t, None, ctx.kernel_size, ctx.dilation
        )
        return attn_0 + attn_1

    @staticmethod
    @custom_bwd
    def backward(
        ctx, grad_out: Tensor
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], NoneType, NoneType]:
        query, key, bias = ctx.saved_tensors
        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        # dbias has to be zero filled
        d_bias = None if bias is None else torch.zeros_like(bias)
        libnatten.na2d_qk_backward(
            d_query,
            d_key,
            d_bias,
            grad_out.contiguous(),
            query,
            key,
            ctx.kernel_size,
            ctx.dilation,
        )
        return d_query, d_key, d_bias, None, None


class NeighborhoodAttention2DAVAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx, attn: Tensor, value: Tensor, kernel_size: int, dilation: int
    ) -> Tensor:
        attn = attn.contiguous()
        value = value.contiguous()
        out = torch.empty_like(value)
        libnatten.na2d_av_forward(out, attn, value, kernel_size, dilation)
        ctx.save_for_backward(attn, value)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        return out

    @staticmethod
    def jvp(ctx, *grad_inputs: Any) -> Tensor:
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the AV operation is:
        av(attn.tangent, value.primal) + av(attn.primal, value.tangent)
        """
        assert len(grad_inputs) == 4
        attn_t: Tensor = grad_inputs[0]
        value_t: Tensor = grad_inputs[1]

        attn_p, value_p = ctx.to_save
        attn_t = attn_t.contiguous()
        value_t = value_t.contiguous()
        out_0 = torch.empty_like(value_p)
        out_1 = torch.empty_like(out_0)
        libnatten.na2d_av_forward(out_0, attn_t, value_p, ctx.kernel_size, ctx.dilation)
        libnatten.na2d_av_forward(out_1, attn_p, value_t, ctx.kernel_size, ctx.dilation)
        return out_0 + out_1

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out: Tensor) -> Tuple[Tensor, Tensor, NoneType, NoneType]:
        attn, value = ctx.saved_tensors
        d_attn = torch.empty_like(attn)
        d_value = torch.empty_like(value)
        libnatten.na2d_av_backward(
            d_attn,
            d_value,
            grad_out.contiguous(),
            attn,
            value,
            ctx.kernel_size,
            ctx.dilation,
        )
        return d_attn, d_value, None, None


class NeighborhoodAttention3DQKAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        bias: Optional[Tensor],
        kernel_size_d: int,
        kernel_size: int,
        dilation_d: int,
        dilation: int,
    ) -> Tensor:
        query = query.contiguous()
        key = key.contiguous()
        if bias is not None:
            bias = bias.to(key.dtype)
        attn = make_attn_tensor_from_input(
            query, kernel_size * kernel_size * kernel_size_d
        )
        libnatten.na3d_qk_forward(
            attn, query, key, bias, kernel_size, dilation, kernel_size_d, dilation_d
        )
        ctx.save_for_backward(query, key, bias)
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
        assert len(grad_inputs) == 7
        query_t: Tensor = grad_inputs[0]
        key_t: Tensor = grad_inputs[1]
        bias: Optional[Tensor] = grad_inputs[2]

        if bias is not None:
            raise ValueError(
                "Positional biases are currently not supported "
                "in forward mode autodiff."
            )
        query_p, key_p, _ = ctx.to_save
        query_t = query_t.contiguous()
        key_t = key_t.contiguous()
        attn_0 = make_attn_tensor_from_input(
            query_t, ctx.kernel_size * ctx.kernel_size * ctx.kernel_size_d
        )
        attn_1 = torch.empty_like(attn_0)
        libnatten.na3d_qk_forward(
            attn_0,
            query_t,
            key_p,
            None,
            ctx.kernel_size,
            ctx.dilation,
            ctx.kernel_size_d,
            ctx.dilation_d,
        )
        libnatten.na3d_qk_forward(
            attn_1,
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
    def backward(
        ctx, grad_out: Tensor
    ) -> Tuple[
        Tensor, Tensor, Optional[Tensor], NoneType, NoneType, NoneType, NoneType
    ]:
        query, key, bias = ctx.saved_tensors
        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        # dbias has to be zero filled
        d_bias = None if bias is None else torch.zeros_like(bias)
        libnatten.na3d_qk_backward(
            d_query,
            d_key,
            d_bias,
            grad_out.contiguous(),
            query,
            key,
            ctx.kernel_size,
            ctx.dilation,
            ctx.kernel_size_d,
            ctx.dilation_d,
        )
        return d_query, d_key, d_bias, None, None, None, None


class NeighborhoodAttention3DAVAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        attn: Tensor,
        value: Tensor,
        kernel_size_d: int,
        kernel_size: int,
        dilation_d: int,
        dilation: int,
    ) -> Tensor:
        attn = attn.contiguous()
        value = value.contiguous()
        out = torch.empty_like(value)
        libnatten.na3d_av_forward(
            out, attn, value, kernel_size, dilation, kernel_size_d, dilation_d
        )
        ctx.save_for_backward(attn, value)
        ctx.kernel_size_d = kernel_size_d
        ctx.kernel_size = kernel_size
        ctx.dilation_d = dilation_d
        ctx.dilation = dilation
        return out

    @staticmethod
    def jvp(ctx, *grad_inputs: Any) -> Tensor:
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the AV operation is:
        av(attn.tangent, value.primal) + av(attn.primal, value.tangent)
        """
        assert len(grad_inputs) == 6
        attn_t: Tensor = grad_inputs[0]
        value_t: Tensor = grad_inputs[1]

        attn_p, value_p = ctx.to_save
        attn_t = attn_t.contiguous()
        value_t = value_t.contiguous()
        out_0 = torch.empty_like(value_p)
        out_1 = torch.empty_like(out_0)
        libnatten.na3d_av_forward(
            out_0,
            attn_t,
            value_p,
            ctx.kernel_size,
            ctx.dilation,
            ctx.kernel_size_d,
            ctx.dilation_d,
        )
        libnatten.na3d_av_forward(
            out_1,
            attn_p,
            value_t,
            ctx.kernel_size,
            ctx.dilation,
            ctx.kernel_size_d,
            ctx.dilation_d,
        )
        return out_0 + out_1

    @staticmethod
    @custom_bwd
    def backward(
        ctx, grad_out: Tensor
    ) -> Tuple[Tensor, Tensor, NoneType, NoneType, NoneType, NoneType]:
        attn, value = ctx.saved_tensors
        d_attn = torch.empty_like(attn)
        d_value = torch.empty_like(value)
        libnatten.na3d_av_backward(
            d_attn,
            d_value,
            grad_out.contiguous(),
            attn,
            value,
            ctx.kernel_size,
            ctx.dilation,
            ctx.kernel_size_d,
            ctx.dilation_d,
        )
        return d_attn, d_value, None, None, None, None


def na1d_qk(query: Tensor, key: Tensor, kernel_size: int, dilation: int) -> Tensor:
    if query.is_nested or key.is_nested:
        return na1d_qk_nested(query, key, None, kernel_size, dilation)
    return NeighborhoodAttention1DQKAutogradFunction.apply(
        query, key, None, kernel_size, dilation
    )


def na1d_qk_with_bias(
    query: Tensor, key: Tensor, bias: Optional[Tensor], kernel_size: int, dilation: int
) -> Tensor:
    if query.is_nested or key.is_nested:
        return na1d_qk_nested(query, key, bias, kernel_size, dilation)
    return NeighborhoodAttention1DQKAutogradFunction.apply(
        query, key, bias, kernel_size, dilation
    )


def na1d_av(attn: Tensor, value: Tensor, kernel_size: int, dilation: int) -> Tensor:
    if attn.is_nested or value.is_nested:
        return na1d_av_nested(attn, value, kernel_size, dilation)
    return NeighborhoodAttention1DAVAutogradFunction.apply(
        attn, value, kernel_size, dilation
    )


def na2d_qk(query: Tensor, key: Tensor, kernel_size: int, dilation: int) -> Tensor:
    if query.is_nested or key.is_nested:
        return na2d_qk_nested(query, key, None, kernel_size, dilation)
    return NeighborhoodAttention2DQKAutogradFunction.apply(
        query, key, None, kernel_size, dilation
    )


def na2d_qk_with_bias(
    query: Tensor, key: Tensor, bias: Optional[Tensor], kernel_size: int, dilation: int
) -> Tensor:
    if query.is_nested or key.is_nested:
        return na2d_qk_nested(query, key, bias, kernel_size, dilation)
    return NeighborhoodAttention2DQKAutogradFunction.apply(
        query, key, bias, kernel_size, dilation
    )


def na2d_av(attn: Tensor, value: Tensor, kernel_size: int, dilation: int) -> Tensor:
    if attn.is_nested or value.is_nested:
        return na2d_av_nested(attn, value, kernel_size, dilation)
    return NeighborhoodAttention2DAVAutogradFunction.apply(
        attn, value, kernel_size, dilation
    )


def na3d_qk_with_bias(
    query: Tensor,
    key: Tensor,
    bias: Optional[Tensor],
    kernel_size_d: int,
    kernel_size: int,
    dilation_d: int,
    dilation: int,
) -> Tensor:
    if query.is_nested or key.is_nested:
        return na3d_qk_nested(
            query, key, bias, kernel_size_d, kernel_size, dilation_d, dilation
        )
    return NeighborhoodAttention3DQKAutogradFunction.apply(
        query, key, bias, kernel_size_d, kernel_size, dilation_d, dilation
    )


def na3d_qk(
    query: Tensor,
    key: Tensor,
    kernel_size_d: int,
    kernel_size: int,
    dilation_d: int,
    dilation: int,
) -> Tensor:
    if query.is_nested or key.is_nested:
        return na3d_qk_nested(
            query, key, None, kernel_size_d, kernel_size, dilation_d, dilation
        )
    return NeighborhoodAttention3DQKAutogradFunction.apply(
        query, key, None, kernel_size_d, kernel_size, dilation_d, dilation
    )


def na3d_av(
    attn: Tensor,
    value: Tensor,
    kernel_size_d: int,
    kernel_size: int,
    dilation_d: int,
    dilation: int,
) -> Tensor:
    if attn.is_nested or value.is_nested:
        return na3d_av_nested(
            attn, value, kernel_size_d, kernel_size, dilation_d, dilation
        )
    return NeighborhoodAttention3DAVAutogradFunction.apply(
        attn, value, kernel_size_d, kernel_size, dilation_d, dilation
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
