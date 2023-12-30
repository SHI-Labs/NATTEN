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
import torch
from torch import Tensor
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    from natten import _C
except ImportError:
    raise ImportError(
        f"Failed to import NATTEN's CPP backend. "
        + f"This could be due to an invalid/incomplete install. "
        + f"Please uninstall NATTEN (pip uninstall natten) and re-install with the"
        f" correct torch build: "
        + f"shi-labs.com/natten"
    )

from .nested import (na1d_av_nested, na1d_qk_nested, na2d_av_nested,
                     na2d_qk_nested, na3d_av_nested, na3d_qk_nested)
from .utils import make_attn_tensor_from_input


def has_cuda():
    return _C.has_cuda()


def has_half():
    return _C.has_half()


def has_bfloat():
    return _C.has_bfloat()


def has_gemm():
    return _C.has_gemm()


def enable_tf32():
    return _C.set_gemm_tf32(True)


def disable_tf32():
    return _C.set_gemm_tf32(False)


def enable_tiled_na():
    return _C.set_tiled_na(True)


def disable_tiled_na():
    return _C.set_tiled_na(False)


def enable_gemm_na():
    return _C.set_gemm_na(True)


def disable_gemm_na():
    return _C.set_gemm_na(False)


class NeighborhoodAttention1DQKAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, query, key, rpb, kernel_size, dilation):
        query = query.contiguous()
        key = key.contiguous()
        if rpb is not None:
            rpb = rpb.to(key.dtype)
        attn = make_attn_tensor_from_input(query, kernel_size)
        _C.na1d_qk_forward(attn, query, key, rpb, kernel_size, dilation)
        ctx.save_for_backward(query, key, rpb)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        return attn

    @staticmethod
    def jvp(ctx, query_t, key_t, rpb, kernel_size, dilation):
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the QK operation is:
        qk(query.tangent, key.primal) + qk(query.primal, key.tangent)
        """
        if rpb is not None:
            raise ValueError(
                "Positional biases are currently not supported "
                "in forward mode autodiff."
            )
        query_p, key_p, _ = ctx.to_save
        query_t = query_t.contiguous()
        key_t = key_t.contiguous()
        attn_0 = make_attn_tensor_from_input(query_t, ctx.kernel_size)
        attn_1 = torch.empty_like(attn_0)
        _C.na1d_qk_forward(attn_0, query_t, key_p, None, ctx.kernel_size, ctx.dilation)
        _C.na1d_qk_forward(attn_1, query_p, key_t, None, ctx.kernel_size, ctx.dilation)
        return attn_0 + attn_1

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        query, key, rpb = ctx.saved_tensors
        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        # dRPB has to be zero filled
        d_rpb = None if rpb is None else torch.zeros_like(rpb)
        _C.na1d_qk_backward(
            d_query,
            d_key,
            d_rpb,
            grad_out.contiguous(),
            query,
            key,
            ctx.kernel_size,
            ctx.dilation,
        )
        return d_query, d_key, d_rpb, None, None


class NeighborhoodAttention1DAVAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, attn, value, kernel_size, dilation):
        attn = attn.contiguous()
        value = value.contiguous()
        out = torch.empty_like(value)
        _C.na1d_av_forward(out, attn, value, kernel_size, dilation)
        ctx.save_for_backward(attn, value)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        return out

    @staticmethod
    def jvp(ctx, attn_t, value_t, kernel_size, dilation):
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the AV operation is:
        av(attn.tangent, value.primal) + av(attn.primal, value.tangent)
        """
        attn_p, value_p = ctx.to_save
        attn_t = attn_t.contiguous()
        value_t = value_t.contiguous()
        out_0 = torch.empty_like(value_p)
        out_1 = torch.empty_like(out_0)
        _C.na1d_av_forward(out_0, attn_t, value_p, ctx.kernel_size, ctx.dilation)
        _C.na1d_av_forward(out_1, attn_p, value_t, ctx.kernel_size, ctx.dilation)
        return out_0 + out_1

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        attn, value = ctx.saved_tensors
        d_attn = torch.empty_like(attn)
        d_value = torch.empty_like(value)
        _C.na1d_av_backward(
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
    def forward(ctx, query, key, rpb, kernel_size, dilation):
        query = query.contiguous()
        key = key.contiguous()
        if rpb is not None:
            rpb = rpb.to(key.dtype)
        attn = make_attn_tensor_from_input(query, kernel_size**2)
        _C.na2d_qk_forward(attn, query, key, rpb, kernel_size, dilation)
        ctx.save_for_backward(query, key, rpb)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        return attn

    @staticmethod
    def jvp(ctx, query_t, key_t, rpb, kernel_size, dilation):
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the QK operation is:
        qk(query.tangent, key.primal) + qk(query.primal, key.tangent)
        """
        if rpb is not None:
            raise ValueError(
                "Positional biases are currently not supported "
                "in forward mode autodiff."
            )
        query_p, key_p, _ = ctx.to_save
        query_t = query_t.contiguous()
        key_t = key_t.contiguous()
        attn_0 = make_attn_tensor_from_input(query_t, ctx.kernel_size**2)
        attn_1 = torch.empty_like(attn_0)
        _C.na2d_qk_forward(attn_0, query_t, key_p, None, ctx.kernel_size, ctx.dilation)
        _C.na2d_qk_forward(attn_1, query_p, key_t, None, ctx.kernel_size, ctx.dilation)
        return attn_0 + attn_1

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        query, key, rpb = ctx.saved_tensors
        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        # dRPB has to be zero filled
        d_rpb = None if rpb is None else torch.zeros_like(rpb)
        _C.na2d_qk_backward(
            d_query,
            d_key,
            d_rpb,
            grad_out.contiguous(),
            query,
            key,
            ctx.kernel_size,
            ctx.dilation,
        )
        return d_query, d_key, d_rpb, None, None


class NeighborhoodAttention2DAVAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, attn, value, kernel_size, dilation):
        attn = attn.contiguous()
        value = value.contiguous()
        out = torch.empty_like(value)
        _C.na2d_av_forward(out, attn, value, kernel_size, dilation)
        ctx.save_for_backward(attn, value)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        return out

    @staticmethod
    def jvp(ctx, attn_t, value_t, kernel_size, dilation):
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the AV operation is:
        av(attn.tangent, value.primal) + av(attn.primal, value.tangent)
        """
        attn_p, value_p = ctx.to_save
        attn_t = attn_t.contiguous()
        value_t = value_t.contiguous()
        out_0 = torch.empty_like(value_p)
        out_1 = torch.empty_like(out_0)
        _C.na2d_av_forward(out_0, attn_t, value_p, ctx.kernel_size, ctx.dilation)
        _C.na2d_av_forward(out_1, attn_p, value_t, ctx.kernel_size, ctx.dilation)
        return out_0 + out_1

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        attn, value = ctx.saved_tensors
        d_attn = torch.empty_like(attn)
        d_value = torch.empty_like(value)
        _C.na2d_av_backward(
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
    def forward(ctx, query, key, rpb, kernel_size_d, kernel_size, dilation_d, dilation):
        query = query.contiguous()
        key = key.contiguous()
        if rpb is not None:
            rpb = rpb.to(key.dtype)
        attn = make_attn_tensor_from_input(
            query, kernel_size * kernel_size * kernel_size_d
        )
        _C.na3d_qk_forward(
            attn, query, key, rpb, kernel_size, dilation, kernel_size_d, dilation_d
        )
        ctx.save_for_backward(query, key, rpb)
        ctx.kernel_size_d = kernel_size_d
        ctx.kernel_size = kernel_size
        ctx.dilation_d = dilation_d
        ctx.dilation = dilation
        return attn

    @staticmethod
    def jvp(ctx, query_t, key_t, rpb, kernel_size_d, kernel_size, dilation_d, dilation):
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the QK operation is:
        qk(query.tangent, key.primal) + qk(query.primal, key.tangent)
        """
        if rpb is not None:
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
        _C.na3d_qk_forward(
            attn_0,
            query_t,
            key_p,
            None,
            ctx.kernel_size,
            ctx.dilation,
            ctx.kernel_size_d,
            ctx.dilation_d,
        )
        _C.na3d_qk_forward(
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
    def backward(ctx, grad_out):
        query, key, rpb = ctx.saved_tensors
        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        # dRPB has to be zero filled
        d_rpb = None if rpb is None else torch.zeros_like(rpb)
        _C.na3d_qk_backward(
            d_query,
            d_key,
            d_rpb,
            grad_out.contiguous(),
            query,
            key,
            ctx.kernel_size,
            ctx.dilation,
            ctx.kernel_size_d,
            ctx.dilation_d,
        )
        return d_query, d_key, d_rpb, None, None, None, None


class NeighborhoodAttention3DAVAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, attn, value, kernel_size_d, kernel_size, dilation_d, dilation):
        attn = attn.contiguous()
        value = value.contiguous()
        out = torch.empty_like(value)
        _C.na3d_av_forward(
            out, attn, value, kernel_size, dilation, kernel_size_d, dilation_d
        )
        ctx.save_for_backward(attn, value)
        ctx.kernel_size_d = kernel_size_d
        ctx.kernel_size = kernel_size
        ctx.dilation_d = dilation_d
        ctx.dilation = dilation
        return out

    @staticmethod
    def jvp(ctx, attn_t, value_t, kernel_size_d, kernel_size, dilation_d, dilation):
        """
        Forward-mode AD support was contributed by @crowsonkb and @Birch-san.
        The Jacobian vector product of the AV operation is:
        av(attn.tangent, value.primal) + av(attn.primal, value.tangent)
        """
        attn_p, value_p = ctx.to_save
        attn_t = attn_t.contiguous()
        value_t = value_t.contiguous()
        out_0 = torch.empty_like(value_p)
        out_1 = torch.empty_like(out_0)
        _C.na3d_av_forward(
            out_0,
            attn_t,
            value_p,
            ctx.kernel_size,
            ctx.dilation,
            ctx.kernel_size_d,
            ctx.dilation_d,
        )
        _C.na3d_av_forward(
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
    def backward(ctx, grad_out):
        attn, value = ctx.saved_tensors
        d_attn = torch.empty_like(attn)
        d_value = torch.empty_like(value)
        _C.na3d_av_backward(
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


def natten1dqkrpb(query, key, rpb, kernel_size, dilation):
    if query.is_nested or key.is_nested:
        return na1d_qk_nested(query, key, rpb, kernel_size, dilation)
    return NeighborhoodAttention1DQKAutogradFunction.apply(
        query, key, rpb, kernel_size, dilation
    )


def natten1dqk(query, key, kernel_size, dilation):
    if query.is_nested or key.is_nested:
        return na1d_qk_nested(query, key, None, kernel_size, dilation)
    return NeighborhoodAttention1DQKAutogradFunction.apply(
        query, key, None, kernel_size, dilation
    )


def natten1dav(attn, value, kernel_size, dilation):
    if attn.is_nested or value.is_nested:
        return na1d_av_nested(attn, value, kernel_size, dilation)
    return NeighborhoodAttention1DAVAutogradFunction.apply(
        attn, value, kernel_size, dilation
    )


def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
    if query.is_nested or key.is_nested:
        return na2d_qk_nested(query, key, rpb, kernel_size, dilation)
    return NeighborhoodAttention2DQKAutogradFunction.apply(
        query, key, rpb, kernel_size, dilation
    )


def natten2dqk(query, key, kernel_size, dilation):
    if query.is_nested or key.is_nested:
        return na2d_qk_nested(query, key, None, kernel_size, dilation)
    return NeighborhoodAttention2DQKAutogradFunction.apply(
        query, key, None, kernel_size, dilation
    )


def natten2dav(attn, value, kernel_size, dilation):
    if attn.is_nested or value.is_nested:
        return na2d_av_nested(attn, value, kernel_size, dilation)
    return NeighborhoodAttention2DAVAutogradFunction.apply(
        attn, value, kernel_size, dilation
    )


def natten3dqkrpb(query, key, rpb, kernel_size_d, kernel_size, dilation_d, dilation):
    if query.is_nested or key.is_nested:
        return na3d_qk_nested(
            query, key, rpb, kernel_size_d, kernel_size, dilation_d, dilation
        )
    return NeighborhoodAttention3DQKAutogradFunction.apply(
        query, key, rpb, kernel_size_d, kernel_size, dilation_d, dilation
    )


def natten3dqk(query, key, kernel_size_d, kernel_size, dilation_d, dilation):
    if query.is_nested or key.is_nested:
        return na3d_qk_nested(
            query, key, None, kernel_size_d, kernel_size, dilation_d, dilation
        )
    return NeighborhoodAttention3DQKAutogradFunction.apply(
        query, key, None, kernel_size_d, kernel_size, dilation_d, dilation
    )


def natten3dav(attn, value, kernel_size_d, kernel_size, dilation_d, dilation):
    if attn.is_nested or value.is_nested:
        return na3d_av_nested(
            attn, value, kernel_size_d, kernel_size, dilation_d, dilation
        )
    return NeighborhoodAttention3DAVAutogradFunction.apply(
        attn, value, kernel_size_d, kernel_size, dilation_d, dilation
    )
