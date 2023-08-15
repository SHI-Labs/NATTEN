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
        attn = _C.na1d_qk_forward(query, key, rpb, kernel_size, dilation)
        ctx.save_for_backward(query, key)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.bias = rpb is not None
        return attn

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = _C.na1d_qk_backward(
            grad_out.contiguous(),
            ctx.saved_tensors[0],
            ctx.saved_tensors[1],
            ctx.bias,
            ctx.kernel_size,
            ctx.dilation,
        )
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb, None, None


class NeighborhoodAttention1DAVAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, attn, value, kernel_size, dilation):
        attn = attn.contiguous()
        value = value.contiguous()
        out = _C.na1d_av_forward(attn, value, kernel_size, dilation)
        ctx.save_for_backward(attn, value)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = _C.na1d_av_backward(
            grad_out.contiguous(),
            ctx.saved_tensors[0],
            ctx.saved_tensors[1],
            ctx.kernel_size,
            ctx.dilation,
        )
        d_attn, d_value = outputs
        return d_attn, d_value, None, None


class NeighborhoodAttention2DQKAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, query, key, rpb, kernel_size, dilation):
        query = query.contiguous()
        key = key.contiguous()
        if rpb is not None:
            rpb = rpb.to(key.dtype)
        attn = _C.na2d_qk_forward(query, key, rpb, kernel_size, dilation)
        ctx.save_for_backward(query, key)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.bias = rpb is not None
        return attn

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = _C.na2d_qk_backward(
            grad_out.contiguous(),
            ctx.saved_tensors[0],
            ctx.saved_tensors[1],
            ctx.bias,
            ctx.kernel_size,
            ctx.dilation,
        )
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb, None, None


class NeighborhoodAttention2DAVAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, attn, value, kernel_size, dilation):
        attn = attn.contiguous().to(value.dtype)
        value = value.contiguous()
        out = _C.na2d_av_forward(attn, value, kernel_size, dilation)
        ctx.save_for_backward(attn, value)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = _C.na2d_av_backward(
            grad_out.contiguous(),
            ctx.saved_tensors[0],
            ctx.saved_tensors[1],
            ctx.kernel_size,
            ctx.dilation,
        )
        d_attn, d_value = outputs
        return d_attn, d_value, None, None


class NeighborhoodAttention3DQKAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, query, key, rpb, kernel_size_d, kernel_size, dilation_d, dilation):
        query = query.contiguous()
        key = key.contiguous()
        attn = _C.na3d_qk_forward(
            query, key, rpb, kernel_size, dilation, kernel_size_d, dilation_d
        )
        ctx.save_for_backward(query, key)
        ctx.kernel_size_d = kernel_size_d
        ctx.kernel_size = kernel_size
        ctx.dilation_d = dilation_d
        ctx.dilation = dilation
        ctx.bias = rpb is not None
        return attn

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = _C.na3d_qk_backward(
            grad_out.contiguous(),
            ctx.saved_tensors[0],
            ctx.saved_tensors[1],
            ctx.bias,
            ctx.kernel_size,
            ctx.dilation,
            ctx.kernel_size_d,
            ctx.dilation_d,
        )
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb, None, None, None, None


class NeighborhoodAttention3DAVAutogradFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, attn, value, kernel_size_d, kernel_size, dilation_d, dilation):
        attn = attn.contiguous()
        value = value.contiguous()
        out = _C.na3d_av_forward(
            attn, value, kernel_size, dilation, kernel_size_d, dilation_d
        )
        ctx.save_for_backward(attn, value)
        ctx.kernel_size_d = kernel_size_d
        ctx.kernel_size = kernel_size
        ctx.dilation_d = dilation_d
        ctx.dilation = dilation
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = _C.na3d_av_backward(
            grad_out.contiguous(),
            ctx.saved_tensors[0],
            ctx.saved_tensors[1],
            ctx.kernel_size,
            ctx.dilation,
            ctx.kernel_size_d,
            ctx.dilation_d,
        )
        d_attn, d_value = outputs
        return d_attn, d_value, None, None, None, None


def natten1dqkrpb(query, key, rpb, kernel_size, dilation):
    return NeighborhoodAttention1DQKAutogradFunction.apply(
        query, key, rpb, kernel_size, dilation
    )


def natten1dqk(query, key, kernel_size, dilation):
    return NeighborhoodAttention1DQKAutogradFunction.apply(
        query, key, None, kernel_size, dilation
    )


def natten1dav(attn, value, kernel_size, dilation):
    return NeighborhoodAttention1DAVAutogradFunction.apply(
        attn, value, kernel_size, dilation
    )


def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
    return NeighborhoodAttention2DQKAutogradFunction.apply(
        query, key, rpb, kernel_size, dilation
    )


def natten2dqk(query, key, kernel_size, dilation):
    return NeighborhoodAttention2DQKAutogradFunction.apply(
        query, key, None, kernel_size, dilation
    )


def natten2dav(attn, value, kernel_size, dilation):
    return NeighborhoodAttention2DAVAutogradFunction.apply(
        attn, value, kernel_size, dilation
    )


def natten3dqkrpb(query, key, rpb, kernel_size_d, kernel_size, dilation_d, dilation):
    return NeighborhoodAttention3DQKAutogradFunction.apply(
        query, key, rpb, kernel_size_d, kernel_size, dilation_d, dilation
    )


def natten3dqk(query, key, kernel_size_d, kernel_size, dilation_d, dilation):
    return NeighborhoodAttention3DQKAutogradFunction.apply(
        query, key, None, kernel_size_d, kernel_size, dilation_d, dilation
    )


def natten3dav(attn, value, kernel_size_d, kernel_size, dilation_d, dilation):
    return NeighborhoodAttention3DAVAutogradFunction.apply(
        attn, value, kernel_size_d, kernel_size, dilation_d, dilation
    )
