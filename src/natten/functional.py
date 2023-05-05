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


class NATTEN1DQKRPBFunction(Function):
    """
    1D QK+RPB autograd function
    Computes neighborhood attention weights given queries and keys,
    and adds relative positional biases.
    This calls the `QKRPB` kernel.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, query, key, rpb, kernel_size, dilation):
        query = query.contiguous()
        key = key.contiguous()
        attn = _C.natten1dqkrpb_forward(query, key, rpb, kernel_size, dilation)
        ctx.save_for_backward(query, key)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.bias = rpb is not None
        return attn

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = _C.natten1dqkrpb_backward(
            grad_out.contiguous(),
            ctx.saved_tensors[0],
            ctx.saved_tensors[1],
            ctx.bias,
            ctx.kernel_size,
            ctx.dilation,
        )
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb, None, None


class NATTEN1DAVFunction(Function):
    """
    1D AV autograd function
    Computes neighborhood attention outputs given attention weights, and values.
    This calls the `AV` kernel.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, attn, value, kernel_size, dilation):
        attn = attn.contiguous()
        value = value.contiguous()
        out = _C.natten1dav_forward(attn, value, kernel_size, dilation)
        ctx.save_for_backward(attn, value)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = _C.natten1dav_backward(
            grad_out.contiguous(),
            ctx.saved_tensors[0],
            ctx.saved_tensors[1],
            ctx.kernel_size,
            ctx.dilation,
        )
        d_attn, d_value = outputs
        return d_attn, d_value, None, None


class NATTEN2DQKRPBFunction(Function):
    """
    2D QK+RPB autograd function
    Computes neighborhood attention weights given queries and keys,
    and adds relative positional biases.
    This calls the `QKRPB` kernel.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, query, key, rpb, kernel_size, dilation):
        query = query.contiguous()
        key = key.contiguous()
        attn = _C.natten2dqkrpb_forward(query, key, rpb, kernel_size, dilation)
        ctx.save_for_backward(query, key)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        ctx.bias = rpb is not None
        return attn

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = _C.natten2dqkrpb_backward(
            grad_out.contiguous(),
            ctx.saved_tensors[0],
            ctx.saved_tensors[1],
            ctx.bias,
            ctx.kernel_size,
            ctx.dilation,
        )
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb, None, None


class NATTEN2DAVFunction(Function):
    """
    2D AV autograd function
    Computes neighborhood attention outputs given attention weights, and values.
    This calls the `AV` kernel.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, attn, value, kernel_size, dilation):
        attn = attn.contiguous()
        value = value.contiguous()
        out = _C.natten2dav_forward(attn, value, kernel_size, dilation)
        ctx.save_for_backward(attn, value)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = _C.natten2dav_backward(
            grad_out.contiguous(),
            ctx.saved_tensors[0],
            ctx.saved_tensors[1],
            ctx.kernel_size,
            ctx.dilation,
        )
        d_attn, d_value = outputs
        return d_attn, d_value, None, None


class NATTEN3DQKRPBFunction(Function):
    """
    3D QK+RPB autograd function
    Computes neighborhood attention weights given queries and keys,
    and adds relative positional biases.
    This calls the `QKRPB` kernel.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, query, key, rpb, kernel_size_d, kernel_size, dilation_d, dilation):
        query = query.contiguous()
        key = key.contiguous()
        attn = _C.natten3dqkrpb_forward(
            query, key, rpb, kernel_size_d, kernel_size, dilation_d, dilation
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
        outputs = _C.natten3dqkrpb_backward(
            grad_out.contiguous(),
            ctx.saved_tensors[0],
            ctx.saved_tensors[1],
            ctx.bias,
            ctx.kernel_size_d,
            ctx.kernel_size,
            ctx.dilation_d,
            ctx.dilation,
        )
        d_query, d_key, d_rpb = outputs
        return d_query, d_key, d_rpb, None, None, None, None


class NATTEN3DAVFunction(Function):
    """
    3D AV autograd function
    Computes neighborhood attention outputs given attention weights, and values.
    This calls the `AV` kernel.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, attn, value, kernel_size_d, kernel_size, dilation_d, dilation):
        attn = attn.contiguous()
        value = value.contiguous()
        out = _C.natten3dav_forward(
            attn, value, kernel_size_d, kernel_size, dilation_d, dilation
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
        outputs = _C.natten3dav_backward(
            grad_out.contiguous(),
            ctx.saved_tensors[0],
            ctx.saved_tensors[1],
            ctx.kernel_size_d,
            ctx.kernel_size,
            ctx.dilation_d,
            ctx.dilation,
        )
        d_attn, d_value = outputs
        return d_attn, d_value, None, None, None, None


def natten1dqkrpb(query, key, rpb, kernel_size, dilation):
    return NATTEN1DQKRPBFunction.apply(query, key, rpb, kernel_size, dilation)


def natten1dqk(query, key, kernel_size, dilation):
    return NATTEN1DQKRPBFunction.apply(query, key, None, kernel_size, dilation)


def natten1dav(attn, value, kernel_size, dilation):
    return NATTEN1DAVFunction.apply(attn, value, kernel_size, dilation)


def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
    return NATTEN2DQKRPBFunction.apply(query, key, rpb, kernel_size, dilation)


def natten2dqk(query, key, kernel_size, dilation):
    return NATTEN2DQKRPBFunction.apply(query, key, None, kernel_size, dilation)


def natten2dav(attn, value, kernel_size, dilation):
    return NATTEN2DAVFunction.apply(attn, value, kernel_size, dilation)


def natten3dqkrpb(query, key, rpb, kernel_size_d, kernel_size, dilation_d, dilation):
    return NATTEN3DQKRPBFunction.apply(
        query, key, rpb, kernel_size_d, kernel_size, dilation_d, dilation
    )


def natten3dqk(query, key, kernel_size_d, kernel_size, dilation_d, dilation):
    return NATTEN3DQKRPBFunction.apply(
        query, key, None, kernel_size_d, kernel_size, dilation_d, dilation
    )


def natten3dav(attn, value, kernel_size_d, kernel_size, dilation_d, dilation):
    return NATTEN3DAVFunction.apply(
        attn, value, kernel_size_d, kernel_size, dilation_d, dilation
    )
