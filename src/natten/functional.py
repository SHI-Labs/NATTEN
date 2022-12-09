"""
Neighborhood Attention Autograd Functions

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from . import backend


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
        attn = backend.natten1dqkrpb_forward(query, key, rpb, kernel_size, dilation)
        ctx.save_for_backward(query, key)
        ctx.dilation = dilation
        ctx.bias = rpb is not None
        return attn

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = backend.natten1dqkrpb_backward(
            grad_out.contiguous(),
            ctx.saved_tensors[0],
            ctx.saved_tensors[1],
            ctx.bias,
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
    def forward(ctx, attn, value, dilation):
        attn = attn.contiguous()
        value = value.contiguous()
        out = backend.natten1dav_forward(attn, value, dilation)
        ctx.save_for_backward(attn, value)
        ctx.dilation = dilation
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = backend.natten1dav_backward(
            grad_out.contiguous(),
            ctx.saved_tensors[0],
            ctx.saved_tensors[1],
            ctx.dilation,
        )
        d_attn, d_value = outputs
        return d_attn, d_value, None


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
        attn = backend.natten2dqkrpb_forward(query, key, rpb, kernel_size, dilation)
        ctx.save_for_backward(query, key)
        ctx.dilation = dilation
        ctx.bias = rpb is not None
        return attn

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = backend.natten2dqkrpb_backward(
            grad_out.contiguous(),
            ctx.saved_tensors[0],
            ctx.saved_tensors[1],
            ctx.bias,
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
    def forward(ctx, attn, value, dilation):
        attn = attn.contiguous()
        value = value.contiguous()
        out = backend.natten2dav_forward(attn, value, dilation)
        ctx.save_for_backward(attn, value)
        ctx.dilation = dilation
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        outputs = backend.natten2dav_backward(
            grad_out.contiguous(),
            ctx.saved_tensors[0],
            ctx.saved_tensors[1],
            ctx.dilation,
        )
        d_attn, d_value = outputs
        return d_attn, d_value, None


def natten1dqkrpb(query, key, rpb, kernel_size, dilation):
    return NATTEN1DQKRPBFunction.apply(query, key, rpb, kernel_size, dilation)


def natten1dav(attn, value, dilation):
    return NATTEN1DAVFunction.apply(attn, value, dilation)


def natten2dqkrpb(query, key, rpb, kernel_size, dilation):
    return NATTEN2DQKRPBFunction.apply(query, key, rpb, kernel_size, dilation)


def natten2dav(attn, value, dilation):
    return NATTEN2DAVFunction.apply(attn, value, dilation)
