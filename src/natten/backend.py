"""
Neighborhood Attention Autograd Functions

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    from natten import _C
except ImportError:
    raise ImportError(
        f"Failed to import NATTEN's Cython backend. "
        + f"This could be due to an invalid/incomplete install. "
        + f"Please uninstall NATTEN (pip uninstall natten) and re-install with the"
        f" correct torch build: "
        + f"shi-labs.com/natten"
    )


def natten1dqkrpb_forward(query, key, rpb, kernel_size, dilation):
    if query.is_cuda:
        with torch.cuda.device(query.device):
            return _C.natten1dqkrpb_forward(query, key, rpb, kernel_size, dilation)
    return _C.natten1dqkrpb_forward(query, key, rpb, kernel_size, dilation)


def natten2dqkrpb_forward(query, key, rpb, kernel_size, dilation):
    if query.is_cuda:
        with torch.cuda.device(query.device):
            return _C.natten2dqkrpb_forward(query, key, rpb, kernel_size, dilation)
    return _C.natten2dqkrpb_forward(query, key, rpb, kernel_size, dilation)


def natten1dav_forward(attn, value, dilation):
    if attn.is_cuda:
        with torch.cuda.device(attn.device):
            return _C.natten1dav_forward(attn, value, dilation)
    return _C.natten1dav_forward(attn, value, dilation)


def natten2dav_forward(attn, value, dilation):
    if attn.is_cuda:
        with torch.cuda.device(attn.device):
            return _C.natten2dav_forward(attn, value, dilation)
    return _C.natten2dav_forward(attn, value, dilation)


def natten1dqkrpb_backward(d_attn, query, key, has_bias, dilation):
    if query.is_cuda:
        with torch.cuda.device(query.device):
            return _C.natten1dqkrpb_backward(d_attn, query, key, has_bias, dilation)
    return _C.natten1dqkrpb_backward(d_attn, query, key, has_bias, dilation)


def natten2dqkrpb_backward(d_attn, query, key, has_bias, dilation):
    if query.is_cuda:
        with torch.cuda.device(query.device):
            return _C.natten2dqkrpb_backward(d_attn, query, key, has_bias, dilation)
    return _C.natten2dqkrpb_backward(d_attn, query, key, has_bias, dilation)


def natten1dav_backward(d_out, attn, value, dilation):
    if attn.is_cuda:
        with torch.cuda.device(attn.device):
            return _C.natten1dav_backward(d_out, attn, value, dilation)
    return _C.natten1dav_backward(d_out, attn, value, dilation)


def natten2dav_backward(d_out, attn, value, dilation):
    if attn.is_cuda:
        with torch.cuda.device(attn.device):
            return _C.natten2dav_backward(d_out, attn, value, dilation)
    return _C.natten2dav_backward(d_out, attn, value, dilation)
