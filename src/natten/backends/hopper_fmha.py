#################################################################################################
# Copyright (c) 2022 - 2026 Ali Hassani.
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
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.amp import custom_bwd, custom_fwd
from torch.autograd import Function

amp_fwd = functools.partial(custom_fwd, device_type="cuda")
amp_bwd = functools.partial(custom_bwd, device_type="cuda")

from natten._libnatten import hopper_fmha_backward, hopper_fmha_forward
from natten.backends.configs.checks import can_run_cutlass_hopper_fmha
from natten.backends.configs.cutlass_hopper import (
    check_cutlass_hopper_fmha_backward_config,
    check_cutlass_hopper_fmha_forward_config,
)
from natten.types import (
    CutlassHopperFmhaBackwardConfigType,
    CutlassHopperFmhaForwardConfigType,
    KernelSchedule,
    NoneType,
)
from natten.utils import log
from natten.utils.checks import fmha_tensor_checks, varlen_tensor_checks

logger = log.get_logger(__name__)


class CutlassHopperFmhaAutogradFn(Function):
    @staticmethod
    @amp_fwd
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool,
        scale: float,
        forward_config: CutlassHopperFmhaForwardConfigType,
        backward_config: CutlassHopperFmhaBackwardConfigType,
        cumulative_seqlen_Q: Optional[Tensor],
        cumulative_seqlen_KV: Optional[Tensor],
        max_seqlen_Q: int,
        max_seqlen_KV: int,
    ) -> Tuple[Tensor, Tensor]:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        (q_tile_size, kv_tile_size), kernel_schedule = forward_config

        output, logsumexp = hopper_fmha_forward(
            query,
            key,
            value,
            is_causal,
            scale,
            q_tile_size,
            kv_tile_size,
            kernel_schedule.value,  # TODO: I don't like this -- write a map with checks?
            cumulative_seqlen_Q,
            cumulative_seqlen_KV,
            max_seqlen_Q,
            max_seqlen_KV,
        )

        ctx.save_for_backward(
            query,
            key,
            value,
            logsumexp,
            output,
            cumulative_seqlen_Q,
            cumulative_seqlen_KV,
        )
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.max_seqlen_Q = max_seqlen_Q
        ctx.max_seqlen_KV = max_seqlen_KV
        ctx.backward_config = backward_config
        # Always record determinism behavior during forward pass (forward pass itself is
        # deterministic anyway).
        # Determinism could be limited to part of the program, which means during forward pass
        # it'll be true, but on .backward() call, if it's been turned off, it will stay off when we
        # get to this operation's backward call.
        ctx.deterministic = torch.are_deterministic_algorithms_enabled()

        return output, logsumexp

    @staticmethod
    @amp_bwd
    def backward(ctx, grad_out: Tensor, grad_lse: Tensor) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
        NoneType,
    ]:
        (
            query,
            key,
            value,
            logsumexp,
            output,
            cumulative_seqlen_Q,
            cumulative_seqlen_KV,
        ) = ctx.saved_tensors
        d_output = grad_out.contiguous()  # noqa: F841

        q_tile_size, k_tile_size = ctx.backward_config

        if ctx.deterministic:
            raise RuntimeError(
                "Hopper FMHA backward pass does not have a deterministic mode, "
                "but PyTorch's deterministic algorithms were enabled. To proceed, "
                "you must either disable torch's deterministic mode, or choose a "
                "different backend."
            )

        d_query, d_key, d_value = hopper_fmha_backward(
            query,
            key,
            value,
            output,
            d_output,
            logsumexp,
            ctx.is_causal,
            ctx.scale,
            q_tile_size,
            k_tile_size,
            cumulative_seqlen_Q,
            cumulative_seqlen_KV,
            ctx.max_seqlen_Q,
            ctx.max_seqlen_KV,
        )

        return d_query, d_key, d_value, None, None, None, None, None, None, None, None


def cutlass_hopper_fmha(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
    kernel_schedule: Optional[KernelSchedule] = None,
    backward_q_tile_size: Optional[int] = None,
    backward_kv_tile_size: Optional[int] = None,
    return_lse: bool = False,
    # varlen parameters
    cumulative_seqlen_Q: Optional[Tensor] = None,
    cumulative_seqlen_KV: Optional[Tensor] = None,
    max_seqlen_Q: int = 0,
    max_seqlen_KV: int = 0,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:

    fmha_tensor_checks(
        query,
        key,
        value,
        must_match_head_dims=True,
        supports_gqa_mqa=True,
        backend_name="Hopper FMHA",
    )

    (
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        max_seqlen_Q,
        max_seqlen_KV,
    ) = varlen_tensor_checks(
        query=query,
        key=key,
        value=value,
        cumulative_seqlen_Q=cumulative_seqlen_Q,
        cumulative_seqlen_KV=cumulative_seqlen_KV,
        max_seqlen_Q=max_seqlen_Q,
        max_seqlen_KV=max_seqlen_KV,
    )
    is_varlen = cumulative_seqlen_Q is not None

    assert can_run_cutlass_hopper_fmha(
        query, key, value, is_causal=is_causal, is_varlen=is_varlen, raise_error=True
    )

    forward_config = check_cutlass_hopper_fmha_forward_config(
        input_tensor=query,
        q_tile_size=q_tile_size,
        kv_tile_size=kv_tile_size,
        kernel_schedule=kernel_schedule,
    )
    backward_config = check_cutlass_hopper_fmha_backward_config(
        input_tensor=query,
        q_tile_size=backward_q_tile_size,
        kv_tile_size=backward_kv_tile_size,
    )

    scale = scale or query.shape[-1] ** -0.5

    # GQA/MQA is not supported by the kernel; only allowed via graph transform
    is_gqa = query.shape[-2] != key.shape[-2]
    if is_gqa:
        heads = query.shape[-2]
        heads_kv = key.shape[-2]
        assert key.shape[-2] == value.shape[-2]
        assert heads >= heads_kv
        assert heads % heads_kv == 0
        h_k = heads // heads_kv

        key = torch.repeat_interleave(key, repeats=h_k, dim=-2, output_size=heads)
        value = torch.repeat_interleave(value, repeats=h_k, dim=-2, output_size=heads)

    output, lse = CutlassHopperFmhaAutogradFn.apply(
        query,
        key,
        value,
        is_causal,
        scale,
        forward_config,
        backward_config,
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        max_seqlen_Q,
        max_seqlen_KV,
    )

    if return_lse:
        return output, lse

    return output
