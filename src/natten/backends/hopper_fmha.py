#################################################################################################
# Copyright (c) 2022-2025 Ali Hassani.
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

from .._libnatten import hopper_fmha_backward, hopper_fmha_forward
from ..types import (
    CutlassHopperFmhaBackwardConfigType,
    CutlassHopperFmhaForwardConfigType,
    KernelSchedule,
    NoneType,
)
from ..utils import log
from ..utils.checks import fmha_tensor_checks

from .configs.checks import can_run_cutlass_hopper_fmha
from .configs.cutlass_hopper import (
    check_cutlass_hopper_fmha_backward_config,
    check_cutlass_hopper_fmha_forward_config,
)

logger = log.get_logger(__name__)


class CutlassHopperFmhaAutogradFn(Function):
    @staticmethod
    @amp_fwd
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: float,
        forward_config: CutlassHopperFmhaForwardConfigType,
        backward_config: CutlassHopperFmhaBackwardConfigType,
    ) -> Tuple[Tensor, Tensor]:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        output = torch.empty_like(query)

        logsumexp = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )

        (q_tile_size, kv_tile_size), kernel_schedule = forward_config
        hopper_fmha_forward(
            output,
            query,
            key,
            value,
            logsumexp,
            scale,
            q_tile_size,
            kv_tile_size,
            kernel_schedule.value,  # TODO: I don't like this -- write a map with checks?
        )

        ctx.save_for_backward(query, key, value, logsumexp, output)
        ctx.scale = scale
        ctx.backward_config = backward_config

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
    ]:
        query, key, value, logsumexp, output = ctx.saved_tensors
        d_output = grad_out.contiguous()  # noqa: F841
        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        d_value = torch.empty_like(value)

        q_tile_size, k_tile_size = ctx.backward_config

        if torch.are_deterministic_algorithms_enabled():
            raise RuntimeError(
                "Hopper FMHA backward pass does not have a deterministic mode, "
                "but PyTorch's deterministic algorithms were enabled. To proceed, "
                "you must either disable torch's deterministic mode, or choose a "
                "different backend."
            )

        # TODO: change the layout in forward pass so we can skip this!
        # Same for padding.
        logsumexp = logsumexp.transpose(-2, -1).contiguous()

        # Padding is required since TMA loads in LSE
        assert query.dtype in [torch.float16, torch.bfloat16]
        elem_bytes = 2
        tma_alignment_bytes = 16 // elem_bytes
        requires_padding = logsumexp.shape[-1] % tma_alignment_bytes != 0
        seq_q_padding = tma_alignment_bytes - (
            logsumexp.shape[-1] % tma_alignment_bytes
        )

        if requires_padding:
            old_shape = logsumexp.shape
            logsumexp = torch.nn.functional.pad(
                logsumexp, (0, seq_q_padding), "constant", 0
            )
            logger.debug(
                f"Padded logsumexp with shape {old_shape} to {logsumexp.shape}."
            )

        hopper_fmha_backward(
            d_query,
            d_key,
            d_value,
            query,
            key,
            value,
            output,
            d_output,
            logsumexp,
            ctx.scale,
            q_tile_size,
            k_tile_size,
        )

        return d_query, d_key, d_value, None, None, None


def cutlass_hopper_fmha(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Optional[float] = None,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
    kernel_schedule: Optional[KernelSchedule] = None,
    backward_q_tile_size: Optional[int] = None,
    backward_kv_tile_size: Optional[int] = None,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:

    fmha_tensor_checks(query, key, value, must_match_head_dims=True)

    assert can_run_cutlass_hopper_fmha(query, key, value, raise_error=True)

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

    output, lse = CutlassHopperFmhaAutogradFn.apply(
        query,
        key,
        value,
        scale,
        forward_config,
        backward_config,
    )

    if return_lse:
        return output, lse

    return output
