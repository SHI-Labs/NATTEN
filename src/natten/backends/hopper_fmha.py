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

from .._libnatten import hopper_fmha_forward
from ..types import CutlassHopperFmhaForwardConfigType, KernelSchedule, NoneType
from ..utils.checks import fmha_tensor_checks

from .configs.checks import can_run_cutlass_hopper_fmha
from .configs.cutlass_hopper import check_cutlass_hopper_fmha_forward_config


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
    ) -> Tuple[Tensor, Tensor]:

        assert isinstance(
            scale, float
        ), f"Expected float attention scale, got {type(scale)}."

        if (
            not isinstance(forward_config, tuple)
            or len(forward_config) != 2
            or not (
                isinstance(forward_config[0], tuple)
                and len(forward_config[0]) == 2
                and all(isinstance(x, int) for x in forward_config[0])
                and isinstance(forward_config[1], KernelSchedule)
            )
        ):
            raise ValueError(
                "Invalid tiling config for Hopper FMHA; expected tuple of "
                f"two tuples: a tuple of two integers, and a kernel configuration type, got {forward_config=}."
            )

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

        return output, logsumexp

    @staticmethod
    @amp_bwd
    def backward(ctx, grad_out: Tensor, grad_lse: Tensor) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        NoneType,
        NoneType,
    ]:
        query, key, value, logsumexp, output = ctx.saved_tensors
        d_output = grad_out.contiguous()  # noqa: F841
        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        d_value = torch.empty_like(value)

        raise NotImplementedError("Hopper FMHA does not support backpropagation yet.")

        return d_query, d_key, d_value, None, None


def cutlass_hopper_fmha(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Optional[float] = None,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
    kernel_schedule: Optional[KernelSchedule] = None,
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

    scale = scale or query.shape[-1] ** -0.5

    output, lse = CutlassHopperFmhaAutogradFn.apply(
        query,
        key,
        value,
        scale,
        forward_config,
    )

    if return_lse:
        return output, lse

    return output
