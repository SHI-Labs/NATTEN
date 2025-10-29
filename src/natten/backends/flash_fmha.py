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

from .._libnatten import flash_fmha_backward, flash_fmha_forward
from ..types import (
    FlashFmhaBackwardConfigType,
    FlashFmhaForwardConfigType,
    NoneType,
)
from ..utils import log
from ..utils.checks import fmha_tensor_checks
from ..context import are_deterministic_algorithms_enabled as\
    natten_are_deterministic_algorithms_enabled

from .configs.checks import can_run_flash_fmha
from .configs.flash import (
    check_flash_fmha_backward_config,
    check_flash_fmha_forward_config,
)

logger = log.get_logger(__name__)


class FlashFmhaAutogradFn(Function):
    @staticmethod
    @amp_fwd
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: float,
        forward_config: FlashFmhaForwardConfigType,
        backward_config: FlashFmhaBackwardConfigType,
        deterministic: bool
    ) -> Tuple[Tensor, Tensor]:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        output = torch.empty_like(query)

        B, Q, H, D = query.shape
        logsumexp = torch.empty(
            (B, H, Q), dtype=torch.float32, device=query.device
        )

        q_tile_size, kv_tile_size = forward_config
        flash_fmha_forward(
            output,
            query,
            key,
            value,
            logsumexp,
            scale,
            q_tile_size,
            kv_tile_size,
        )

        logsumexp = logsumexp.transpose(1, 2).contiguous()

        ctx.save_for_backward(query, key, value, logsumexp, output)
        ctx.scale = scale
        ctx.backward_config = backward_config
        ctx.deterministic = natten_are_deterministic_algorithms_enabled()

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
    ]:
        query, key, value, logsumexp, output = ctx.saved_tensors
        logsumexp = logsumexp.transpose(1, 2).contiguous()

        d_output = grad_out.contiguous()  # noqa: F841
        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        d_value = torch.empty_like(value)

        q_tile_size, k_tile_size = ctx.backward_config

        flash_fmha_backward(
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
            ctx.deterministic
        )

        return d_query, d_key, d_value, None, None, None, None


def flash_fmha(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Optional[float] = None,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
    backward_q_tile_size: Optional[int] = None,
    backward_kv_tile_size: Optional[int] = None,
    return_lse: bool = False,
    deterministic: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:

    fmha_tensor_checks(query, key, value, must_match_head_dims=True)

    assert can_run_flash_fmha(query, key, value, raise_error=True)

    forward_config = check_flash_fmha_forward_config(
        input_tensor=query,
        q_tile_size=q_tile_size,
        kv_tile_size=kv_tile_size,
    )
    backward_config = check_flash_fmha_backward_config(
        input_tensor=query,
        q_tile_size=backward_q_tile_size,
        kv_tile_size=backward_kv_tile_size,
    )

    scale = scale or query.shape[-1] ** -0.5

    torch_deterministic = torch.are_deterministic_algorithms_enabled()
    natten_deterministic = natten_are_deterministic_algorithms_enabled()
    if natten_deterministic ^ torch_deterministic:
        raise RuntimeError(
            "The provided deterministic argument does not "
            f"match with PyTorch's global setting: \n \t PyTorch: {torch_deterministic}, "
            f"NATTEN: {natten_deterministic}"
        )

    output, lse = FlashFmhaAutogradFn.apply(
        query,
        key,
        value,
        scale,
        forward_config,
        backward_config,
        natten_deterministic
    )

    if return_lse:
        return output, lse

    return output
