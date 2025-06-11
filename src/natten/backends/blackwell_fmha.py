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

from .._libnatten import blackwell_fmha_forward
from ..types import CutlassBlackwellFmhaForwardConfigType, NoneType
from ..utils.checks import fmha_tensor_checks

from .configs.checks import can_run_cutlass_blackwell_fmha
from .configs.cutlass_blackwell import check_cutlass_blackwell_fmha_forward_config


class CutlassBlackwellFmhaAutogradFn(Function):
    @staticmethod
    @amp_fwd
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: float,
        tiling_config: CutlassBlackwellFmhaForwardConfigType,
        run_persistent_kernel: bool,
    ) -> Tuple[Tensor, Tensor]:

        assert isinstance(
            scale, float
        ), f"Expected float attention scale, got {type(scale)}."

        if (
            not isinstance(tiling_config, tuple)
            or len(tiling_config) != 2
            or not all(isinstance(x, int) for x in tiling_config)
        ):
            raise ValueError(
                "Invalid tiling config for Blackwell FMHA; expected tuple of "
                f"integers of size 2, got {tiling_config=}."
            )

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        output = torch.empty_like(query)

        logsumexp = torch.empty(
            query.shape[:-1], dtype=torch.float32, device=query.device
        )

        q_tile_size, kv_tile_size = tiling_config
        blackwell_fmha_forward(
            output,
            query,
            key,
            value,
            logsumexp,
            scale,
            q_tile_size,
            kv_tile_size,
            run_persistent_kernel,
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
        NoneType,
    ]:
        query, key, value, logsumexp, output = ctx.saved_tensors
        d_output = grad_out.contiguous()  # noqa: F841
        d_query = torch.empty_like(query)
        d_key = torch.empty_like(key)
        d_value = torch.empty_like(value)

        raise NotImplementedError(
            "Blackwell FMHA does not support backpropagation yet."
        )

        return d_query, d_key, d_value, None, None, None


def cutlass_blackwell_fmha(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Optional[float] = None,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
    run_persistent_kernel: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:

    fmha_tensor_checks(query, key, value, must_match_head_dims=True)

    assert can_run_cutlass_blackwell_fmha(query, key, value, raise_error=True)

    tiling_config_forward = check_cutlass_blackwell_fmha_forward_config(
        input_tensor=query, q_tile_size=q_tile_size, kv_tile_size=kv_tile_size
    )

    scale = scale or query.shape[-1] ** -0.5

    output, lse = CutlassBlackwellFmhaAutogradFn.apply(
        query,
        key,
        value,
        scale,
        tiling_config_forward,
        run_persistent_kernel,
    )

    if return_lse:
        return output, lse

    return output
