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

from .._libnatten import fmha_backward, fmha_forward
from ..types import (
    CutlassFmhaBackwardConfigType,
    CutlassFmhaForwardConfigType,
    NoneType,
)
from ..utils import log
from ..utils.checks import fmha_tensor_checks, varlen_tensor_checks
from .configs.checks import can_run_cutlass_fmha
from .configs.cutlass import (
    check_cutlass_fmha_backward_config,
    check_cutlass_fmha_forward_config,
)

logger = log.get_logger(__name__)


class CutlassFmhaAutogradFn(Function):
    @staticmethod
    @amp_fwd
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool,
        scale: float,
        forward_config: CutlassFmhaForwardConfigType,
        backward_config: CutlassFmhaBackwardConfigType,
        cumulative_seqlen_Q: Optional[Tensor],
        cumulative_seqlen_KV: Optional[Tensor],
        max_seqlen_Q: int,
        max_seqlen_KV: int,
    ) -> Tuple[Tensor, Tensor]:

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()

        assert query.dim() == value.dim() == 4
        assert query.shape[0] == value.shape[0]
        assert query.shape[2] == value.shape[2]

        q_tile_size, kv_tile_size = forward_config
        output, logsumexp = fmha_forward(
            query,
            key,
            value,
            is_causal,
            scale,
            q_tile_size,
            kv_tile_size,
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
        # varlen
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
        d_output = grad_out.contiguous()

        q_tile_size, k_tile_size, kv_splits, compute_delta_with_pt = ctx.backward_config

        num_kv_splits = kv_splits
        if kv_splits > 1 and torch.are_deterministic_algorithms_enabled():
            num_kv_splits = 1
            logger.warning(
                "You enabled PyTorch's deterministic mode, but tried to train with FNA's KV "
                "parallelism, which is non-deterministic. "
                f"Overriding {kv_splits=} to {num_kv_splits}."
            )

        if not compute_delta_with_pt and torch.are_deterministic_algorithms_enabled():
            compute_delta_with_pt = True
            # Silent override
            # logger.warning(
            #    "You enabled PyTorch's deterministic mode, but tried to use CUTLASS's reduction kernel"
            #    ", which is non-deterministic. Overriding."
            # )

        seqlen_kv = key.shape[1] if cumulative_seqlen_KV is None else ctx.max_seqlen_KV
        num_kv_tiles = (seqlen_kv + k_tile_size - 1) // k_tile_size
        assert num_kv_tiles > 0
        if num_kv_splits > num_kv_tiles:
            logger.warning(
                "Number of KV splits cannot exceed number of KV tiles, got "
                f"{num_kv_splits=}, {num_kv_tiles=}."
            )
            num_kv_splits = num_kv_tiles

        d_query, d_key, d_value = fmha_backward(
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
            num_kv_splits,
            compute_delta_with_pt,
            cumulative_seqlen_Q,
            cumulative_seqlen_KV,
            ctx.max_seqlen_Q,
            ctx.max_seqlen_KV,
        )

        return d_query, d_key, d_value, None, None, None, None, None, None, None, None


def cutlass_fmha(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
    backward_q_tile_size: Optional[int] = None,
    backward_kv_tile_size: Optional[int] = None,
    backward_kv_splits: Optional[int] = None,
    backward_use_pt_reduction: bool = False,
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
        must_match_head_dims=False,
        supports_gqa_mqa=True,
        backend_name="CUTLASS FMHA",
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

    assert can_run_cutlass_fmha(
        query, key, value, is_causal=is_causal, is_varlen=is_varlen, raise_error=True
    )

    forward_config = check_cutlass_fmha_forward_config(
        input_tensor=query if value.shape[-1] <= query.shape[-1] else value,
        q_tile_size=q_tile_size,
        kv_tile_size=kv_tile_size,
    )

    backward_config = check_cutlass_fmha_backward_config(
        input_tensor=key if value.shape[-1] <= key.shape[-1] else value,
        q_tile_size=backward_q_tile_size,
        kv_tile_size=backward_kv_tile_size,
        kv_splits=backward_kv_splits,
        use_pt_reduction=backward_use_pt_reduction,
        max_seqlen=max_seqlen_KV if is_varlen else None,
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

    output, lse = CutlassFmhaAutogradFn.apply(
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
