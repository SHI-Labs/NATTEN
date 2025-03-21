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

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from .utils.testing import _IS_TORCH_COMPILE_SUPPORTED


def maybe_torch_compile(*args, **kwargs):
    def decorator(f):
        if _IS_TORCH_COMPILE_SUPPORTED:
            return torch.compile(f, *args, **kwargs)
        return f

    return decorator


def qk_cross_forward(query: Tensor, key: Tensor, out: Tensor):
    """
    Performs cross attention between arbitrary-rank tensors ([batch, heads, ..., dim]).
    Allows output to be a view.
    """
    query_bmm_view = query.view(query.shape[0], query.shape[1], -1, query.shape[-1])
    key_transposed_bmm_view = key.view(
        key.shape[0], key.shape[1], -1, key.shape[-1]
    ).transpose(-2, -1)
    out_bmm_view = out.view(out.shape[0], out.shape[1], -1, out.shape[-1])
    torch.matmul(query_bmm_view, key_transposed_bmm_view, out=out_bmm_view)


def qk_cross_backward(
    query: Tensor, d_attn: Tensor, key: Tensor, d_query: Tensor, d_key: Tensor
):
    """
    Backward pass for qk_cross_forward.
    """
    query_bmm_view = query.view(query.shape[0], query.shape[1], -1, query.shape[-1])
    key_bmm_view = key.view(key.shape[0], key.shape[1], -1, key.shape[-1])
    d_attn_bmm_view = d_attn.view(
        d_attn.shape[0], d_attn.shape[1], -1, d_attn.shape[-1]
    )
    d_attn_transposed_bmm_view = d_attn_bmm_view.transpose(-2, -1)
    d_query_bmm_view = d_query.view(
        d_query.shape[0], d_query.shape[1], -1, d_query.shape[-1]
    )
    d_key_bmm_view = d_key.view(d_key.shape[0], d_key.shape[1], -1, d_key.shape[-1])
    torch.matmul(d_attn_bmm_view, key_bmm_view, out=d_query_bmm_view)
    torch.matmul(d_attn_transposed_bmm_view, query_bmm_view, out=d_key_bmm_view)


def av_cross_forward(attn: Tensor, value: Tensor, output: Tensor):
    """
    Applies cross attention weights.
    """
    attn_bmm_view = attn.view(attn.shape[0], attn.shape[1], -1, attn.shape[-1])
    value_bmm_view = value.view(value.shape[0], value.shape[1], -1, value.shape[-1])
    output_bmm_view = output.view(
        output.shape[0], output.shape[1], -1, output.shape[-1]
    )
    torch.matmul(attn_bmm_view, value_bmm_view, out=output_bmm_view)


def av_cross_backward(
    d_out: Tensor, value: Tensor, attn: Tensor, d_attn: Tensor, d_value: Tensor
):
    """
    Backward pass for av_cross_forward.
    """
    d_out_bmm_view = d_out.view(d_out.shape[0], d_out.shape[1], -1, d_out.shape[-1])
    value_transposed_bmm_view = value.view(
        value.shape[0], value.shape[1], -1, value.shape[-1]
    ).transpose(-2, -1)
    d_value_bmm_view = d_value.view(
        d_value.shape[0], d_value.shape[1], -1, d_value.shape[-1]
    )
    attn_transposed_bmm_view = attn.view(
        attn.shape[0], attn.shape[1], -1, attn.shape[-1]
    ).transpose(-2, -1)
    d_attn_bmm_view = d_attn.view(
        d_attn.shape[0], d_attn.shape[1], -1, d_attn.shape[-1]
    )
    torch.matmul(attn_transposed_bmm_view, d_out_bmm_view, out=d_value_bmm_view)
    torch.matmul(d_out_bmm_view, value_transposed_bmm_view, out=d_attn_bmm_view)


def additional_sdpa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Optional[float] = None,
    attn_kwargs: Optional[Dict] = None,
) -> Tuple[Tensor, Tensor]:
    try:
        from xformers.ops.fmha import memory_efficient_attention_partial  # type: ignore
    except ImportError:
        raise ImportError(
            "Using additional KVs with Fused NA requires xFormers "
            ">= v0.0.25. Please make sure this requirement is "
            "satisfied, and if so, open an issue."
        )

    batch = query.shape[0]
    heads, dim = query.shape[-2], query.shape[-1]
    query = query.reshape(batch, -1, heads, dim)

    key = key.reshape(batch, -1, heads, dim)
    value = value.reshape(batch, -1, heads, dim)

    additional_kwargs = attn_kwargs or {}
    if "scale" not in additional_kwargs:
        additional_kwargs["scale"] = scale

    output, lse = memory_efficient_attention_partial(
        query=query,
        key=key,
        value=value,
        **additional_kwargs,
    )

    return output, lse


@maybe_torch_compile(fullgraph=True)
def merge_attentions(
    output_fna: Tensor, output_sdpa: Tensor, lse_fna: Tensor, lse_sdpa: Tensor
) -> Tensor:

    assert (
        output_sdpa.dim() == 4
    ), "SDPA output must be a rank-4 tensor with (batch, seq, heads, dim) modes."
    input_shape = output_sdpa.shape
    output_shape = output_fna.shape
    batch, seqlen, heads, dim = input_shape

    accum_type = torch.float32
    output_type = output_fna.dtype

    lse_0 = lse_fna.reshape(batch, seqlen, heads).to(accum_type)
    lse_1 = lse_sdpa.reshape(batch, heads, seqlen).to(accum_type).permute(0, 2, 1)

    output_0 = output_fna.reshape(input_shape).to(accum_type)
    output_1 = output_sdpa.reshape(input_shape).to(accum_type)

    lse_max = torch.maximum(lse_0, lse_1)
    exp_diff_0 = torch.exp(lse_0 - lse_max).unsqueeze(-1)
    exp_diff_1 = torch.exp(lse_1 - lse_max).unsqueeze(-1)

    output_0_rescaled = output_0 * exp_diff_0
    output_1_rescaled = output_1 * exp_diff_1

    assert output_0_rescaled.shape == output_1_rescaled.shape == output_0.shape

    sum_of_exps = exp_diff_0 + exp_diff_1

    output = (output_0_rescaled + output_1_rescaled) / sum_of_exps

    assert (
        output.shape == output_0.shape
    ), f"Output shape {output.shape=} doesn't match input shape {output_0.shape=}"

    output = output.reshape(output_shape).to(output_type)

    return output
