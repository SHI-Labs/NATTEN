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

from typing import Optional, Tuple, Union

import torch  # noqa: F401
from torch import Tensor

from natten.types import NoneType
from natten.utils.environment import is_torch_compiling


def generate_varlen_parameters(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    seqlens_Q: Optional[Tensor] = None,
    seqlens_KV: Optional[Tensor] = None,
) -> Union[
    Tuple[NoneType, NoneType, int, int],
    Tuple[Tensor, Tensor, int, int],
]:
    # NOTE: max_seqlen_{Q,KV} require a device-host sync, since they're expected to be ints (with
    # which we launch the varlen kernel) and not device tensors.
    # .item() introduces control flow and breaks the graph.
    # It is also inefficient to repeat this per-op, and mostly there for convenience.
    # generate_varlen_parameters should ideally always be called by the user ahead of model
    # forward / backward.
    if is_torch_compiling():
        raise RuntimeError(
            "Running 'generate_varlen_parameters' in a torch-compiled region is disallowed as it "
            "results in graph breaks. Please consider calling ahead of time and pass "
            "'cumulative_seqlen_{Q,KV}' and 'max_seqlen_{Q,KV}' instead of 'seqlens_{Q,KV}' to "
            "'attention'. "
        )

    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError(
            "Q, K, and V must match in batch size, got "
            f"{query.shape[0]=}, {key.shape[0]=}, {value.shape[0]=}."
        )

    if (seqlens_Q is None) ^ (seqlens_KV is None):
        raise ValueError(
            "Variable length Attention requires both of seqlens_Q and seqlens_KV to be set, got "
            f"{seqlens_Q=}, {seqlens_KV=}."
        )

    if seqlens_Q is None and seqlens_KV is None:
        # Not varlen
        return None, None, 0, 0

    assert seqlens_Q is not None
    assert seqlens_KV is not None

    if not isinstance(seqlens_Q, Tensor) or not isinstance(seqlens_KV, Tensor):
        raise ValueError("seqlens_Q and seqlens_KV must both be tensors.")

    if seqlens_Q.device != query.device or seqlens_KV.device != query.device:
        raise ValueError(
            "seqlens_Q and seqlens_KV must be on the same device as QKV, but "
            f"{seqlens_Q.device=}, {seqlens_KV.device=}, {query.device=}."
        )

    if seqlens_Q.dtype != torch.int32 or seqlens_KV.dtype != torch.int32:
        raise ValueError(
            "seqlens_Q and seqlens_KV must both be torch.int32 tensors, got "
            f"{seqlens_Q.dtype=}, {seqlens_KV.dtype=}."
        )

    if seqlens_Q.dim() != 1 or seqlens_KV.dim() != 1:
        raise ValueError(
            "seqlens_Q and seqlens_KV must both be 1-D tensors, got "
            f"{seqlens_Q.dim()=}, {seqlens_KV.dim()=}."
        )

    if seqlens_Q.shape[0] != seqlens_KV.shape[0]:
        raise ValueError(
            "seqlens_Q and seqlens_KV must match in size, got "
            f"{seqlens_Q.shape=}, {seqlens_KV.shape=}."
        )

    if seqlens_Q.shape[0] < 1:
        raise ValueError(
            "seqlens_Q and seqlens_KV must contain at least one element, got "
            f"{seqlens_Q.shape=}, {seqlens_KV.shape=}."
        )

    if query.shape[0] != 1:
        raise ValueError(
            "Variable length attention only supports sequence-packed memory layout "
            f"(batch = 1), got {query.shape[0]=}."
        )

    assert seqlens_Q.dim() == seqlens_KV.dim() == 1
    assert seqlens_Q.shape[0] == seqlens_KV.shape[0] >= 1
    assert seqlens_Q.dtype == seqlens_KV.dtype == torch.int32

    max_seqlen_Q = seqlens_Q.max().item()  # type: ignore
    max_seqlen_KV = seqlens_KV.max().item()  # type: ignore

    # NOTE: we have to prepend with 0 manually :(
    z = torch.tensor([0], dtype=torch.int32, device=seqlens_Q.device)
    cumulative_seqlen_Q = torch.cat([z, seqlens_Q.cumsum(0).to(torch.int32)], dim=0)
    cumulative_seqlen_KV = torch.cat([z, seqlens_KV.cumsum(0).to(torch.int32)], dim=0)

    assert isinstance(max_seqlen_Q, int)
    assert isinstance(max_seqlen_KV, int)

    return (
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        max_seqlen_Q,
        max_seqlen_KV,
    )
