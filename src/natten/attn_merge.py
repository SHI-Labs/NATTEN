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

from typing import List

import torch
from torch import Tensor

from ._environment import _IS_TORCH_COMPILE_SUPPORTED


def maybe_torch_compile(*args, **kwargs):
    def decorator(f):
        if _IS_TORCH_COMPILE_SUPPORTED:
            return torch.compile(f, *args, **kwargs)
        return f

    return decorator


# TODO: if use cases for this grow, we might want to do a custom kernel
def merge_attentions_fn(outputs: List[Tensor], lse_tensors: List[Tensor]) -> Tensor:

    assert len(outputs) >= 2, "Expected at least two tensors."
    assert len(outputs) == len(
        lse_tensors
    ), "Expected number of outputs and LSE tensors to match."

    assert all(
        output.dim() == 4 and output.is_contiguous() for output in outputs
    ), "Output tensors must be rank-4 tensors with (batch, seq, heads, dim) contiguous layout."

    batch, seqlen, heads, dim = outputs[0].shape

    assert all(
        [x for x in output.shape] == [batch, seqlen, heads, dim] for output in outputs
    ), "Output tensors must match in shape."

    assert all(
        lse.dim() == 3
        and lse.is_contiguous()
        and [x for x in lse.shape] == [batch, seqlen, heads]
        for lse in lse_tensors
    ), "LSE tensors must be rank-3 tensors with (batch, seq, heads) contiguous layout, and match in shape."

    accum_type = torch.float32
    output_type = outputs[0].dtype

    lse_tensors = [lse.to(accum_type) for lse in lse_tensors]

    outputs = [output.to(accum_type) for output in outputs]

    lse_max = torch.maximum(lse_tensors[0], lse_tensors[1])
    for i in range(2, len(lse_tensors)):
        lse_max = torch.maximum(lse_max, lse_tensors[i])

    exp_diffs = [torch.exp(lse - lse_max).unsqueeze(-1) for lse in lse_tensors]

    outputs_rescaled = [
        output * exp_diff for output, exp_diff in zip(outputs, exp_diffs)
    ]

    assert all(
        [
            output_rescaled.shape == output.shape
            for output, output_rescaled in zip(outputs, outputs_rescaled)
        ]
    )

    sum_of_exps = exp_diffs[0] + exp_diffs[1]
    sum_of_outs = outputs_rescaled[0] + outputs_rescaled[1]
    for i in range(2, len(exp_diffs)):
        sum_of_exps += exp_diffs[i]
        sum_of_outs += outputs_rescaled[i]

    output = sum_of_outs / sum_of_exps

    output = output.to(output_type)

    return output


# TODO: if use cases for this grow, we might want to do a custom kernel
@maybe_torch_compile(fullgraph=True)
def merge_attentions_compile(
    outputs: List[Tensor], lse_tensors: List[Tensor]
) -> Tensor:
    return merge_attentions_fn(outputs, lse_tensors)
