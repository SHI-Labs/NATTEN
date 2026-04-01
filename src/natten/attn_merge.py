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
from typing import List, Tuple

import torch
from torch import Tensor
from torch.amp import custom_bwd, custom_fwd
from torch.autograd import Function

amp_fwd = functools.partial(custom_fwd, device_type="cuda")
amp_bwd = functools.partial(custom_bwd, device_type="cuda")

from natten._environment import _IS_TORCH_COMPILE_SUPPORTED


def _maybe_torch_compile(*args, **kwargs):
    def decorator(f):
        if _IS_TORCH_COMPILE_SUPPORTED:
            return torch.compile(f, *args, **kwargs)
        return f

    return decorator


# TODO: if use cases for this grow, we might want to do a custom kernel
def _merge_attentions_fn(
    outputs: List[Tensor], lse_tensors: List[Tensor]
) -> Tuple[Tensor, Tensor]:

    assert len(outputs) >= 2, "Expected at least two tensors."
    num_splits = len(outputs)
    assert (
        len(lse_tensors) == num_splits
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

    lse_tensors = [lse.to(accum_type).unsqueeze(-1) for lse in lse_tensors]

    outputs = [output.to(accum_type) for output in outputs]

    # New approach based on https://github.com/zhuzilin/ring-flash-attention/pull/34
    output = outputs[0] - torch.nn.functional.sigmoid(
        lse_tensors[1] - lse_tensors[0]
    ) * (outputs[0] - outputs[1])
    logsumexp = lse_tensors[0] - torch.nn.functional.logsigmoid(
        lse_tensors[0] - lse_tensors[1]
    )
    for i in range(2, num_splits):
        output = output - torch.nn.functional.sigmoid(lse_tensors[i] - logsumexp) * (
            output - outputs[i]
        )
        logsumexp = logsumexp - torch.nn.functional.logsigmoid(
            logsumexp - lse_tensors[i]
        )

    output = output.to(output_type)
    logsumexp = logsumexp.squeeze(-1)

    assert logsumexp.dim() == 3
    assert logsumexp.shape[0] == batch
    assert logsumexp.shape[1] == seqlen
    assert logsumexp.shape[2] == heads

    return output, logsumexp


@_maybe_torch_compile(fullgraph=True)
def _merge_attentions_compile(
    outputs: List[Tensor], lse_tensors: List[Tensor]
) -> Tuple[Tensor, Tensor]:
    return _merge_attentions_fn(outputs, lse_tensors)


def _merge_attentions_op(
    outputs: List[Tensor], lse_tensors: List[Tensor], torch_compile: bool = True
) -> Tuple[Tensor, Tensor]:

    if not torch_compile:
        return _merge_attentions_fn(
            [output.contiguous() for output in outputs],
            [lse.contiguous() for lse in lse_tensors],
        )

    return _merge_attentions_compile(
        [output.contiguous() for output in outputs],
        [lse.contiguous() for lse in lse_tensors],
    )


class MergeAttentionsAutogradFn(Function):
    @staticmethod
    @amp_fwd
    def forward(
        ctx,
        *args,
    ) -> Tuple[Tensor, Tensor]:

        assert len(args) >= 5, (
            "Expected at least 5 args (two outputs, two lse tensors, 1 torch compile flag) "
            + f"in attention merge, got {len(args)}."
        )
        assert (len(args) - 1) % 2 == 0, (
            f"Expected pairs of outputs and lse tensors, got {len(args)-1} args "
            + "(excluding torch compile flag)"
        )
        num_pairs = (len(args) - 1) // 2
        assert num_pairs >= 2

        torch_compile = args[-1]
        outputs = args[:num_pairs]
        lses = args[num_pairs:-1]

        assert len(outputs) == len(lses), (
            "Expected the same number of outputs as logsumexp tensors, "
            + f"got {len(outputs)=}, {len(lses)=}"
        )

        merged_output, merged_lse = _merge_attentions_op(
            outputs,  # type: ignore[arg-type]
            lses,  # type: ignore[arg-type]
            torch_compile=torch_compile,
        )

        ctx.num_pairs = num_pairs
        ctx.save_for_backward(merged_output, merged_lse, *outputs, *lses)

        return merged_output, merged_lse

    @staticmethod
    @amp_bwd
    def backward(ctx, grad_out: Tensor, grad_lse: Tensor) -> Tuple:

        num_pairs = ctx.num_pairs
        merged_output, merged_lse = ctx.saved_tensors[:2]
        outputs = ctx.saved_tensors[2 : num_pairs + 2]
        lses = ctx.saved_tensors[num_pairs + 2 :]

        # Outputs and LSEs from the originating attention ops must be replaced with
        # the merged ones inplace so that we get correct behavior, and not break torch.compile
        # graphs in the process.
        for output, lse in zip(outputs, lses):
            output.data.copy_(merged_output.data.reshape(output.shape))
            lse.data.copy_(merged_lse.data.reshape(lse.shape))

        return (
            *(grad_out for _ in range(num_pairs)),
            *(grad_lse for _ in range(num_pairs)),
            None,
        )


def merge_attentions(
    outputs: List[Tensor],
    lse_tensors: List[Tensor],
    torch_compile: bool = True,
    use_autograd_fix: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Takes multiple attention *outputs* originating from the same query tensor, and their
    corresponding logsumexps, and merges them as if their context (key/value pair) had been
    concatenated.

    This operation is used to implement cross-neighborhood attention, and can also be used for
    distributed setups, such as context-parallelism.

    This operation also attempts to use `torch.compile` to fuse the elementwise operations. This
    can be disabled by passing `torch_compile=False`.

    Parameters:
        outputs (List[Tensor]): List of 4-D attention output tensors, with the heads last layout
            (`[batch, seqlen, heads, head_dim]`)

        lse_tensors (List[Tensor]): List of 3-D logsumexp tensors, with the heads last layout
            (`[batch, seqlen, heads]`)

        torch_compile (bool): Attempt to use `torch.compile` to fuse the underlying elementwise
            operations. Default: True.

        use_autograd_fix (bool): fix backpropagation by using a custom autograd function. Only
            compatible with fused attention operations (Flash/FMHA/FNA), only as long as the inputs
            of this function are (views) of outputs from said attention operation.
            NATTEN's tests (tests/test_attn_merge.py) only verify correctness for when using
            attention operations from NATTEN. Integration for non-NATTEN ops must be verified by the
            end user.
            This must be disabled when using unfused Attention, which includes Flex without
            torch.compile. Default: True.

    Returns:
        output (Tensor): merged attention output.

        logsumexp (Tensor): updated logsumexp.
    """

    if len(outputs) < 2:
        raise ValueError("`merge_attentions` expects at least two tensors.")

    if len(outputs) != len(lse_tensors):
        raise ValueError(
            "`merge_attentions` expected number of outputs and LSE tensors to match, "
            f"got {len(outputs)=} != {len(lse_tensors)}."
        )
    assert len(outputs) == len(lse_tensors)

    requires_grad = outputs[0].requires_grad
    shape = outputs[0].shape

    for i, (output, lse) in enumerate(zip(outputs, lse_tensors)):
        if output.dim() != 4 or not output.is_contiguous():
            raise ValueError(
                "Output tensors must be rank-4 tensors with (batch, seq, heads, dim), "
                f"but got output {i} with rank={output.dim()}."
            )

        if output.shape != shape:
            raise ValueError(
                f"Output tensors must must match in shape, but got output {i} "
                f"with shape={output.shape}."
            )

        if lse.dim() != 3:
            raise ValueError(
                "LSE tensors must be rank-3 tensors with (batch, seq, heads)"
                f"but got LSE {i} with rank={lse.dim()}."
            )

        if lse.shape != shape[:3]:
            raise ValueError(
                f"LSE tensors must must match outputs in shape except last dim "
                f"({shape=}), but got LSE {i} with shape={lse.shape}."
            )

        if output.requires_grad and not requires_grad:
            raise ValueError(
                "Either all attentions must require grad, or none of them."
            )

    # This path is the correct way to do backward pass, but since we can't have lists as inputs to
    # autograd functions, we're forced to specialize it for 2-way for now.
    if use_autograd_fix:
        merged_output, merged_lse = MergeAttentionsAutogradFn.apply(
            *outputs, *lse_tensors, torch_compile
        )
        return merged_output, merged_lse

    return _merge_attentions_op(
        [output.contiguous() for output in outputs],
        [lse.contiguous() for lse in lse_tensors],
        torch_compile=torch_compile,
    )


__all__ = ["merge_attentions"]
