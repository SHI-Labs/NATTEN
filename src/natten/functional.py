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
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from .attn_merge import merge_attentions
from .backends import (
    choose_backend,
    choose_fmha_backend,
    cutlass_blackwell_fmha,
    cutlass_blackwell_fna_generic,
    cutlass_fmha,
    cutlass_fna_generic,
    cutlass_hopper_fmha,
    cutlass_hopper_fna_generic,
    flex_fmha,
    flex_fna_generic,
    flash_fmha,
    flash_fna_generic
)
from .types import (
    CausalArg1DTypeOrDed,
    CausalArg2DTypeOrDed,
    CausalArg3DTypeOrDed,
    CausalArgTypeOrDed,
    Dimension1DType,
    Dimension1DTypeOrDed,
    Dimension2DType,
    Dimension2DTypeOrDed,
    Dimension3DType,
    Dimension3DTypeOrDed,
    DimensionType,
    DimensionTypeOrDed,
    KernelSchedule,
)
from .utils import log
from .utils.checks import (
    additional_kv_tensor_checks,
    check_all_args,
    check_args_against_input,
    check_kernel_schedule,
    fmha_tensor_checks,
    is_self_attention,
    na_tensor_checks,
    varlen_tensor_checks,
)

logger = log.get_logger(__name__)


# Standard Attention


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
    # varlen parameters
    seqlens_Q: Optional[Tensor] = None,
    seqlens_KV: Optional[Tensor] = None,
    cumulative_seqlen_Q: Optional[Tensor] = None,
    cumulative_seqlen_KV: Optional[Tensor] = None,
    max_seqlen_Q: Optional[int] = None,
    max_seqlen_KV: Optional[int] = None,
    # backend parameters
    backend: Optional[str] = None,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
    backward_q_tile_size: Optional[int] = None,
    backward_kv_tile_size: Optional[int] = None,
    backward_kv_splits: Optional[int] = None,
    backward_use_pt_reduction: bool = False,
    run_persistent_kernel: bool = True,
    kernel_schedule: Optional[Union[str, KernelSchedule]] = None,
    torch_compile: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Runs standard dot product attention.

    This operation is used to implement neighborhood cross attention, in which we allow every
    token to interact with some additional context (`additional_keys` and `additional_values`
    tensors in [na1d][natten.na1d], [na2d][natten.na2d], and [na3d][natten.na3d]).
    This operator is also used as a fast path for cases where neighborhood attention is equivalent
    to self attention (not causal along any dims, and `kernel_size` is equal to the number of input
    tokens).

    This operation does not call into PyTorch's SDPA, and only runs one of the NATTEN backends
    (`cutlass-fmha`, `hopper-fmha`, `blackwell-fmha`, `flex-fmha`, `flash-fmha`). Reasons for that include being
    able to control performance-related arguments, return logsumexp, and more.
    For more information refer to [backends](backends.md).

    Causal mask, and Variable length (varlen) Attention are also supported in some backends
    (`cutlass-fmha` and `blackwell-fmha`).

    Varlen Attention is only supported for the sequence-packed layout: QKV tensors have batch size
    1, and tokens from different batches are concatenated without any padding along the sequence
    dimension. Sequence lengths for different batches can be provided in two ways:
        1. `seqlens_Q` and `seqlens_KV` (less efficient): only provide the sequence lengths as
            integer tensors (must be on the same device as QKV), and NATTEN will compute cumulative
            and maximum sequence lengths on each call.
            This is **incompatible** with full-graph `torch.compile` since it requires a
            synchronization.
        2. `cumulative_seqlen_{Q,KV}` and `max_seqlen_{Q,KV}` (more efficient):
            compute cumulative and maximum sequence lengths. `cumulative_seqlen_{Q,KV}` are integer
            tensors on the same device as QKV containing the cumulative sum of `seqlens_{Q,KV}`,
            with an additional `0` element in the beginning, therefore sized `batch+1`.
            `max_seqlen_{Q,KV}` are integers (not Tensors) that represent the maximum sequence
            lengths for Q and KV among all sequence batches.
            You can use `natten.utils.varlen.generate_varlen_parameters` to generate these
            parameters:
                ```python3
                from natten.utils.varlen import generate_varlen_parameters
                (
                    cumulative_seqlen_Q,
                    cumulative_seqlen_KV,
                    max_seqlen_Q,
                    max_seqlen_KV,
                ) = generate_varlen_parameters(q, k, v, seqlens_Q, seqlens_KV)
                ```
            As long as `generate_varlen_parameters` is called ahead of torch.compiling the model, it
            is supported without any graph breaks.

    GQA/MQA support (`heads != heads_kv`) is available. For now, `blackwell-fmha` and
    `flex-fmha` support GQA/MQA natively, and `cutlass-fmha` and `hopper-fmha` support it with
    explicit repeats (increases memory usage and runtime).

    Parameters:
        query (Tensor): 4-D query tensor, with the heads last layout
            (`[batch, seqlen, heads, head_dim]`)

        key (Tensor): 4-D key tensor, with the heads last layout
            (`[batch, seqlen_kv, heads_kv, head_dim]`)

        value (Tensor): 4-D value tensor, with the heads last layout
            (`[batch, seqlen_kv, heads_kv, head_dim_v]`)

        is_causal (bool): Toggle causal masking. Defaults to `False` (bi-directional).

        scale (float): Attention scale. Defaults to `head_dim ** -0.5`.

        seqlens_Q (Optional[Tensor]): (varlen) Optional 1-D tensor with size `batch`
            indicating the number of query tokens in each batch. Must be passed together with
            `seqlens_KV`.

        seqlens_KV (Optional[Tensor]): (varlen) Optional 1-D tensor with size `batch`
            indicating the number of key/value tokens in each batch. Must be passed together with
            `seqlens_Q`.

        cumulative_seqlen_Q (Optional[Tensor]): (varlen) Optional 1-D tensor with size `batch + 1`
            indicating the cumulative sum of number of query tokens in each batch, with an
            additional 0 element in the beginning. Must be passed together with
            `cumulative_seqlen_KV` and `max_seqlen_{Q,KV}`.

        cumulative_seqlen_KV (Optional[Tensor]): (varlen) Optional 1-D tensor with size `batch + 1`
            indicating the cumulative sum of number of key/value tokens in each batch, with an
            additional 0 element in the beginning. Must be passed together with
            `cumulative_seqlen_Q` and `max_seqlen_{Q,KV}`.

        max_seqlen_Q (Optional[int]): (varlen) Optional integer indicating the maximum query
            sequence length in all batches. Must be passed together with `cumulative_seqlen_{Q,KV}`
            and `max_seqlen_KV`.

        max_seqlen_KV (Optional[int]): (varlen) Optional integer indicating the maximum key/value
            sequence length in all batches. Must be passed together with `cumulative_seqlen_{Q,KV}`
            and `max_seqlen_Q`.

    Other Parameters:
        backend (str): Backend implementation to run with. Choices are: `None` (pick the best
            available one), `"cutlass-fmha"`, `"hopper-fmha"`, `"blackwell-fmha"`, `"flex-fmha"`,
            `"flash-fmha"`.
            Refer to [backends](backends.md) for more information.

        q_tile_size (int): Tile size along query sequence length in the forward pass kernel.
            You can use [profiler](profiler.md) to find valid choices for your use case.

        kv_tile_size (int): Tile size along key/value sequence length in the forward pass kernel.
            You can use [profiler](profiler.md) to find valid choices for your use case.

        backward_q_tile_size (int): Tile size along query sequence length in the backward pass
            kernel. This is ignored by `"flex-fmha"`.
            You can use [profiler](profiler.md) to find valid choices for your use case.

        backward_kv_tile_size (int): Tile size along key/value sequence length in the backward pass
            kernel. This is ignored by `"flex-fmha"`.
            You can use [profiler](profiler.md) to find valid choices for your use case.

        backward_kv_splits (int): Number of key/value tiles allowed to work in parallel in the
            backward pass kernel. This is only respected by the `"cutlass-fmha"` backend, only when
            [KV parallelism](context.md#kv-parallelism-in-fna) is enabled.

        backward_use_pt_reduction (bool): Whether to use PyTorch eager for computing the `dO * O`
            product required by the backward pass, over the CUTLASS kernel. This only applies to
            the `"cutlass-fmha"` backend.

        run_persistent_kernel (bool): Whether to use persistent tile scheduling in the forward pass
            kernel. This only applies to the `"blackwell-fmha"` backend.

        kernel_schedule (Optional[str]): Kernel type (Hopper architecture only). Choices are
            `None`: pick the default, `"non"` (non-persistent), `"coop"` (warp-specialized
            cooperative), or `"pp"` (warp-specialized ping-ponging). Refer to
            [Hopper FMHA/FNA backend](backends.md#hopper-fna-fmha) for more information.

        torch_compile (bool): Applies only to the `"flex-fmha"` backend. Whether or not to JIT
            compile the attention kernel. Due to this being an experimental feature in PyTorch, we
            do not recommend it, and it is guarded by context flags. Read more in
            [Flex Attention + `torch.compile`](context.md#flex-attention-torchcompile).

        return_lse (bool): Whether or not to return the `logsumexp` tensor. `logsumexp` can be used
            in the backward pass, and for [attention merging][natten.merge_attentions].

    Returns:
        output (Tensor): 4-D output tensor, with the heads last layout
            (`[batch, seqlen, heads, head_dim_v]`).

        logsumexp (Tensor): only returned when `return_lse=True`. 3-D logsumexp tensor, with the
            heads last layout (`[batch, seqlen, heads]`).
    """

    fmha_tensor_checks(query, key, value)

    (
        cumulative_seqlen_Q,
        cumulative_seqlen_KV,
        max_seqlen_Q,
        max_seqlen_KV,
    ) = varlen_tensor_checks(
        query=query,
        key=key,
        value=value,
        seqlens_Q=seqlens_Q,
        seqlens_KV=seqlens_KV,
        cumulative_seqlen_Q=cumulative_seqlen_Q,
        cumulative_seqlen_KV=cumulative_seqlen_KV,
        max_seqlen_Q=max_seqlen_Q,
        max_seqlen_KV=max_seqlen_KV,
    )
    is_varlen = cumulative_seqlen_Q is not None

    scale = scale or query.shape[-1] ** -0.5

    kernel_schedule = check_kernel_schedule(kernel_schedule)

    backend = backend or choose_fmha_backend(
        query,
        key,
        value,
        is_causal=is_causal,
        is_varlen=is_varlen,
        torch_compile=torch_compile,
    )

    if backend == "blackwell-fmha":
        return cutlass_blackwell_fmha(
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            scale=scale,
            cumulative_seqlen_Q=cumulative_seqlen_Q,
            cumulative_seqlen_KV=cumulative_seqlen_KV,
            max_seqlen_Q=max_seqlen_Q,
            max_seqlen_KV=max_seqlen_KV,
            q_tile_size=q_tile_size,
            kv_tile_size=kv_tile_size,
            backward_q_tile_size=backward_q_tile_size,
            backward_kv_tile_size=backward_kv_tile_size,
            run_persistent_kernel=run_persistent_kernel,
            return_lse=return_lse,
        )

    if backend == "hopper-fmha":
        return cutlass_hopper_fmha(
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            scale=scale,
            q_tile_size=q_tile_size,
            kv_tile_size=kv_tile_size,
            backward_q_tile_size=backward_q_tile_size,
            backward_kv_tile_size=backward_kv_tile_size,
            kernel_schedule=kernel_schedule,
            return_lse=return_lse,
            cumulative_seqlen_Q=cumulative_seqlen_Q,
            cumulative_seqlen_KV=cumulative_seqlen_KV,
            max_seqlen_Q=max_seqlen_Q,
            max_seqlen_KV=max_seqlen_KV,
        )

    elif backend == "cutlass-fmha":
        return cutlass_fmha(
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            scale=scale,
            q_tile_size=q_tile_size,
            kv_tile_size=kv_tile_size,
            backward_q_tile_size=backward_q_tile_size,
            backward_kv_tile_size=backward_kv_tile_size,
            backward_kv_splits=backward_kv_splits,
            backward_use_pt_reduction=backward_use_pt_reduction,
            return_lse=return_lse,
            cumulative_seqlen_Q=cumulative_seqlen_Q,
            cumulative_seqlen_KV=cumulative_seqlen_KV,
            max_seqlen_Q=max_seqlen_Q,
            max_seqlen_KV=max_seqlen_KV,
        )

    elif backend == "flex-fmha":
        return flex_fmha(
            query=query,
            key=key,
            value=value,
            is_causal=is_causal,
            scale=scale,
            q_tile_size=q_tile_size,
            kv_tile_size=kv_tile_size,
            torch_compile=torch_compile,
            return_lse=return_lse,
            cumulative_seqlen_Q=cumulative_seqlen_Q,
            cumulative_seqlen_KV=cumulative_seqlen_KV,
            max_seqlen_Q=max_seqlen_Q,
            max_seqlen_KV=max_seqlen_KV,
        )

    elif backend == "flash-fmha":
        return flash_fmha(
            query=query,
            key=key,
            value=value,
            scale=scale,
            q_tile_size=q_tile_size,
            kv_tile_size=kv_tile_size,
            backward_q_tile_size=backward_q_tile_size,
            backward_kv_tile_size=backward_kv_tile_size,
            return_lse=return_lse,
        )

    raise NotImplementedError(f"Unrecognized NATTEN FMHA backend {backend}.")


# Neighborhood Attention


def neighborhood_attention_generic(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: DimensionTypeOrDed,
    stride: DimensionTypeOrDed = 1,
    dilation: DimensionTypeOrDed = 1,
    is_causal: Optional[CausalArgTypeOrDed] = False,
    scale: Optional[float] = None,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    attention_kwargs: Optional[Dict] = None,
    # Perf-related args
    backend: Optional[str] = None,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    backward_q_tile_shape: Optional[DimensionType] = None,
    backward_kv_tile_shape: Optional[DimensionType] = None,
    backward_kv_splits: Optional[DimensionType] = None,
    backward_use_pt_reduction: bool = False,
    run_persistent_kernel: bool = True,
    kernel_schedule: Optional[Union[str, KernelSchedule]] = None,
    torch_compile: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:

    na_tensor_checks(query, key, value)
    additional_kv_tensor_checks(query, key, value, additional_keys, additional_values)
    kernel_schedule = check_kernel_schedule(kernel_schedule)

    na_dim = query.dim() - 3  # batch, heads, head_dim

    assert na_dim in [1, 2, 3]

    kernel_size, stride, dilation, is_causal = check_all_args(
        na_dim, kernel_size, stride, dilation, is_causal
    )

    check_args_against_input(
        query,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
    )

    has_additional_attention = (
        additional_keys is not None and additional_values is not None
    )

    if is_self_attention(
        query,
        kernel_size=kernel_size,
        is_causal=is_causal,
        has_additional_attention=has_additional_attention,
    ) and backend != "flash-fna":
        logger.debug(
            f"{query.shape=} with {kernel_size=}, {has_additional_attention=} and {is_causal=} is "
            "self attention. Calling attention instead of neighborhood attention directly."
        )

        query_shape = query.shape
        query = query.flatten(1, na_dim)
        key = key.flatten(1, na_dim)
        value = value.flatten(1, na_dim)

        if has_additional_attention:
            assert additional_keys is not None
            assert additional_values is not None
            key = torch.cat([key, additional_keys], dim=1)
            value = torch.cat([value, additional_values], dim=1)

        attn_kwargs = attention_kwargs or {}
        out, lse = attention(
            query,
            key,
            value,
            is_causal=is_causal[0],  # NOTE: special case
            scale=scale,
            return_lse=True,
            **attn_kwargs,
        )
        lse_shape = [s for s in query_shape[:-1]]
        output_shape = lse_shape + [value.shape[-1]]
        out = out.reshape(*output_shape)
        lse = lse.reshape(*lse_shape)

        if return_lse:
            return out, lse

        return out

    scale = scale or query.shape[-1] ** -0.5

    backend = backend or choose_backend(query, key, value, torch_compile=torch_compile)

    if backend == "blackwell-fna":
        output, lse = cutlass_blackwell_fna_generic(
            query=query,
            key=key,
            value=value,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            scale=scale,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            backward_q_tile_shape=backward_q_tile_shape,
            backward_kv_tile_shape=backward_kv_tile_shape,
            run_persistent_kernel=run_persistent_kernel,
            return_lse=True,
        )

    elif backend == "hopper-fna":
        output, lse = cutlass_hopper_fna_generic(
            query=query,
            key=key,
            value=value,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            scale=scale,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            backward_q_tile_shape=backward_q_tile_shape,
            backward_kv_tile_shape=backward_kv_tile_shape,
            kernel_schedule=kernel_schedule,
            return_lse=True,
        )

    elif backend == "flash-fna":
        output, lse = flash_fna_generic(
            query=query,
            key=key,
            value=value,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            scale=scale,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            backward_q_tile_shape=backward_q_tile_shape,
            backward_kv_tile_shape=backward_kv_tile_shape,
            return_lse=True,
        )

    elif backend == "cutlass-fna":
        output, lse = cutlass_fna_generic(
            query=query,
            key=key,
            value=value,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            scale=scale,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            backward_q_tile_shape=backward_q_tile_shape,
            backward_kv_tile_shape=backward_kv_tile_shape,
            backward_kv_splits=backward_kv_splits,
            backward_use_pt_reduction=backward_use_pt_reduction,
            return_lse=True,
        )

    elif backend == "flex-fna":
        output, lse = flex_fna_generic(
            query=query,
            key=key,
            value=value,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            scale=scale,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            torch_compile=torch_compile,
            return_lse=True,
        )

    else:
        raise NotImplementedError(f"Unrecognized NATTEN backend {backend}.")

    if has_additional_attention:
        assert additional_keys is not None
        assert additional_values is not None

        attention_kwargs = attention_kwargs or {}
        if "torch_compile" not in attention_kwargs:
            attention_kwargs["torch_compile"] = torch_compile

        additional_output, additional_lse = attention(
            query.flatten(1, na_dim),
            additional_keys,
            additional_values,
            is_causal=False,
            scale=scale,
            return_lse=True,
            **attention_kwargs,
        )

        # NOTE: Flex unfused should not use the autograd fix
        is_flex = backend == "flex-fna" or (
            "backend" in attention_kwargs and attention_kwargs["backend"] == "flex-fmha"
        )
        use_autograd_fix = not is_flex or torch_compile

        merged_output, merged_lse = merge_attentions(
            [output.flatten(1, na_dim), additional_output],
            [lse.flatten(1, na_dim), additional_lse],
            use_autograd_fix=use_autograd_fix,
        )
        merged_output = merged_output.reshape(output.shape)
        merged_lse = merged_lse.reshape(output.shape[:-1])

        if return_lse:
            return merged_output, merged_lse

        return merged_output

    if return_lse:
        return output, lse

    return output


def na1d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension1DTypeOrDed,
    stride: Dimension1DTypeOrDed = 1,
    dilation: Dimension1DTypeOrDed = 1,
    is_causal: Optional[CausalArg1DTypeOrDed] = False,
    scale: Optional[float] = None,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    attention_kwargs: Optional[Dict] = None,
    backend: Optional[str] = None,
    q_tile_shape: Optional[Dimension1DType] = None,
    kv_tile_shape: Optional[Dimension1DType] = None,
    backward_q_tile_shape: Optional[Dimension1DType] = None,
    backward_kv_tile_shape: Optional[Dimension1DType] = None,
    backward_kv_splits: Optional[Dimension1DType] = None,
    backward_use_pt_reduction: bool = False,
    run_persistent_kernel: bool = True,
    kernel_schedule: Optional[Union[str, KernelSchedule]] = None,
    torch_compile: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Computes 1-D neighborhood attention.

    GQA/MQA support (`heads != heads_kv`) is available. For now, `blackwell-fna` and
    `flex-fna` support GQA/MQA natively, and `cutlass-fna` and `hopper-fna` support it with
    explicit repeats (increases memory usage and runtime).

    Parameters:
        query (Tensor): 4-D query tensor, with the heads last layout
            (`[batch, seqlen, heads, head_dim]`)

        key (Tensor): 4-D key tensor, with the heads last layout
            (`[batch, seqlen, heads_kv, head_dim]`)

        value (Tensor): 4-D value tensor, with the heads last layout
            (`[batch, seqlen, heads_kv, head_dim_v]`)

        kernel_size (Tuple[int] | int): Neighborhood window (kernel) size.

            !!! note
                `kernel_size` must be smaller than or equal to `seqlen`.

        stride (Tuple[int] | int): Sliding window step size. Defaults to `1` (standard sliding
            window).

            !!! note
                `stride` must be smaller than or equal to `kernel_size`.
                When `stride == kernel_size`, there will be no overlap between sliding windows,
                which is equivalent to blocked attention (a.k.a.
                [window self attention](https://arxiv.org/abs/2103.14030)).

        dilation (Tuple[int] | int): Dilation step size. Defaults to `1` (standard sliding window).

            !!! note
                The product of `dilation` and `kernel_size` must be smaller than or equal to
                `seqlen`.

        is_causal (Tuple[bool] | bool): Toggle causal masking. Defaults to `False`
            (bi-directional).

        scale (float): Attention scale. Defaults to `head_dim ** -0.5`.

        additional_keys: `None` or 4-D key tensor, with the heads last layout
            (`[batch, seqlen_kv, heads_kv, head_dim]`), corresponding to key tokens from some
            additional context. Used when performing neighborhood cross-attention, where `query`
            tokens attend to their neighborhood, as well as some fixed additional set of tokens.

        additional_values: `None` or 4-D value tensor, with the heads last layout
            (`[batch, seqlen_kv, heads_kv, head_dim_v]`), corresponding to value tokens from some
            additional context. Used when performing neighborhood cross-attention, where `query`
            tokens attend to their neighborhood, as well as some fixed additional set of tokens.

            !!! note
                `additional_keys` and `additional_values` must both either be `Tensor`s, or both
                `None`s, and must match in shape.

    Other Parameters:
        backend (str): Backend implementation to run with. Choices are: `None` (pick the best
            available one), `"cutlass-fna"`, `"hopper-fna"`, `"blackwell-fna"`, `"flex-fna"`.
            Refer to [backends](backends.md) for more information.

        q_tile_shape (Tuple[int]): 1-D Tile shape for the query token layout in the forward pass
            kernel. You can use [profiler](profiler.md) to find valid choices for your use case,
            and search for the best combination.

        kv_tile_shape (Tuple[int]): 1-D Tile shape for the key-value token layout in the forward
            pass kernel. You can use [profiler](profiler.md) to find valid choices for your use
            case, and search for the best combination.

        backward_q_tile_shape (Tuple[int]): 1-D Tile shape for the query token layout in the
            backward pass kernel. This is ignored by `"flex-fna"`.
            You can use [profiler](profiler.md) to find valid choices for your use case, and
            search for the best combination.

        backward_kv_tile_shape (Tuple[int]): 1-D Tile shape for the key/value token layout in the
            backward pass kernel. This is ignored by `"flex-fna"`.
            You can use [profiler](profiler.md) to find valid choices for your use case, and
            search for the best combination.

        backward_kv_splits (Tuple[int]): Number of key/value tiles allowed to work in parallel in
            the backward pass kernel. Like tile shapes, this is a tuple and not an integer for
            neighborhood attention operations, and the size of the tuple corresponds to the number
            of dimensions / rank of the layout of tokens. This is only respected by the
            `"cutlass-fna"` backend, and only when
            [KV parallelism](context.md#kv-parallelism-in-fna) is enabled.

        backward_use_pt_reduction (bool): Whether to use PyTorch eager for computing the `dO * O`
            product required by the backward pass, over the CUTLASS kernel. This only applies to
            the `"cutlass-fna"` backend.

        run_persistent_kernel (bool): Whether to use persistent tile scheduling in the forward pass
            kernel. This only applies to the `"blackwell-fna"` backend.

        kernel_schedule (Optional[str]): Kernel type (Hopper architecture only). Choices are
            `None`: pick the default, `"non"` (non-persistent), `"coop"` (warp-specialized
            cooperative), or `"pp"` (warp-specialized ping-ponging). Refer to
            [Hopper FMHA/FNA backend](backends.md#hopper-fna-fmha) for more information.

        torch_compile (bool): Applies only to the `"flex-fna"` backend. Whether or not to JIT
            compile the attention kernel. Due to this being an experimental feature in PyTorch, we
            do not recommend it, and it is guarded by context flags. Read more in
            [Flex Attention + `torch.compile`](context.md#flex-attention-torchcompile).

        attention_kwargs: arguments to the [attention][natten.attention] operator, if used to
            implement neighborhood cross-attention, or self attention as a fast path for
            neighborhood attention.

            If `additional_{keys,values}` are specified, NATTEN usually performs a separate
            cross-attention using our [attention][natten.attention] operator, and
            [merges][natten.merge_attentions] the results.

            If for a given use case, the neighborhood attention problem is equivalent to self
            attention (not causal, `kernel_size == seqlen`), NATTEN will also attempt to directly
            use [attention][natten.attention].

            You can override arguments to [attention][natten.attention] by passing a
            dictionary here.

            !!! example
                ```python
                out = na1d(
                    q, k, v, kernel_size=kernel_size,
                    ...,
                    attention_kwargs={
                        "backend": "blackwell-fmha",
                        "run_persistent_kernel": True,
                    }
                )
                ```

        return_lse (bool): Whether or not to return the `logsumexp` tensor. `logsumexp` can be used
            in the backward pass, and for [attention merging][natten.merge_attentions].

    Returns:
        output (Tensor): 4-D output tensor, with the heads last layout
            (`[batch, seqlen, heads, head_dim_v]`).

        logsumexp (Tensor): only returned when `return_lse=True`. 3-D logsumexp tensor, with the
            heads last layout (`[batch, seqlen, heads]`).
    """
    return neighborhood_attention_generic(
        query=query,
        key=key,
        value=value,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
        additional_keys=additional_keys,
        additional_values=additional_values,
        attention_kwargs=attention_kwargs,
        backend=backend,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
        backward_q_tile_shape=backward_q_tile_shape,
        backward_kv_tile_shape=backward_kv_tile_shape,
        backward_kv_splits=backward_kv_splits,
        backward_use_pt_reduction=backward_use_pt_reduction,
        run_persistent_kernel=run_persistent_kernel,
        kernel_schedule=kernel_schedule,
        torch_compile=torch_compile,
        return_lse=return_lse,
    )


def na2d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension2DTypeOrDed,
    stride: Dimension2DTypeOrDed = 1,
    dilation: Dimension2DTypeOrDed = 1,
    is_causal: Optional[CausalArg2DTypeOrDed] = False,
    scale: Optional[float] = None,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    attention_kwargs: Optional[Dict] = None,
    backend: Optional[str] = None,
    q_tile_shape: Optional[Dimension2DType] = None,
    kv_tile_shape: Optional[Dimension2DType] = None,
    backward_q_tile_shape: Optional[Dimension2DType] = None,
    backward_kv_tile_shape: Optional[Dimension2DType] = None,
    backward_kv_splits: Optional[Dimension2DType] = None,
    backward_use_pt_reduction: bool = False,
    run_persistent_kernel: bool = True,
    kernel_schedule: Optional[Union[str, KernelSchedule]] = None,
    torch_compile: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Computes 2-D neighborhood attention.

    GQA/MQA support (`heads != heads_kv`) is available. For now, `blackwell-fna` and
    `flex-fna` support GQA/MQA natively, and `cutlass-fna` and `hopper-fna` support it with
    explicit repeats (increases memory usage and runtime).

    Parameters:
        query (Tensor): 2-D query tensor, with the heads last layout:
            `[batch, X, Y, heads, head_dim]`, where token layout shape (feature map shape) is
            `(X, Y)`.

        key (Tensor): 2-D key tensor, with the heads last layout:
            `[batch, X, Y, heads_kv, head_dim]`, where token layout shape (feature map shape) is
            `(X, Y)`.

        value (Tensor): 2-D value tensor, with the heads last layout:
            `[batch, X, Y, heads_kv, head_dim_v]`, where token layout shape (feature map shape) is
            `(X, Y)`.

        kernel_size (Tuple[int, int] | int): Neighborhood window (kernel) size/shape. If an
            integer, it will be repeated for all 2 dimensions. For example `kernel_size=3` is
            reinterpreted as `kernel_size=(3, 3)`.

            !!! note
                `kernel_size` must be smaller than or equal to token layout shape (`(X, Y)`) along
                every dimension.

        stride (Tuple[int, int] | int): Sliding window step size/shape. Defaults to `1` (standard
            sliding window). If an integer, it will be repeated for all 2 dimensions. For example
            `stride=2` is reinterpreted as `stride=(2, 2)`.

            !!! note
                `stride` must be smaller than or equal to `kernel_size` along every dimension.
                When `stride == kernel_size`, there will be no overlap between sliding windows,
                which is equivalent to blocked attention (a.k.a.
                [window self attention](https://arxiv.org/abs/2103.14030)).

        dilation (Tuple[int, int] | int): Dilation step size/shape. Defaults to `1` (standard
            sliding window). If an integer, it will be repeated for all 2 dimensions. For example
            `dilation=4` is reinterpreted as `dilation=(4, 4)`.

            !!! note
                The product of `dilation` and `kernel_size` must be smaller than or equal to
                token layout shape (`(X, Y)`) along every dimension.

        is_causal (Tuple[bool, bool] | bool): Toggle causal masking. Defaults to `False`
            (bi-directional). If a boolean, it will be repeated for all 2 dimensions. For example
            `is_causal=True` is reinterpreted as `is_causal=(True, True)`.

        scale (float): Attention scale. Defaults to `head_dim ** -0.5`.

        additional_keys: `None` or 4-D key tensor, with the heads last layout
            (`[batch, seqlen_kv, heads_kv, head_dim]`), corresponding to key tokens from some
            additional context. Used when performing neighborhood cross-attention, where `query`
            tokens attend to their neighborhood, as well as some fixed additional set of tokens.

        additional_values: `None` or 4-D value tensor, with the heads last layout
            (`[batch, seqlen_kv, heads_kv, head_dim_v]`), corresponding to value tokens from some
            additional context. Used when performing neighborhood cross-attention, where `query`
            tokens attend to their neighborhood, as well as some fixed additional set of tokens.

            !!! note
                `additional_keys` and `additional_values` must both either be `Tensor`s, or both
                `None`s, and must match in shape.

    Other Parameters:
        backend (str): Backend implementation to run with. Choices are: `None` (pick the best
            available one), `"cutlass-fna"`, `"hopper-fna"`, `"blackwell-fna"`, `"flex-fna"`.
            Refer to [backends](backends.md) for more information.

        q_tile_shape (Tuple[int, int]): 2-D Tile shape for the query token layout in the forward
            pass kernel. You can use [profiler](profiler.md) to find valid choices for your use
            case, and search for the best combination.

        kv_tile_shape (Tuple[int, int]): 2-D Tile shape for the key-value token layout in the
            forward pass kernel. You can use [profiler](profiler.md) to find valid choices for your
            use case, and search for the best combination.

        backward_q_tile_shape (Tuple[int, int]): 2-D Tile shape for the query token layout in the
            backward pass kernel. This is ignored by `"flex-fna"`.
            You can use [profiler](profiler.md) to find valid choices for your use case, and
            search for the best combination.

        backward_kv_tile_shape (Tuple[int, int]): 2-D Tile shape for the key/value token layout in
            the backward pass kernel. This is ignored by `"flex-fna"`.
            You can use [profiler](profiler.md) to find valid choices for your use case, and
            search for the best combination.

        backward_kv_splits (Tuple[int, int]): Number of key/value tiles allowed to work in parallel
            in the backward pass kernel. Like tile shapes, this is a tuple and not an integer for
            neighborhood attention operations, and the size of the tuple corresponds to the number
            of dimensions / rank of the layout of tokens. This is only respected by the
            `"cutlass-fna"` backend, and only when
            [KV parallelism](context.md#kv-parallelism-in-fna) is enabled.

        backward_use_pt_reduction (bool): Whether to use PyTorch eager for computing the `dO * O`
            product required by the backward pass, over the CUTLASS kernel. This only applies to
            the `"cutlass-fna"` backend.

        run_persistent_kernel (bool): Whether to use persistent tile scheduling in the forward pass
            kernel. This only applies to the `"blackwell-fna"` backend.

        kernel_schedule (Optional[str]): Kernel type (Hopper architecture only). Choices are
            `None`: pick the default, `"non"` (non-persistent), `"coop"` (warp-specialized
            cooperative), or `"pp"` (warp-specialized ping-ponging). Refer to
            [Hopper FMHA/FNA backend](backends.md#hopper-fna-fmha) for more information.

        torch_compile (bool): Applies only to the `"flex-fna"` backend. Whether or not to JIT
            compile the attention kernel. Due to this being an experimental feature in PyTorch, we
            do not recommend it, and it is guarded by context flags. Read more in
            [Flex Attention + `torch.compile`](context.md#flex-attention-torchcompile).

        attention_kwargs: arguments to the [attention][natten.attention] operator, if used to
            implement neighborhood cross-attention, or self attention as a fast path for
            neighborhood attention.

            If `additional_{keys,values}` are specified, NATTEN usually performs a separate
            cross-attention using our [attention][natten.attention] operator, and
            [merges][natten.merge_attentions] the results.

            If for a given use case, the neighborhood attention problem is equivalent to self
            attention (not causal along any dims, `kernel_size == (X, Y)`), NATTEN will also
            attempt to directly use [attention][natten.attention].

            You can override arguments to [attention][natten.attention] by passing a
            dictionary here.

            !!! example
                ```python
                out = na2d(
                    q, k, v, kernel_size=kernel_size,
                    ...,
                    attention_kwargs={
                        "backend": "blackwell-fmha",
                        "run_persistent_kernel": True,
                    }
                )
                ```

        return_lse (bool): Whether or not to return the `logsumexp` tensor. `logsumexp` can be used
            in the backward pass, and for [attention merging][natten.merge_attentions].

    Returns:
        output (Tensor): 5-D output tensor, with the heads last layout
            (`[batch, X, Y, heads, head_dim_v]`).

        logsumexp (Tensor): only returned when `return_lse=True`. 4-D logsumexp tensor, with the
            heads last layout (`[batch, X, Y, heads]`).
    """
    return neighborhood_attention_generic(
        query=query,
        key=key,
        value=value,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
        additional_keys=additional_keys,
        additional_values=additional_values,
        attention_kwargs=attention_kwargs,
        backend=backend,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
        backward_q_tile_shape=backward_q_tile_shape,
        backward_kv_tile_shape=backward_kv_tile_shape,
        backward_kv_splits=backward_kv_splits,
        backward_use_pt_reduction=backward_use_pt_reduction,
        run_persistent_kernel=run_persistent_kernel,
        kernel_schedule=kernel_schedule,
        torch_compile=torch_compile,
        return_lse=return_lse,
    )


def na3d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension3DTypeOrDed,
    stride: Dimension3DTypeOrDed = 1,
    dilation: Dimension3DTypeOrDed = 1,
    is_causal: Optional[CausalArg3DTypeOrDed] = False,
    scale: Optional[float] = None,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    attention_kwargs: Optional[Dict] = None,
    backend: Optional[str] = None,
    q_tile_shape: Optional[Dimension3DType] = None,
    kv_tile_shape: Optional[Dimension3DType] = None,
    backward_q_tile_shape: Optional[Dimension3DType] = None,
    backward_kv_tile_shape: Optional[Dimension3DType] = None,
    backward_kv_splits: Optional[Dimension3DType] = None,
    backward_use_pt_reduction: bool = False,
    run_persistent_kernel: bool = True,
    kernel_schedule: Optional[Union[str, KernelSchedule]] = None,
    torch_compile: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Computes 3-D neighborhood attention.

    GQA/MQA support (`heads != heads_kv`) is available. For now, `blackwell-fna` and
    `flex-fna` support GQA/MQA natively, and `cutlass-fna` and `hopper-fna` support it with
    explicit repeats (increases memory usage and runtime).

    Parameters:
        query (Tensor): 3-D query tensor, with the heads last layout:
            `[batch, X, Y, Z, heads, head_dim]`, where token layout shape (feature map shape) is
            `(X, Y, Z)`.

        key (Tensor): 3-D key tensor, with the heads last layout:
            `[batch, X, Y, Z, heads_kv, head_dim]`, where token layout shape (feature map shape) is
            `(X, Y, Z)`.

        value (Tensor): 3-D value tensor, with the heads last layout:
            `[batch, X, Y, Z, heads_kv, head_dim_V]`, where token layout shape (feature map shape) is
            `(X, Y, Z)`.

        kernel_size (Tuple[int, int, int] | int): Neighborhood window (kernel) size/shape. If an
            integer, it will be repeated for all 3 dimensions. For example `kernel_size=3` is
            reinterpreted as `kernel_size=(3, 3, 3)`.

            !!! note
                `kernel_size` must be smaller than or equal to token layout shape (`(X, Y, Z)`)
                along every dimension.

        stride (Tuple[int, int, int] | int): Sliding window step size/shape. Defaults to `1`
            (standard sliding window). If an integer, it will be repeated for all 3 dimensions.
            For example `stride=2` is reinterpreted as `stride=(2, 2, 2)`.

            !!! note
                `stride` must be smaller than or equal to `kernel_size` along every dimension.
                When `stride == kernel_size`, there will be no overlap between sliding windows,
                which is equivalent to blocked attention (a.k.a.
                [window self attention](https://arxiv.org/abs/2103.14030)).

        dilation (Tuple[int, int, int] | int): Dilation step size/shape. Defaults to `1` (standard
            sliding window). If an integer, it will be repeated for all 3 dimensions. For example
            `dilation=4` is reinterpreted as `dilation=(4, 4, 4)`.

            !!! note
                The product of `dilation` and `kernel_size` must be smaller than or equal to
                token layout shape (`(X, Y, Z)`) along every dimension.

        is_causal (Tuple[bool, bool, bool] | bool): Toggle causal masking. Defaults to `False`
            (bi-directional). If a boolean, it will be repeated for all 3 dimensions. For example
            `is_causal=True` is reinterpreted as `is_causal=(True, True, True)`.

        scale (float): Attention scale. Defaults to `head_dim ** -0.5`.

        additional_keys: `None` or 4-D key tensor, with the heads last layout
            (`[batch, seqlen_kv, heads_kv, head_dim_v]`), corresponding to key tokens from some
            additional context. Used when performing neighborhood cross-attention, where `query`
            tokens attend to their neighborhood, as well as some fixed additional set of tokens.

        additional_values: `None` or 4-D value tensor, with the heads last layout
            (`[batch, seqlen_kv, heads_kv, head_dim_v]`), corresponding to value tokens from some
            additional context. Used when performing neighborhood cross-attention, where `query`
            tokens attend to their neighborhood, as well as some fixed additional set of tokens.

            !!! note
                `additional_keys` and `additional_values` must both either be `Tensor`s, or both
                `None`s, and must match in shape.

    Other Parameters:
        backend (str): Backend implementation to run with. Choices are: `None` (pick the best
            available one), `"cutlass-fna"`, `"hopper-fna"`, `"blackwell-fna"`, `"flex-fna"`.
            Refer to [backends](backends.md) for more information.

        q_tile_shape (Tuple[int, int, int]): 3-D Tile shape for the query token layout in the
            forward pass kernel. You can use [profiler](profiler.md) to find valid choices for your
            use case, and search for the best combination.

        kv_tile_shape (Tuple[int, int, int]): 3-D Tile shape for the key-value token layout in the
            forward pass kernel. You can use [profiler](profiler.md) to find valid choices for your
            use case, and search for the best combination.

        backward_q_tile_shape (Tuple[int, int, int]): 3-D Tile shape for the query token layout in
            the backward pass kernel. This is ignored by `"flex-fna"`.
            You can use [profiler](profiler.md) to find valid choices for your use case, and
            search for the best combination.

        backward_kv_tile_shape (Tuple[int, int, int]): 3-D Tile shape for the key/value token
            layout in the backward pass kernel. This is ignored by `"flex-fna"`.
            You can use [profiler](profiler.md) to find valid choices for your use case,
            and search for the best combination.

        backward_kv_splits (Tuple[int, int, int]): Number of key/value tiles allowed to work in
            parallel in the backward pass kernel. Like tile shapes, this is a tuple and not an
            integer for neighborhood attention operations, and the size of the tuple corresponds to
            the number of dimensions / rank of the layout of tokens. This is only respected by the
            `"cutlass-fna"` backend, and only when
            [KV parallelism](context.md#kv-parallelism-in-fna) is enabled.

        backward_use_pt_reduction (bool): Whether to use PyTorch eager for computing the `dO * O`
            product required by the backward pass, over the CUTLASS kernel. This only applies to
            the `"cutlass-fna"` backend.

        run_persistent_kernel (bool): Whether to use persistent tile scheduling in the forward pass
            kernel. This only applies to the `"blackwell-fna"` backend.

        kernel_schedule (Optional[str]): Kernel type (Hopper architecture only). Choices are
            `None`: pick the default, `"non"` (non-persistent), `"coop"` (warp-specialized
            cooperative), or `"pp"` (warp-specialized ping-ponging). Refer to
            [Hopper FMHA/FNA backend](backends.md#hopper-fna-fmha) for more information.

        torch_compile (bool): Applies only to the `"flex-fna"` backend. Whether or not to JIT
            compile the attention kernel. Due to this being an experimental feature in PyTorch, we
            do not recommend it, and it is guarded by context flags. Read more in
            [Flex Attention + `torch.compile`](context.md#flex-attention-torchcompile).

        attention_kwargs: arguments to the [attention][natten.attention] operator, if used to
            implement neighborhood cross-attention, or self attention as a fast path for
            neighborhood attention.

            If `additional_{keys,values}` are specified, NATTEN usually performs a separate
            cross-attention using our [attention][natten.attention] operator, and
            [merges][natten.merge_attentions] the results.

            If for a given use case, the neighborhood attention problem is equivalent to self
            attention (not causal along any dims, `kernel_size == (X, Y, Z)`), NATTEN will also
            attempt to directly use [attention][natten.attention].

            You can override arguments to [attention][natten.attention] by passing a
            dictionary here.

            !!! example
                ```python
                out = na3d(
                    q, k, v, kernel_size=kernel_size,
                    ...,
                    attention_kwargs={
                        "backend": "blackwell-fmha",
                        "run_persistent_kernel": True,
                    }
                )
                ```

        return_lse (bool): Whether or not to return the `logsumexp` tensor. `logsumexp` can be used
            in the backward pass, and for [attention merging][natten.merge_attentions].

    Returns:
        output (Tensor): 6-D output tensor, with the heads last layout
            (`[batch, X, Y, Z, heads, head_dim_v]`).

        logsumexp (Tensor): only returned when `return_lse=True`. 5-D logsumexp tensor, with the
            heads last layout (`[batch, X, Y, Z, heads]`).
    """
    return neighborhood_attention_generic(
        query=query,
        key=key,
        value=value,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
        additional_keys=additional_keys,
        additional_values=additional_values,
        attention_kwargs=attention_kwargs,
        backend=backend,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
        backward_q_tile_shape=backward_q_tile_shape,
        backward_kv_tile_shape=backward_kv_tile_shape,
        backward_kv_splits=backward_kv_splits,
        backward_use_pt_reduction=backward_use_pt_reduction,
        run_persistent_kernel=run_persistent_kernel,
        kernel_schedule=kernel_schedule,
        torch_compile=torch_compile,
        return_lse=return_lse,
    )
