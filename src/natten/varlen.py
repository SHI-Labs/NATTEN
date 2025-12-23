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

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

#    choose_backend,
#    choose_fmha_backend,
#    cutlass_blackwell_fmha,
# )
from natten.backends import cutlass_blackwell_fna_varlen_generic
from natten.backends.configs.cutlass_blackwell import (
    check_cutlass_blackwell_fna_backward_config_tensorless,
    check_cutlass_blackwell_fna_forward_config_tensorless,
)
from natten.token_permute import generate_fna_varlen_metadata, get_na_dim
from natten.types import CausalArgTypeOrDed, DimensionType, DimensionTypeOrDed
from natten.utils import log
from natten.utils.checks import fmha_tensor_checks

#    check_all_args,
#    check_args_against_input,
#    check_kernel_schedule,
#    is_self_attention,
#    na_tensor_checks,
#    varlen_tensor_checks,
# )

logger = log.get_logger(__name__)


def configure_varlen(
    token_layout_list: List[DimensionType],
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    requires_grad: bool,
    #
    kernel_size: DimensionTypeOrDed,
    stride: DimensionTypeOrDed = 1,
    dilation: DimensionTypeOrDed = 1,
    is_causal: Optional[CausalArgTypeOrDed] = False,
    #
    backend: Optional[str] = None,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    backward_q_tile_shape: Optional[DimensionType] = None,
    backward_kv_tile_shape: Optional[DimensionType] = None,
) -> dict:
    """
    Selects a Variable-Length FNA backend, along with forward and backward configurations, and
    generates metadata necessary to run the operation.
    This function is NOT torch-compilable and must be done ahead of time.

    Parameters:
        token_layout_list (list[tuple]): list of token layouts that describe the various independent
            sets of tokens / sequences in QKV. All elements must be integer tuples of size 1, 2, or
            3, and match each other in size as well.

        head_dim (int): Attention head dimension.

        device (torch.device): Target PyTorch device for runtime.

        dtype (torch.dtype): Tensor element type.

        requires_grad (bool): Whether or not tensors will require backward pass.

        kernel_size (tuple): kernel / window size must be provided for verification.

        stride (Optional[tuple]): stride parameter, if used, must be provided for verification.

        dilation (Optional[tuple]): dilation parameter, if used, must be provided for verification.

        is_causal (Optional[tuple]): is_causal parameter, if used, must be provided for verification.

    Other Parameters:
        backend (str): Backend implementation to run with. Picks the best available one if
            not specified. Refer to [backends](backends.md) for more information.

        q_tile_shape (tuple): Tile shape for the query token layout in the forward pass kernel.

        kv_tile_shape (tuple): Tile shape for the key-value token layout in the forward pass kernel.

        backward_q_tile_shape (tuple): Tile shape for the query token layout in the backward pass
            kernel.

        backward_kv_tile_shape (tuple): Tile shape for the key/value token layout in the backward
            pass kernel.

    Returns:
        varlen_metadata (dict): Runtime metadata for the current use case.
    """

    # get_na_dim acts as a type verifier
    na_dim = get_na_dim(token_layout_list=token_layout_list)
    assert na_dim in [1, 2, 3]

    # TODO: make proper backend selectors when we extend to more backends
    # TODO: use natten.backends.choose_backend when we extend the tensor-less APIs.
    if backend is not None and backend != "blackwell-fna":
        raise NotImplementedError(
            "Varlen FNA is only available in the Blackwell FNA API."
        )
    backend = "blackwell-fna"
    fwd_checker = check_cutlass_blackwell_fna_forward_config_tensorless
    bwd_checker = check_cutlass_blackwell_fna_backward_config_tensorless
    flip_tiled_dims = True

    q_tile_shape, kv_tile_shape = fwd_checker(
        na_dim=na_dim,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
    )
    if requires_grad:
        backward_q_tile_shape, backward_kv_tile_shape = bwd_checker(
            na_dim=na_dim,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
            q_tile_shape=backward_q_tile_shape,
            kv_tile_shape=backward_kv_tile_shape,
        )
    else:
        backward_q_tile_shape, backward_kv_tile_shape = None, None

    varlen_metadata = generate_fna_varlen_metadata(
        token_layout_list=token_layout_list,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
        backward_q_tile_shape=backward_q_tile_shape,
        backward_kv_tile_shape=backward_kv_tile_shape,
        device=device,
        flip_tiled_dims=flip_tiled_dims,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
    )

    return varlen_metadata


def neighborhood_attention_varlen(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: DimensionTypeOrDed,
    stride: DimensionTypeOrDed = 1,
    dilation: DimensionTypeOrDed = 1,
    is_causal: Optional[CausalArgTypeOrDed] = False,
    # Varlen-specific args: at least one must be specified
    token_layout_list: Optional[List[DimensionType]] = None,
    varlen_metadata: Optional[dict] = None,
    #
    scale: Optional[float] = None,
    # Perf-related args
    backend: Optional[str] = None,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    backward_q_tile_shape: Optional[DimensionType] = None,
    backward_kv_tile_shape: Optional[DimensionType] = None,
    run_persistent_kernel: bool = True,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    # TODO: docstring

    # We use FMHA verifiers here because tensors are sequence-packed
    fmha_tensor_checks(
        query,
        key,
        value,
        must_match_head_dims=True,
        supports_gqa_mqa=True,
        backend_name="Variable-length Neighborhood Attention",
    )

    # get_na_dim acts as a type verifier
    na_dim = get_na_dim(
        varlen_metadata=varlen_metadata, token_layout_list=token_layout_list
    )
    assert na_dim in [1, 2, 3]

    # TODO: make proper backend selectors when we extend to more backends
    # TODO: use natten.backends.choose_backend when we extend the tensor-less APIs.
    if backend is not None and backend != "blackwell-fna":
        raise NotImplementedError(
            "Varlen FNA is only available in the Blackwell FNA API."
        )
    backend = "blackwell-fna"

    if varlen_metadata is None and token_layout_list is not None:
        varlen_metadata = configure_varlen(
            token_layout_list=token_layout_list,
            head_dim=query.shape[-1],
            device=query.device,
            dtype=query.dtype,
            requires_grad=query.requires_grad,
            #
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            #
            backend=backend,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            backward_q_tile_shape=backward_q_tile_shape,
            backward_kv_tile_shape=backward_kv_tile_shape,
        )
    assert varlen_metadata is not None

    # kernel_size, stride, dilation, and is_causal are verified in
    # generate_fna_varlen_metadata

    scale = scale or query.shape[-1] ** -0.5

    if backend == "blackwell-fna":
        outputs = cutlass_blackwell_fna_varlen_generic(
            query=query,
            key=key,
            value=value,
            varlen_metadata=varlen_metadata,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            scale=scale,
            run_persistent_kernel=run_persistent_kernel,
            return_lse=return_lse,
        )

    else:
        raise NotImplementedError(f"Backend {backend} does not implement varlen FNA.")

    return outputs
