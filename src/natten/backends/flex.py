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
import math
import time
from typing import Callable, Optional, Tuple, Union

import torch
from torch import BoolTensor, IntTensor, Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)

from ..token_permute import maybe_pad, maybe_unpad, token_permute, token_unpermute
from ..types import (
    CausalArg1DTypeOrDed,
    CausalArg2DTypeOrDed,
    CausalArg3DTypeOrDed,
    CausalArgType,
    CausalArgTypeOrDed,
    Dimension1DType,
    Dimension1DTypeOrDed,
    Dimension2DType,
    Dimension2DTypeOrDed,
    Dimension3DType,
    Dimension3DTypeOrDed,
    DimensionType,
    DimensionTypeOrDed,
)
from ..utils import log
from ..utils.checks import (
    check_all_args,
    check_args_against_input,
    check_input_size_arg,
    fmha_tensor_checks,
    na_tensor_checks,
)

from .configs.checks import (  # noqa: F401
    _FLEX_COMPILE_SUPPORTED,
    _FLEX_SUPPORTED,
    can_run_flex_attention,
)
from .configs.flex import check_flex_fmha_forward_config, check_flex_fna_forward_config

logger = log.get_logger(__name__)


def get_flex_attention_fn(
    torch_compile: bool, torch_compile_args: Optional[dict] = None
) -> Callable:
    if not torch_compile:
        return flex_attention

    additional_args = torch_compile_args or {}
    additional_args["dynamic"] = False

    return torch.compile(flex_attention, **additional_args)


def _run_flex_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_mask: BlockMask,
    scale: float,
    torch_compile: bool,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
    torch_compile_args: Optional[dict] = None,
) -> Tuple[Tensor, Tensor]:

    # We may need to override the default flex config.
    # Default ones are not guaranteed to work out of the box across architectures.
    # Some oversubscribe shmem even on the B200!
    torch_compile_args = {}

    # Disable flex decoding path
    kernel_options = {
        "FORCE_USE_FLEX_ATTENTION": True,
    }

    if q_tile_size is not None and torch_compile:
        kv_tile_size = kv_tile_size or q_tile_size

        # Have to auto-tune, otherwise torch will only allow the default config.
        torch_compile_args["mode"] = "max-autotune-no-cudagraphs"

        kernel_options["SPARSE_Q_BLOCK_SIZE"] = q_tile_size  # type: ignore[assignment]
        kernel_options["SPARSE_KV_BLOCK_SIZE"] = kv_tile_size  # type: ignore[assignment]
        kernel_options["BLOCK_M"] = q_tile_size  # type: ignore[assignment]
        kernel_options["BLOCK_N"] = kv_tile_size  # type: ignore[assignment]

    flex_fn = get_flex_attention_fn(
        torch_compile=torch_compile, torch_compile_args=torch_compile_args
    )

    return flex_fn(
        q,
        k,
        v,
        block_mask=block_mask,
        return_lse=True,
        scale=scale,
        kernel_options=kernel_options,
    )


def run_flex_attn(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    block_mask: BlockMask,
    scale: float,
    torch_compile: bool,
    torch_compile_args: Optional[dict] = None,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:

    if q_tile_size is not None and kv_tile_size is not None:
        return _run_flex_attn(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=scale,
            torch_compile=torch_compile,
            q_tile_size=q_tile_size,
            kv_tile_size=kv_tile_size,
            torch_compile_args=torch_compile_args,
        )

    # Use smallest tile size combo to try and evade shmem oversubscription
    # The defaults just fail very frequently.
    return _run_flex_attn(
        q,
        k,
        v,
        block_mask=block_mask,
        scale=scale,
        torch_compile=torch_compile,
        q_tile_size=64,
        kv_tile_size=64,
        torch_compile_args=torch_compile_args,
    )


def flex_fmha(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Optional[float] = None,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
    torch_compile: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:

    fmha_tensor_checks(query, key, value, must_match_head_dims=True)

    q_tile_size, kv_tile_size = check_flex_fmha_forward_config(
        input_tensor=query,
        q_tile_size=q_tile_size,
        kv_tile_size=kv_tile_size,
    )

    scale = scale or query.shape[-1] ** -0.5

    assert can_run_flex_attention(
        query, key, value, torch_compile=torch_compile, raise_error=True
    )

    batch_size, seqlen_q, num_heads, head_dim = query.shape
    seqlen_kv = key.shape[1]

    # Flex and torch attention use heads first layout
    query_ = query.reshape(batch_size, seqlen_q, num_heads, head_dim).transpose(1, 2)
    key_ = key.reshape(batch_size, seqlen_kv, num_heads, head_dim).transpose(1, 2)
    value_ = value.reshape(batch_size, seqlen_kv, num_heads, head_dim).transpose(1, 2)

    out_, lse_ = run_flex_attn(
        query_,
        key_,
        value_,
        block_mask=None,  # type: ignore[arg-type]
        scale=scale,
        torch_compile=torch_compile,
        q_tile_size=q_tile_size,
        kv_tile_size=kv_tile_size,
    )

    out = out_.transpose(1, 2).reshape(batch_size, seqlen_q, num_heads, head_dim)
    lse = lse_.transpose(1, 2).reshape(batch_size, seqlen_q, num_heads)

    if return_lse:
        return out, lse

    return out


# TODO: move me elsewhere?
def idx2crd(index, shape) -> tuple:
    rank = len(shape)
    coord = []
    residual = index
    for i in range(rank - 1, -1, -1):
        coord.append(residual % shape[i])
        residual = residual // shape[i]

    # assert residual == 0
    return tuple(coord[::-1])


def get_na_flex_mask(
    device: str,
    na_dim: int,
    num_heads: int,
    qkv_shape: DimensionType,
    kernel_size: DimensionType,
    stride: DimensionType,
    dilation: DimensionType,
    is_causal: CausalArgType,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    q_shape: Optional[DimensionType] = None,
    kv_shape: Optional[DimensionType] = None,
    torch_compile: bool = False,
):
    flex_mask_start_time = time.perf_counter()
    do_token_permute = q_tile_shape is not None and kv_tile_shape is not None
    if do_token_permute:
        if q_tile_shape is None or kv_tile_shape is None:
            raise ValueError(
                "Please specify Q and KV tile shapes for multi dimensional tiling. "
                f"Got {q_tile_shape=}, {kv_tile_shape=}."
            )

        if q_shape is None or kv_shape is None:
            raise ValueError(
                "Please specify q_shape and kv_shape for multi dimensional tiling."
            )

        if len(q_tile_shape) != na_dim or len(kv_tile_shape) != na_dim:
            raise ValueError(
                "Q and KV tile shapes must match the number of dimensions in the "
                f"token layout ({na_dim}, got {q_tile_shape=}, {kv_tile_shape=}."
            )

        if any(x % t != 0 for x, t in zip(q_shape, q_tile_shape)):
            raise ValueError(
                "Input must be divisible by Q tile shape, but got "
                f"{q_shape=}, {q_tile_shape=}."
            )

        if any(x % t != 0 for x, t in zip(kv_shape, kv_tile_shape)):
            raise ValueError(
                "Input must be divisible by KV tile shape, but got "
                f"{kv_shape=}, {kv_tile_shape=}."
            )

        q_rest_shape = tuple(x // t for x, t in zip(q_shape, q_tile_shape))
        kv_rest_shape = tuple(x // t for x, t in zip(kv_shape, kv_tile_shape))

    def single_dim_tiling_mask(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
        qkv_shape,
        kernel_size,
        stride,
        dilation,
        is_causal,
    ) -> BoolTensor:

        # Reconstruct global Q and KV coordinates
        q_crd = idx2crd(q_idx, qkv_shape)
        kv_crd = idx2crd(kv_idx, qkv_shape)

        # Coordinates within dilation group
        q_crd_di = tuple(x // d for x, d in zip(q_crd, dilation))
        kv_crd_di = tuple(x // d for x, d in zip(kv_crd, dilation))

        # Dilation group coordinates
        q_dilation_group_crd = tuple(x % d for x, d in zip(q_crd, dilation))
        kv_dilation_group_crd = tuple(x % d for x, d in zip(kv_crd, dilation))

        # Fixup input shape according to dilation group
        dilation_group_padding = tuple(
            1 - ((dg + (d - (x % d))) // d)
            for dg, d, x in zip(q_dilation_group_crd, dilation, qkv_shape)
        )
        qkv_shape_corrected = tuple(
            (x // d) + p for p, d, x in zip(dilation_group_padding, dilation, qkv_shape)
        )

        # Window size left and right (non-causal only)
        window_size_left = tuple(w // 2 for w in kernel_size)
        window_size_right = tuple(w // 2 + (w % 2 - 1) for w in kernel_size)

        masks = []
        for i in range(na_dim):
            if is_causal[i]:
                # Leader is the last (right-most) query in the stride group.
                stride_group_leader = torch.min(
                    (q_crd_di[i] // stride[i]) * stride[i] + stride[i] - 1,
                    qkv_shape_corrected[i] - 1,
                )

                mask = (
                    (
                        q_crd_di[i] - kv_crd_di[i] >= 0
                    )  # window still ends at query index
                    & (stride_group_leader - kv_crd_di[i] < kernel_size[i])
                    & (q_dilation_group_crd[i] == kv_dilation_group_crd[i])
                )
            else:
                # Leader is the center-most query in the stride group.
                # If stride is even, choose the right hand side center query.
                stride_group_leader = torch.min(
                    (q_crd_di[i] // stride[i]) * stride[i] + (stride[i] // 2),
                    qkv_shape_corrected[i] - 1,
                )

                window_center = stride_group_leader.clamp(
                    window_size_left[i] * torch.ones_like(qkv_shape_corrected[i]),
                    qkv_shape_corrected[i] - 1 - window_size_right[i],
                )
                w0 = window_center - kv_crd_di[i]
                w1 = kv_crd_di[i] - window_center
                mask = (
                    ((0 <= w0) & (w0 <= window_size_left[i]))
                    | ((0 <= w1) & (w1 <= window_size_right[i]))
                ) & (q_dilation_group_crd[i] == kv_dilation_group_crd[i])

            masks.append(mask)

        return functools.reduce(lambda x, y: x & y, masks)  # type: ignore

    def multi_dim_tiling_mask(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
        q_tile_size: int,
        kv_tile_size: int,
        q_tile_shape,
        kv_tile_shape,
        qkv_shape,
        kernel_size,
        stride,
        dilation,
        is_causal,
    ) -> BoolTensor:

        # Reconstruct global Q and KV coordinates
        q_tile_idx = q_idx // q_tile_size
        kv_tile_idx = kv_idx // kv_tile_size
        q_tile_offset = q_idx % q_tile_size
        kv_tile_offset = kv_idx % q_tile_size
        q_tile_coord = idx2crd(q_tile_idx, q_rest_shape)
        kv_tile_coord = idx2crd(kv_tile_idx, kv_rest_shape)
        q_tile_offset_coord = idx2crd(q_tile_offset, q_tile_shape)
        kv_tile_offset_coord = idx2crd(kv_tile_offset, kv_tile_shape)

        q_crd = tuple(
            tile_crd * tile_sz + tile_off
            for tile_crd, tile_sz, tile_off in zip(
                q_tile_coord, q_tile_shape, q_tile_offset_coord
            )
        )
        kv_crd = tuple(
            tile_crd * tile_sz + tile_off
            for tile_crd, tile_sz, tile_off in zip(
                kv_tile_coord, kv_tile_shape, kv_tile_offset_coord
            )
        )

        # Dilation group coordinates
        # h_actual = h % num_heads
        dilation_group_idx = h // num_heads
        dilation_group_crd = idx2crd(dilation_group_idx, dilation)

        # Fixup input shape according to dilation group
        dilation_group_padding = tuple(
            1 - ((dg + (d - (x % d))) // d)
            for dg, d, x in zip(dilation_group_crd, dilation, qkv_shape)
        )
        qkv_shape_corrected = tuple(
            (x // d) + p for p, d, x in zip(dilation_group_padding, dilation, qkv_shape)
        )

        # Window size left and right (non-causal only)
        window_size_left = tuple(w // 2 for w in kernel_size)
        window_size_right = tuple(w // 2 + (w % 2 - 1) for w in kernel_size)

        masks = []
        for i in range(na_dim):
            if is_causal[i]:
                # Leader is the last (right-most) query in the stride group.
                stride_group_leader = torch.min(
                    (q_crd[i] // stride[i]) * stride[i] + stride[i] - 1,
                    qkv_shape_corrected[i] - 1,
                )

                mask = (
                    q_crd[i] - kv_crd[i] >= 0
                ) & (  # window still ends at query index
                    stride_group_leader - kv_crd[i] < kernel_size[i]
                )
            else:
                # Leader is the center-most query in the stride group.
                # If stride is even, choose the right hand side center query.
                stride_group_leader = torch.min(
                    (q_crd[i] // stride[i]) * stride[i] + (stride[i] // 2),
                    qkv_shape_corrected[i] - 1,
                )

                window_center = stride_group_leader.clamp(
                    window_size_left[i] * torch.ones_like(qkv_shape_corrected[i]),
                    qkv_shape_corrected[i] - 1 - window_size_right[i],
                )
                w0 = window_center - kv_crd[i]
                w1 = kv_crd[i] - window_center

                mask = ((0 <= w0) & (w0 <= window_size_left[i])) | (
                    (0 <= w1) & (w1 <= window_size_right[i])
                )

            masks.append(mask)

        return functools.reduce(lambda x, y: x & y, masks)  # type: ignore

    mask_mod = None
    seq_length_q = seq_length_kv = math.prod(qkv_shape)
    q_tile_size, kv_tile_size = 64, 64
    if do_token_permute:
        assert q_shape is not None
        assert kv_shape is not None
        assert q_tile_shape is not None
        assert kv_tile_shape is not None

        seq_length_q = math.prod(q_shape)
        seq_length_kv = math.prod(kv_shape)
        q_tile_size, kv_tile_size = math.prod(q_tile_shape), math.prod(kv_tile_shape)

        mask_mod = functools.partial(
            multi_dim_tiling_mask,
            q_tile_size=q_tile_size,
            kv_tile_size=kv_tile_size,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            qkv_shape=qkv_shape,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
        )
    else:
        mask_mod = functools.partial(
            single_dim_tiling_mask,
            qkv_shape=qkv_shape,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
        )

    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=seq_length_q,
        KV_LEN=seq_length_kv,
        _compile=torch_compile,
        BLOCK_SIZE=(q_tile_size, kv_tile_size),
        device=device,
    )
    flex_mask_end_time = time.perf_counter()
    flex_mask_time = flex_mask_end_time - flex_mask_start_time
    logger.debug(
        f"Flex Attention block mask ({torch_compile=}) created in {flex_mask_time:.2f} seconds."
    )
    return block_mask


def flex_fna_generic(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: DimensionTypeOrDed,
    stride: DimensionTypeOrDed = 1,
    dilation: DimensionTypeOrDed = 1,
    is_causal: Optional[CausalArgTypeOrDed] = False,
    scale: Optional[float] = None,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
    torch_compile: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:

    na_tensor_checks(query, key, value, must_match_head_dims=True)

    na_dim = query.dim() - 3  # batch, heads, head_dim

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

    batch_size, *qkv_shape_in, num_heads, head_dim = query.shape
    qkv_shape = check_input_size_arg(na_dim, qkv_shape_in)

    scale = scale or query.shape[-1] ** -0.5

    assert can_run_flex_attention(
        query, key, value, torch_compile=torch_compile, raise_error=True
    )

    if (q_tile_shape is None) ^ (kv_tile_shape is None):
        raise ValueError(
            "Please specify both q_tile_shape and kv_tile_shape, or neither one. "
            f"Got {q_tile_shape=}, {kv_tile_shape=}."
        )

    do_token_permute = q_tile_shape is not None and kv_tile_shape is not None

    q_shape = kv_shape = qkv_shape
    q_tile_size: Optional[int] = None
    kv_tile_size: Optional[int] = None
    if do_token_permute:
        q_tile_shape, kv_tile_shape = check_flex_fna_forward_config(
            input_tensor=query,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
        )

        q_tile_size = math.prod(q_tile_shape)
        kv_tile_size = math.prod(kv_tile_shape)

        query_pad, padding = maybe_pad(query, q_tile_shape, dilation=dilation)
        key, _ = maybe_pad(key, kv_tile_shape, dilation=dilation)
        value, _ = maybe_pad(value, kv_tile_shape, dilation=dilation)

        query_perm, q_shape, qR = token_permute(
            query_pad, q_tile_shape, dilation=dilation, flip_tiled_dims=False
        )
        key_perm, k_shape, kR = token_permute(
            key, kv_tile_shape, dilation=dilation, flip_tiled_dims=False
        )
        value_perm, v_shape, vR = token_permute(
            value, kv_tile_shape, dilation=dilation, flip_tiled_dims=False
        )

        assert k_shape == v_shape
        kv_shape = k_shape

        # Token permute already flattens to 1-D
        # Flex uses heads first layout
        query_ = query_perm.transpose(1, 2)
        key_ = key_perm.transpose(1, 2)
        value_ = value_perm.transpose(1, 2)

    else:
        seqlen = math.prod(qkv_shape)
        # Flex uses heads first layout
        query_ = query.reshape(batch_size, seqlen, num_heads, head_dim).transpose(1, 2)
        key_ = key.reshape(batch_size, seqlen, num_heads, head_dim).transpose(1, 2)
        value_ = value.reshape(batch_size, seqlen, num_heads, head_dim).transpose(1, 2)

    na_block_mask = get_na_flex_mask(
        device=query.device.type,
        na_dim=na_dim,
        num_heads=num_heads,
        qkv_shape=qkv_shape,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        q_tile_shape=q_tile_shape,
        kv_tile_shape=kv_tile_shape,
        q_shape=q_shape,
        kv_shape=kv_shape,
        torch_compile=torch_compile,
    )

    out_, lse_ = run_flex_attn(
        query_,
        key_,
        value_,
        na_block_mask,
        scale,
        torch_compile=torch_compile,
        q_tile_size=q_tile_size,
        kv_tile_size=kv_tile_size,
    )

    if do_token_permute:
        out = out_.transpose(1, 2)
        lse = lse_.transpose(1, 2).unsqueeze(-1)

        out = maybe_unpad(
            token_unpermute(
                out, q_tile_shape, q_shape, qR, dilation=dilation, flip_tiled_dims=False
            ),
            padding,
        )
        lse = maybe_unpad(
            token_unpermute(
                lse, q_tile_shape, q_shape, qR, dilation=dilation, flip_tiled_dims=False
            ),
            padding,
        ).squeeze(-1)
    else:
        out = out_.transpose(1, 2).reshape(batch_size, *qkv_shape, num_heads, head_dim)
        lse = lse_.transpose(1, 2).reshape(batch_size, *qkv_shape, num_heads)

    if return_lse:
        return out, lse

    return out


def na1d_flex(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension1DTypeOrDed,
    stride: Dimension1DTypeOrDed = 1,
    dilation: Dimension1DTypeOrDed = 1,
    is_causal: Optional[CausalArg1DTypeOrDed] = False,
    scale: Optional[float] = None,
    q_tile_shape: Optional[Dimension1DType] = None,
    kv_tile_shape: Optional[Dimension1DType] = None,
    torch_compile: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    return flex_fna_generic(
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
        return_lse=return_lse,
    )


def na2d_flex(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension2DTypeOrDed,
    stride: Dimension2DTypeOrDed = 1,
    dilation: Dimension2DTypeOrDed = 1,
    is_causal: Optional[CausalArg2DTypeOrDed] = False,
    scale: Optional[float] = None,
    q_tile_shape: Optional[Dimension2DType] = None,
    kv_tile_shape: Optional[Dimension2DType] = None,
    torch_compile: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    return flex_fna_generic(
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
        return_lse=return_lse,
    )


def na3d_flex(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension3DTypeOrDed,
    stride: Dimension3DTypeOrDed = 1,
    dilation: Dimension3DTypeOrDed = 1,
    is_causal: Optional[CausalArg3DTypeOrDed] = False,
    scale: Optional[float] = None,
    q_tile_shape: Optional[Dimension3DType] = None,
    kv_tile_shape: Optional[Dimension3DType] = None,
    torch_compile: bool = False,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    return flex_fna_generic(
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
        return_lse=return_lse,
    )
