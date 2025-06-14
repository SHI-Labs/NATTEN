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
from typing import Optional

import torch  # noqa: F401
from torch import nn, Tensor

from .functional import neighborhood_attention_generic
from .types import (
    CausalArg1DTypeOrDed,
    CausalArg2DTypeOrDed,
    CausalArg3DTypeOrDed,
    CausalArgTypeOrDed,
    Dimension1DTypeOrDed,
    Dimension2DTypeOrDed,
    Dimension3DTypeOrDed,
    DimensionTypeOrDed,
)
from .utils.checks import check_all_args


class NeighborhoodAttentionGeneric(nn.Module):
    def __init__(
        self,
        na_dim: int,
        embed_dim: int,
        num_heads: int,
        kernel_size: DimensionTypeOrDed,
        stride: DimensionTypeOrDed = 1,
        dilation: DimensionTypeOrDed = 1,
        is_causal: CausalArgTypeOrDed = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        kernel_size, stride, dilation, is_causal = check_all_args(
            na_dim, kernel_size, stride, dilation, is_causal
        )

        if embed_dim % num_heads != 0:
            raise ValueError(
                "Number of attention heads must evenly divide embedding dimension, "
                f"got {embed_dim=}, {num_heads=}."
            )

        self.na_dim = na_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.is_causal = is_causal

        self.expected_input_tensor_rank = self.na_dim + 2  # batch, embedding dim

        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != self.expected_input_tensor_rank:
            raise ValueError(
                f"NeighborhoodAttention{self.na_dim}D expected a tensor with rank "
                f"{self.expected_input_tensor_rank} ({self.na_dim} for token layout, 1 for batch, "
                f"1 for embedding dimension), got {x.dim()=}."
            )

        B, *input_shape, C = x.shape

        if C != self.embed_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embed_dim}, got {C} ({x.shape=})."
            )

        # 3, batch, *input_shape, heads, head_dim
        permutation = (
            [self.na_dim + 1, 0]
            + [x + 1 for x in range(self.na_dim)]
            + [self.na_dim + 2, self.na_dim + 3]
        )
        qkv = (
            self.qkv(x)
            .reshape(B, *input_shape, 3, self.num_heads, self.head_dim)
            .permute(*permutation)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        x = neighborhood_attention_generic(
            q,
            k,
            v,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            is_causal=self.is_causal,
            scale=self.scale,
        )
        x = x.reshape(B, *input_shape, C)

        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, "
            + f"stride={self.stride}, "
            + f"dilation={self.dilation}, "
            + f"is_causal={self.is_causal}"
        )


class NeighborhoodAttention1D(NeighborhoodAttentionGeneric):
    """
    1-D Neighborhood Attention torch module.

    Performs QKV and output linear projections in addition to the [na1d][natten.na1d] operation.

    Args:
        embed_dim: Embedding dimension size (a.k.a. number of channels, latent size).
            !!! note
                This is not `head_dim`. It's `head_dim * num_heads`.

        num_heads: Number of attention heads.

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

        qkv_bias: Enable bias in the QKV linear projection.

        qk_scale: Attention scale. Defaults to `head_dim ** -0.5`.

        proj_drop: Dropout score for projection layer. Defaults is `0.0` (no dropout).

    Example:
        ```python3
        import torch
        from natten import NeighborhoodAttention1D

        num_heads = 4
        head_dim = 128
        embed_dim = num_heads * head_dim

        model = NeighborhoodAttention1D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kernel_size=2048,
            stride=2,
            dilation=4,
            is_causal=True
        )

        batch = 1
        seqlen = 4096 # (1)!

        x = torch.randn(batch, seqlen, embed_dim) # (2)!
        y = model(x) # (3)!
        ```

        1. Tokens are arranged in a sequential layout of size 4096, to which we apply a
            kernel size of 2048, stride 2, dilation 4, and apply causal masking.

        2. `x.shape == [1, 4096, 512]`
        3. `y.shape == [1, 4096, 512]`
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kernel_size: Dimension1DTypeOrDed,
        stride: Dimension1DTypeOrDed = 1,
        dilation: Dimension1DTypeOrDed = 1,
        is_causal: CausalArg1DTypeOrDed = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        proj_drop: float = 0.0,
    ):
        super().__init__(
            na_dim=1,
            embed_dim=embed_dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_drop=proj_drop,
        )


class NeighborhoodAttention2D(NeighborhoodAttentionGeneric):
    """
    2-D Neighborhood Attention torch module.

    Performs QKV and output linear projections in addition to the [na2d][natten.na2d] operation.

    Args:
        embed_dim: Embedding dimension size (a.k.a. number of channels, latent size).
            !!! note
                This is not `head_dim`. It's `head_dim * num_heads`.

        num_heads: Number of attention heads.

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

        qkv_bias: Enable bias in the QKV linear projection.

        qk_scale: Attention scale. Defaults to `head_dim ** -0.5`.

        proj_drop: Dropout score for projection layer. Defaults is `0.0` (no dropout).

    Example:
        ```python3
        import torch
        from natten import NeighborhoodAttention2D

        num_heads = 4
        head_dim = 128
        embed_dim = num_heads * head_dim

        model = NeighborhoodAttention2D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kernel_size=(8, 16),
            stride=(1, 2),
            dilation=(2, 1),
            is_causal=False
        )

        batch = 1
        token_layout_shape = (16, 32) # (1)!

        x = torch.randn(batch, *token_layout_shape, embed_dim) # (2)!
        y = model(x) # (3)!
        ```

        1. Tokens are arranged in a 16 x 32 layout, to which we apply a
            kernel size of 8 x 16,
            stride 1 x 2,
            and dilation 2 x 1.

        2. `x.shape == [1, 16, 32, 512]`
        3. `y.shape == [1, 16, 32, 512]`
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kernel_size: Dimension2DTypeOrDed,
        stride: Dimension2DTypeOrDed = 1,
        dilation: Dimension2DTypeOrDed = 1,
        is_causal: CausalArg2DTypeOrDed = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        proj_drop: float = 0.0,
    ):
        super().__init__(
            na_dim=2,
            embed_dim=embed_dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_drop=proj_drop,
        )


class NeighborhoodAttention3D(NeighborhoodAttentionGeneric):
    """
    3-D Neighborhood Attention torch module.

    Performs QKV and output linear projections in addition to the [na3d][natten.na3d] operation.

    Args:
        embed_dim: Embedding dimension size (a.k.a. number of channels, latent size).
            !!! note
                This is not `head_dim`. It's `head_dim * num_heads`.

        num_heads: Number of attention heads.

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

        qkv_bias: Enable bias in the QKV linear projection.

        qk_scale: Attention scale. Defaults to `head_dim ** -0.5`.

        proj_drop: Dropout score for projection layer. Defaults is `0.0` (no dropout).

    Example:
        ```python3
        import torch
        from natten import NeighborhoodAttention3D

        num_heads = 4
        head_dim = 128
        embed_dim = num_heads * head_dim

        model = NeighborhoodAttention3D(
            embed_dim=embed_dim,
            num_heads=num_heads,
            kernel_size=(4, 8, 12),
            stride=(1, 1, 4),
            dilation=(1, 2, 1),
            is_causal=(True, False, False)
        )

        batch = 1
        token_layout_shape = (12, 16, 20) # (1)!

        x = torch.randn(batch, *token_layout_shape, embed_dim) # (2)!
        y = model(x) # (3)!
        ```

        1. Tokens are arranged in a 12 x 16 x 20 layout, to which we apply a
            kernel size of 4 x 8 x 12,
            stride 1 x 1 x 4,
            dilation 1 x 2 x 1, and apply causal masking to the left-most dimension (12).

        2. `x.shape == [1, 12, 16, 20, 512]`
        3. `y.shape == [1, 12, 16, 20, 512]`
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kernel_size: Dimension3DTypeOrDed,
        stride: Dimension3DTypeOrDed = 1,
        dilation: Dimension3DTypeOrDed = 1,
        is_causal: CausalArg3DTypeOrDed = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        proj_drop: float = 0.0,
    ):
        super().__init__(
            na_dim=3,
            embed_dim=embed_dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            proj_drop=proj_drop,
        )
