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

from __future__ import annotations

import math
from typing import Iterable, Tuple

import pytest
import torch

import natten


def _corrected_axis_size(axis_size: int, dilation: int, dilation_group_coord: int) -> int:
    # Equivalent to slicing: x[dg::dilation], i.e. number of tokens in a dilation group.
    return (axis_size - dilation_group_coord + dilation - 1) // dilation


def _axis_mask(
    q_di: int,
    kv_di: int,
    *,
    corrected_size: int,
    kernel_size: int,
    stride: int,
    is_causal: bool,
) -> bool:
    if is_causal:
        stride_group_leader = min((q_di // stride) * stride + (stride - 1), corrected_size - 1)
        return (q_di - kv_di >= 0) and (stride_group_leader - kv_di < kernel_size)

    window_size_left = kernel_size // 2
    window_size_right = kernel_size // 2 + (kernel_size % 2 - 1)
    stride_group_leader = min((q_di // stride) * stride + (stride // 2), corrected_size - 1)
    window_center = max(
        window_size_left,
        min(stride_group_leader, corrected_size - 1 - window_size_right),
    )
    return (window_center - window_size_left) <= kv_di <= (window_center + window_size_right)


def _bruteforce_na3d(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    is_causal: Tuple[bool, bool, bool],
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    b, x, y, z, h, d = q.shape
    dv = v.shape[-1]
    n = x * y * z

    qf = q.reshape(b, n, h, d)
    kf = k.reshape(b, n, h, d)
    vf = v.reshape(b, n, h, dv)

    out = torch.empty((b, n, h, dv), dtype=q.dtype)
    lse = torch.empty((b, n, h), dtype=torch.float32)

    coords = []
    for xi in range(x):
        for yi in range(y):
            for zi in range(z):
                coords.append((xi, yi, zi))

    for qi, (qx, qy, qz) in enumerate(coords):
        dgx, dgy, dgz = (qx % dilation[0], qy % dilation[1], qz % dilation[2])
        cx = _corrected_axis_size(x, dilation[0], dgx)
        cy = _corrected_axis_size(y, dilation[1], dgy)
        cz = _corrected_axis_size(z, dilation[2], dgz)

        qx_di, qy_di, qz_di = (qx // dilation[0], qy // dilation[1], qz // dilation[2])

        # Collect masked logits over all KV tokens.
        logits = torch.full((b, h, n), float("-inf"), dtype=torch.float32)
        for kvi, (kx, ky, kz) in enumerate(coords):
            if (kx % dilation[0], ky % dilation[1], kz % dilation[2]) != (dgx, dgy, dgz):
                continue

            kx_di, ky_di, kz_di = (kx // dilation[0], ky // dilation[1], kz // dilation[2])
            if not _axis_mask(
                qx_di,
                kx_di,
                corrected_size=cx,
                kernel_size=kernel_size[0],
                stride=stride[0],
                is_causal=is_causal[0],
            ):
                continue
            if not _axis_mask(
                qy_di,
                ky_di,
                corrected_size=cy,
                kernel_size=kernel_size[1],
                stride=stride[1],
                is_causal=is_causal[1],
            ):
                continue
            if not _axis_mask(
                qz_di,
                kz_di,
                corrected_size=cz,
                kernel_size=kernel_size[2],
                stride=stride[2],
                is_causal=is_causal[2],
            ):
                continue

            # logits[b, h] for this kv
            logits[:, :, kvi] = torch.einsum(
                "bhd,bhd->bh", qf[:, qi, :, :], kf[:, kvi, :, :]
            ).float() * scale

        lse[:, qi, :] = torch.logsumexp(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)  # [B, H, N]
        out[:, qi, :, :] = torch.einsum(
            "bhn,bhnd->bhd", probs, vf.float().permute(0, 2, 1, 3)
        ).to(q.dtype)

    out = out.reshape(b, x, y, z, h, dv)
    lse = lse.reshape(b, x, y, z, h)
    return out, lse


@pytest.mark.parametrize(
    "input_shape,kernel_size,stride,dilation,is_causal",
    [
        ((4, 4, 4), (3, 3, 3), (1, 1, 1), (1, 1, 1), (False, False, False)),
        ((4, 3, 2), (3, 3, 2), (2, 1, 1), (1, 1, 1), (False, False, False)),
        ((4, 6, 3), (3, 3, 3), (2, 2, 1), (1, 2, 1), (False, False, False)),
        ((4, 4, 3), (4, 2, 2), (2, 2, 1), (1, 1, 1), (False, False, False)),
        ((4, 6, 3), (3, 3, 3), (2, 2, 1), (1, 2, 1), (True, False, True)),
    ],
)
def test_torch_fna_na3d_matches_bruteforce(
    input_shape: Tuple[int, int, int],
    kernel_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    dilation: Tuple[int, int, int],
    is_causal: Tuple[bool, bool, bool],
):
    torch.manual_seed(0)

    b = 2
    h = 2
    d = 8
    dv = 4
    x, y, z = input_shape

    q = torch.randn((b, x, y, z, h, d), dtype=torch.float32)
    k = torch.randn((b, x, y, z, h, d), dtype=torch.float32)
    v = torch.randn((b, x, y, z, h, dv), dtype=torch.float32)
    scale = 1.0 / math.sqrt(d)

    out_ref, lse_ref = _bruteforce_na3d(
        q,
        k,
        v,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
    )

    out, lse = natten.na3d(
        q,
        k,
        v,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
        backend="torch-fna",
        return_lse=True,
    )

    torch.testing.assert_close(out, out_ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-4, atol=1e-4)
