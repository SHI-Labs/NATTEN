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

import os
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from natten.types import CausalArgTypeOrDed, Dimension3DType, DimensionTypeOrDed
from natten.utils.checks import (
    check_all_args,
    check_args_against_input,
    na_tensor_checks,
)


def _get_torch_fna_chunk_size() -> int:
    # Chunking is essential to keep memory bounded in eager PyTorch.
    # Environment override lets users tune for their workload/device.
    val = os.getenv("NATTEN_TORCH_FNA_CHUNK_SIZE", "").strip()
    if not val:
        return 1024
    try:
        chunk_size = int(val)
    except ValueError as e:
        raise ValueError(
            f"Invalid NATTEN_TORCH_FNA_CHUNK_SIZE={val!r}; expected an integer."
        ) from e
    if chunk_size <= 0:
        raise ValueError(
            f"Invalid NATTEN_TORCH_FNA_CHUNK_SIZE={val!r}; expected a positive integer."
        )
    return chunk_size


def _axis_kv_positions(
    q_coord: Tensor,
    *,
    axis_size: int,
    kernel_size: int,
    stride: int,
    is_causal: bool,
) -> Tuple[Tensor, Tensor]:
    device = q_coord.device
    dtype = q_coord.dtype
    k = kernel_size

    ar = torch.arange(k, device=device, dtype=dtype).unsqueeze(0)  # [1, k]
    axis_max = torch.tensor(axis_size - 1, device=device, dtype=dtype)

    if is_causal:
        stride_group_leader = torch.minimum(
            (q_coord // stride) * stride + (stride - 1), axis_max
        )
        kv = stride_group_leader.unsqueeze(1) - (k - 1) + ar
        mask = (kv >= 0) & (kv < axis_size) & (kv <= q_coord.unsqueeze(1))
        return kv, mask

    window_size_left = k // 2
    window_size_right = k // 2 + (k % 2 - 1)
    stride_group_leader = torch.minimum(
        (q_coord // stride) * stride + (stride // 2), axis_max
    )

    window_center = stride_group_leader.clamp(
        window_size_left,
        axis_size - 1 - window_size_right,
    )
    kv = window_center.unsqueeze(1) - window_size_left + ar
    # clamp ensures this is always in bounds, but keep the mask for safety.
    mask = (kv >= 0) & (kv < axis_size)
    return kv, mask


def _torch_na3d_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    kernel_size: Dimension3DType,
    stride: Dimension3DType,
    dilation: Dimension3DType,
    is_causal: Tuple[bool, bool, bool],
    scale: float,
    return_lse: bool,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    batch_size, x, y, z, heads, head_dim = query.shape
    head_dim_v = value.shape[-1]

    output = torch.empty(
        (batch_size, x, y, z, heads, head_dim_v),
        dtype=value.dtype,
        device=value.device,
    )
    lse = torch.empty(
        (batch_size, x, y, z, heads), dtype=torch.float32, device=value.device
    )

    # Dilation groups only attend to themselves (same "mod dilation" coordinates).
    dx, dy, dz = dilation
    kx, ky, kz = kernel_size
    sx, sy, sz = stride

    q_chunk_size = _get_torch_fna_chunk_size()
    k_prod = kx * ky * kz

    for gx in range(dx):
        for gy in range(dy):
            for gz in range(dz):
                q_g = query[:, gx::dx, gy::dy, gz::dz].contiguous()
                k_g = key[:, gx::dx, gy::dy, gz::dz].contiguous()
                v_g = value[:, gx::dx, gy::dy, gz::dz].contiguous()

                bx, xg, yg, zg, hq, dq = q_g.shape
                assert bx == batch_size
                assert hq == heads and dq == head_dim

                # GQA/MQA: match head dims by explicit repeats (like cutlass-fna).
                if k_g.shape[-2] != hq:
                    heads_kv = k_g.shape[-2]
                    if heads_kv != v_g.shape[-2]:
                        raise ValueError(
                            "Expected key/value to have the same number of KV heads, got "
                            f"{k_g.shape[-2]=} and {v_g.shape[-2]=}."
                        )
                    if hq < heads_kv or (hq % heads_kv != 0):
                        raise ValueError(
                            "GQA/MQA requires heads to be a multiple of heads_kv, got "
                            f"{hq=} and {heads_kv=}."
                        )
                    repeats = hq // heads_kv
                    k_g = torch.repeat_interleave(k_g, repeats=repeats, dim=-2, output_size=hq)
                    v_g = torch.repeat_interleave(v_g, repeats=repeats, dim=-2, output_size=hq)

                n = xg * yg * zg
                q_flat = q_g.reshape(batch_size, n, heads, head_dim)
                k_flat = k_g.reshape(batch_size, n, heads, head_dim)
                v_flat = v_g.reshape(batch_size, n, heads, head_dim_v)

                # NOTE: output slices are strided when dilation > 1, so we cannot reshape a view
                # and write into it (reshape would allocate a copy). Write into contiguous group
                # buffers and scatter back into the full output.
                out_g = torch.empty(
                    (batch_size, xg, yg, zg, heads, head_dim_v),
                    dtype=value.dtype,
                    device=value.device,
                )
                lse_g = torch.empty(
                    (batch_size, xg, yg, zg, heads),
                    dtype=torch.float32,
                    device=value.device,
                )
                out_flat = out_g.reshape(batch_size, n, heads, head_dim_v)
                lse_flat = lse_g.reshape(batch_size, n, heads)

                yz = yg * zg
                for q0 in range(0, n, q_chunk_size):
                    q1 = min(n, q0 + q_chunk_size)
                    q_idx = torch.arange(q0, q1, device=query.device, dtype=torch.int64)

                    xq = q_idx // yz
                    r = q_idx - xq * yz
                    yq = r // zg
                    zq = r - yq * zg

                    kv_x, mx = _axis_kv_positions(
                        xq, axis_size=xg, kernel_size=kx, stride=sx, is_causal=is_causal[0]
                    )
                    kv_y, my = _axis_kv_positions(
                        yq, axis_size=yg, kernel_size=ky, stride=sy, is_causal=is_causal[1]
                    )
                    kv_z, mz = _axis_kv_positions(
                        zq, axis_size=zg, kernel_size=kz, stride=sz, is_causal=is_causal[2]
                    )

                    kv_x_b = kv_x[:, :, None, None]
                    kv_y_b = kv_y[:, None, :, None]
                    kv_z_b = kv_z[:, None, None, :]

                    kv_lin = (kv_x_b * yz + kv_y_b * zg + kv_z_b).reshape(-1, k_prod)

                    valid = (
                        mx[:, :, None, None] & my[:, None, :, None] & mz[:, None, None, :]
                    ).reshape(-1, k_prod)

                    # Avoid out-of-bounds / negative indices (which would wrap around) by gathering
                    # from a safe dummy index and masking it out in logits.
                    kv_lin = kv_lin.masked_fill(~valid, 0)

                    k_patch = k_flat[:, kv_lin, :, :]  # [B, Q, K, H, D]
                    v_patch = v_flat[:, kv_lin, :, :]  # [B, Q, K, H, Dv]
                    k_patch = k_patch.permute(0, 1, 3, 2, 4)  # [B, Q, H, K, D]
                    v_patch = v_patch.permute(0, 1, 3, 2, 4)  # [B, Q, H, K, Dv]

                    q_chunk = q_flat[:, q0:q1, :, :]  # [B, Q, H, D]
                    logits = torch.einsum("bqhd,bqhkd->bqhk", q_chunk, k_patch) * scale
                    logits = logits.masked_fill(~valid[None, :, None, :], float("-inf"))

                    logits_f = logits.float()
                    lse_flat[:, q0:q1, :] = torch.logsumexp(logits_f, dim=-1)

                    probs = torch.softmax(logits_f, dim=-1)
                    out_chunk = torch.einsum("bqhk,bqhkd->bqhd", probs, v_patch.float())
                    out_flat[:, q0:q1, :, :] = out_chunk.to(dtype=value.dtype)

                output[:, gx::dx, gy::dy, gz::dz] = out_g
                lse[:, gx::dx, gy::dy, gz::dz] = lse_g

    if return_lse:
        return output, lse
    return output


def torch_fna_generic(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: DimensionTypeOrDed,
    stride: DimensionTypeOrDed = 1,
    dilation: DimensionTypeOrDed = 1,
    is_causal: Optional[CausalArgTypeOrDed] = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Eager PyTorch (unfused) neighborhood attention.

    Notes:
    - This backend is intended for correctness/debuggability and CPU-only environments.
    - It is not optimized, and relies on query chunking to avoid memory blow-ups.
      Chunk size can be tuned via `NATTEN_TORCH_FNA_CHUNK_SIZE` (default: 1024).
    """
    na_tensor_checks(query, key, value, must_match_head_dims=False, supports_gqa_mqa=True)

    na_dim = query.dim() - 3
    if na_dim != 3:
        raise NotImplementedError(
            f"torch-fna backend currently supports NA3D only, got na_dim={na_dim}."
        )

    kernel_size, stride, dilation, is_causal_ = check_all_args(
        na_dim, kernel_size, stride, dilation, is_causal
    )
    check_args_against_input(
        query,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        is_causal=is_causal_,
    )

    scale = float(scale or query.shape[-1] ** -0.5)

    # For NA3D we already have a 3-tuple of bools here.
    is_causal_3 = (bool(is_causal_[0]), bool(is_causal_[1]), bool(is_causal_[2]))
    return _torch_na3d_forward(
        query,
        key,
        value,
        kernel_size=kernel_size,  # type: ignore[arg-type]
        stride=stride,  # type: ignore[arg-type]
        dilation=dilation,  # type: ignore[arg-type]
        is_causal=is_causal_3,
        scale=scale,
        return_lse=return_lse,
    )
