#################################################################################################
# Copyright (c) 2022-2024 Ali Hassani.
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
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.cuda import _device_t

try:
    from natten import libnatten  # type: ignore
except ImportError:
    raise ImportError(
        "Failed to import NATTEN's CPP backend. "
        "This could be due to an invalid/incomplete install. "
        "Please uninstall NATTEN (pip uninstall natten) and re-install with the"
        " correct torch build: shi-labs.com/natten ."
    )
from .utils import check_all_args


def get_device_cc(device_index: Optional[_device_t] = None) -> int:
    major, minor = torch.cuda.get_device_capability(device_index)
    return major * 10 + minor


class Autotuner:
    enabled: bool = False
    warmup_steps: int = 5

    _CACHE: Dict[int, Any] = {}


_TILE_SIZES_1D = [
    ((64,), (64,)),
    ((32,), (128,)),
]

_TILE_SIZES_1D_64x128 = [
    ((64,), (128,)),
]

_TILE_SIZES_2D = [
    # 64x64
    ((64, 1), (64, 1)),
    ((32, 2), (32, 2)),
    ((16, 4), (16, 4)),
    ((8, 8), (8, 8)),
    ((4, 16), (4, 16)),
    ((2, 32), (2, 32)),
    ((1, 64), (1, 64)),
    # 32x128
    ((32, 1), (128, 1)),
    ((32, 1), (64, 2)),
    ((32, 1), (32, 4)),
    ((16, 2), (64, 2)),
    ((16, 2), (32, 4)),
    ((16, 2), (16, 8)),
    ((8, 4), (32, 4)),
    ((8, 4), (16, 8)),
    ((8, 4), (8, 16)),
    ((4, 8), (16, 8)),
    ((4, 8), (8, 16)),
    ((4, 8), (4, 32)),
    ((2, 16), (8, 16)),
    ((2, 16), (4, 32)),
    ((2, 16), (2, 64)),
    ((1, 32), (4, 32)),
    ((1, 32), (2, 64)),
    ((1, 32), (1, 128)),
]

_TILE_SIZES_2D_64x128 = [
    ((64, 1), (128, 1)),
    ((64, 1), (64, 2)),
    ((32, 2), (64, 2)),
    ((32, 2), (32, 4)),
    ((16, 4), (32, 4)),
    ((16, 4), (16, 8)),
    ((8, 8), (16, 8)),
    ((8, 8), (8, 16)),
    ((4, 16), (8, 16)),
    ((4, 16), (4, 32)),
    ((2, 32), (4, 32)),
    ((2, 32), (2, 64)),
    ((1, 64), (2, 64)),
    ((1, 64), (1, 128)),
]

_TILE_SIZES_3D = [
    # 64x64
    ((64, 1, 1), (64, 1, 1)),
    ((32, 2, 1), (32, 2, 1)),
    ((32, 1, 2), (32, 1, 2)),
    ((16, 4, 1), (16, 4, 1)),
    ((16, 2, 2), (16, 2, 2)),
    ((16, 1, 4), (16, 1, 4)),
    ((8, 8, 1), (8, 8, 1)),
    ((8, 4, 2), (8, 4, 2)),
    ((8, 2, 4), (8, 2, 4)),
    ((8, 1, 8), (8, 1, 8)),
    ((4, 16, 1), (4, 16, 1)),
    ((4, 8, 2), (4, 8, 2)),
    ((4, 4, 4), (4, 4, 4)),
    ((4, 2, 8), (4, 2, 8)),
    ((4, 1, 16), (4, 1, 16)),
    ((2, 32, 1), (2, 32, 1)),
    ((2, 16, 2), (2, 16, 2)),
    ((2, 8, 4), (2, 8, 4)),
    ((2, 4, 8), (2, 4, 8)),
    ((2, 2, 16), (2, 2, 16)),
    ((2, 1, 32), (2, 1, 32)),
    ((1, 64, 1), (1, 64, 1)),
    ((1, 32, 2), (1, 32, 2)),
    ((1, 16, 4), (1, 16, 4)),
    ((1, 8, 8), (1, 8, 8)),
    ((1, 4, 16), (1, 4, 16)),
    ((1, 2, 32), (1, 2, 32)),
    ((1, 1, 64), (1, 1, 64)),
    # 32x128
    ((32, 1, 1), (128, 1, 1)),
    ((32, 1, 1), (64, 2, 1)),
    ((32, 1, 1), (64, 1, 2)),
    ((32, 1, 1), (32, 4, 1)),
    ((32, 1, 1), (32, 2, 2)),
    ((32, 1, 1), (32, 1, 4)),
    ((16, 2, 1), (64, 2, 1)),
    ((16, 2, 1), (32, 4, 1)),
    ((16, 2, 1), (32, 2, 2)),
    ((16, 2, 1), (16, 8, 1)),
    ((16, 2, 1), (16, 4, 2)),
    ((16, 2, 1), (16, 2, 4)),
    ((16, 1, 2), (64, 1, 2)),
    ((16, 1, 2), (32, 2, 2)),
    ((16, 1, 2), (32, 1, 4)),
    ((16, 1, 2), (16, 4, 2)),
    ((16, 1, 2), (16, 2, 4)),
    ((16, 1, 2), (16, 1, 8)),
    ((8, 4, 1), (32, 4, 1)),
    ((8, 4, 1), (16, 8, 1)),
    ((8, 4, 1), (16, 4, 2)),
    ((8, 4, 1), (8, 16, 1)),
    ((8, 4, 1), (8, 8, 2)),
    ((8, 4, 1), (8, 4, 4)),
    ((8, 2, 2), (32, 2, 2)),
    ((8, 2, 2), (16, 4, 2)),
    ((8, 2, 2), (16, 2, 4)),
    ((8, 2, 2), (8, 8, 2)),
    ((8, 2, 2), (8, 4, 4)),
    ((8, 2, 2), (8, 2, 8)),
    ((8, 1, 4), (32, 1, 4)),
    ((8, 1, 4), (16, 2, 4)),
    ((8, 1, 4), (16, 1, 8)),
    ((8, 1, 4), (8, 4, 4)),
    ((8, 1, 4), (8, 2, 8)),
    ((8, 1, 4), (8, 1, 16)),
    ((4, 8, 1), (16, 8, 1)),
    ((4, 8, 1), (8, 16, 1)),
    ((4, 8, 1), (8, 8, 2)),
    ((4, 8, 1), (4, 32, 1)),
    ((4, 8, 1), (4, 16, 2)),
    ((4, 8, 1), (4, 8, 4)),
    ((4, 4, 2), (16, 4, 2)),
    ((4, 4, 2), (8, 8, 2)),
    ((4, 4, 2), (8, 4, 4)),
    ((4, 4, 2), (4, 16, 2)),
    ((4, 4, 2), (4, 8, 4)),
    ((4, 4, 2), (4, 4, 8)),
    ((4, 2, 4), (16, 2, 4)),
    ((4, 2, 4), (8, 4, 4)),
    ((4, 2, 4), (8, 2, 8)),
    ((4, 2, 4), (4, 8, 4)),
    ((4, 2, 4), (4, 4, 8)),
    ((4, 2, 4), (4, 2, 16)),
    ((4, 1, 8), (16, 1, 8)),
    ((4, 1, 8), (8, 2, 8)),
    ((4, 1, 8), (8, 1, 16)),
    ((4, 1, 8), (4, 4, 8)),
    ((4, 1, 8), (4, 2, 16)),
    ((4, 1, 8), (4, 1, 32)),
    ((2, 16, 1), (8, 16, 1)),
    ((2, 16, 1), (4, 32, 1)),
    ((2, 16, 1), (4, 16, 2)),
    ((2, 16, 1), (2, 64, 1)),
    ((2, 16, 1), (2, 32, 2)),
    ((2, 16, 1), (2, 16, 4)),
    ((2, 8, 2), (8, 8, 2)),
    ((2, 8, 2), (4, 16, 2)),
    ((2, 8, 2), (4, 8, 4)),
    ((2, 8, 2), (2, 32, 2)),
    ((2, 8, 2), (2, 16, 4)),
    ((2, 8, 2), (2, 8, 8)),
    ((2, 4, 4), (8, 4, 4)),
    ((2, 4, 4), (4, 8, 4)),
    ((2, 4, 4), (4, 4, 8)),
    ((2, 4, 4), (2, 16, 4)),
    ((2, 4, 4), (2, 8, 8)),
    ((2, 4, 4), (2, 4, 16)),
    ((2, 2, 8), (8, 2, 8)),
    ((2, 2, 8), (4, 4, 8)),
    ((2, 2, 8), (4, 2, 16)),
    ((2, 2, 8), (2, 8, 8)),
    ((2, 2, 8), (2, 4, 16)),
    ((2, 2, 8), (2, 2, 32)),
    ((2, 1, 16), (8, 1, 16)),
    ((2, 1, 16), (4, 2, 16)),
    ((2, 1, 16), (4, 1, 32)),
    ((2, 1, 16), (2, 4, 16)),
    ((2, 1, 16), (2, 2, 32)),
    ((2, 1, 16), (2, 1, 64)),
    ((1, 32, 1), (4, 32, 1)),
    ((1, 32, 1), (2, 64, 1)),
    ((1, 32, 1), (2, 32, 2)),
    ((1, 32, 1), (1, 128, 1)),
    ((1, 32, 1), (1, 64, 2)),
    ((1, 32, 1), (1, 32, 4)),
    ((1, 16, 2), (4, 16, 2)),
    ((1, 16, 2), (2, 32, 2)),
    ((1, 16, 2), (2, 16, 4)),
    ((1, 16, 2), (1, 64, 2)),
    ((1, 16, 2), (1, 32, 4)),
    ((1, 16, 2), (1, 16, 8)),
    ((1, 8, 4), (4, 8, 4)),
    ((1, 8, 4), (2, 16, 4)),
    ((1, 8, 4), (2, 8, 8)),
    ((1, 8, 4), (1, 32, 4)),
    ((1, 8, 4), (1, 16, 8)),
    ((1, 8, 4), (1, 8, 16)),
    ((1, 4, 8), (4, 4, 8)),
    ((1, 4, 8), (2, 8, 8)),
    ((1, 4, 8), (2, 4, 16)),
    ((1, 4, 8), (1, 16, 8)),
    ((1, 4, 8), (1, 8, 16)),
    ((1, 4, 8), (1, 4, 32)),
    ((1, 2, 16), (4, 2, 16)),
    ((1, 2, 16), (2, 4, 16)),
    ((1, 2, 16), (2, 2, 32)),
    ((1, 2, 16), (1, 8, 16)),
    ((1, 2, 16), (1, 4, 32)),
    ((1, 2, 16), (1, 2, 64)),
    ((1, 1, 32), (4, 1, 32)),
    ((1, 1, 32), (2, 2, 32)),
    ((1, 1, 32), (2, 1, 64)),
    ((1, 1, 32), (1, 4, 32)),
    ((1, 1, 32), (1, 2, 64)),
    ((1, 1, 32), (1, 1, 128)),
]

_TILE_SIZES_3D_64x128 = [
    ((64, 1, 1), (128, 1, 1)),
    ((64, 1, 1), (64, 2, 1)),
    ((64, 1, 1), (64, 1, 2)),
    ((32, 2, 1), (64, 2, 1)),
    ((32, 2, 1), (32, 4, 1)),
    ((32, 2, 1), (32, 2, 2)),
    ((32, 1, 2), (64, 1, 2)),
    ((32, 1, 2), (32, 2, 2)),
    ((32, 1, 2), (32, 1, 4)),
    ((16, 4, 1), (32, 4, 1)),
    ((16, 4, 1), (16, 8, 1)),
    ((16, 4, 1), (16, 4, 2)),
    ((16, 2, 2), (32, 2, 2)),
    ((16, 2, 2), (16, 4, 2)),
    ((16, 2, 2), (16, 2, 4)),
    ((16, 1, 4), (32, 1, 4)),
    ((16, 1, 4), (16, 2, 4)),
    ((16, 1, 4), (16, 1, 8)),
    ((8, 8, 1), (16, 8, 1)),
    ((8, 8, 1), (8, 16, 1)),
    ((8, 8, 1), (8, 8, 2)),
    ((8, 4, 2), (16, 4, 2)),
    ((8, 4, 2), (8, 8, 2)),
    ((8, 4, 2), (8, 4, 4)),
    ((8, 2, 4), (16, 2, 4)),
    ((8, 2, 4), (8, 4, 4)),
    ((8, 2, 4), (8, 2, 8)),
    ((8, 1, 8), (16, 1, 8)),
    ((8, 1, 8), (8, 2, 8)),
    ((8, 1, 8), (8, 1, 16)),
    ((4, 16, 1), (8, 16, 1)),
    ((4, 16, 1), (4, 32, 1)),
    ((4, 16, 1), (4, 16, 2)),
    ((4, 8, 2), (8, 8, 2)),
    ((4, 8, 2), (4, 16, 2)),
    ((4, 8, 2), (4, 8, 4)),
    ((4, 4, 4), (8, 4, 4)),
    ((4, 4, 4), (4, 8, 4)),
    ((4, 4, 4), (4, 4, 8)),
    ((4, 2, 8), (8, 2, 8)),
    ((4, 2, 8), (4, 4, 8)),
    ((4, 2, 8), (4, 2, 16)),
    ((4, 1, 16), (8, 1, 16)),
    ((4, 1, 16), (4, 2, 16)),
    ((4, 1, 16), (4, 1, 32)),
    ((2, 32, 1), (4, 32, 1)),
    ((2, 32, 1), (2, 64, 1)),
    ((2, 32, 1), (2, 32, 2)),
    ((2, 16, 2), (4, 16, 2)),
    ((2, 16, 2), (2, 32, 2)),
    ((2, 16, 2), (2, 16, 4)),
    ((2, 8, 4), (4, 8, 4)),
    ((2, 8, 4), (2, 16, 4)),
    ((2, 8, 4), (2, 8, 8)),
    ((2, 4, 8), (4, 4, 8)),
    ((2, 4, 8), (2, 8, 8)),
    ((2, 4, 8), (2, 4, 16)),
    ((2, 2, 16), (4, 2, 16)),
    ((2, 2, 16), (2, 4, 16)),
    ((2, 2, 16), (2, 2, 32)),
    ((2, 1, 32), (4, 1, 32)),
    ((2, 1, 32), (2, 2, 32)),
    ((2, 1, 32), (2, 1, 64)),
    ((1, 64, 1), (2, 64, 1)),
    ((1, 64, 1), (1, 128, 1)),
    ((1, 64, 1), (1, 64, 2)),
    ((1, 32, 2), (2, 32, 2)),
    ((1, 32, 2), (1, 64, 2)),
    ((1, 32, 2), (1, 32, 4)),
    ((1, 16, 4), (2, 16, 4)),
    ((1, 16, 4), (1, 32, 4)),
    ((1, 16, 4), (1, 16, 8)),
    ((1, 8, 8), (2, 8, 8)),
    ((1, 8, 8), (1, 16, 8)),
    ((1, 8, 8), (1, 8, 16)),
    ((1, 4, 16), (2, 4, 16)),
    ((1, 4, 16), (1, 8, 16)),
    ((1, 4, 16), (1, 4, 32)),
    ((1, 2, 32), (2, 2, 32)),
    ((1, 2, 32), (1, 4, 32)),
    ((1, 2, 32), (1, 2, 64)),
    ((1, 1, 64), (2, 1, 64)),
    ((1, 1, 64), (1, 2, 64)),
    ((1, 1, 64), (1, 1, 128)),
]


def enable_autotuner():
    Autotuner.enabled = True
    warnings.warn(
        "You're enabling NATTEN auto-tuner. This is an experimental "
        "feature intended only for fused neighborhood attention. "
        "Proceed with caution."
    )


def disable_autotuner():
    Autotuner.enabled = False


def get_default_tiling_config(
    na_dim: int,
) -> (
    Tuple[Tuple[int], Tuple[int]]
    | Tuple[Tuple[int, int], Tuple[int, int]]
    | Tuple[Tuple[int, int, int], Tuple[int, int, int]]
):
    assert na_dim > 0 and na_dim < 4
    if na_dim == 2:
        return ((8, 8), (8, 8))
    if na_dim == 3:
        return ((4, 4, 4), (4, 4, 4))
    return ((64,), (64,))


def _get_all_tiling_configs(
    na_dim: int, torch_device: Any
) -> (
    List[Tuple[Tuple[int], Tuple[int]]]
    | List[Tuple[Tuple[int, int], Tuple[int, int]]]
    | List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]
):
    assert na_dim > 0 and na_dim < 4
    # SM80 and SM90 have more shared memory than SM86 and SM89
    # and their tensor core GEMMs can therefore target larger
    # tile shapes.
    # SM80 and 86 have been tested, but I don't have an SM89.
    # However, I suspect SM89 is to SM90 what SM86 was to SM80
    # in terms of shared memory (and only that).
    # Better to disable the larger tile configs for SM89 as well
    # as 86 until we can test it.
    if get_device_cc(torch_device) in [86, 89]:
        if na_dim == 2:
            return _TILE_SIZES_2D
        if na_dim == 3:
            return _TILE_SIZES_3D
        return _TILE_SIZES_1D
    if na_dim == 2:
        return _TILE_SIZES_2D + _TILE_SIZES_2D_64x128
    if na_dim == 3:
        return _TILE_SIZES_3D + _TILE_SIZES_3D_64x128
    return _TILE_SIZES_1D + _TILE_SIZES_1D_64x128


def _problem_to_hash(
    na_dim: int,
    shape: torch.Size,
    device: Any,
    dtype: Any,
    kernel_size: Any,
    dilation: Any,
    is_causal: Any,
) -> int:
    kernel_size, dilation, is_causal = check_all_args(
        na_dim, kernel_size, dilation, is_causal
    )
    key = (
        hash(na_dim)
        ^ hash(shape)
        ^ hash(device)
        ^ hash(dtype)
        ^ hash(kernel_size)
        ^ hash(dilation)
        ^ hash(is_causal)
    )
    return key


def _get_fna_func(na_dim: int):
    assert na_dim > 0 and na_dim < 4
    if na_dim == 2:
        return libnatten.na2d_forward
    if na_dim == 3:
        return libnatten.na3d_forward
    return libnatten.na1d_forward


def autotune_fna(
    na_dim: int,
    input_tensor: Tensor,
    kernel_size: Any,
    dilation: Any,
    is_causal: Any,
):
    if not Autotuner.enabled:
        return get_default_tiling_config(na_dim)

    kernel_size, dilation, is_causal = check_all_args(
        na_dim, kernel_size, dilation, is_causal
    )
    problem_hash = _problem_to_hash(
        na_dim=na_dim,
        shape=input_tensor.shape,
        device=input_tensor.device,
        dtype=input_tensor.dtype,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
    )
    if problem_hash in Autotuner._CACHE:
        return Autotuner._CACHE[problem_hash]

    fna_func = _get_fna_func(na_dim)
    with torch.no_grad():
        # NOTE: do NOT use empty_like; we care about dtype and device, not mem layout,
        # which the autograd function should decide.
        q = torch.empty(
            input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device
        )
        k = torch.empty(
            input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device
        )
        v = torch.empty(
            input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device
        )
        out = torch.empty(
            input_tensor.shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        best_time = 1e9
        best_config = None
        for tile_config in _get_all_tiling_configs(na_dim, input_tensor.device):
            for _ in range(Autotuner.warmup_steps):
                fna_func(
                    out,
                    q,
                    k,
                    v,
                    None,
                    kernel_size,
                    dilation,
                    is_causal,
                    1.0,
                    *tile_config
                )
                torch.cuda.synchronize()
            torch.cuda.synchronize()
            starter.record()
            torch.cuda.synchronize()
            fna_func(
                out, q, k, v, None, kernel_size, dilation, is_causal, 1.0, *tile_config
            )
            torch.cuda.synchronize()
            ender.record()
            torch.cuda.synchronize()
            time_ms = starter.elapsed_time(ender)
            if time_ms < best_time:
                best_time = time_ms
                best_config = tile_config

        Autotuner._CACHE[problem_hash] = best_config
        return best_config
