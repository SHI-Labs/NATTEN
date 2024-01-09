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

from .functional import (
    disable_gemm_na,
    disable_tf32,
    disable_tiled_na,
    enable_gemm_na,
    enable_tf32,
    enable_tiled_na,
    has_bfloat,
    has_cuda,
    has_fp32_gemm,
    has_fp64_gemm,
    has_gemm,
    has_half,
    has_tf32_gemm,
)
from .natten1d import NeighborhoodAttention1D
from .natten2d import NeighborhoodAttention2D
from .natten3d import NeighborhoodAttention3D

__all__ = [
    "NeighborhoodAttention1D",
    "NeighborhoodAttention2D",
    "NeighborhoodAttention3D",
    "disable_gemm_na",
    "disable_tf32",
    "disable_tiled_na",
    "enable_gemm_na",
    "enable_tf32",
    "enable_tiled_na",
    "has_bfloat",
    "has_cuda",
    "has_gemm",
    "has_half",
    "has_tf32_gemm",
    "has_fp32_gemm",
    "has_fp64_gemm",
]

__version__ = "0.15.0"
