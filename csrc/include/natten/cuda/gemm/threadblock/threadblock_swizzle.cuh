/***************************************************************************************************
 * Copyright (c) 2022-2024 Ali Hassani.
 * NATTEN's CUTLASS kernels started off from CUTLASS 2.X's implicit GEMM kernels
 *for convolution.
 **************************************************************************************************/
/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Implements a threadblock-swizzling function mapping blockIdx to
      NA problems.
*/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/index_remat.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/platform/platform.h>

#include "natten/cuda/gemm/na1d_problem_size.cuh"
#include "natten/cuda/gemm/na2d_problem_size.cuh"
#include "natten/cuda/gemm/neighborhood_attention.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace natten {
namespace cuda {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Threadblock swizzling function for GEMMs
struct NAIdentityThreadblockSwizzle {
  CUTLASS_HOST_DEVICE
  NAIdentityThreadblockSwizzle() {}

  /// Tile shape
  /// Returns the shape of the problem in units of logical tiles
  /// *Gemm* problem size: gemm(M, N, K)
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord get_tiled_shape(
      cutlass::gemm::GemmCoord problem_size,
      cutlass::gemm::GemmCoord tile_size,
      int na_tile_size_sq) const {
    int grid_x = problem_size.m() *
        ((na_tile_size_sq + tile_size.m() - 1) / tile_size.m());
    int grid_y = (problem_size.n() + tile_size.n() - 1) / tile_size.n();
    return cutlass::gemm::GemmCoord(grid_x, grid_y, 1);
  }

  /// Returns the x-block multiplier.
  /// If our NA tile is greater than our GEMM-N (i.e. 16x16 NA tile, 128 GEMM-N)
  /// we'd need to split the tiles according to GEMM-N.
  CUTLASS_HOST_DEVICE
  int block_x_multiplier(
      cutlass::gemm::GemmCoord problem_size,
      cutlass::gemm::GemmCoord tile_size,
      int na_tile_size_sq) const {
    return ((na_tile_size_sq + tile_size.m() - 1) / tile_size.m());
  }

  /// Returns the shape of the problem in units of logical tiles
  /// *ImplicitGemm* NA1d problem size: na_operator(PN, NN, IN)
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord get_tiled_shape(
      natten::cuda::gemm::Operator na_operator,
      natten::cuda::gemm::NA1dProblemSize const& problem_size,
      cutlass::gemm::GemmCoord tile_size,
      int na_tile_size,
      int na_tile_extent,
      int l_tiled_shape) const {
    cutlass::gemm::GemmCoord implicit_gemm_problem_size =
        natten::cuda::gemm::implicit_gemm_problem_size(
            na_operator, problem_size, na_tile_extent, l_tiled_shape);
    int na_tile_size_out = natten::cuda::gemm::a_tile_size(
        na_operator, na_tile_size, na_tile_extent);

    return get_tiled_shape(
        implicit_gemm_problem_size, tile_size, na_tile_size_out);
  }

  /// Returns the shape of the problem in units of logical tiles
  /// *ImplicitGemm* NA2d problem size: na_operator(PN, NN, IN)
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord get_tiled_shape(
      natten::cuda::gemm::Operator na_operator,
      natten::cuda::gemm::NA2dProblemSize const& problem_size,
      cutlass::gemm::GemmCoord tile_size,
      int na_tile_size,
      int na_tile_extent,
      cutlass::MatrixCoord hw_tiled_shape) const {
    cutlass::gemm::GemmCoord implicit_gemm_problem_size =
        natten::cuda::gemm::implicit_gemm_problem_size(
            na_operator, problem_size, na_tile_extent, hw_tiled_shape);
    int na_tile_size_sq = natten::cuda::gemm::a_tile_size_sq(
        na_operator, na_tile_size, na_tile_extent);

    return get_tiled_shape(
        implicit_gemm_problem_size, tile_size, na_tile_size_sq);
  }

  /// Computes CUDA grid dimensions given a size in units of logical tiles
  CUTLASS_HOST_DEVICE
  dim3 get_grid_shape(cutlass::gemm::GemmCoord tiled_shape) const {
    return dim3(tiled_shape.m(), tiled_shape.n(), tiled_shape.k());
  }

  /// Obtains the threadblock offset (in units of threadblock-scoped tiles)
  CUTLASS_DEVICE
  cutlass::gemm::GemmCoord get_tile_offset() const {
    int block_idx_x = cutlass::gemm::threadblock::RematerializeBlockIdxX();
    int block_idx_y = cutlass::gemm::threadblock::RematerializeBlockIdxY();
    int block_idx_z = cutlass::gemm::threadblock::RematerializeBlockIdxZ();

    return cutlass::gemm::GemmCoord(block_idx_x, block_idx_y, block_idx_z);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cuda
} // namespace natten
