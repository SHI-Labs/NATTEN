/***************************************************************************************************
 * Copyright (c) 2022-2024 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 **************************************************************************************************/

#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cmath>
#else
#include <cmath>
#endif

#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/functional.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/tensor_coord.h>

namespace natten {
namespace cuda {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Problem size structure
struct NA1dProblemSize {
  // NA1d strictly problem size parameters
  int B, N, L, D, R, RH, Ldiv, Lmod;
  int RPB_R;
  int dilation;
  int kTile, kStride;

  //
  // Methods
  //

 public:
  CUTLASS_HOST_DEVICE
  NA1dProblemSize()
      : B(0),
        N(0),
        L(0),
        D(0),
        R(0),
        RH(0),
        RPB_R(0),
        Ldiv(0),
        Lmod(0),
        kTile(0),
        kStride(0),
        dilation(1) {}

  /// Constructor for default dilation
  CUTLASS_HOST_DEVICE
  NA1dProblemSize(
      int B,
      int N,
      int L,
      int D,
      int R,
      int dilation,
      int kTile,
      int kStride)
      : B(B),
        N(N),
        L(L),
        D(D),
        R(R),
        RH(int(R * 0.5)),
        RPB_R(R * 2 - 1),
        kTile(kTile),
        kStride(kStride),
        dilation(dilation) {
    Ldiv = L / dilation;
    Lmod = L % dilation;
  }

  /// Equality operator
  CUTLASS_HOST_DEVICE
  bool operator==(NA1dProblemSize const& na) const {
    return (
        (B == na.B) && (N == na.N) && (L == na.L) && (D == na.D) &&
        (R == na.R) && (kTile == na.kTile) && (kStride == na.kStride) &&
        (dilation == na.dilation));
  }

  /// Inequality operator
  CUTLASS_HOST_DEVICE
  bool operator!=(NA1dProblemSize const& rhs) const {
    return !(*this == rhs);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                  ImplicitGemm helper functions //
////////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE
int get_l_tiled_shape(NA1dProblemSize const& problem_size, int tile_size) {
  int PL = problem_size.Ldiv + (tile_size - problem_size.Ldiv % tile_size);
  PL *= problem_size.dilation;
  return PL / tile_size;
}

CUTLASS_HOST_DEVICE
int a_tile_size(Operator na_operator, int tile_size, int tile_extent) {
  // Compute problem size
  switch (na_operator) {
    case Operator::kPN:
      return tile_extent;
    case Operator::kNN:
      return tile_size;
    case Operator::kIN:
      return tile_size;
    default:
      break;
  }
  return 0;
}

/// Determine the problem size of the implicit GEMM operation
CUTLASS_HOST_DEVICE
cutlass::gemm::GemmCoord implicit_gemm_problem_size(
    Operator na_operator,
    NA1dProblemSize const& problem_size,
    int extent,
    int l_tiled_shape) {
  // Compute problem size
  switch (na_operator) {
    case Operator::kPN:
      return cutlass::gemm::GemmCoord(
          problem_size.B * problem_size.N * l_tiled_shape,
          extent,
          problem_size.D);
    case Operator::kNN:
      return cutlass::gemm::GemmCoord(
          problem_size.B * problem_size.N * l_tiled_shape,
          problem_size.D,
          extent);
    case Operator::kIN:
      return cutlass::gemm::GemmCoord(
          problem_size.B * problem_size.N * l_tiled_shape,
          problem_size.D,
          extent);
    default:
      break;
  }
  return cutlass::gemm::GemmCoord();
}

// Determine the number of gemm_k iterations for na1d problem using implicit
// gemm algorithm
CUTLASS_HOST_DEVICE
int implicit_gemm_k_iterations(
    Operator na_operator,
    int threadblock_K,
    int extent,
    NA1dProblemSize const& problem_size) {
  int iterations = 0;
  switch (na_operator) {
    case Operator::kPN:
      iterations = (problem_size.D + threadblock_K - 1) / threadblock_K;
      break;
    case Operator::kNN:
      iterations = (extent + threadblock_K - 1) / threadblock_K;
      break;
    case Operator::kIN:
      iterations = (extent + threadblock_K - 1) / threadblock_K;
      break;

    default:
      break;
  }

  return iterations;
}

} // namespace gemm
} // namespace cuda
} // namespace natten

////////////////////////////////////////////////////////////////////////////////////////////////////
