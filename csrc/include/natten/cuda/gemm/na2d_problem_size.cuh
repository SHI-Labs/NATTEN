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
struct NA2dProblemSize {
  // NA2d strictly problem size parameters
  int B, N, H, W, D, R, S, RH, SH, Hdiv, Wdiv, Hmod, Wmod;
  int RPB_R, RPB_S;
  int dilation_h, dilation_w;

  //
  // Methods
  //

 public:
  CUTLASS_HOST_DEVICE
  NA2dProblemSize()
      : B(0),
        N(0),
        H(0),
        W(0),
        D(0),
        R(0),
        S(0),
        RH(0),
        SH(0),
        RPB_R(0),
        RPB_S(0),
        Hdiv(0),
        Wdiv(0),
        Hmod(0),
        Wmod(0),
        dilation_h(1),
        dilation_w(1) {}

  /// Constructor for default dilation
  CUTLASS_HOST_DEVICE
  NA2dProblemSize(
      int B,
      int N,
      int H,
      int W,
      int D,
      int R,
      int S,
      int dilation_h,
      int dilation_w)
      : B(B),
        N(N),
        H(H),
        W(W),
        D(D),
        R(R),
        S(S),
        RH(int(R * 0.5)),
        SH(int(S * 0.5)),
        RPB_R(R * 2 - 1),
        RPB_S(S * 2 - 1),
        dilation_h(dilation_h),
        dilation_w(dilation_w) {
    Hdiv = H / dilation_h;
    Wdiv = W / dilation_w;
    Hmod = H % dilation_h;
    Wmod = W % dilation_w;
  }

  /// Equality operator
  CUTLASS_HOST_DEVICE
  bool operator==(NA2dProblemSize const& na) const {
    return (
        (B == na.B) && (N == na.N) && (H == na.H) && (W == na.W) &&
        (D == na.D) && (R == na.R) && (S == na.S) &&
        (dilation_h == na.dilation_h) && (dilation_w == na.dilation_w));
  }

  /// Inequality operator
  CUTLASS_HOST_DEVICE
  bool operator!=(NA2dProblemSize const& rhs) const {
    return !(*this == rhs);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//                                  ImplicitGemm helper functions //
////////////////////////////////////////////////////////////////////////////////////////////////////

CUTLASS_HOST_DEVICE
cutlass::MatrixCoord get_hw_tiled_shape(
    NA2dProblemSize const& problem_size,
    cutlass::MatrixCoord tile_size) {
  int PH = problem_size.Hdiv +
      (tile_size.row() - problem_size.Hdiv % tile_size.row());
  int PW = problem_size.Wdiv +
      (tile_size.column() - problem_size.Wdiv % tile_size.column());
  PH *= problem_size.dilation_h;
  PW *= problem_size.dilation_w;
  int length_h = PH / tile_size.row();
  int length_w = PW / tile_size.column();
  return cutlass::MatrixCoord({length_h, length_w});
}

CUTLASS_HOST_DEVICE
int a_tile_size_sq(Operator na_operator, int tile_size, int tile_extent) {
  // Compute problem size
  switch (na_operator) {
    case Operator::kPN:
      return tile_extent * tile_extent;
    case Operator::kNN:
      return tile_size * tile_size;
    case Operator::kIN:
      return tile_size * tile_size;
    default:
      break;
  }
  return 0;
}

/// Determine the problem size of the implicit GEMM operation
CUTLASS_HOST_DEVICE
cutlass::gemm::GemmCoord implicit_gemm_problem_size(
    Operator na_operator,
    NA2dProblemSize const& problem_size,
    int extent,
    cutlass::MatrixCoord hw_tiled_shape) {
  // Compute problem size
  switch (na_operator) {
    case Operator::kPN:
      return cutlass::gemm::GemmCoord(
          problem_size.B * problem_size.N * hw_tiled_shape.row() *
              hw_tiled_shape.column(),
          extent * extent,
          problem_size.D);
    case Operator::kNN:
      return cutlass::gemm::GemmCoord(
          problem_size.B * problem_size.N * hw_tiled_shape.row() *
              hw_tiled_shape.column(),
          problem_size.D,
          extent * extent);
    case Operator::kIN:
      return cutlass::gemm::GemmCoord(
          problem_size.B * problem_size.N * hw_tiled_shape.row() *
              hw_tiled_shape.column(),
          problem_size.D,
          extent * extent);
    default:
      break;
  }
  return cutlass::gemm::GemmCoord();
}

// Determine the number of gemm_k iterations for na2d problem using implicit
// gemm algorithm
CUTLASS_HOST_DEVICE
int implicit_gemm_k_iterations(
    Operator na_operator,
    int threadblock_K,
    int extent,
    NA2dProblemSize const& problem_size) {
  int iterations = 0;
  switch (na_operator) {
    case Operator::kPN:
      iterations = (problem_size.D + threadblock_K - 1) / threadblock_K;
      break;
    case Operator::kNN:
      iterations = ((extent * extent) + threadblock_K - 1) / threadblock_K;
      break;
    case Operator::kIN:
      iterations = ((extent * extent) + threadblock_K - 1) / threadblock_K;
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
