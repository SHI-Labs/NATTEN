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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <cutlass/tfloat32.h>

#include "natten/dtypes.cuh"

namespace natten {
namespace gemm {
namespace detail {

template <
    int _kM,
    int _kN,
    int _kK,
    int _kWarpM,
    int _kWarpN,
    int _kWarpK,
    int _kMathM,
    int _kMathN,
    int _kMathK,
    int _kStages>
struct GemmConfig {
  // CTA shape
  static constexpr int kM = _kM;
  static constexpr int kN = _kN;
  static constexpr int kK = _kK;

  // Warp shape
  static constexpr int kWarpM = _kWarpM;
  static constexpr int kWarpN = _kWarpN;
  static constexpr int kWarpK = _kWarpK;

  // Instruction shape
  static constexpr int kMathM = _kMathM;
  static constexpr int kMathN = _kMathN;
  static constexpr int kMathK = _kMathK;

  // Number of gemm stages
  static constexpr int kStages = _kStages;
};

template <int _AlignmentA, int _AlignmentB, int _AlignmentC>
struct AlignmentConfig {
  static constexpr int AlignmentA = _AlignmentA;
  static constexpr int AlignmentB = _AlignmentB;
  static constexpr int AlignmentC = _AlignmentC;
};

template <typename dtype>
struct DTypeConfig;

template <>
struct DTypeConfig<natten::float64> {
  using Element = double;
  using ElementOutput = double;
  using ElementAccumulator = double;
  using ElementCompute = double;
};

template <>
struct DTypeConfig<natten::float32> {
  using Element = float;
  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
};

template <>
struct DTypeConfig<natten::tf32> {
  using Element = cutlass::tfloat32_t;
  using ElementOutput = float;
  using ElementAccumulator = float;
  using ElementCompute = float;
};

template <>
struct DTypeConfig<natten::float16> {
  using Element = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = cutlass::half_t;
  using ElementCompute = cutlass::half_t;
};

template <>
struct DTypeConfig<natten::bfloat16> {
  using Element = cutlass::bfloat16_t;
  using ElementOutput = cutlass::bfloat16_t;
  using ElementAccumulator = float;
  using ElementCompute = cutlass::bfloat16_t;
};

template <
    int _kM,
    int _kN,
    int _kK,
    int _kWarpM,
    int _kWarpN,
    int _kWarpK,
    int _kMathM,
    int _kMathN,
    int _kMathK,
    int _kStages,
    int _kTile,
    int _kExt,
    int _kNeighborhood>
struct GemmConfig2D {
  // CTA shape
  static constexpr int kM = _kM;
  static constexpr int kN = _kN;
  static constexpr int kK = _kK;

  // Warp shape
  static constexpr int kWarpM = _kWarpM;
  static constexpr int kWarpN = _kWarpN;
  static constexpr int kWarpK = _kWarpK;

  // Instruction shape
  static constexpr int kMathM = _kMathM;
  static constexpr int kMathN = _kMathN;
  static constexpr int kMathK = _kMathK;

  // Number of gemm stages
  static constexpr int kStages = _kStages;

  // NA2d problem shape
  static constexpr int kTile = _kTile;
  static constexpr int kExt = _kExt;
  static constexpr int kNeighborhood = _kNeighborhood;
};

template <int SM, typename T>
struct ArchArgs;

template <typename T>
struct ArchArgs<80, T> {
  using OpClass = typename cutlass::arch::OpClassTensorOp;
  using Tag = typename cutlass::arch::Sm80;
};

template <typename T>
struct ArchArgs<75, T> {
  using OpClass = typename cutlass::arch::OpClassSimt;
  using Tag = typename cutlass::arch::Sm75;
};

template <typename T>
struct ArchArgs<70, T> {
  using OpClass = typename cutlass::arch::OpClassSimt;
  using Tag = typename cutlass::arch::Sm70;
};

template <>
struct ArchArgs<75, natten::float16> {
  using OpClass = typename cutlass::arch::OpClassTensorOp;
  using Tag = typename cutlass::arch::Sm75;
};

template <>
struct ArchArgs<70, natten::float16> {
  using OpClass = typename cutlass::arch::OpClassTensorOp;
  using Tag = typename cutlass::arch::Sm70;
};

} // namespace detail
} // namespace gemm
} // namespace natten
