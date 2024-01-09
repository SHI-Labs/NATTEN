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

#pragma once

#include <cutlass/cutlass.h>

#include <cutlass/aligned_buffer.h>
#include <cutlass/array.h>
#include <cutlass/fast_math.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>

#include "natten/cuda/gemm/na1d_problem_size.cuh"
#include "natten/cuda/gemm/neighborhood_attention.cuh"
#include "natten/cuda/gemm/threadblock/na1d_tile.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace natten {
namespace cuda {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    Operator NAOperator, ///! NA operator (PN, NN, IN)
    typename Mma_, ///! Threadblock-scoped matrix multiply-accumulate
    typename Epilogue_, ///! Epilogue
    typename ThreadblockSwizzle_, ///! Threadblock swizzling function
    typename NAProblemSize_ = NA1dProblemSize ///! NA problem size
    >
struct ImplicitGemmNA1d {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static Operator const kNAOperator = NAOperator;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename EpilogueOutputOp::ElementOutput;

  /// Tile info
  using TileInfo = natten::cuda::gemm::threadblock::NA1dTileInfo<NAOperator>;

  /// Set output tensor C layout
  using LayoutC = LayoutA;

  using ElementAccumulator = typename EpilogueOutputOp::ElementAccumulator;
  using ElementCompute = typename EpilogueOutputOp::ElementCompute;

  using WarpMmaOperator = typename Mma::Policy::Operator;

  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator = typename ArchMmaOperator::Operator;

  using OperatorClass = typename WarpMmaOperator::OperatorClass;
  using ArchTag = typename WarpMmaOperator::ArchTag;

  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename WarpMmaOperator::Shape;
  using InstructionShape = typename ArchMmaOperator::Shape;

  static int const kStages = Mma::kStages;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using TensorRefA = typename Mma::IteratorA::TensorRef;
  using TensorRefB = typename Mma::IteratorB::TensorRef;
  using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;
  using TensorRefBias = cutlass::TensorRef<ElementC, cutlass::layout::RowMajor>;

  /// NA dimension and problem size structure
  using NAProblemSize = NAProblemSize_;

  /// Argument structure
  struct Arguments {
    //
    // Data members
    //

    NAProblemSize problem_size;
    int l_tiled_shape;
    TensorRefA ref_A;
    TensorRefB ref_B;
    TensorRefC ref_C;
    TensorRefC ref_D;
    TensorRefBias ref_BIAS;
    typename EpilogueOutputOp::Params output_op;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() {}

    CUTLASS_HOST_DEVICE
    Arguments(NAProblemSize const& problem_size)
        : problem_size(problem_size),
          l_tiled_shape(get_l_tiled_shape(problem_size, problem_size.kTile)) {}

    CUTLASS_HOST_DEVICE
    Arguments(
        NAProblemSize const& problem_size,
        TensorRefA const& ref_A,
        TensorRefB const& ref_B,
        TensorRefC const& ref_C,
        TensorRefC const& ref_D,
        TensorRefBias const& ref_BIAS,
        typename EpilogueOutputOp::Params const& output_op)
        : problem_size(problem_size),
          ref_A(ref_A),
          ref_B(ref_B),
          ref_C(ref_C),
          ref_D(ref_D),
          ref_BIAS(ref_BIAS),
          output_op(output_op),
          l_tiled_shape(get_l_tiled_shape(problem_size, problem_size.kTile)) {}
  };

  /// Parameters structure
  struct Params {
    NAProblemSize problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    cutlass::gemm::GemmCoord implicit_gemm_size;

    int gemm_k_iterations;
    typename Mma::IteratorA::Params iterator_A;
    typename Mma::IteratorA::Element const* ptr_A;
    typename Mma::IteratorB::Params iterator_B;
    typename Mma::IteratorB::Element const* ptr_B;
    typename Epilogue::OutputTileIterator::Params iterator_C;
    typename Epilogue::OutputTileIterator::Element* ptr_C;
    typename Epilogue::OutputTileIterator::Params iterator_D;
    typename Epilogue::OutputTileIterator::Element* ptr_D;
    typename EpilogueOutputOp::Params output_op;

    typename Epilogue::OutputTileIterator::Element* ptr_bias;
    natten::cuda::gemm::threadblock::NA1dTileInfoParams tile_info_params;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() : gemm_k_iterations(0) {}

    ///
    CUTLASS_HOST_DEVICE
    Params(Arguments const& args)
        : problem_size(args.problem_size),
          implicit_gemm_size(implicit_gemm_problem_size(
              kNAOperator,
              args.problem_size,
              args.problem_size.kStride,
              args.l_tiled_shape)),
          iterator_A(args.problem_size, args.ref_A.layout()),
          ptr_A(args.ref_A.data()),
          iterator_B(args.problem_size, args.ref_B.layout()),
          ptr_B(args.ref_B.data()),
          iterator_C(args.ref_C.stride(0)),
          ptr_C(args.ref_C.data()),
          iterator_D(args.ref_D.stride(0)),
          ptr_D(args.ref_D.data()),
          output_op(args.output_op),
          ptr_bias(args.ref_BIAS.data()) {
      gemm_k_iterations = implicit_gemm_k_iterations(
          kNAOperator,
          ThreadblockShape::kK,
          args.problem_size.kStride,
          args.problem_size);

      ThreadblockSwizzle threadblock_swizzle;
      int na_tile_size = natten::cuda::gemm::a_tile_size(
          kNAOperator, args.problem_size.kTile, args.problem_size.kStride);

      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
          implicit_gemm_size,
          {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
          na_tile_size);

      int block_x_multiplier = threadblock_swizzle.block_x_multiplier(
          implicit_gemm_size,
          {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
          na_tile_size);
      tile_info_params = natten::cuda::gemm::threadblock::NA1dTileInfoParams(
          args.l_tiled_shape,
          block_x_multiplier,
          args.ref_D.stride(2),
          args.ref_D.stride(1),
          args.ref_D.stride(0),
          args.ref_BIAS.good());
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  ImplicitGemmNA1d() {}

  /// Executes one Implicit GEMM
  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& shared_storage) {
    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset();

    // Early exit if CTA is out of range
    TileInfo tile_info(params.tile_info_params, params.problem_size);
    if (!tile_info.valid_batch ||
        params.grid_tiled_shape.m() <= threadblock_tile_idx.m() ||
        params.grid_tiled_shape.n() <= threadblock_tile_idx.n() ||
        params.grid_tiled_shape.k() <= threadblock_tile_idx.k()) {
      return;
    }

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A(
        params.iterator_A,
        params.problem_size,
        params.ptr_A,
        tile_info,
        thread_idx,
        cutlass::MatrixCoord(
            tile_info.gemm_m_offset * Mma::Shape::kM,
            threadblock_tile_idx.k() * Mma::Shape::kK));

    typename Mma::IteratorB iterator_B(
        params.iterator_B,
        params.problem_size,
        params.ptr_B,
        tile_info,
        thread_idx,
        cutlass::MatrixCoord(
            threadblock_tile_idx.n() * Mma::Shape::kN,
            threadblock_tile_idx.k() * Mma::Shape::kK));

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    mma(params.gemm_k_iterations,
        accumulators,
        iterator_A,
        iterator_B,
        accumulators);

    //
    // Epilogue
    //

    EpilogueOutputOp output_op(params.output_op);

    // Tile iterator writing to destination tensor
    typename Epilogue::OutputTileIterator iterator_D(
        params.iterator_D,
        params.ptr_D,
        tile_info,
        thread_idx,
        params.problem_size,
        cutlass::MatrixCoord(
            tile_info.gemm_m_offset * Mma::Shape::kM,
            threadblock_tile_idx.n() * Mma::Shape::kN),
        nullptr);

    typename Epilogue::OutputTileIterator iterator_C(
        params.iterator_C,
        params.ptr_C,
        tile_info,
        thread_idx,
        params.problem_size,
        cutlass::MatrixCoord(
            tile_info.gemm_m_offset * Mma::Shape::kM,
            threadblock_tile_idx.n() * Mma::Shape::kN),
        params.ptr_bias);

    // Construct the epilogue
    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

    // Run (almost) efficient epilogue
    epilogue(output_op, iterator_D, accumulators, iterator_C);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cuda
} // namespace natten

/////////////////////////////////////////////////////////////////////////////////////////////////
