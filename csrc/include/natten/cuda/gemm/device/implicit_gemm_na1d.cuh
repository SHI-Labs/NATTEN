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
/* \file
   \brief Template for device-level Implicit GEMM Neighborhood Attention (1d)
*/

#pragma once

#include <limits>

#include <cutlass/cutlass.h>
#include <natten/cuda/gemm/neighborhood_attention.cuh>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace natten {
namespace cuda {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename NAGemmKernel_>
class ImplicitGemmNA1d {
 public:
  using UnderlyingKernel = NAGemmKernel_;

  using ElementA = typename UnderlyingKernel::ElementA;
  using LayoutA = typename UnderlyingKernel::LayoutA;
  using ElementB = typename UnderlyingKernel::ElementB;
  using LayoutB = typename UnderlyingKernel::LayoutB;
  using ElementC = typename UnderlyingKernel::ElementC;
  using LayoutC = typename UnderlyingKernel::LayoutC;
  using ElementAccumulator = typename UnderlyingKernel::ElementAccumulator;
  using ElementCompute = typename UnderlyingKernel::ElementCompute;
  using OperatorClass = typename UnderlyingKernel::OperatorClass;
  using ArchTag = typename UnderlyingKernel::ArchTag;
  using ThreadblockShape = typename UnderlyingKernel::ThreadblockShape;
  using WarpShape = typename UnderlyingKernel::WarpShape;
  using InstructionShape = typename UnderlyingKernel::InstructionShape;
  using ThreadblockSwizzle = typename UnderlyingKernel::ThreadblockSwizzle;
  using EpilogueOutputOp = typename UnderlyingKernel::EpilogueOutputOp;
  static int const kStages = UnderlyingKernel::kStages;
  using WarpMmaOperator = typename UnderlyingKernel::WarpMmaOperator;
  using ArchMmaOperator = typename UnderlyingKernel::ArchMmaOperator;
  using MathOperator = typename UnderlyingKernel::MathOperator;

  static Operator const kNAOperator = UnderlyingKernel::kNAOperator;

  static int const kWarpCount = (ThreadblockShape::kM / WarpShape::kM) *
      (ThreadblockShape::kN / WarpShape::kN) *
      (ThreadblockShape::kK / WarpShape::kK);

  /// Argument structure
  using Arguments = typename UnderlyingKernel::Arguments;

 private:
  /// Kernel parameters object
  typename UnderlyingKernel::Params params_;

 public:
  /// Constructs Implicit GEMM NA
  ImplicitGemmNA1d() {}

  /// Determines whether the Implicit GEMM can execute the given problem.
  static cutlass::Status can_implement(Arguments const& args) {
    // dispatch to iterators
    cutlass::Status status =
        UnderlyingKernel::Mma::IteratorA::can_implement(args.problem_size);
    if (cutlass::Status::kSuccess != status) {
      return status;
    }

    status = UnderlyingKernel::Mma::IteratorB::can_implement(args.problem_size);
    if (cutlass::Status::kSuccess != status) {
      return status;
    }

    static int const kAlignmentC =
        UnderlyingKernel::Epilogue::OutputTileIterator::kElementsPerAccess;
    if (kNAOperator == Operator::kPN) {
      if (args.problem_size.R % kAlignmentC)
        return cutlass::Status::kErrorMisalignedOperand;
    } else if (kNAOperator == Operator::kNN) {
      if (args.problem_size.D % kAlignmentC)
        return cutlass::Status::kErrorMisalignedOperand;
    } else if (kNAOperator == Operator::kIN) {
      if (args.problem_size.D % kAlignmentC)
        return cutlass::Status::kErrorMisalignedOperand;
    }

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid =
        threadblock_swizzle.get_grid_shape(threadblock_swizzle.get_tiled_shape(
            kNAOperator,
            args.problem_size,
            {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
            args.problem_size.kTile,
            args.problem_size.kStride,
            args.l_tiled_shape));

    if (!(grid.y <= std::numeric_limits<uint16_t>::max() &&
          grid.z <= std::numeric_limits<uint16_t>::max())) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    return cutlass::Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const& args) {
    // TODO: see if split-K / stream-K are possible?
    return 0;
  }

  /// Initializes GEMM state from arguments.
  cutlass::Status initialize(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr) {
    // initialize the params structure from the arguments
    params_ =
        typename UnderlyingKernel::Params(args //,
                                               // static_cast<int *>(workspace)
        );

    int smem_size = int(sizeof(typename UnderlyingKernel::SharedStorage));

    if (smem_size >= (48 << 10)) {
      cudaError_t result = cudaFuncSetAttribute(
          natten::cuda::gemm::Kernel<ArchTag, UnderlyingKernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);

      if (result != cudaSuccess) {
        return cutlass::Status::kErrorInternal;
      }
    }

    return cutlass::Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  cutlass::Status run(cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(32 * kWarpCount, 1, 1);

    int smem_size = int(sizeof(typename UnderlyingKernel::SharedStorage));

    natten::cuda::gemm::Kernel<ArchTag, UnderlyingKernel>
        <<<grid, block, smem_size, stream>>>(params_);

    cudaError_t result = cudaGetLastError();

    return result == cudaSuccess ? cutlass::Status::kSuccess
                                 : cutlass::Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  cutlass::Status operator()(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr) {
    cutlass::Status status = initialize(args, workspace, stream);

    if (status == cutlass::Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cuda
} // namespace natten

/////////////////////////////////////////////////////////////////////////////////////////////////
