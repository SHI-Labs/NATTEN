/***************************************************************************************************
 * Copyright (c) 2022-2024 Ali Hassani.
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
    \brief
    Default kernel-level implicit batched GEMM NA definitions combine
   threadblock-scoped matrix multiply-add with the appropriate
   threadblock-scoped epilogue.
*/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/threadblock/default_thread_map_simt.h>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>

#include "natten/cuda/gemm/kernel/default_na.cuh"

#include "natten/cuda/gemm/threadblock/default_epilogue_simt.cuh"
#include "natten/cuda/gemm/threadblock/default_epilogue_tensor_op.cuh"
#include "natten/cuda/gemm/threadblock/na1d_in_input_tile_iterator.cuh"
#include "natten/cuda/gemm/threadblock/na1d_in_output_tile_iterator.cuh"
#include "natten/cuda/gemm/threadblock/na1d_in_value_tile_iterator.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace natten {
namespace cuda {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for NA1dIN
template <
    typename ElementA,
    typename LayoutA,
    typename ElementB,
    typename LayoutB,
    typename ElementC,
    typename LayoutC,
    typename ElementAccumulator,
    typename OperatorClass,
    typename ArchTag,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename EpilogueOutputOp,
    typename ThreadblockSwizzle,
    int Stages,
    typename MathOperatorTag,
    /// Access granularity of A matrix in units of elements
    int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value,
    /// Access granularity of B matrix in units of elements
    int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value>
struct DefaultNA1dIN;

/////////////////////////////////////////////////////////////////////////////////////////////////
//                         OpClassTensorOp NAs
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename ElementA,
    typename LayoutA,
    typename ElementB,
    typename LayoutB,
    typename ElementC,
    typename LayoutC,
    typename ElementAccumulator,
    typename ArchTag,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename EpilogueOutputOp,
    typename ThreadblockSwizzle,
    int Stages,
    typename MathOperatorTag,
    int AlignmentA,
    int AlignmentB>
struct DefaultNA1dIN<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    MathOperatorTag,
    AlignmentA,
    AlignmentB> {
  static_assert(AlignmentA == 1); // NA requirement for IN

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::layout::RowMajor,
      cutlass::arch::OpClassTensorOp,
      Stages,
      MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA = natten::cuda::gemm::threadblock::NA1dINInputTileIterator<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      LayoutA,
      ThreadMapA,
      AccessTypeA>;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB = natten::cuda::gemm::threadblock::NA1dINValueTileIterator<
      cutlass::MatrixShape<ThreadblockShape::kN, ThreadblockShape::kK>,
      ElementB,
      LayoutB,
      ThreadMapB,
      AccessTypeB>;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((cutlass::sizeof_bits<ElementB>::value * AlignmentB) == 128)
      ? cutlass::arch::CacheOperation::Global
      : cutlass::arch::CacheOperation::Always;

  // Define the Mma
  using Mma = natten::cuda::gemm::threadblock::ImplicitGemmMultistage<
      ThreadblockShape,
      IteratorA,
      SmemIteratorA,
      cutlass::arch::CacheOperation::Always,
      IteratorB,
      SmemIteratorB,
      CacheOpB,
      MmaPolicy,
      Stages>;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  using EpilogueElementOutput = typename EpilogueOutputOp::ElementOutput;

  // Threadmap and output iterator were moved out of the default epilogue,
  // which may seem counter-intuitive, but it was all done pretty lazily to
  // allow different kernels to have their correct output iterator instantiated
  // at the kernel level.
  // May move this back and customize the epilogue in the future.
  using OutputTileThreadMap =
      typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
          ThreadblockShape,
          typename WarpMmaTensorOp::Shape,
          kPartitionsK,
          EpilogueElementOutput,
          EpilogueOutputOp::kCount>::Type;

  static bool const UseCUDAStore =
      cutlass::platform::is_same<EpilogueElementOutput, double>::value;

  using OutputTileIterator = natten::cuda::gemm::threadblock::
      NA1dINOutputTileIterator<OutputTileThreadMap, EpilogueElementOutput>;

  // Define the epilogue
  using Epilogue =
      typename natten::cuda::gemm::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape,
          WarpMmaTensorOp,
          kPartitionsK,
          EpilogueOutputOp,
          EpilogueOutputOp::kCount,
          OutputTileIterator>::Epilogue;

  // Define the kernel
  using Kernel =
      ImplicitGemmNA1d<Operator::kIN, Mma, Epilogue, ThreadblockSwizzle>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//                         Pipelined implicit gemm
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename ElementA,
    typename LayoutA,
    typename ElementB,
    typename LayoutB,
    typename ElementC,
    typename LayoutC,
    typename ElementAccumulator,
    typename ArchTag,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename EpilogueOutputOp,
    typename ThreadblockSwizzle,
    typename MathOperatorTag,
    int AlignmentA,
    int AlignmentB>
struct DefaultNA1dIN<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    2,
    MathOperatorTag,
    AlignmentA,
    AlignmentB> {
  static_assert(AlignmentA == 1); // NA requirement for IN

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::layout::RowMajor,
      cutlass::arch::OpClassTensorOp,
      2,
      MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA = natten::cuda::gemm::threadblock::TileIterator<
      natten::cuda::gemm::threadblock::NA1dINInputTileIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA,
          LayoutA,
          ThreadMapA,
          AccessTypeA>,
      1>;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB = natten::cuda::gemm::threadblock::TileIterator<
      natten::cuda::gemm::threadblock::NA1dINValueTileIterator<
          cutlass::MatrixShape<ThreadblockShape::kN, ThreadblockShape::kK>,
          ElementB,
          LayoutB,
          ThreadMapB,
          AccessTypeB>,
      1>;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = cutlass::conv::threadblock::ImplicitGemmPipelined<
      ThreadblockShape,
      IteratorA,
      SmemIteratorA,
      IteratorB,
      SmemIteratorB,
      ElementC,
      LayoutC,
      MmaPolicy>;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  using EpilogueElementOutput = typename EpilogueOutputOp::ElementOutput;

  // Threadmap and output iterator were moved out of the default epilogue,
  // which may seem counter-intuitive, but it was all done pretty lazily to
  // allow different kernels to have their correct output iterator instantiated
  // at the kernel level.
  // May move this back and customize the epilogue in the future.
  using OutputTileThreadMap =
      typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
          ThreadblockShape,
          typename WarpMmaTensorOp::Shape,
          kPartitionsK,
          EpilogueElementOutput,
          EpilogueOutputOp::kCount>::Type;

  static bool const UseCUDAStore =
      cutlass::platform::is_same<EpilogueElementOutput, double>::value;

  using OutputTileIterator = natten::cuda::gemm::threadblock::
      NA1dINOutputTileIterator<OutputTileThreadMap, EpilogueElementOutput>;

  // Define the epilogue
  using Epilogue =
      typename natten::cuda::gemm::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape,
          WarpMmaTensorOp,
          kPartitionsK,
          EpilogueOutputOp,
          EpilogueOutputOp::kCount,
          OutputTileIterator>::Epilogue;

  // Define the kernel
  using Kernel =
      ImplicitGemmNA1d<Operator::kIN, Mma, Epilogue, ThreadblockSwizzle>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//                         SIMT implicit gemm
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename ElementA,
    typename LayoutA,
    typename ElementB,
    typename LayoutB,
    typename ElementC,
    typename LayoutC,
    typename ElementAccumulator,
    typename ArchTag,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename EpilogueOutputOp,
    typename ThreadblockSwizzle,
    int Stages,
    typename MathOperatorTag,
    int AlignmentA,
    int AlignmentB>
struct DefaultNA1dIN<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    MathOperatorTag,
    AlignmentA,
    AlignmentB> {
  static_assert(AlignmentA == 1 && AlignmentB == 1);

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::layout::RowMajor,
      cutlass::arch::OpClassSimt,
      Stages,
      MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA = natten::cuda::gemm::threadblock::NA1dINInputTileIterator<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      LayoutA,
      ThreadMapA>;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB = natten::cuda::gemm::threadblock::NA1dINValueTileIterator<
      cutlass::MatrixShape<ThreadblockShape::kN, ThreadblockShape::kK>,
      ElementB,
      LayoutB,
      ThreadMapB>;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimt = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((cutlass::sizeof_bits<ElementB>::value * AlignmentB) == 128)
      ? cutlass::arch::CacheOperation::Global
      : cutlass::arch::CacheOperation::Always;

  // Define the Mma
  using Mma = natten::cuda::gemm::threadblock::ImplicitGemmMultistage<
      ThreadblockShape,
      IteratorA,
      SmemIteratorA,
      cutlass::arch::CacheOperation::Always,
      IteratorB,
      SmemIteratorB,
      CacheOpB,
      MmaPolicy,
      Stages>;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  using EpilogueElementOutput = typename EpilogueOutputOp::ElementOutput;

  // Threadmap and output iterator were moved out of the default epilogue,
  // which may seem counter-intuitive, but it was all done pretty lazily to
  // allow different kernels to have their correct output iterator instantiated
  // at the kernel level.
  // May move this back and customize the epilogue in the future.
  using OutputTileThreadMap =
      typename cutlass::epilogue::threadblock::DefaultThreadMapSimt<
          ThreadblockShape,
          typename WarpMmaSimt::Shape,
          typename WarpMmaSimt::Policy,
          kPartitionsK,
          EpilogueElementOutput,
          EpilogueOutputOp::kCount>::Type;

  static bool const UseCUDAStore =
      cutlass::platform::is_same<EpilogueElementOutput, double>::value;

  using OutputTileIterator = natten::cuda::gemm::threadblock::
      NA1dINOutputTileIterator<OutputTileThreadMap, EpilogueElementOutput>;

  // Define the epilogue
  using Epilogue =
      typename natten::cuda::gemm::threadblock::DefaultEpilogueSimt<
          ThreadblockShape,
          WarpMmaSimt,
          EpilogueOutputOp,
          EpilogueOutputOp::kCount,
          OutputTileIterator>::Epilogue;

  // Define the kernel
  using Kernel =
      ImplicitGemmNA1d<Operator::kIN, Mma, Epilogue, ThreadblockSwizzle>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//                         Pipelined SIMT implicit gemm
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename ElementA,
    typename LayoutA,
    typename ElementB,
    typename LayoutB,
    typename ElementC,
    typename LayoutC,
    typename ElementAccumulator,
    typename ArchTag,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename EpilogueOutputOp,
    typename ThreadblockSwizzle,
    typename MathOperatorTag,
    int AlignmentA,
    int AlignmentB>
struct DefaultNA1dIN<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    2,
    MathOperatorTag,
    AlignmentA,
    AlignmentB> {
  static_assert(AlignmentA == 1 && AlignmentB == 1);

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      ElementA,
      cutlass::layout::ColumnMajor,
      ElementB,
      cutlass::layout::RowMajor,
      ElementAccumulator,
      cutlass::layout::RowMajor,
      cutlass::arch::OpClassSimt,
      2,
      MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA = natten::cuda::gemm::threadblock::TileIterator<
      natten::cuda::gemm::threadblock::NA1dINInputTileIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA,
          LayoutA,
          ThreadMapA>,
      1>;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB = natten::cuda::gemm::threadblock::TileIterator<
      natten::cuda::gemm::threadblock::NA1dINValueTileIterator<
          cutlass::MatrixShape<ThreadblockShape::kN, ThreadblockShape::kK>,
          ElementB,
          LayoutB,
          ThreadMapB>,
      1>;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimt = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = cutlass::conv::threadblock::ImplicitGemmPipelined<
      ThreadblockShape,
      IteratorA,
      SmemIteratorA,
      IteratorB,
      SmemIteratorB,
      ElementC,
      LayoutC,
      MmaPolicy>;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  using EpilogueElementOutput = typename EpilogueOutputOp::ElementOutput;

  // Threadmap and output iterator were moved out of the default epilogue,
  // which may seem counter-intuitive, but it was all done pretty lazily to
  // allow different kernels to have their correct output iterator instantiated
  // at the kernel level.
  // May move this back and customize the epilogue in the future.
  using OutputTileThreadMap =
      typename cutlass::epilogue::threadblock::DefaultThreadMapSimt<
          ThreadblockShape,
          typename WarpMmaSimt::Shape,
          typename WarpMmaSimt::Policy,
          kPartitionsK,
          EpilogueElementOutput,
          EpilogueOutputOp::kCount>::Type;

  static bool const UseCUDAStore =
      cutlass::platform::is_same<EpilogueElementOutput, double>::value;

  using OutputTileIterator = natten::cuda::gemm::threadblock::
      NA1dINOutputTileIterator<OutputTileThreadMap, EpilogueElementOutput>;

  // Define the epilogue
  using Epilogue =
      typename natten::cuda::gemm::threadblock::DefaultEpilogueSimt<
          ThreadblockShape,
          WarpMmaSimt,
          EpilogueOutputOp,
          EpilogueOutputOp::kCount,
          OutputTileIterator>::Epilogue;

  // Define the kernel
  using Kernel =
      ImplicitGemmNA1d<Operator::kIN, Mma, Epilogue, ThreadblockSwizzle>;
};

} // namespace kernel
} // namespace gemm
} // namespace cuda
} // namespace natten

/////////////////////////////////////////////////////////////////////////////////////////////////
