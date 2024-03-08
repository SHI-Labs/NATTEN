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
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>
#include <cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h>

#include "natten/cuda/gemm/kernel/default_na.cuh"

#include "natten/cuda/gemm/threadblock/na2d_pn_input_tile_iterator.cuh"
#include "natten/cuda/gemm/threadblock/na2d_pn_output_tile_iterator.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace natten {
namespace cuda {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for NA2dPN
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
    typename NAShape,
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
struct DefaultNA2dPN;

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
    typename NAShape,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename EpilogueOutputOp,
    typename ThreadblockSwizzle,
    int Stages,
    typename MathOperatorTag,
    int AlignmentA,
    int AlignmentB>
struct DefaultNA2dPN<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    ArchTag,
    NAShape,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    MathOperatorTag,
    AlignmentA,
    AlignmentB> {
  static_assert(
      ArchTag::kMinComputeCapability >= 80,
      "This specialization is intended for SM80 and above.");

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      ElementA,
      cutlass::layout::RowMajor,
      ElementB,
      cutlass::layout::ColumnMajor,
      ElementAccumulator,
      cutlass::layout::RowMajor,
      cutlass::arch::OpClassTensorOp,
      Stages,
      MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA = natten::cuda::gemm::threadblock::NA2dPNInputTileIterator<
      NAShape,
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      LayoutA,
      ThreadMapA,
      AccessTypeA>;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB = natten::cuda::gemm::threadblock::NA2dPNInputTileIterator<
      NAShape,
      cutlass::MatrixShape<ThreadblockShape::kN, ThreadblockShape::kK>,
      ElementB,
      LayoutB,
      ThreadMapB,
      AccessTypeB>;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((cutlass::sizeof_bits<ElementA>::value * AlignmentA) == 128)
      ? cutlass::arch::CacheOperation::Global
      : cutlass::arch::CacheOperation::Always;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((cutlass::sizeof_bits<ElementB>::value * AlignmentB) == 128)
      ? cutlass::arch::CacheOperation::Global
      : cutlass::arch::CacheOperation::Always;

  // Define the Mma
  using Mma = natten::cuda::gemm::threadblock::ImplicitGemmMultistage<
      ThreadblockShape,
      IteratorA,
      SmemIteratorA,
      CacheOpA,
      IteratorB,
      SmemIteratorB,
      CacheOpB,
      MmaPolicy,
      Stages>;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using EpilogueBase =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape,
          WarpMmaTensorOp,
          kPartitionsK,
          EpilogueOutputOp,
          EpilogueOutputOp::kCount>;

  using OutputTileIterator =
      natten::cuda::gemm::threadblock::NA2dPNOutputTileIterator<
          NAShape,
          typename EpilogueBase::OutputTileThreadMap,
          typename EpilogueOutputOp::ElementOutput>;

  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
      typename EpilogueBase::Shape,
      typename EpilogueBase::WarpMmaTensorOp,
      EpilogueBase::kPartitionsK,
      OutputTileIterator,
      typename EpilogueBase::AccumulatorFragmentIterator,
      typename EpilogueBase::WarpTileIterator,
      typename EpilogueBase::SharedLoadIterator,
      typename EpilogueBase::OutputOp,
      typename EpilogueBase::Padding,
      GetFragmentsPerIter<ArchTag, EpilogueBase>::value>;

  // Define the kernel
  using Kernel = ImplicitGemmNA2d<
      Operator::kPN,
      NAShape,
      Mma,
      Epilogue,
      ThreadblockSwizzle>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//                         Pipelined implicit gemm mainloop
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
    typename NAShape,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename EpilogueOutputOp,
    typename ThreadblockSwizzle,
    typename MathOperatorTag,
    int AlignmentA,
    int AlignmentB>
struct DefaultNA2dPN<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    ArchTag,
    NAShape,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    2,
    MathOperatorTag,
    AlignmentA,
    AlignmentB> {
  static_assert(
      ArchTag::kMinComputeCapability == 70 ||
          ArchTag::kMinComputeCapability == 75,
      "This specialization is intended for SM70 and SM75.");

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      ElementA,
      cutlass::layout::RowMajor,
      ElementB,
      cutlass::layout::ColumnMajor,
      ElementAccumulator,
      cutlass::layout::RowMajor,
      cutlass::arch::OpClassTensorOp,
      2,
      MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA = natten::cuda::gemm::threadblock::TileIterator<
      natten::cuda::gemm::threadblock::NA2dPNInputTileIterator<
          NAShape,
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA,
          LayoutA,
          ThreadMapA,
          AccessTypeA>,
      2>;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB = natten::cuda::gemm::threadblock::TileIterator<
      natten::cuda::gemm::threadblock::NA2dPNInputTileIterator<
          NAShape,
          cutlass::MatrixShape<ThreadblockShape::kN, ThreadblockShape::kK>,
          ElementB,
          LayoutB,
          ThreadMapB,
          AccessTypeB>,
      2>;

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

  // Define the epilogue
  using EpilogueBase = typename std::conditional<
      ArchTag::kMinComputeCapability == 75,
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape,
          WarpMmaTensorOp,
          kPartitionsK,
          EpilogueOutputOp,
          EpilogueOutputOp::kCount>,
      typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
          ThreadblockShape,
          WarpMmaTensorOp,
          kPartitionsK,
          EpilogueOutputOp,
          EpilogueOutputOp::kCount>>::type;

  using OutputTileIterator =
      natten::cuda::gemm::threadblock::NA2dPNOutputTileIterator<
          NAShape,
          typename EpilogueBase::OutputTileThreadMap,
          typename EpilogueOutputOp::ElementOutput>;

  using Epilogue = cutlass::epilogue::threadblock::Epilogue<
      typename EpilogueBase::Shape,
      typename EpilogueBase::WarpMmaTensorOp,
      EpilogueBase::kPartitionsK,
      OutputTileIterator,
      typename EpilogueBase::AccumulatorFragmentIterator,
      typename EpilogueBase::WarpTileIterator,
      typename EpilogueBase::SharedLoadIterator,
      typename EpilogueBase::OutputOp,
      typename EpilogueBase::Padding,
      GetFragmentsPerIter<ArchTag, EpilogueBase>::value>;

  // Define the kernel
  using Kernel = ImplicitGemmNA2d<
      Operator::kPN,
      NAShape,
      Mma,
      Epilogue,
      ThreadblockSwizzle>;
};

} // namespace kernel
} // namespace gemm
} // namespace cuda
} // namespace natten

/////////////////////////////////////////////////////////////////////////////////////////////////
