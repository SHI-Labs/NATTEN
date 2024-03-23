/*
 * Copied from xFormers (https://github.com/facebookresearch/xformers/) and
 * edited
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * BSD 3-Clause License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC
 * Laboratories America and IDIAP Research Institute nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <natten/cuda/fna/iterators/predicated_tile_access_iterator.h>
#include <natten/cuda/fna/iterators/predicated_tile_iterator.h>

#include <cutlass/gemm/threadblock/mma_multistage.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>

#include <natten/cuda/fna/gemm/custom_mma_multistage.h>
#include <natten/cuda/fna/gemm/custom_mma_pipelined.h>

namespace natten {
namespace cuda {
namespace fna {

template <typename Mma, int kMaxK>
struct MakeCustomMma;

template <
    typename Shape,
    typename IteratorA,
    typename SmemIteratorA,
    cutlass::arch::CacheOperation::Kind CacheOpA,
    typename IteratorB,
    typename SmemIteratorB,
    cutlass::arch::CacheOperation::Kind CacheOpB,
    typename ElementC,
    typename LayoutC,
    typename Policy,
    int Stages,
    cutlass::gemm::SharedMemoryClearOption SharedMemoryClear,
    int kMaxK>
struct MakeCustomMma<
    cutlass::gemm::threadblock::MmaMultistage<
        Shape,
        IteratorA,
        SmemIteratorA,
        CacheOpA,
        IteratorB,
        SmemIteratorB,
        CacheOpB,
        ElementC,
        LayoutC,
        Policy,
        Stages,
        SharedMemoryClear>,
    kMaxK> {
  // Reduce the number of stages if we don't need that many
  static int constexpr kStages =
      kMaxK == cutlass::platform::numeric_limits<int>::max()
      ? Stages
      : cutlass::const_min(
            Stages,
            (kMaxK + int(Shape::kK) - 1) / int(Shape::kK));
  using Mma = cutlass::gemm::threadblock::CustomMmaMultistage<
      Shape,
      IteratorA,
      SmemIteratorA,
      CacheOpA,
      IteratorB,
      SmemIteratorB,
      CacheOpB,
      ElementC,
      LayoutC,
      Policy,
      kStages,
      SharedMemoryClear,
      kMaxK>;
};

template <
    typename Shape,
    typename IteratorA,
    typename SmemIteratorA,
    typename IteratorB,
    typename SmemIteratorB,
    typename ElementC,
    typename LayoutC,
    typename Policy,
    int kMaxK>
struct MakeCustomMma<
    cutlass::gemm::threadblock::MmaPipelined<
        Shape,
        IteratorA,
        SmemIteratorA,
        IteratorB,
        SmemIteratorB,
        ElementC,
        LayoutC,
        Policy>,
    kMaxK> {
  using Mma = cutlass::gemm::threadblock::CustomMmaPipelined<
      Shape,
      IteratorA,
      SmemIteratorA,
      IteratorB,
      SmemIteratorB,
      ElementC,
      LayoutC,
      Policy>;
};

template <int NADim, typename Mma, int kMaxK>
struct MakeCustomMmaAndReplaceIterators;

template <
    int NADim,
    typename Shape,
    typename IteratorA,
    typename SmemIteratorA,
    cutlass::arch::CacheOperation::Kind CacheOpA,
    typename IteratorB,
    typename SmemIteratorB,
    cutlass::arch::CacheOperation::Kind CacheOpB,
    typename ElementC,
    typename LayoutC,
    typename Policy,
    int Stages,
    cutlass::gemm::SharedMemoryClearOption SharedMemoryClear,
    int kMaxK>
struct MakeCustomMmaAndReplaceIterators<
    NADim,
    cutlass::gemm::threadblock::MmaMultistage<
        Shape,
        IteratorA,
        SmemIteratorA,
        CacheOpA,
        IteratorB,
        SmemIteratorB,
        CacheOpB,
        ElementC,
        LayoutC,
        Policy,
        Stages,
        SharedMemoryClear>,
    kMaxK> {
  using NewIteratorA =
      cutlass::transform::threadblock::CustomPredicatedTileAccessIterator<
          NADim,
          typename IteratorA::Shape,
          typename IteratorA::Element,
          typename IteratorA::Layout,
          IteratorA::kAdvanceRank,
          typename IteratorA::ThreadMap,
          typename IteratorA::AccessType>;

  using NewIteratorB =
      cutlass::transform::threadblock::CustomPredicatedTileAccessIterator<
          NADim,
          typename IteratorB::Shape,
          typename IteratorB::Element,
          typename IteratorB::Layout,
          IteratorB::kAdvanceRank,
          typename IteratorB::ThreadMap,
          typename IteratorB::AccessType>;
  // Reduce the number of stages if we don't need that many
  static int constexpr kStages =
      kMaxK == cutlass::platform::numeric_limits<int>::max()
      ? Stages
      : cutlass::const_min(
            Stages,
            (kMaxK + int(Shape::kK) - 1) / int(Shape::kK));
  using Mma = cutlass::gemm::threadblock::CustomMmaMultistage<
      Shape,
      NewIteratorA,
      SmemIteratorA,
      CacheOpA,
      NewIteratorB,
      SmemIteratorB,
      CacheOpB,
      ElementC,
      LayoutC,
      Policy,
      kStages,
      SharedMemoryClear,
      kMaxK>;
};

template <
    int NADim,
    typename Shape,
    typename IteratorA,
    typename SmemIteratorA,
    typename IteratorB,
    typename SmemIteratorB,
    typename ElementC,
    typename LayoutC,
    typename Policy,
    int kMaxK>
struct MakeCustomMmaAndReplaceIterators<
    NADim,
    cutlass::gemm::threadblock::MmaPipelined<
        Shape,
        IteratorA,
        SmemIteratorA,
        IteratorB,
        SmemIteratorB,
        ElementC,
        LayoutC,
        Policy>,
    kMaxK> {
  using NewIteratorA =
      cutlass::transform::threadblock::CustomPredicatedTileIterator<
          NADim,
          typename IteratorA::Shape,
          typename IteratorA::Element,
          typename IteratorA::Layout,
          IteratorA::kAdvanceRank,
          typename IteratorA::ThreadMap,
          IteratorA::AccessType::kElements>;

  using NewIteratorB =
      cutlass::transform::threadblock::CustomPredicatedTileIterator<
          NADim,
          typename IteratorB::Shape,
          typename IteratorB::Element,
          typename IteratorB::Layout,
          IteratorB::kAdvanceRank,
          typename IteratorB::ThreadMap,
          IteratorB::AccessType::kElements>;
  using Mma = cutlass::gemm::threadblock::CustomMmaPipelined<
      Shape,
      NewIteratorA,
      SmemIteratorA,
      NewIteratorB,
      SmemIteratorB,
      ElementC,
      LayoutC,
      Policy>;
};

} // namespace fna
} // namespace cuda
} // namespace natten
