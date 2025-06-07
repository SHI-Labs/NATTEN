/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "natten/cuda/fna_hopper/collective/fna_collective_tma.hpp"
#include "natten/cuda/fna_hopper/collective/fna_collective_tma_warpspecialized.hpp"
#include "natten/cuda/fna_hopper/collective/fna_epilogue.hpp"
#include "natten/cuda/fna_hopper/kernel/fna_kernel_tma.hpp"
#include "natten/cuda/fna_hopper/kernel/fna_kernel_tma_warpspecialized.hpp"
#include "natten/cuda/fna_hopper/kernel/fna_options.hpp"

namespace cutlass::fna::kernel {

template <
    class NADim,
    class QTileShape,
    class KVTileShape,
    class Element_,
    class ElementAccumulatorQK_,
    class ElementAccumulatorPV_,
    class TileShape_, // BlockQO, BlockKV, BlockHead
    class LayoutQ_,
    class LayoutK_,
    class LayoutV_,
    class Fusion,
    class DispatchPolicy,
    class... Options>
struct FnaBuilder;

template <
    class NADim,
    class QTileShape,
    class KVTileShape,
    class Element,
    class ElementAccumulator,
    class TileShape, // BlockQO, BlockKV, BlockHead
    class Fusion,
    class... Options>
struct FnaBuilder<
    NADim,
    QTileShape,
    KVTileShape,
    Element,
    ElementAccumulator,
    ElementAccumulator,
    TileShape,
    cute::tuple<int, _1, cute::tuple<int, int>>,
    cute::tuple<int, _1, cute::tuple<int, int>>,
    cute::tuple<int, _1, cute::tuple<int, int>>,
    Fusion,
    cutlass::gemm::KernelTma,
    Options...> {
  using CollectiveMainloop = cutlass::fna::collective::FnaMainloopTmaSm90<
      NADim,
      QTileShape,
      KVTileShape,
      Element,
      ElementAccumulator,
      TileShape,
      Fusion,
      Options...>;

  using CollectiveEpilogue = cutlass::fna::collective::FnaFwdEpilogueSm90<
      Element,
      ElementAccumulator,
      typename CollectiveMainloop::TileShapePV>;

  using Kernel = cutlass::fna::kernel::
      FnaKernelTma<CollectiveMainloop, CollectiveEpilogue, Options...>;
};

template <
    class NADim,
    class QTileShape,
    class KVTileShape,
    class Element,
    class ElementAccumulatorQK,
    class ElementAccumulatorPV,
    class TileShape, // BlockQO, BlockKV, BlockHead
    class LayoutQ,
    class LayoutK,
    class LayoutV,
    class Fusion,
    class... Options>
struct FnaBuilder<
    NADim,
    QTileShape,
    KVTileShape,
    Element,
    ElementAccumulatorQK,
    ElementAccumulatorPV,
    TileShape,
    LayoutQ,
    LayoutK,
    LayoutV,
    Fusion,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative,
    Options...> {
  using CollectiveMainloop =
      cutlass::fna::collective::FnaMainloopTmaWarpSpecializedSm90<
          NADim,
          QTileShape,
          KVTileShape,
          Element,
          ElementAccumulatorQK,
          ElementAccumulatorPV,
          TileShape,
          LayoutQ,
          LayoutK,
          LayoutV,
          Fusion,
          Options...>;

  using CollectiveEpilogue = cutlass::fna::collective::FnaFwdEpilogueSm90<
      Element,
      ElementAccumulatorPV,
      typename CollectiveMainloop::TileShapePV>;

  static constexpr bool kIsPersistent =
      find_option_t<Tag::kIsPersistent, false_type, Options...>::value;
  using TileScheduler = std::conditional_t<
      kIsPersistent,
      cutlass::fna::kernel::PersistentTileScheduler,
      cutlass::fna::kernel::IndividualTileScheduler>;

  using Kernel = cutlass::fna::kernel::FnaKernelTmaWarpSpecialized<
      CollectiveMainloop,
      CollectiveEpilogue,
      TileScheduler,
      Options...>;
};

template <
    class NADim,
    class QTileShape,
    class KVTileShape,
    class Element,
    class ElementAccumulatorQK,
    class ElementAccumulatorPV,
    class TileShape, // BlockQO, BlockKV, BlockHead
    class LayoutQ,
    class LayoutK,
    class LayoutV,
    class Fusion,
    class... Options>
struct FnaBuilder<
    NADim,
    QTileShape,
    KVTileShape,
    Element,
    ElementAccumulatorQK,
    ElementAccumulatorPV,
    TileShape,
    LayoutQ,
    LayoutK,
    LayoutV,
    Fusion,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    Options...> {
  using Kernel = typename FnaBuilder<
      NADim,
      QTileShape,
      KVTileShape,
      Element,
      ElementAccumulatorQK,
      ElementAccumulatorPV,
      TileShape,
      LayoutQ,
      LayoutK,
      LayoutV,
      Fusion,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative,
      Options...,
      Option<Tag::kIsPersistent, true_type>,
      Option<Tag::kLoadsQSeparately, true_type>>::Kernel;
};

} // namespace cutlass::fna::kernel
