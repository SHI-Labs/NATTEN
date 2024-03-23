/***************************************************************************************************
 * Copyright (c) 2022-2024 Ali Hassani.
 *
 * Fused Neighborhood Attention kernels are heavily based on the
 * memory-efficient attention kernels from the xFormers project by Meta
 * Platforms, Inc.
 *
 * Copyright (c) Facebook, Inc. and its affiliates
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

#include <natten/cuda/fna/iterators/predicated_tile_access_iterator_residual_last.h>
#include <natten/cuda/fna/iterators/predicated_tile_iterator_residual_last.h>

namespace cutlass {
namespace transform {
namespace threadblock {

template <int NADim, typename BaseIterator>
struct MakeIteratorResidualLast;

template <
    int NADim,
    typename Shape,
    typename Element,
    typename Layout,
    int AdvanceRank,
    typename ThreadMap,
    int AccessSize>
struct MakeIteratorResidualLast<
    NADim,
    PredicatedTileIterator<
        Shape,
        Element,
        Layout,
        AdvanceRank,
        ThreadMap,
        AccessSize>> {
  using Iterator = PredicatedTileIteratorResidualLast<
      NADim,
      Shape,
      Element,
      Layout,
      AdvanceRank,
      ThreadMap,
      AccessSize>;
};

template <
    int NADim,
    typename Shape,
    typename Element,
    typename Layout,
    int AdvanceRank,
    typename ThreadMap,
    typename AccessType>
struct MakeIteratorResidualLast<
    NADim,
    PredicatedTileAccessIterator<
        Shape,
        Element,
        Layout,
        AdvanceRank,
        ThreadMap,
        AccessType>> {
  using Iterator = PredicatedTileAccessIteratorResidualLast<
      NADim,
      Shape,
      Element,
      Layout,
      AdvanceRank,
      ThreadMap,
      AccessType>;
};

} // namespace threadblock
} // namespace transform
} // namespace cutlass
