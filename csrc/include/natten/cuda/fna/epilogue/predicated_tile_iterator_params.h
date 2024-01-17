/*
 * Copied from CUTLASS (https://github.com/NVIDIA/cutlass/) and edited.
 */
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
*/

#pragma once

#include <cutlass/cutlass.h>

#include <cutlass/layout/matrix.h>
#include <cutlass/layout/pitch_linear.h>

#include <natten/cuda/fna/na_utils.cuh>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

//
// Parameters struct for PredicatedTileIterator
//

template <int NADim>
struct CustomPredicatedTileIteratorParams {
  using Dim = typename natten::cuda::fna::GetDim<NADim>::type;

  using Index = int32_t;
  using LongIndex = int64_t;

  //
  // Data members
  //

  Dim stride; ///< stride in bytes between rows

  // LongIndex increment_row;        ///< increment quantity (in bytes) to
  // advance when moving between rows LongIndex increment_group;      ///<
  // increment quantity (in bytes) to advance when moving to the next group
  // LongIndex increment_cluster;    ///< increment quantity (in bytes) to
  // advance when moving to the next cluster

  // LongIndex advance_row;          ///< amount to add to move to the next
  // 'row' position LongIndex advance_group;        ///< amount to add to move
  // to the next 'group' position LongIndex advance_cluster;      ///< amount to
  // add to move to the next 'cluster' position LongIndex advance_tile; ///<
  // amount to add to move to the next 'tile'

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Status initialize(
      Dim stride_,
      Index elem,
      Dim extent_row,
      OutputTileThreadMapDesc thread_map) {
    stride = stride_ * elem;

    // increment_row = (stride *
    // natten::cuda::fna::map_index_to_coord(thread_map.delta.row,
    // extent_row)).sum();

    // increment_group = (stride *
    // natten::cuda::fna::map_index_to_coord(thread_map.delta.group, extent_row)
    //   - stride * natten::cuda::fna::map_index_to_coord(thread_map.delta.row *
    //   (thread_map.iterations.row - 1), extent_row)).sum();

    // increment_cluster = (stride *
    // natten::cuda::fna::map_index_to_coord(thread_map.delta.cluster,
    // extent_row)
    //   - stride * natten::cuda::fna::map_index_to_coord(thread_map.delta.group
    //   * (thread_map.iterations.group - 1), extent_row)
    //   - stride * natten::cuda::fna::map_index_to_coord(thread_map.delta.row *
    //   (thread_map.iterations.row - 1), extent_row)).sum();

    // advance_row = (stride *
    // natten::cuda::fna::map_index_to_coord(thread_map.shape.row,
    // extent_row)).sum();

    // advance_group =
    //   (stride *
    //   natten::cuda::fna::map_index_to_coord((thread_map.shape.group - 1) *
    //   thread_map.shape.row * thread_map.count.row, extent_row)).sum();
    //
    // advance_cluster = (
    //   stride *
    //   natten::cuda::fna::map_index_to_coord(thread_map.count.group *
    //   thread_map.shape.group *
    //   thread_map.count.row *
    //   thread_map.shape.row, extent_row)).sum();
    //
    // advance_tile =(
    //   stride *
    //   natten::cuda::fna::map_index_to_coord(thread_map.shape.group *
    //   thread_map.shape.row *
    //   thread_map.shape.cluster *
    //   thread_map.shape.tile, extent_row)).sum();

    return Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  CustomPredicatedTileIteratorParams(
      Dim stride,
      Index elem,
      Dim extent_row,
      OutputTileThreadMapDesc thread_map) {
    initialize(stride, elem, extent_row, thread_map);
  }
};

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
