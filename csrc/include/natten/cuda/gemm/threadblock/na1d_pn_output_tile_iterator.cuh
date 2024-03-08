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
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory
  to match canonical tensor layouts in global memory. Epilogues support
  conversion and reduction operations.

*/

#pragma once

#include <cutlass/arch/arch.h>
#include <cutlass/arch/memory.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/threadblock/output_tile_thread_map.h>
#include <cutlass/epilogue/threadblock/predicated_tile_iterator_params.h>
#include <cutlass/fast_math.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/permute.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/transform/pitch_linear_thread_map.h>

#include "natten/cuda/gemm/na1d_problem_size.cuh"
#include "natten/cuda/gemm/neighborhood_attention.cuh"
#include "natten/cuda/gemm/threadblock/na1d_params.cuh"
#include "natten/cuda/gemm/threadblock/na1d_tile.cuh"

////////////////////////////////////////////////////////////////////////////////

namespace natten {
namespace cuda {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load and store output tile from global memory in
/// epilogue.
template <
    typename ThreadMap_, ///< Thread map (conept: OutputTileThreadMap)
    typename Element_, ///< Element data type
    bool UseCUDAStore = false>
class NA1dPNOutputTileIterator {
 public:
  using TileInfo = NA1dTileInfo<Operator::kPN>;
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_;

  using Layout = cutlass::layout::RowMajor;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = cutlass::MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  static_assert(ThreadMap::kElementsPerAccess == 1); // NA requirement for PN

  static_assert(
      ThreadMap::Iterations::kRow > 0,
      "ThreadMap::Iterations::kRow must be > 0");
  static_assert(
      ThreadMap::Iterations::kGroup > 0,
      "ThreadMap::Iterations::kGroup must be > 0");
  static_assert(
      ThreadMap::Iterations::kCluster > 0,
      "ThreadMap::Iterations::kCluster must be > 0");
  static_assert(
      ThreadMap::Iterations::kColumn > 0,
      "ThreadMap::Iterations::kColumn must be > 0");

  /// Fragment object
  using Fragment = cutlass::Array<
      Element,
      ThreadMap::Iterations::kColumn * ThreadMap::Iterations::kRow *
          ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster *
          ThreadMap::kElementsPerAccess>;

  /// Memory access size
  using AccessType =
      cutlass::AlignedArray<Element, ThreadMap::kElementsPerAccess>;

  //
  // Parameters struct
  //

  /// Uses a non-template class
  struct Params : cutlass::epilogue::threadblock::PredicatedTileIteratorParams {
    using Base = cutlass::epilogue::threadblock::PredicatedTileIteratorParams;

    CUTLASS_HOST_DEVICE
    Params() {}

    CUTLASS_HOST_DEVICE
    Params(Layout const& layout)
        : cutlass::epilogue::threadblock::PredicatedTileIteratorParams(
              layout.stride(0) * int(sizeof(AccessType)) / kElementsPerAccess,
              cutlass::epilogue::threadblock::make_OutputTileThreadMapDesc<
                  ThreadMap>()) {}

    CUTLASS_HOST_DEVICE
    Params(Base const& base) : Base(base) {}
  };

 private:
  //
  // Data members
  //

  /// Parameters structure containing reference and precomputed state.
  cutlass::epilogue::threadblock::PredicatedTileIteratorParams params_;

  /// Byte-level pointer for load().
  uint8_t* rpb_byte_pointer_;

  /// Byte-level pointer for store().
  uint8_t* store_byte_pointer_;

  /// A thread's starting row position (assuming steady-state predicates have
  /// been computed)
  Index thread_start_row_;

  /// A thread's starting column
  Index thread_start_column_;

  /// Internal state counter
  int state_[3];

  // Neighborhood tile information and problem size
  NA1dProblemSize const& problem_size_;
  TileInfo const& tile_info;

  //
  // Static asserts about internal strides
  //

  static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");
  static_assert(
      sizeof(cutlass::epilogue::threadblock::PredicatedTileIteratorParams::
                 stride) == 8,
      "Expected 64b strides");

 private:
  //
  // Methods
  //

 public:
  //
  // Methods
  //

  /// Constructor
  CUTLASS_DEVICE
  NA1dPNOutputTileIterator(
      cutlass::epilogue::threadblock::PredicatedTileIteratorParams const&
          params,
      Element* pointer,
      TileInfo const& tile_info,
      int thread_idx,
      NA1dProblemSize const& problem_size,
      TensorCoord threadblock_offset,
      Element* rpb_pointer)
      : params_(params), problem_size_(problem_size), tile_info(tile_info) {
    TensorCoord thread_offset =
        ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    thread_start_row_ = thread_offset.row();
    thread_start_column_ = thread_offset.column();

    // Initialize rpb_byte_pointer_
    rpb_byte_pointer_ =
        reinterpret_cast<uint8_t*>(rpb_pointer + tile_info.bias_offset);

    // Initialize byte_pointer_
    store_byte_pointer_ =
        reinterpret_cast<uint8_t*>(pointer + tile_info.out_offset);

    // Initialize internal state counter
    state_[0] = state_[1] = state_[2] = 0;
  }

  ///// Adds a pointer offset in units of Element
  // CUTLASS_HOST_DEVICE
  // void add_pointer_offset(LongIndex pointer_offset) {
  //  store_byte_pointer_ +=
  //      pointer_offset * cutlass::sizeof_bits<Element>::value / 8;
  //}

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment& frag, int64_t byte_offset) const {
    if (rpb_byte_pointer_ == nullptr)
      return;
    uint8_t* byte_pointer = rpb_byte_pointer_;
    AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
         ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int frag_row_idx =
              (row +
               ThreadMap::Iterations::kRow *
                   (group + ThreadMap::Iterations::kGroup * cluster));

          int row_r = row * ThreadMap::Delta::kRow +
              group * ThreadMap::Delta::kGroup +
              cluster * ThreadMap::Delta::kCluster + thread_start_row_;

          int row_start_l = window_start(
              row_r + tile_info.l_ext,
              tile_info.tile_l_length,
              problem_size_.R,
              problem_size_.RH);

          int pi = pb_start(
              row_r + tile_info.l_ext,
              tile_info.tile_l_length,
              problem_size_.RH);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn;
               ++column) {
            int col_init =
                column * ThreadMap::Delta::kColumn + thread_start_column_;

            int ni = col_init + tile_info.l_ext - row_start_l;

            bool guard =
                (ni >= 0 && ni < problem_size_.R &&
                 row_r + tile_info.l_ext < tile_info.tile_l_length);
            LongIndex perm_idx = LongIndex(pi + ni);

            AccessType* memory_pointer = reinterpret_cast<AccessType*>(
                byte_pointer + byte_offset +
                perm_idx * sizeof(AccessType) / kElementsPerAccess);

            cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                frag_ptr
                    [frag_row_idx * ThreadMap::Iterations::kColumn + column],
                (void*)&memory_pointer[0],
                guard);
          }
        }

        // if (group + 1 < ThreadMap::Iterations::kGroup) {
        //  byte_pointer += params_.increment_group;
        //}
      }

      // if (cluster + 1 < ThreadMap::Iterations::kCluster) {
      //  byte_pointer += params_.increment_cluster;
      //}
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment& frag) const {
    if (rpb_byte_pointer_ == nullptr)
      return;
    load_with_byte_offset(frag, 0);
  }

  // Returns the neighborhood start coordinate
  CUTLASS_HOST_DEVICE
  int pb_start(const int index, const int length, const int window_radius)
      const {
    return window_radius + (index < window_radius) * (window_radius - index) +
        (index + window_radius >= length) *
        (length - index - 1 - window_radius);
  }

  // Returns the neighborhood start coordinate
  CUTLASS_HOST_DEVICE
  int window_start(int index, int length, int window_size, int window_radius)
      const {
    return cutlass::fast_max<int>(index - window_radius, 0) +
        (index + window_radius >= length) *
        (length - index - window_radius - 1);
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) const {
    uint8_t* byte_pointer = store_byte_pointer_;
    AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster;
         ++cluster) {
      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {
        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {
          int frag_row_idx =
              (row +
               ThreadMap::Iterations::kRow *
                   (group + ThreadMap::Iterations::kGroup * cluster));

          int row_r = row * ThreadMap::Delta::kRow +
              group * ThreadMap::Delta::kGroup +
              cluster * ThreadMap::Delta::kCluster + thread_start_row_;

          int row_start_l = window_start(
              row_r + tile_info.l_ext,
              tile_info.tile_l_length,
              problem_size_.R,
              problem_size_.RH);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn;
               ++column) {
            int col_init =
                column * ThreadMap::Delta::kColumn + thread_start_column_;

            int ni = col_init + tile_info.l_ext - row_start_l;

            bool guard =
                (ni >= 0 && ni < problem_size_.R &&
                 row_r + tile_info.l_ext < tile_info.tile_l_length);
            LongIndex perm_idx = LongIndex(row_r * problem_size_.dilation) *
                    LongIndex(params_.stride * kElementsPerAccess /
                              sizeof(AccessType)) +
                LongIndex(ni);

            // Locate memory_pointer
            AccessType* memory_pointer = reinterpret_cast<AccessType*>(
                byte_pointer + byte_offset +
                perm_idx * sizeof(AccessType) / kElementsPerAccess);

            if (UseCUDAStore) {
              if (guard) {
                memory_pointer[0] = frag_ptr
                    [frag_row_idx * ThreadMap::Iterations::kColumn + column];
              }
            } else {
              cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                  frag_ptr
                      [frag_row_idx * ThreadMap::Iterations::kColumn + column],
                  (void*)&memory_pointer[0],
                  guard);
            }
          }
        }

        // if (group + 1 < ThreadMap::Iterations::kGroup) {
        //  byte_pointer += params_.increment_group;
        //}
      }

      // if (cluster + 1 < ThreadMap::Iterations::kCluster) {
      //  byte_pointer += params_.increment_cluster;
      //}
    }
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const& frag) const {
    store_with_byte_offset(frag, 0);
  }

  CUTLASS_DEVICE
  cutlass::MatrixCoord thread_start() const {
    return cutlass::MatrixCoord(thread_start_row_, thread_start_column_);
  }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_row() const {
    return thread_start_row_;
  }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_column() const {
    return thread_start_column_;
  }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  NA1dPNOutputTileIterator& operator++() {
    ++state_[0];

    thread_start_row_ += ThreadMap::Shape::kRow;

    if (state_[0] == ThreadMap::Count::kRow) {
      state_[0] = 0;
      ++state_[1];
      // store_byte_pointer_ += params_.advance_group;

      thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
          ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {
        state_[1] = 0;
        ++state_[2];
        // store_byte_pointer_ += params_.advance_cluster;

        thread_start_row_ += ThreadMap::Count::kGroup *
            ThreadMap::Shape::kGroup * ThreadMap::Count::kRow *
            ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;
          // store_byte_pointer_ += params_.advance_tile;

          thread_start_row_ += ThreadMap::Shape::kGroup *
              ThreadMap::Shape::kRow * ThreadMap::Shape::kCluster *
              ThreadMap::Shape::kTile;
        }
      }
    }

    return *this;
  }

  /// Advances a number of positions to load or store
  CUTLASS_HOST_DEVICE
  NA1dPNOutputTileIterator& operator+=(int increment) {
    // Row
    state_[0] += increment;
    int increment_row = state_[0] / ThreadMap::Count::kRow;
    state_[0] = state_[0] % ThreadMap::Count::kRow;

    // store_byte_pointer_ += (params_.advance_row * increment);
    thread_start_row_ += (ThreadMap::Shape::kRow * increment);

    // Group
    state_[1] += increment_row;
    int increment_group = state_[1] / ThreadMap::Count::kGroup;
    state_[1] = state_[1] % ThreadMap::Count::kGroup;

    // store_byte_pointer_ += (params_.advance_group * increment_row);
    thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
        ThreadMap::Shape::kRow * ThreadMap::Count::kRow * increment_row;

    // Cluster
    state_[2] += increment_group;
    int increment_cluster = state_[2] / ThreadMap::Count::kCluster;
    state_[2] = state_[2] % ThreadMap::Count::kCluster;

    // store_byte_pointer_ += (params_.advance_cluster * increment_group);
    thread_start_row_ += ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup *
        ThreadMap::Count::kRow * ThreadMap::Shape::kRow * increment_group;

    // Tile
    // store_byte_pointer_ += (params_.advance_tile * increment_cluster);
    thread_start_row_ += ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow *
        ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile *
        increment_cluster;

    return *this;
  }

  ///< Efficiently disables all accesses guarded by mask
  CUTLASS_DEVICE void clear_mask() {}

  ///< Efficiently enables all accesses guarded by mask
  CUTLASS_DEVICE void enable_mask() {}
};

///////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cuda
} // namespace natten

////////////////////////////////////////////////////////////////////////////////
