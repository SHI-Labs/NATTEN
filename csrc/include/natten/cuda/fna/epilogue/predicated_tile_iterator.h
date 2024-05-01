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
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/numeric_types.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/transform/pitch_linear_thread_map.h>

#include <natten/cuda/fna/epilogue/predicated_tile_iterator_params.h>
#include <natten/cuda/fna/na_utils.cuh>

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////

namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator used to load and store output tile from global memory in
/// epilogue.
///
/// Satisfies: ReadableTileIterator | CustomPredicatedTileIterator |
/// ForwardTileIterator
///
template <
    int NADim,
    typename ThreadMap_, ///< Thread map (conept: OutputTileThreadMap)
    typename Element_ ///< Element data type
    >
class CustomPredicatedTileIterator {
 public:
  static_assert(NADim >= 1 && NADim < 4);
  using Dim = typename natten::cuda::fna::GetDim<NADim>::type;

  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_;

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

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
  using Fragment = Array<
      Element,
      ThreadMap::Iterations::kColumn * ThreadMap::Iterations::kRow *
          ThreadMap::Iterations::kGroup * ThreadMap::Iterations::kCluster *
          ThreadMap::kElementsPerAccess>;

  /// Memory access size
  using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

  //
  // Parameters struct
  //
  using ParamsBase = CustomPredicatedTileIteratorParams<NADim>;

  /// Uses a non-template class
  struct Params : ParamsBase {
    using Base = ParamsBase;

    CUTLASS_HOST_DEVICE
    Params() {}

    CUTLASS_HOST_DEVICE
    Params(Dim stride, Dim extent_row)
        : ParamsBase(
              stride,
              int32_t(sizeof(AccessType)) / kElementsPerAccess,
              extent_row,
              make_OutputTileThreadMapDesc<ThreadMap>()) {}

    CUTLASS_HOST_DEVICE
    Params(Base const& base) : Base(base) {}
  };

  /// Mask object
  struct Mask {
    static int const kCount = ThreadMap::Iterations::kColumn;

    /// Predicate state
    bool predicates[kCount];

    //
    // Mask
    //
    CUTLASS_HOST_DEVICE
    Mask() {
      enable();
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_HOST_DEVICE void clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = false;
      }
    }

    ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
    CUTLASS_DEVICE void enable() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = true;
      }
    }
  };

 private:
  //
  // Data members
  //

  /// Parameters structure containing reference and precomputed state.
  ParamsBase params_;

  /// Byte-level pointer. This pointer is usually for both load() and store(),
  /// unless PermuteD is performed. When having PermuteD, byte_pointer_ is only
  /// for load().
  uint8_t* byte_pointer_;

  // /// Byte-level pointer for store(). Due to PermuteD Op, store_byte_pointer_
  // may be with different address computation compared to byte_pointer_.
  // uint8_t *store_byte_pointer_;

  /// Array of boolean values to contain steady-state predicates
  Mask mask_;

  /// Extent of the matrix tile in rows
  Dim extent_row_;
  int32_t extent_row_int;

  /// Extent of the matrix tile in rows
  Index extent_column_;

  /// A thread's starting row position (assuming steady-state predicates have
  /// been computed)
  Index thread_start_row_;

  /// A thread's starting column
  Index thread_start_column_;

  /// Internal state counter
  int state_[3];

  //
  // Static asserts about internal strides
  //

  static_assert(sizeof(extent_column_) == 4, "Expected 32b extents");
  static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");

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
  CustomPredicatedTileIterator(
      ParamsBase const& params,
      Element* pointer,
      Dim extent_row,
      int32_t extent_col,
      int thread_idx,
      TensorCoord threadblock_offset = TensorCoord())
      : params_(params) {
    TensorCoord thread_offset =
        ThreadMap::initial_offset(thread_idx) + threadblock_offset;

    extent_row_ = extent_row;
    extent_column_ = extent_col;
    extent_row_int = extent_row.prod32();

    thread_start_row_ = thread_offset.row();
    thread_start_column_ = thread_offset.column();

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {
      mask_.predicates[c] =
          ((thread_offset.column() + ThreadMap::Delta::kColumn * c) <
           extent_col);
    }

    // Null pointer performs no accesses
    if (!pointer) {
      mask_.clear();
    }

    // if (ScatterD && !indices) {
    //   mask_.clear();
    // }

    // Initialize byte_pointer_
    byte_pointer_ = reinterpret_cast<uint8_t*>(pointer) +
        //(natten::cuda::fna::map_index_to_coord(thread_offset.row(),
        // extent_row_) * params_.stride).sum() +
        LongIndex(thread_offset.column()) * sizeof(AccessType) /
            kElementsPerAccess;

    // if (ScatterD) {
    //  byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) +
    //    LongIndex(thread_offset.column()) * sizeof(AccessType) /
    //    kElementsPerAccess;
    //}

    // store_byte_pointer_ is set to be the same with byte_pointer_ unless
    // PermuteD is used.
    // store_byte_pointer_ = PermuteD ? reinterpret_cast<uint8_t *>(pointer) :
    // byte_pointer_;
    // store_byte_pointer_ = byte_pointer_;

    // Initialize internal state counter
    state_[0] = state_[1] = state_[2] = 0;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    // store_byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
    byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment& frag, int64_t byte_offset) const {
    uint8_t* byte_pointer = byte_pointer_;
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

          int32_t row_offset = row * ThreadMap::Delta::kRow +
              group * ThreadMap::Delta::kGroup +
              cluster * ThreadMap::Delta::kCluster;

          auto row_offset_full_ = natten::cuda::fna::map_index_to_coord(
              row_offset + thread_start_row_, extent_row_);
          // bool row_guard =
          // natten::cuda::fna::is_coord_within_upper_bound(row_offset_full_,
          // extent_row_);
          bool row_guard = row_offset + thread_start_row_ < extent_row_int;

          AccessType* memory_pointer = reinterpret_cast<AccessType*>(
              byte_pointer + byte_offset +
              (row_offset_full_ * params_.stride).sum());

          // if (ScatterD && row_guard) {
          //  assert(indices_);

          //  memory_pointer = reinterpret_cast<AccessType *>(byte_pointer +
          //  byte_offset +
          //    LongIndex(indices_[row_offset + thread_start_row_]) *
          //    LongIndex(params_.stride));
          //}

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn;
               ++column) {
            bool guard = row_guard && mask_.predicates[column];

            cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
                frag_ptr
                    [frag_row_idx * ThreadMap::Iterations::kColumn + column],
                (void*)&memory_pointer
                    [column * ThreadMap::Delta::kColumn / kElementsPerAccess],
                guard);
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            // if (!ScatterD) {
            // byte_pointer += params_.increment_row;
            //}
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          // byte_pointer += params_.increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        // byte_pointer += params_.increment_cluster;
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment& frag) const {
    load_with_byte_offset(frag, 0);
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const& frag, int64_t byte_offset) const {
    // uint8_t *byte_pointer = store_byte_pointer_;
    uint8_t* byte_pointer = byte_pointer_;
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

          int32_t row_offset = row * ThreadMap::Delta::kRow +
              group * ThreadMap::Delta::kGroup +
              cluster * ThreadMap::Delta::kCluster;

          auto row_offset_full_ = natten::cuda::fna::map_index_to_coord(
              row_offset + thread_start_row_, extent_row_);
          // bool row_guard =
          // natten::cuda::fna::is_coord_within_upper_bound(row_offset_full_,
          // extent_row_);
          bool row_guard = row_offset + thread_start_row_ < extent_row_int;

          AccessType* memory_pointer = reinterpret_cast<AccessType*>(
              byte_pointer + byte_offset +
              (row_offset_full_ * params_.stride).sum());

          // if (ScatterD && row_guard) {
          //  assert(indices_);

          //  memory_pointer = reinterpret_cast<AccessType *>(byte_pointer +
          //  byte_offset +
          //    LongIndex(indices_[row_offset + thread_start_row_]) *
          //    LongIndex(params_.stride));
          //}

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn;
               ++column) {
            bool guard = row_guard && mask_.predicates[column];

            // if (PermuteD) {

            //   int col_offset = column * ThreadMap::Delta::kColumn;

            //   int col = col_offset + thread_start_column_;
            //   int row = row_offset + thread_start_row_;

            //   // Locate memory_pointer
            //   memory_pointer = reinterpret_cast<AccessType *>(byte_pointer +
            //   byte_offset
            //      + permute_layout_(PitchLinearCoord(col, row)) *
            //      sizeof(AccessType) / kElementsPerAccess);
            // }

            // if (UseCUDAStore) {
            //  if (guard) {
            //    memory_pointer[0] =
            //        frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn +
            //        column];
            //  }
            //} else {
            cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                frag_ptr
                    [frag_row_idx * ThreadMap::Iterations::kColumn + column],
                (void*)&memory_pointer[0],
                guard);
            //}

            // if (!PermuteD) {
            memory_pointer += (ThreadMap::Delta::kColumn / kElementsPerAccess);
            //}
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            // if (!ScatterD && !PermuteD) {
            // byte_pointer += params_.increment_row;
            //}
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          // byte_pointer += params_.increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        // byte_pointer += params_.increment_cluster;
      }
    }
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const& frag) const {
    store_with_byte_offset(frag, 0);
  }

  CUTLASS_DEVICE
  MatrixCoord thread_start() const {
    return MatrixCoord(thread_start_row_, thread_start_column_);
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

  /// Extent of the matrix in columns
  CUTLASS_DEVICE
  Index extent_column() const {
    return extent_column_;
  }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileIterator& operator++() {
    ++state_[0];

    // if (!ScatterD && !PermuteD) {
    //  store_byte_pointer_ += params_.advance_row;
    //}

    // if (!ScatterD) {
    //  byte_pointer_ += params_.advance_row;
    //}

    thread_start_row_ += ThreadMap::Shape::kRow;

    if (state_[0] == ThreadMap::Count::kRow) {
      state_[0] = 0;
      ++state_[1];
      // byte_pointer_ += params_.advance_group;
      // store_byte_pointer_ += params_.advance_group;

      thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
          ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {
        state_[1] = 0;
        ++state_[2];
        // byte_pointer_ += params_.advance_cluster;
        // store_byte_pointer_ += params_.advance_cluster;

        thread_start_row_ += ThreadMap::Count::kGroup *
            ThreadMap::Shape::kGroup * ThreadMap::Count::kRow *
            ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;
          // byte_pointer_ += params_.advance_tile;
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
  CustomPredicatedTileIterator& operator+=(int increment) {
    // Row
    state_[0] += increment;
    int increment_row = state_[0] / ThreadMap::Count::kRow;
    state_[0] = state_[0] % ThreadMap::Count::kRow;

    // byte_pointer_ += (params_.advance_row * increment);
    // store_byte_pointer_ += (params_.advance_row * increment);
    thread_start_row_ += (ThreadMap::Shape::kRow * increment);

    // Group
    state_[1] += increment_row;
    int increment_group = state_[1] / ThreadMap::Count::kGroup;
    state_[1] = state_[1] % ThreadMap::Count::kGroup;

    // byte_pointer_ += (params_.advance_group * increment_row);
    // store_byte_pointer_ += (params_.advance_group * increment_row);
    thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
        ThreadMap::Shape::kRow * ThreadMap::Count::kRow * increment_row;

    // Cluster
    state_[2] += increment_group;
    int increment_cluster = state_[2] / ThreadMap::Count::kCluster;
    state_[2] = state_[2] % ThreadMap::Count::kCluster;

    // byte_pointer_ += (params_.advance_cluster * increment_group);
    // store_byte_pointer_ += (params_.advance_cluster * increment_group);
    thread_start_row_ += ThreadMap::Count::kGroup * ThreadMap::Shape::kGroup *
        ThreadMap::Count::kRow * ThreadMap::Shape::kRow * increment_group;

    // Tile
    // byte_pointer_ += (params_.advance_tile * increment_cluster);
    // store_byte_pointer_ += (params_.advance_tile * increment_cluster);
    thread_start_row_ += ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow *
        ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile *
        increment_cluster;

    return *this;
  }

  ///< Efficiently disables all accesses guarded by mask
  CUTLASS_DEVICE void clear_mask() {
    mask_.clear();
  }

  ///< Efficiently enables all accesses guarded by mask
  CUTLASS_DEVICE void enable_mask() {
    mask_.enable();
  }

  ///< Sets the mask
  CUTLASS_DEVICE void get_mask(Mask& mask) const {
    mask = mask_;
  }

  ///< Sets the mask
  CUTLASS_DEVICE void set_mask(Mask const& mask) {
    mask_ = mask;
  }
};

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
