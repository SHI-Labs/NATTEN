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
    \brief Templates calculating the address and predicates to the load of tiles
    from pitch-linear rank=2 tensors.

    This iterator uses masks to guard out-of-bounds accesses. The first tile
   this iterator visits maybe partial, then the remaining tiles are complete.
   So, we only need to compute the predicates twice, once before the first tile
   and once for the remaining full tiles which can share the same predicates.

    A precomputed "Params" object minimizes the amount of state that must be
    stored in registers, and integer addition is used to advance the pointer
    through memory.
*/

#pragma once

#include <cutlass/array.h>
#include <cutlass/coord.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/pitch_linear.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/predicate_vector.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/tensor_view.h>
#include <cutlass/transform/threadblock/predicated_tile_access_iterator_params.h>

#include <natten/cuda/fna/na_utils.cuh>

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

template <int NADim, typename Shape>
struct CustomPredicatedTileAccessIteratorParams {
  using Dim = typename natten::cuda::fna::GetDim<NADim>::type;

  using Index = int32_t;
  using LongIndex =
      typename std::conditional<NADim == 1, int64_t, int32_t>::type;

  //
  // Data members
  //
  /// stride of pitch-linear layout (units of Element)
  Dim stride_;
  /// amount (in byte) to increment pointer to move to next access along
  /// strided dimension
  LongIndex inc_strided_;
  /// amount (in byte) to increment pointer from last access to first access
  /// of next tile
  LongIndex inc_next_;
  /// amount (in byte) to increment pointer from first access of current tile
  /// to first access of next tile
  LongIndex inc_advance_;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  Status initialize(
      Dim stride,
      Dim extent_row,
      PredicatedTileAccessIteratorDesc desc) {
    stride_ = stride;

    if constexpr (NADim == 1) {
      inc_strided_ = (LongIndex(stride_.x) * desc.threadmap_delta.strided()) *
          desc.element_size_bits / 8;

      if (desc.advance_rank) {
        // advance along strided dimension
        inc_advance_ = desc.threadblock_shape.strided() * LongIndex(stride_.x) *
            desc.element_size_bits / 8;
      } else {
        // advance along contiguous dimension
        inc_advance_ =
            desc.threadblock_shape.contiguous() * desc.element_size_bits / 8;
      }

      inc_next_ = inc_advance_ -
          LongIndex(desc.threadmap_iterations.strided() - 1) *
              desc.threadmap_delta.strided() * LongIndex(stride_.x) *
              desc.element_size_bits / 8;
      // inc_strided_ =
      // (natten::cuda::fna::map_index_to_coord(desc.threadmap_delta.strided(),
      // extent_row) * stride_).sum() *
      //                 desc.element_size_bits / 8;
      //// inc_strided_ = (LongIndex(stride_) * desc.threadmap_delta.strided())
      ///* /                  desc.element_size_bits / 8;

      // if (desc.advance_rank) {
      //  // advance along strided dimension
      //  inc_advance_ = (natten::cuda::fna::map_index_to_coord(
      //      desc.threadblock_shape.strided(), extent_row) * stride_).sum() *
      //      desc.element_size_bits / 8;
      //  // inc_advance_ =
      //  //     desc.threadblock_shape.strided() * LongIndex(stride_) *
      //  desc.element_size_bits / 8;
      //} else {
      //  // advance along contiguous dimension
      //  inc_advance_ = desc.threadblock_shape.contiguous() *
      //  desc.element_size_bits / 8;
      //}

      // inc_next_ = inc_advance_ -
      // (natten::cuda::fna::map_index_to_coord((desc.threadmap_iterations.strided()
      // - 1) *
      //                               desc.threadmap_delta.strided(),
      //                               extent_row) * stride_).sum() *
      //                               desc.element_size_bits / 8;
      //// inc_next_ = inc_advance_ -
      /// LongIndex(desc.threadmap_iterations.strided() - 1) * /
      /// desc.threadmap_delta.strided() * LongIndex(stride_) * /
      /// desc.element_size_bits / 8;
    } else {
      // inc_strided_ = LongIndex(desc.threadmap_delta.strided());

      // if (desc.advance_rank) {
      //   // advance along strided dimension
      //   inc_advance_ = LongIndex(desc.threadblock_shape.strided());
      // } else {
      //   // advance along contiguous dimension
      //   inc_advance_ = LongIndex(desc.threadblock_shape.contiguous());
      // }

      // inc_next_ = inc_advance_ -
      // LongIndex(desc.threadmap_iterations.strided() - 1) *
      // desc.threadmap_delta.strided();
    }

    return Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIteratorParams(
      Dim stride,
      Dim extent_row,
      PredicatedTileAccessIteratorDesc desc) {
    initialize(stride, extent_row, desc);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// CustomPredicatedTileAccessIteratorPredicates
///
template <
    int NADim,
    typename Shape_,
    typename Element_,
    typename Layout_,
    int AdvanceRank,
    typename ThreadMap_,
    typename AccessType_>
class CustomPredicatedTileAccessIteratorPredicates {
 public:
  static_assert(NADim >= 1 && NADim < 4);
  using Dim = typename natten::cuda::fna::GetDim<NADim>::type;

  using Shape = Shape_;
  using Element = Element_;
  using Layout = Layout_;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorCoord = typename Layout::TensorCoord;

  static int const kAccessesPerVector =
      ThreadMap::kElementsPerAccess / AccessType::kElements;

  static_assert(
      !(ThreadMap::kElementsPerAccess % AccessType::kElements),
      "Vectors implied by the thread map must be divisible by the access type.");

  static int const kPredicatesPerByte = 4;
  static int const kPredicatesPerWord = 4 * kPredicatesPerByte;

  static int const kPredicateCount =
      ThreadMap::Iterations::kCount * kAccessesPerVector;

  /// Number of 32b words containing predicates
  static int const kPredicateByteCount =
      (kPredicateCount + kPredicatesPerByte - 1) / kPredicatesPerByte;
  static int const kPredicateWordCount = (kPredicateByteCount + 3) / 4;

  static unsigned const kPredicateMask = (1u << kPredicatesPerByte) - 1u;

  static_assert(kPredicateWordCount <= 4, "Too many predicates.");

  /// Predicate vector stores mask to guard accesses
  using Mask = Array<uint32_t, kPredicateWordCount>;

  // private:
  /// Guard predicates
  uint32_t predicates_[kPredicateWordCount];

  /// Size of tensor
  Dim extent_row_dim;
  TensorCoord extent_;

  /// Initial offset for each thread
  TensorCoord thread_offset_;

  /// Offset to the first steady-state tile
  TensorCoord residue_offset_;

  /// Iteration along vectors implied by the thread map
  int iteration_vector_;

  /// Iteration in the contiguous dimension
  int iteration_contiguous_;

  /// Iteration in the strided dimension
  int iteration_strided_;

 public:
  /// Computes predicates based on internally tracked per-thread offset.
  CUTLASS_DEVICE
  void compute_predicates_(
      /// Extent of the matrix window
      TensorCoord extent,
      /// optionally, simplify predicate calculation during 'steady state' phase
      bool is_steady_state = false) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      predicates_[i] = 0u;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int access_idx = 0;
         access_idx < ThreadMap::Iterations::kCount * kAccessesPerVector;
         ++access_idx) {
      int s = access_idx /
          (ThreadMap::Iterations::kContiguous * kAccessesPerVector);

      int access_residual = access_idx %
          (ThreadMap::Iterations::kContiguous * kAccessesPerVector);

      int c = access_residual / kAccessesPerVector;
      int v = access_residual % kAccessesPerVector;

      TensorCoord iteration_coord(
          c * ThreadMap::Delta::kContiguous + v * AccessType::kElements,
          s * ThreadMap::Delta::kStrided);

      TensorCoord coord = thread_offset_ + iteration_coord;

      bool guard;

      // Dim coord_row;
      // if constexpr (kAdvanceRank == 0) {
      //  coord_row = natten::cuda::fna::map_index_to_coord(coord.strided(),
      //  extent_.strided());
      //} else {
      //  coord_row = natten::cuda::fna::map_index_to_coord(coord.contiguous(),
      //  natten::cuda::fna::MapTileSizeToNd<Shape::kContiguous, NADim>::value);
      //}

      // if (is_steady_state) {
      //   guard = coord.strided() < extent.strided();
      // } else {
      ////if constexpr (kAdvanceRank == 0) {
      //  guard = (coord.strided() < extent.strided() &&
      //           coord.contiguous() < extent.contiguous());
      ////} else {
      ////  guard = (coord.strided() < extent_col &&
      ////           coord_row < extent_row);
      ////}
      //}
      if (is_steady_state) {
        if (kAdvanceRank == 0) {
          guard = (coord.strided() < extent.strided());
        } else {
          guard = (coord.contiguous() < extent.contiguous());
        }
      } else {
        guard =
            (coord.strided() < extent.strided() &&
             coord.contiguous() < extent.contiguous());
      }

      int pred_idx =
          v + kAccessesPerVector * (c + ThreadMap::Iterations::kContiguous * s);

      int word_idx = pred_idx / kPredicatesPerWord;
      int residual = pred_idx % kPredicatesPerWord;
      int byte_idx = residual / kPredicatesPerByte;
      int bit_idx = residual % kPredicatesPerByte;

      predicates_[word_idx] |= (unsigned(guard) << (byte_idx * 8 + bit_idx));
    }
  }

  CUTLASS_HOST_DEVICE
  void set_predicates(int thread_id, TensorCoord const& threadblock_offset) {
    TensorCoord residue_extent;
    // int32_t residue_extent_row = 0;
    // int32_t residue_extent_col = 0;
    // if constexpr (kAdvanceRank == 0) {
    //  // Column-major right handside operand (only instance is K in QK^T).
    //  // extent_[0] == K (dim per head) == extent_.contiguous()
    //  // extent_.contiguous() == extent_[0] == extent_.contiguous()
    //  //
    //  // Row-major left handside operand (only instance is Q in QK^T)
    //  // extent_[0] == K (dim per head) == extent_.contiguous()
    //  // extent_.contiguous() == extent_[0] == extent_.contiguous()
    //  typename TensorCoord::Index residue_size = (extent_.contiguous() -
    //  threadblock_offset.contiguous()) % Shape::kContiguous; if
    //  (!residue_size) {
    //    residue_size = Shape::kContiguous;
    //  }
    //  residue_offset_ = make_Coord(residue_size, 0);
    //
    //  residue_extent.strided() = extent_.strided();
    //  residue_extent.contiguous() = min(extent_.contiguous(),
    //  threadblock_offset.contiguous() + residue_size);
    //} else {
    //  // Row-major right-handside operand (only instance is V in PV)
    //  // extent_[1] == M == extent_.strided()
    //  // extent_.contiguous() == extent_[0] == extent_.contiguous()
    //  // extent_.strided() == extent_[1] == extent_.strided()
    //  //
    //  typename TensorCoord::Index residue_size = (extent_.strided() -
    //  threadblock_offset.strided()) % Shape::kStrided; if (!residue_size) {
    //    residue_size = Shape::kStrided;
    //  }
    //  residue_offset_ = make_Coord(0, residue_size);

    //  residue_extent.strided() = min(threadblock_offset.strided() +
    //  residue_size, extent_.strided()); residue_extent.contiguous() =
    //  extent_.contiguous();
    //}
    if (kAdvanceRank) {
      typename TensorCoord::Index residue_size =
          (extent_[kAdvanceRank] - threadblock_offset.strided()) %
          Shape::kStrided;
      if (!residue_size) {
        residue_size = Shape::kStrided;
      }

      residue_offset_ = make_Coord(0, residue_size);
      residue_extent = make_Coord(
          extent_.contiguous(),
          min(threadblock_offset.strided() + residue_size, extent_.strided()));
    } else {
      typename TensorCoord::Index residue_size =
          (extent_[kAdvanceRank] - threadblock_offset.contiguous()) %
          Shape::kContiguous;
      if (!residue_size) {
        residue_size = Shape::kContiguous;
      }

      residue_offset_ = make_Coord(residue_size, 0);

      residue_extent = make_Coord(
          min(extent_.contiguous(),
              threadblock_offset.contiguous() + residue_size),
          extent_.strided());
    }

    // Per-thread offset in logical coordinates of tensor
    thread_offset_ = threadblock_offset + ThreadMap::initial_offset(thread_id);

    compute_predicates_(residue_extent, false);

    set_iteration_index(0);
  }

  /// Default constructor
  CustomPredicatedTileAccessIteratorPredicates() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIteratorPredicates(
      /// Extent of tensor (rows)
      Dim extent_row,
      /// Extent of tensor (columns)
      int32_t extent_col)
      : extent_row_dim(extent_row), extent_(extent_col, extent_row.prod32()) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iteration_vector_ = index % kAccessesPerVector;
    int residual_access = index / kAccessesPerVector;

    iteration_contiguous_ =
        residual_access % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = residual_access / ThreadMap::Iterations::kContiguous;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIteratorPredicates& operator++() {
    return *this;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      predicates_[i] = enable ? 0u : predicates_[i];
    }
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      predicates_[i] = 0xffffffff;
    }
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      predicates_[i] = mask[i];
    }
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPredicateWordCount; ++i) {
      mask[i] = predicates_[i];
    }
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() const {
    int pred_idx = iteration_vector_ +
        kAccessesPerVector *
            (iteration_contiguous_ +
             iteration_strided_ * ThreadMap::Iterations::kContiguous);

    int word_idx = pred_idx / kPredicatesPerWord;
    int residual = pred_idx % kPredicatesPerWord;
    int byte_idx = residual / kPredicatesPerByte;
    int bit_idx = residual % kPredicatesPerByte;

    bool pred = (predicates_[word_idx] & (1u << (byte_idx * 8 + bit_idx))) != 0;
    return pred;
  }
};

////////////////////////////////////////////////////////////////////////////////

/// CustomPredicatedTileAccessIterator
///
template <
    int NADim,
    typename Shape,
    typename Element,
    typename Layout,
    int AdvanceRank,
    typename ThreadMap,
    typename AccessType>
class CustomPredicatedTileAccessIterator;

////////////////////////////////////////////////////////////////////////////////

/// CustomPredicatedTileAccessIterator for possibly multiple axes (NA1D, 2D, 3D)
template <
    int NADim,
    typename Shape_,
    typename Element_,
    int AdvanceRank,
    typename ThreadMap_,
    typename AccessType_>
class CustomPredicatedTileAccessIterator<
    NADim,
    Shape_,
    Element_,
    layout::PitchLinear,
    AdvanceRank,
    ThreadMap_,
    AccessType_> {
 public:
  static_assert(NADim >= 1 && NADim < 4);
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Dim = typename natten::cuda::fna::GetDim<NADim>::type;

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename platform::remove_const<Element>::type*;

  using UnderlyingPredicates = CustomPredicatedTileAccessIteratorPredicates<
      NADim,
      Shape,
      Element,
      Layout,
      AdvanceRank,
      ThreadMap,
      AccessType>;

  static int const kAccessesPerVector =
      ThreadMap::kElementsPerAccess / AccessType::kElements;

  static_assert(
      !(ThreadMap::kElementsPerAccess % AccessType::kElements),
      "Vectors implied by the thread map must be divisible by the access type.");

  using Mask = typename UnderlyingPredicates::Mask;

  /// Uses a non-template class
  struct Params : CustomPredicatedTileAccessIteratorParams<NADim, Shape> {
    using Base = CustomPredicatedTileAccessIteratorParams<NADim, Shape>;

    /// Default constructor
    Params() = default;

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Dim stride, Dim extent_row)
        : Base(
              stride,
              extent_row,
              MakePredicatedTileAccessIteratorDesc<
                  Shape,
                  Element,
                  Layout,
                  kAdvanceRank,
                  ThreadMap>()()) {}

    CUTLASS_HOST_DEVICE
    Params(Base const& base) : Base(base) {}
  };

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char*;

 private:
  //
  // Data members
  //

  UnderlyingPredicates the_predicates;

  /// Parameters object with precomputed internal state
  Params params_;

  /// Internal pointer to first access of tile
  BytePointer pointer_;
  TensorCoord coord_offset_;

  /// Used for out-of-order visitation
  bool is_residue_tile_;

 private:
  /// Computes predicates based on internally tracked per-thread offset.
  CUTLASS_DEVICE
  void compute_predicates_(
      /// Extent of the matrix window
      TensorCoord extent,
      /// optionally, simplify predicate calculation during 'steady state' phase
      bool is_steady_state = false) {
    the_predicates.compute_predicates_(extent, is_steady_state);
  }

 public:
  /// Default constructor
  CustomPredicatedTileAccessIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIterator(
      /// Precomputed parameters object
      Params const& params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor (rows)
      Dim extent_row,
      /// Extent of tensor (columns)
      int32_t extent_col,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const& threadblock_offset)
      : params_(params),
        pointer_(reinterpret_cast<BytePointer>(
            const_cast<NonConstPointer>(pointer))),
        the_predicates(extent_row, extent_col),
        is_residue_tile_(true) {
    the_predicates.set_predicates(thread_id, threadblock_offset);

    // update internal pointers
    coord_offset_ = the_predicates.thread_offset_;
    // add_pointer_offset(the_predicates.thread_offset_.strided(),
    // the_predicates.thread_offset_.contiguous());
  }

  /// Construct a CustomPredicatedTileAccessIterator with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIterator(
      /// Precomputed parameters object
      Params const& params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor (rows)
      Dim extent_row,
      /// Extent of tensor (columns)
      int32_t extent_col,
      ///< ID of each participating thread
      int thread_id)
      : CustomPredicatedTileAccessIterator(
            params,
            pointer,
            extent_row,
            extent_col,
            thread_id,
            make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    the_predicates.set_iteration_index(index);
  }

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(int32_t row_offset, int32_t col_offset) {
    coord_offset_.strided() += row_offset;
    coord_offset_.contiguous() += col_offset;
  }

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex linear_offset) {
    printf("FATAL!");
    // if constexpr (kAdvanceRank) {
    //   auto col_offset = linear_offset % Shape::kStrided;
    //   auto row_offset = linear_offset / Shape::kStrided;
    //   add_pointer_offset(row_offset, (int32_t)col_offset);
    // } else {
    //   auto col_offset = linear_offset % Shape::kContiguous;
    //   auto row_offset = linear_offset / Shape::kContiguous;
    //   add_pointer_offset(row_offset, (int32_t)col_offset);
    // }
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    if (is_residue_tile_) {
      the_predicates.thread_offset_ += the_predicates.residue_offset_;

      the_predicates.compute_predicates_(the_predicates.extent_, true);

      // add_pointer_offset(the_predicates.residue_offset_.strided(),
      // the_predicates.residue_offset_.contiguous());

      coord_offset_.strided() = the_predicates.thread_offset_.strided() +
          Shape::kStrided * (tile_offset.strided() - kAdvanceRank);
      coord_offset_.contiguous() = the_predicates.thread_offset_.contiguous() +
          Shape::kContiguous * (tile_offset.contiguous() - (1 - kAdvanceRank));

      // if (kAdvanceRank) {
      //   add_pointer_offset(
      //       params_.inc_advance_ * (tile_offset.strided() - 1) +
      //       Shape::kContiguous * tile_offset.contiguous());
      //   // pointer_ += params_.inc_advance_ * LongIndex(tile_offset.strided()
      //   - 1);
      //   // pointer_ += Shape::kContiguous * tile_offset.contiguous() *
      //   sizeof_bits<Element>::value / 8;
      // } else {
      //   // Advance along contiguous dimension
      //   add_pointer_offset(
      //       params_.inc_advance_ * (tile_offset.contiguous() - 1) +
      //       Shape::kStrided * tile_offset.strided());
      //   // pointer_ += params_.inc_advance_ *
      //   LongIndex(tile_offset.contiguous() - 1);
      //   // pointer_ += Shape::kStrided * tile_offset.strided() *
      //   sizeof_bits<Element>::value / 8;
      // }
    } else {
      add_pointer_offset(
          Shape::kStrided * tile_offset.strided(),
          Shape::kContiguous * tile_offset.contiguous());
      // if (kAdvanceRank) {
      //   add_pointer_offset(
      //       params_.inc_advance_ * tile_offset.strided() +
      //       Shape::kContiguous * tile_offset.contiguous());
      //   // pointer_ += params_.inc_advance_ *
      //   LongIndex(tile_offset.strided());
      //   // pointer_ += Shape::kContiguous * tile_offset.contiguous();
      // } else {
      //   // Advance along contiguous dimension
      //   add_pointer_offset(
      //       params_.inc_advance_ * tile_offset.contiguous() +
      //       Shape::kStrided * tile_offset.strided());
      //   // pointer_ += params_.inc_advance_ *
      //   LongIndex(tile_offset.contiguous());
      //   // pointer_ += Shape::kStrided * tile_offset.strided();
      // }
    }

    is_residue_tile_ = false;
  }

  CUTLASS_HOST_DEVICE
  LongIndex get_pointer_offset() const {
    LongIndex coord_contig = coord_offset_.contiguous() +
        the_predicates.iteration_contiguous_ * ThreadMap::Delta::kContiguous;
    LongIndex coord_strided =
        (natten::cuda::fna::map_index_to_coord(
             coord_offset_.strided() +
                 the_predicates.iteration_strided_ * ThreadMap::Delta::kStrided,
             the_predicates.extent_row_dim) *
         params_.stride_)
            .sum();
    return ((coord_strided + coord_contig) * sizeof_bits<Element>::value) / 8;
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    if (!valid()) {
      return nullptr;
    }
    return reinterpret_cast<AccessType*>(pointer_ + get_pointer_offset()) +
        the_predicates.iteration_vector_;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIterator& operator++() {
    the_predicates.operator++();

    ++the_predicates.iteration_vector_;
    if (the_predicates.iteration_vector_ < kAccessesPerVector) {
      return *this;
    }

    the_predicates.iteration_vector_ = 0;
    ++the_predicates.iteration_contiguous_;

    if (the_predicates.iteration_contiguous_ <
        ThreadMap::Iterations::kContiguous) {
      return *this;
    }

    // Enter here only if (iteration_contiguous_ ==
    // ThreadMap::Iteration::kContiguous)
    the_predicates.iteration_contiguous_ = 0;
    ++the_predicates.iteration_strided_;

    if (the_predicates.iteration_strided_ < ThreadMap::Iterations::kStrided) {
      // add_pointer_offset(params_.inc_strided_, 0);
      // pointer_ += params_.inc_strided_;

      return *this;
    }

    // Enter here only if (iteration_stride_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next tile.
    the_predicates.iteration_strided_ = 0;

    // // advance to next tile
    // add_pointer_offset(params_.inc_next_, 0);
    // // pointer_ += params_.inc_next_;

    // //if constexpr (kAdvanceRank) {
    // add_pointer_offset(-1 * params_.inc_advance_);
    // //} else {
    // //  // along the contiguous axis.
    // //  add_pointer_offset(0, -1 * params_.inc_advance_);
    // //}
    // // now return to start tile - if the iterator is subsequently advanced,
    // this
    // // subtraction as well as the subsequent integer addition are both elided
    // by
    // // the compiler.
    // // pointer_ -= params_.inc_advance_;

    return *this;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIterator operator++(int) {
    CustomPredicatedTileAccessIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    the_predicates.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    the_predicates.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    the_predicates.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    the_predicates.get_mask(mask);
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() const {
    return the_predicates.valid();
  }
};

// Specialization for NA1D, where we only have one stride,
// and therefore the typical pitch-linear layout, and won't have
// to manually convert offsets on every get().
template <
    typename Shape_,
    typename Element_,
    int AdvanceRank,
    typename ThreadMap_,
    typename AccessType_>
class CustomPredicatedTileAccessIterator<
    1,
    Shape_,
    Element_,
    layout::PitchLinear,
    AdvanceRank,
    ThreadMap_,
    AccessType_> {
 public:
  static constexpr int NADim = 1;
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Dim = typename natten::cuda::fna::GetDim<NADim>::type;

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename platform::remove_const<Element>::type*;

  using UnderlyingPredicates = CustomPredicatedTileAccessIteratorPredicates<
      NADim,
      Shape,
      Element,
      Layout,
      AdvanceRank,
      ThreadMap,
      AccessType>;

  static int const kAccessesPerVector =
      ThreadMap::kElementsPerAccess / AccessType::kElements;

  static_assert(
      !(ThreadMap::kElementsPerAccess % AccessType::kElements),
      "Vectors implied by the thread map must be divisible by the access type.");

  using Mask = typename UnderlyingPredicates::Mask;

  /// Uses a non-template class
  struct Params : CustomPredicatedTileAccessIteratorParams<NADim, Shape> {
    using Base = CustomPredicatedTileAccessIteratorParams<NADim, Shape>;

    /// Default constructor
    Params() = default;

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Dim stride, Dim extent_row)
        : Base(
              stride,
              extent_row,
              MakePredicatedTileAccessIteratorDesc<
                  Shape,
                  Element,
                  Layout,
                  kAdvanceRank,
                  ThreadMap>()()) {}

    CUTLASS_HOST_DEVICE
    Params(Base const& base) : Base(base) {}
  };

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char*;

 private:
  //
  // Data members
  //

  UnderlyingPredicates the_predicates;

  /// Parameters object with precomputed internal state
  Params params_;

  /// Internal pointer to first access of tile
  BytePointer pointer_;

  /// Used for out-of-order visitation
  bool is_residue_tile_;

 private:
  /// Computes predicates based on internally tracked per-thread offset.
  CUTLASS_DEVICE
  void compute_predicates_(
      /// Extent of the matrix window
      Dim extent_row,
      int32_t extent_col,
      /// optionally, simplify predicate calculation during 'steady state' phase
      bool is_steady_state = false) {
    the_predicates.compute_predicates_(extent_row, extent_col, is_steady_state);
  }

 public:
  /// Default constructor
  CustomPredicatedTileAccessIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIterator(
      /// Precomputed parameters object
      Params const& params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor (rows)
      Dim extent_row,
      /// Extent of tensor (columns)
      int32_t extent_col,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const& threadblock_offset)
      : params_(params),
        pointer_(reinterpret_cast<BytePointer>(
            const_cast<NonConstPointer>(pointer))),
        the_predicates(extent_row, extent_col),
        is_residue_tile_(true) {
    the_predicates.set_predicates(thread_id, threadblock_offset);

    // update internal pointers
    auto offset = (natten::cuda::fna::map_index_to_coord(
                       the_predicates.thread_offset_.strided(),
                       the_predicates.extent_row_dim) *
                   params_.stride_)
                      .sum() +
        the_predicates.thread_offset_.contiguous();

    add_pointer_offset(offset);
  }

  /// Construct a CustomPredicatedTileAccessIterator with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIterator(
      /// Precomputed parameters object
      Params const& params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor (rows)
      Dim extent_row,
      /// Extent of tensor (columns)
      int32_t extent_col,
      ///< ID of each participating thread
      int thread_id)
      : CustomPredicatedTileAccessIterator(
            params,
            pointer,
            extent_row,
            extent_col,
            thread_id,
            make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    the_predicates.set_iteration_index(index);
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    pointer_ += sizeof_bits<Element>::value * pointer_offset / 8;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    if (is_residue_tile_) {
      the_predicates.thread_offset_ += the_predicates.residue_offset_;

      the_predicates.compute_predicates_(the_predicates.extent_, true);

      auto offset = (natten::cuda::fna::map_index_to_coord(
                         the_predicates.residue_offset_.strided(),
                         the_predicates.extent_row_dim) *
                     params_.stride_)
                        .sum() +
          the_predicates.residue_offset_.contiguous();

      add_pointer_offset(offset);

      if (kAdvanceRank) {
        pointer_ += params_.inc_advance_ * LongIndex(tile_offset.strided() - 1);
        pointer_ += Shape::kContiguous * tile_offset.contiguous() *
            sizeof_bits<Element>::value / 8;
      } else {
        pointer_ +=
            params_.inc_advance_ * LongIndex(tile_offset.contiguous() - 1);
        pointer_ += Shape::kStrided * tile_offset.strided() *
            sizeof_bits<Element>::value / 8;
      }
    } else {
      if (kAdvanceRank) {
        pointer_ += params_.inc_advance_ * LongIndex(tile_offset.strided());
        pointer_ += Shape::kContiguous * tile_offset.contiguous();
      } else {
        pointer_ += params_.inc_advance_ * LongIndex(tile_offset.contiguous());
        pointer_ += Shape::kStrided * tile_offset.strided();
      }
    }

    is_residue_tile_ = false;
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    return reinterpret_cast<AccessType*>(
               pointer_ +
               the_predicates.iteration_contiguous_ *
                   (ThreadMap::Delta::kContiguous *
                    sizeof_bits<Element>::value) /
                   8) +
        the_predicates.iteration_vector_;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIterator& operator++() {
    the_predicates.operator++();

    ++the_predicates.iteration_vector_;
    if (the_predicates.iteration_vector_ < kAccessesPerVector) {
      return *this;
    }

    the_predicates.iteration_vector_ = 0;
    ++the_predicates.iteration_contiguous_;

    if (the_predicates.iteration_contiguous_ <
        ThreadMap::Iterations::kContiguous) {
      return *this;
    }

    // Enter here only if (iteration_contiguous_ ==
    // ThreadMap::Iteration::kContiguous)
    the_predicates.iteration_contiguous_ = 0;
    ++the_predicates.iteration_strided_;

    if (the_predicates.iteration_strided_ < ThreadMap::Iterations::kStrided) {
      pointer_ += params_.inc_strided_;

      return *this;
    }

    // Enter here only if (iteration_stride_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next tile.
    the_predicates.iteration_strided_ = 0;

    // advance to next tile
    pointer_ += params_.inc_next_;

    // now return to start tile - if the iterator is subsequently advanced, this
    // subtraction as well as the subsequent integer addition are both elided by
    // the compiler.
    pointer_ -= params_.inc_advance_;

    return *this;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIterator operator++(int) {
    CustomPredicatedTileAccessIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    the_predicates.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    the_predicates.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    the_predicates.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    the_predicates.get_mask(mask);
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() const {
    return the_predicates.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of CustomPredicatedTileAccessIterator for column-major data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
    int NADim,
    typename Shape_,
    typename Element_,
    int AdvanceRank,
    typename ThreadMap_,
    typename AccessType_>
class CustomPredicatedTileAccessIterator<
    NADim,
    Shape_,
    Element_,
    layout::ColumnMajor,
    AdvanceRank,
    ThreadMap_,
    AccessType_> {
 public:
  static_assert(NADim >= 1 && NADim < 4);
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Dim = typename natten::cuda::fna::GetDim<NADim>::type;

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename platform::remove_const<Element>::type*;

  using UnderlyingIterator = CustomPredicatedTileAccessIterator<
      NADim,
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
      Element,
      layout::PitchLinear,
      (kAdvanceRank == 0 ? 0 : 1),
      ThreadMap,
      AccessType>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend CustomPredicatedTileAccessIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    /// Default constructor
    Params() = default;

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Dim stride, Dim extent_row) : params_(stride, extent_row){};

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params::Base const& base)
        : params_(base) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Default constructor
  CustomPredicatedTileAccessIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIterator(
      ///< Precomputed parameters object
      Params const& params,
      ///< Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor (rows)
      Dim extent_row,
      /// Extent of tensor (columns)
      int32_t extent_col,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const& threadblock_offset)
      : iterator_(
            params.params_,
            pointer,
            extent_row,
            extent_col,
            thread_id,
            layout::PitchLinearCoord(
                threadblock_offset.row(),
                threadblock_offset.column())) {}

  /// Construct a CustomPredicatedTileAccessIterator with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIterator(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      Dim extent_row, ///< Extent of tensor (rows)
      int32_t extent_col, ///< Extent of tensor (columns)
      int thread_id ///< ID of each participating thread
      )
      : CustomPredicatedTileAccessIterator(
            params,
            pointer,
            extent_row,
            extent_col,
            thread_id,
            make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iterator_.set_iteration_index(index);
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    return reinterpret_cast<AccessType*>(iterator_.get());
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIterator& operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIterator operator++(int) {
    CustomPredicatedTileAccessIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    iterator_.get_mask(mask);
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return iterator_.valid();
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of CustomPredicatedTileAccessIterator for row-major data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <
    int NADim,
    typename Shape_,
    typename Element_,
    int AdvanceRank,
    typename ThreadMap_,
    typename AccessType_>
class CustomPredicatedTileAccessIterator<
    NADim,
    Shape_,
    Element_,
    layout::RowMajor,
    AdvanceRank,
    ThreadMap_,
    AccessType_> {
 public:
  static_assert(NADim >= 1 && NADim < 4);
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Dim = typename natten::cuda::fna::GetDim<NADim>::type;

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajor;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename platform::remove_const<Element>::type*;

  using UnderlyingIterator = CustomPredicatedTileAccessIterator<
      NADim,
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
      Element,
      layout::PitchLinear,
      (kAdvanceRank == 0 ? 1 : 0),
      ThreadMap,
      AccessType>;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend CustomPredicatedTileAccessIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    /// Default constructor
    Params() = default;

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Dim stride, Dim extent_row) : params_(stride, extent_row){};

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params::Base const& base)
        : params_(base) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;

 public:
  /// Default constructor
  CustomPredicatedTileAccessIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIterator(
      ///< Precomputed parameters object
      Params const& params,
      ///< Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor (rows)
      Dim extent_row,
      /// Extent of tensor (columns)
      int32_t extent_col,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const& threadblock_offset)
      : iterator_(
            params.params_,
            pointer,
            extent_row,
            extent_col,
            thread_id,
            layout::PitchLinearCoord(
                threadblock_offset.column(),
                threadblock_offset.row())) {}

  /// Construct a CustomPredicatedTileAccessIterator with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIterator(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      Dim extent_row, ///< Extent of tensor (rows)
      int32_t extent_col, ///< Extent of tensor (columns)
      int thread_id ///< ID of each participating thread
      )
      : CustomPredicatedTileAccessIterator(
            params,
            pointer,
            extent_row,
            extent_col,
            thread_id,
            make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iterator_.set_iteration_index(index);
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType* get() const {
    return reinterpret_cast<AccessType*>(iterator_.get());
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIterator& operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileAccessIterator operator++(int) {
    CustomPredicatedTileAccessIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    iterator_.get_mask(mask);
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid() {
    return iterator_.valid();
  }
};

} // namespace threadblock
} // namespace transform
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
