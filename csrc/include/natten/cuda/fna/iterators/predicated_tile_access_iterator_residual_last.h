/*
 * Copied from xFormers (https://github.com/facebookresearch/xformers/) and
 * edited.
 *
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 */
/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <natten/cuda/fna/iterators/predicated_tile_access_iterator.h>
#include <natten/cuda/fna/na_utils.cuh>

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// PredicatedTileAccessIteratorResidualLast
///
template <
    int NADim,
    typename Shape,
    typename Element,
    typename Layout,
    int AdvanceRank,
    typename ThreadMap,
    typename AccessType>
class PredicatedTileAccessIteratorResidualLast;

////////////////////////////////////////////////////////////////////////////////

/// PredicatedTileAccessIteratorResidualLast for possibly multiple axes (NA1D,
/// 2D, 3D)
template <
    int NADim,
    typename Shape_,
    typename Element_,
    int AdvanceRank,
    typename ThreadMap_,
    typename AccessType_>
class PredicatedTileAccessIteratorResidualLast<
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

    // Default ctor
    CUTLASS_HOST_DEVICE
    Params() {}

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
  Mask residual_tile_mask;

  /// Parameters object with precomputed internal state
  Params params_;

  /// Internal pointer to first access of tile
  BytePointer pointer_;
  TensorCoord coord_offset_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
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
        the_predicates(extent_row, extent_col) {
    the_predicates.set_predicates(thread_id, threadblock_offset);
    the_predicates.get_mask(residual_tile_mask);

    // Working around a weird compiler bug happening on P100 for the backward.
    // I've seen together: the_predicates.predicates_[0] = 14 (instead of 15)
    // residual_tile_mask[0] = 15 (correct)
    //
    // Adding prints when the value is calculated (in `compute_predicates_`)
    // sometimes removes the bug. The consequence is that we skip some
    // element of a tensor, leading to wrong results
    // Setting `compute_predicates_`'s second argument (`is_steady_state`) to
    // true also seems to get rid of the bug - at the cost of twice as many
    // comparisons.
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)
    constexpr bool kWorkAroundCompilerBug = false;
#else
    constexpr bool kWorkAroundCompilerBug = true;
#endif
    the_predicates.compute_predicates_(
        the_predicates.extent_, true && !kWorkAroundCompilerBug);

    // update internal pointers
    coord_offset_ = the_predicates.thread_offset_;
    // add_pointer_offset(the_predicates.thread_offset_.strided(),
    // the_predicates.thread_offset_.contiguous());
  }

  /// Construct a PredicatedTileAccessIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
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
      : PredicatedTileAccessIteratorResidualLast(
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
  void set_residual_tile(bool is_residual_tile) {
    if (is_residual_tile) {
      the_predicates.set_mask(residual_tile_mask);
    }
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
    //   add_pointer_offset((int32_t)row_offset, (int32_t)col_offset);
    // } else {
    //   auto col_offset = linear_offset % Shape::kContiguous;
    //   auto row_offset = linear_offset / Shape::kContiguous;
    //   add_pointer_offset((int32_t)row_offset, (int32_t)col_offset);
    // }
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    add_pointer_offset(
        Shape::kStrided * tile_offset.strided(),
        Shape::kContiguous * tile_offset.contiguous());
    // if (kAdvanceRank) {
    //   add_pointer_offset(
    //       params_.inc_advance_ * (tile_offset.strided()) +
    //       Shape::kContiguous * tile_offset.contiguous());
    //   // pointer_ += params_.inc_advance_ * LongIndex(tile_offset.strided());
    //   // pointer_ += Shape::kContiguous * tile_offset.contiguous();
    // } else {
    //     // Advance along contiguous dimension
    //     add_pointer_offset(
    //       params_.inc_advance_ * (tile_offset.contiguous()) +
    //       Shape::kStrided * tile_offset.strided());
    //   // pointer_ += params_.inc_advance_ *
    //   LongIndex(tile_offset.contiguous());
    //   // pointer_ += Shape::kStrided * tile_offset.strided();
    // }
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
  PredicatedTileAccessIteratorResidualLast& operator++() {
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
      // add_pointer_offset(params_.inc_strided_);
      // pointer_ += params_.inc_strided_;

      return *this;
    }

    // Enter here only if (iteration_stride_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next tile.
    the_predicates.iteration_strided_ = 0;

    //// advance to next tile
    // add_pointer_offset(params_.inc_next_);
    //// pointer_ += params_.inc_next_;

    // add_pointer_offset(-1 * params_.inc_advance_);
    ////if constexpr (kAdvanceRank) {
    ////  add_pointer_offset(-1 * params_.inc_advance_, 0);
    ////} else {
    ////  // along the contiguous axis.
    ////  add_pointer_offset(0, -1 * params_.inc_advance_);
    ////}
    //// now return to start tile - if the iterator is subsequently advanced,
    /// this / subtraction as well as the subsequent integer addition are both
    /// elided by / the compiler. / pointer_ -= params_.inc_advance_;

    return *this;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast operator++(int) {
    PredicatedTileAccessIteratorResidualLast self(*this);
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
class PredicatedTileAccessIteratorResidualLast<
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

    // Default ctor
    CUTLASS_HOST_DEVICE
    Params() {}

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
  Mask residual_tile_mask;

  /// Parameters object with precomputed internal state
  Params params_;

  /// Internal pointer to first access of tile
  BytePointer pointer_;

 public:
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
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
        the_predicates(extent_row, extent_col) {
    the_predicates.set_predicates(thread_id, threadblock_offset);
    the_predicates.get_mask(residual_tile_mask);

    // Working around a weird compiler bug happening on P100 for the backward.
    // I've seen together: the_predicates.predicates_[0] = 14 (instead of 15)
    // residual_tile_mask[0] = 15 (correct)
    //
    // Adding prints when the value is calculated (in `compute_predicates_`)
    // sometimes removes the bug. The consequence is that we skip some
    // element of a tensor, leading to wrong results
    // Setting `compute_predicates_`'s second argument (`is_steady_state`) to
    // true also seems to get rid of the bug - at the cost of twice as many
    // comparisons.
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)
    constexpr bool kWorkAroundCompilerBug = false;
#else
    constexpr bool kWorkAroundCompilerBug = true;
#endif
    the_predicates.compute_predicates_(
        the_predicates.extent_, true && !kWorkAroundCompilerBug);

    // update internal pointers
    auto offset = (natten::cuda::fna::map_index_to_coord(
                       the_predicates.thread_offset_.strided(), extent_row) *
                   params_.stride_)
                      .sum() +
        the_predicates.thread_offset_.contiguous();

    add_pointer_offset(offset);
  }

  /// Construct a PredicatedTileAccessIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
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
      : PredicatedTileAccessIteratorResidualLast(
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
  void set_residual_tile(bool is_residual_tile) {
    if (is_residual_tile) {
      the_predicates.set_mask(residual_tile_mask);
    }
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
    if (kAdvanceRank) {
      pointer_ += params_.inc_advance_ * LongIndex(tile_offset.strided());
      pointer_ += Shape::kContiguous * tile_offset.contiguous();
    } else {
      pointer_ += params_.inc_advance_ * LongIndex(tile_offset.contiguous());
      pointer_ += Shape::kStrided * tile_offset.strided();
    }
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
  PredicatedTileAccessIteratorResidualLast& operator++() {
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

    // now return to start tile - if the iterator is subsequently advanced,
    // this subtraction as well as the subsequent integer addition are both
    // elided by the compiler.
    pointer_ -= params_.inc_advance_;

    return *this;
  }

  /// Increment and return an instance to self.
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast operator++(int) {
    PredicatedTileAccessIteratorResidualLast self(*this);
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

/// Specialization of PredicatedTileAccessIteratorResidualLast for row-major
/// data.
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
class PredicatedTileAccessIteratorResidualLast<
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

  using UnderlyingIterator = PredicatedTileAccessIteratorResidualLast<
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
    friend PredicatedTileAccessIteratorResidualLast;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    /// Default ctor
    CUTLASS_HOST_DEVICE
    Params() {}

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
  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
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

  /// Construct a PredicatedTileAccessIteratorResidualLast with zero threadblock
  /// offset
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIteratorResidualLast(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      Dim extent_row, ///< Extent of tensor (rows)
      int32_t extent_col, ///< Extent of tensor (columns)
      int thread_id ///< ID of each participating thread
      )
      : PredicatedTileAccessIteratorResidualLast(
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

  CUTLASS_HOST_DEVICE
  void set_residual_tile(bool enable) {
    iterator_.set_residual_tile(enable);
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
  PredicatedTileAccessIteratorResidualLast& operator++() {
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
  PredicatedTileAccessIteratorResidualLast operator++(int) {
    PredicatedTileAccessIteratorResidualLast self(*this);
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
