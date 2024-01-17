/*
 * Copied from CUTLASS (https://github.com/NVIDIA/cutlass/) and edited.
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
    \brief Templates implementing loading of tiles from pitch-linear rank=2
   tensors.

    This iterator uses masks to guard out-of-bounds accesses. The first tile
   this iterator visits maybe partial, then the remaining tiles are complete.
   So, we only need to compute the predicates twice, once before the first tile
   and once for the remaining full tiles which can share the same predicates.

    A precomputed "Params" object minimizes the amount of state that must be
   stored in registers, and integer addition is used to advance the pointer
   through memory.
*/

#pragma once

#include <cutlass/arch/memory.h>

#include <natten/cuda/fna/iterators/predicated_tile_access_iterator.h>
#include <natten/cuda/fna/na_utils.cuh>

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

template <
    int NADim,
    typename Shape,
    typename Element,
    typename Layout,
    int AdvanceRank,
    typename ThreadMap,
    int AccessSize = ThreadMap::kElementsPerAccess>
class CustomPredicatedTileIterator;

////////////////////////////////////////////////////////////////////////////////

template <
    int NADim,
    typename Shape_,
    typename Element_,
    int AdvanceRank,
    typename ThreadMap_,
    int AccessSize>
class CustomPredicatedTileIterator<
    NADim,
    Shape_,
    Element_,
    layout::PitchLinear,
    AdvanceRank,
    ThreadMap_,
    AccessSize> {
 public:
  static_assert(NADim >= 1 && NADim < 4);
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Dim = typename natten::cuda::fna::GetDim<NADim>::type;

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  static int const kAdvanceRank = AdvanceRank;
  using ThreadMap = ThreadMap_;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename platform::remove_const<Element>::type*;

  /// Type used for internal memory accesses
  using AccessType = AlignedArray<
      Element,
      AccessSize,
      (AccessSize * sizeof_bits<Element>::value / 8)>;

  /// Underlying iterator to compute the addresses
  using TileAccessIterator = CustomPredicatedTileAccessIterator<
      NADim,
      Shape,
      Element,
      Layout,
      kAdvanceRank,
      ThreadMap,
      AccessType>;

  static int const kAccessesPerVector = TileAccessIterator::kAccessesPerVector;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<
      Element,
      ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename TileAccessIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   public:
    using Base = typename TileAccessIterator::Params::Base;

    friend CustomPredicatedTileIterator;

   private:
    /// Parameters object
    typename TileAccessIterator::Params params_;

   public:
    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Dim stride, Dim extent_row) : params_(stride, extent_row) {}

    /// Default constructor
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(Base const& base) : params_(base) {}
  };

 private:
  /// Internal pointer type permits fast address arithmetic
  using BytePointer = char*;

 private:
  //
  // Data members
  //

  /// Data member to the tile access iterator
  TileAccessIterator address_iterator_;

 public:
  /// Default constructor
  CustomPredicatedTileIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileIterator(
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
      : address_iterator_(
            params.params_,
            pointer,
            extent_row,
            extent_col,
            thread_id,
            threadblock_offset) {}

  /// Construct a CustomPredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileIterator(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      Dim extent_row, ///< Extent of tensor (rows)
      int32_t extent_col, ///< Extent of tensor (columns)
      int thread_id ///< ID of each participating thread
      )
      : CustomPredicatedTileIterator(
            params,
            pointer,
            extent_row,
            extent_col,
            thread_id,
            make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    address_iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileIterator& operator++() {
    if (kAdvanceRank)
      address_iterator_.add_tile_offset({0, 1});
    else
      address_iterator_.add_tile_offset({1, 0});

    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileIterator operator++(int) {
    CustomPredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    address_iterator_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    address_iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& mask) {
    address_iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& mask) {
    address_iterator_.get_mask(mask);
  }

  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
    load_with_byte_offset(
        frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
    AccessType* frag_ptr = reinterpret_cast<AccessType*>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {
          int idx = v +
              kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);

          address_iterator_.set_iteration_index(idx);
          char const* byte_ptr =
              reinterpret_cast<char const*>(address_iterator_.get()) +
              byte_offset;

          AccessType const* access_ptr =
              reinterpret_cast<AccessType const*>(byte_ptr);

          cutlass::arch::global_load<AccessType, sizeof(AccessType)>(
              frag_ptr[idx], access_ptr, address_iterator_.valid());

          ++address_iterator_;
        }
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment& frag) {
    load_with_byte_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
    store_with_byte_offset(
        frag, pointer_offset * sizeof_bits<Element>::value / 8);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
    address_iterator_.set_iteration_index(0);
    AccessType const* frag_ptr = reinterpret_cast<AccessType const*>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < kAccessesPerVector; ++v) {
          int idx = v +
              kAccessesPerVector * (c + s * ThreadMap::Iterations::kContiguous);

          char* byte_ptr =
              reinterpret_cast<char*>(address_iterator_.get()) + byte_offset;
          AccessType* access_ptr = reinterpret_cast<AccessType*>(byte_ptr);

          if (address_iterator_.valid()) {
            *access_ptr = frag_ptr[idx];
          }
          ++address_iterator_;
        }
      }
    }
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const& frag) {
    store_with_byte_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of CustomPredicatedTileIterator for column-major data.
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
    int AccessSize>
class CustomPredicatedTileIterator<
    NADim,
    Shape_,
    Element_,
    layout::ColumnMajor,
    AdvanceRank,
    ThreadMap_,
    AccessSize> {
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

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename platform::remove_const<Element>::type*;

  using UnderlyingIterator = CustomPredicatedTileIterator<
      NADim,
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>,
      Element,
      layout::PitchLinear,
      (kAdvanceRank == 0 ? 0 : 1),
      ThreadMap,
      AccessSize>;

  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<
      Element,
      ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend CustomPredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    /// Default constructor
    Params() = default;

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Dim stride, Dim extent_row) : params_(stride, extent_row) {}

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
  CustomPredicatedTileIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileIterator(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      Dim extent_row, ///< Extent of tensor (rows)
      int32_t extent_col, ///< Extent of tensor (columns)
      int thread_id, ///< ID of each participating thread
      TensorCoord const& threadblock_offset ///< Initial offset of threadblock
      )
      : iterator_(
            params.params_,
            pointer,
            extent_row,
            extent_col,
            thread_id,
            layout::PitchLinearCoord(
                threadblock_offset.row(),
                threadblock_offset.column())) {}

  /// Construct a CustomPredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileIterator(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      Dim extent_row, ///< Extent of tensor (rows)
      int32_t extent_col, ///< Extent of tensor (columns)
      int thread_id ///< ID of each participating thread
      )
      : CustomPredicatedTileIterator(
            params,
            pointer,
            extent_row,
            extent_col,
            thread_id,
            make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileIterator& operator++() {
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
  CustomPredicatedTileIterator operator++(int) {
    CustomPredicatedTileIterator self(*this);
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

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment& frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
    iterator_.store_with_byte_offset(frag, byte_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const& frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization of CustomPredicatedTileIterator for row-major data.
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
    int AccessSize>
class CustomPredicatedTileIterator<
    NADim,
    Shape_,
    Element_,
    layout::RowMajor,
    AdvanceRank,
    ThreadMap_,
    AccessSize> {
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

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorView = TensorView<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using Pointer = Element*;
  using NonConstPointer = typename platform::remove_const<Element>::type*;

  using UnderlyingIterator = CustomPredicatedTileIterator<
      NADim,
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>,
      Element,
      layout::PitchLinear,
      (kAdvanceRank == 0 ? 1 : 0),
      ThreadMap,
      AccessSize>;

  using AccessType = typename UnderlyingIterator::AccessType;

  /// Fragment object to be loaded or stored
  using Fragment = cutlass::Array<
      Element,
      ThreadMap::Iterations::kCount * ThreadMap::kElementsPerAccess>;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend CustomPredicatedTileIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:
    /// Default constructor
    Params() = default;

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Dim stride, Dim extent_row) : params_(stride, extent_row) {}

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
  CustomPredicatedTileIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileIterator(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      Dim extent_row, ///< Extent of tensor (rows)
      int32_t extent_col, ///< Extent of tensor (columns)
      int thread_id, ///< ID of each participating thread
      TensorCoord const& threadblock_offset ///< Initial offset of threadblock
      )
      : iterator_(
            params.params_,
            pointer,
            extent_row,
            extent_col,
            thread_id,
            layout::PitchLinearCoord(
                threadblock_offset.column(),
                threadblock_offset.row())) {}

  /// Construct a CustomPredicatedTileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileIterator(
      Params const& params, ///< Precomputed parameters object
      Pointer pointer, ///< Pointer to start of tensor
      Dim extent_row, ///< Extent of tensor (rows)
      int32_t extent_col, ///< Extent of tensor (columns)
      int thread_id ///< ID of each participating thread
      )
      : CustomPredicatedTileIterator(
            params,
            pointer,
            extent_row,
            extent_col,
            thread_id,
            make_Coord(0, 0)) {}

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  CustomPredicatedTileIterator& operator++() {
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
  CustomPredicatedTileIterator operator++(int) {
    CustomPredicatedTileIterator self(*this);
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

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment& frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment& frag, LongIndex byte_offset) {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment& frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const& frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const& frag, LongIndex byte_offset) {
    iterator_.store_with_byte_offset(frag, byte_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const& frag) {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace transform
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
