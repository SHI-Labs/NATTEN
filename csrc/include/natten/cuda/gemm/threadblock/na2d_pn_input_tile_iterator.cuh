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
    \brief Templates implementing loading of NA tiles mapped to GEMM
   (neighborhood x channels tile) matrix from memory.

    This iterator assumes TensorNDHWC layout of tensors in Global Memory.

    The iterator is specialized for each of the three NA operators.
*/

#pragma once

#include <cutlass/array.h>
#include <cutlass/coord.h>
#include <cutlass/cutlass.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/pitch_linear.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/matrix_shape.h>
#include <cutlass/predicate_vector.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/tensor_view.h>

#include "natten/cuda/gemm/na2d_problem_size.cuh"
#include "natten/cuda/gemm/neighborhood_attention.cuh"
#include "natten/cuda/gemm/threadblock/na2d_params.cuh"
#include "natten/cuda/gemm/threadblock/na2d_tile.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace natten {
namespace cuda {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    typename NAShape_,
    typename Shape_,
    typename Element_,
    typename Layout_,
    typename ThreadMap_,
    typename AccessType_ =
        cutlass::AlignedArray<Element_, ThreadMap_::kElementsPerAccess>>
class NA2dPNInputTileIterator {
 public:
  //
  // Types
  //

  using NAShape = NAShape_;
  using TileInfo = NA2dTileInfo<NAShape, Operator::kPN>;
  using Shape = Shape_;
  using Element = Element_;
  using Layout = Layout_;
  using TensorCoord = typename Layout::TensorCoord;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;
  using TensorRef = cutlass::TensorRef<Element, Layout>;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  static int const kAccessesPerVector =
      ThreadMap::kElementsPerAccess / AccessType::kElements;

  static_assert(
      !(ThreadMap::kElementsPerAccess % AccessType::kElements),
      "Vectors implied by the thread map must be divisible by the access type.");

  static_assert(
      cutlass::sizeof_bits<Element>::value >= 8,
      "NA-PN requires elements of size 8b or greater.");

  static_assert(
      ThreadMap::Iterations::kContiguous == 1,
      "Require Iterations::kContiguous == 1");

  //
  // Parameters structure
  //

  using Params = NA2dOptimizedParams<NAShape, Layout>;

 private:
  Params const& params_;
  NA2dProblemSize const& problem_size_;
  LongIndex iteration_contiguous_;
  LongIndex iteration_strided_;
  LongIndex iteration_vector_;

  char const* pointer_[ThreadMap::Iterations::kStrided];

  bool mask_[ThreadMap::Iterations::kStrided];
  bool mask_d_[kAccessesPerVector];

  int offset_d_;

 public:
  CUTLASS_HOST_DEVICE
  NA2dPNInputTileIterator(
      Params const& params,
      NA2dProblemSize const& problem_size,
      Element const* ptr,
      TileInfo const& tile_info,
      int thread_idx,
      cutlass::MatrixCoord const& threadblock_offset = cutlass::MatrixCoord())
      : params_(params), problem_size_(problem_size) {
    cutlass::layout::PitchLinearCoord thread_coord =
        ThreadMap::initial_offset(thread_idx);

    // initialize offset_d
    offset_d_ = threadblock_offset.column() + thread_coord.contiguous();

    // initialize rs for every strided iteration
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      pointer_[s] = reinterpret_cast<char const*>(ptr);
      int offset_r, offset_s;
      params_.stride_divmod(
          offset_r,
          offset_s,
          (threadblock_offset.row() + thread_coord.strided() +
           s * ThreadMap::Delta::kStrided));

      TensorCoord coord(
          tile_info.b,
          tile_info.n,
          tile_info.h_ext_abs +
              offset_r * problem_size_.dilation_h, /* Extended tiles in PN */
          tile_info.w_ext_abs +
              offset_s * problem_size_.dilation_w, /* Extended tiles in PN */
          offset_d_);
      pointer_[s] +=
          params_.layout(coord) * cutlass::sizeof_bits<Element>::value / 8;
      mask_[s] = coord.h() >= 0 && coord.w() >= 0 &&
          coord.h() < problem_size_.H && coord.w() < problem_size_.W;
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kAccessesPerVector; ++i) {
      mask_d_[i] = (offset_d_ + i * AccessType::kElements) < problem_size_.D;
    }
  }

  CUTLASS_HOST_DEVICE
  static Params getParams(
      NA2dProblemSize const& problem_size,
      Layout const& layout) {
    return Params(problem_size, layout);
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(Index index) {
    iteration_vector_ = index % kAccessesPerVector;
    int residual_access = index / kAccessesPerVector;
    iteration_contiguous_ =
        residual_access % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = residual_access / ThreadMap::Iterations::kContiguous;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    // pointer_ += pointer_offset * cutlass::sizeof_bits<Element>::value / 8;
  }

  CUTLASS_HOST_DEVICE
  void advance() {
    // moves to the next tile
    offset_d_ += Shape::kColumn;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kAccessesPerVector; ++i) {
      mask_d_[i] = (offset_d_ + i * AccessType::kElements) < problem_size_.D;
    }
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      pointer_[s] += Shape::kColumn * cutlass::sizeof_bits<Element>::value / 8;
    }
  }

  /// Returns true if the current coordinate is within the activations tensor X
  CUTLASS_HOST_DEVICE
  bool valid() const {
    return mask_d_[iteration_vector_] && mask_[iteration_strided_];
  }

  /// Returns a pointer to the vector starting at the current coordinate
  CUTLASS_HOST_DEVICE
  AccessType const* get() const {
    return reinterpret_cast<AccessType const*>(pointer_[iteration_strided_]) +
        iteration_vector_;
  }

  /// Increments to the next memory access
  CUTLASS_HOST_DEVICE
  NA2dPNInputTileIterator& operator++() {
    ++iteration_vector_;
    if (iteration_vector_ < kAccessesPerVector) {
      return *this;
    }
    iteration_vector_ = 0;

    ++iteration_contiguous_;
    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous) {
      return *this;
    }
    iteration_contiguous_ = 0;

    ++iteration_strided_;
    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      return *this;
    }
    iteration_strided_ = 0;

    return *this;
  }

  /// Determines whether the Implicit GEMM can execute the given problem.
  CUTLASS_HOST_DEVICE
  static cutlass::Status can_implement(NA2dProblemSize const& problem_size) {
    // check alignment constraint on iterator's contiguous dimension
    if (problem_size.D % AccessType::kElements) {
      return cutlass::Status::kErrorInvalidProblem;
    }

    return cutlass::Status::kSuccess;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cuda
} // namespace natten

/////////////////////////////////////////////////////////////////////////////////////////////////
