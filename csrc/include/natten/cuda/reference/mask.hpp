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

#include "cute/layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace natten {
namespace cuda {
namespace reference {

namespace mask {

using namespace cute;

template <class NADim>
CUTE_DEVICE constexpr auto get_window_left(NADim const& window_size) {
  return transform_leaf(window_size, [&](auto const& w) { return w / 2; });
}

template <class NADim>
CUTE_DEVICE constexpr auto get_window_right(NADim const& window_size) {
  return transform_leaf(
      window_size, [&](auto const& w) { return (w / 2) + ((w % 2) - 1); });
}

template <bool IsCausal>
CUTE_DEVICE int get_win_start(
    int index,
    int window_left,
    int window_right,
    int stride,
    int length) {
  if constexpr (IsCausal) {
    int32_t stride_group_leader_idx =
        cutlass::fast_min((index / stride) * stride + stride - 1, length - 1);
    return cutlass::fast_max(
        stride_group_leader_idx - window_left - window_right, 0);

  } else {
    auto stride_group_leader_idx = cutlass::fast_min(
        ((index / stride) * stride) + (stride / 2), length - 1);

    return cutlass::fast_max(stride_group_leader_idx - window_left, 0) +
        ((stride_group_leader_idx + window_right >= length) *
         (length - window_right - stride_group_leader_idx - 1));
  }
}

template <bool IsCausal>
CUTE_DEVICE int get_win_end(int index, int start, int window_size, int length) {
  if constexpr (IsCausal) {
    return cutlass::fast_min(index + 1, length);
  } else {
    return start + window_size;
  }
}

template <class Causal, class NADim>
CUTE_DEVICE auto get_window_start(
    NADim index,
    NADim window_left,
    NADim window_right,
    NADim stride,
    NADim length) {
  static_assert(rank(index) > 0 && rank(index) < 4);
  static_assert(rank(index) == rank(Causal{}));
  if constexpr (rank(index) == 1) {
    return make_tuple(get_win_start<get<0>(Causal{})>(
        get<0>(index),
        get<0>(window_left),
        get<0>(window_right),
        get<0>(stride),
        get<0>(length)));
  } else if constexpr (rank(index) == 2) {
    return make_tuple(
        get_win_start<get<0>(Causal{})>(
            get<0>(index),
            get<0>(window_left),
            get<0>(window_right),
            get<0>(stride),
            get<0>(length)),
        get_win_start<get<1>(Causal{})>(
            get<1>(index),
            get<1>(window_left),
            get<1>(window_right),
            get<1>(stride),
            get<1>(length)));
  } else {
    return make_tuple(
        get_win_start<get<0>(Causal{})>(
            get<0>(index),
            get<0>(window_left),
            get<0>(window_right),
            get<0>(stride),
            get<0>(length)),
        get_win_start<get<1>(Causal{})>(
            get<1>(index),
            get<1>(window_left),
            get<1>(window_right),
            get<1>(stride),
            get<1>(length)),
        get_win_start<get<2>(Causal{})>(
            get<2>(index),
            get<2>(window_left),
            get<2>(window_right),
            get<2>(stride),
            get<2>(length)));
  }
}

template <class Causal, class NADim>
CUTE_DEVICE auto get_window_end(
    NADim index,
    NADim start,
    NADim window_size,
    NADim length) {
  static_assert(rank(index) > 0 && rank(index) < 4);
  static_assert(rank(index) == rank(Causal{}));
  if constexpr (rank(index) == 1) {
    return make_tuple(get_win_end<get<0>(Causal{})>(
        get<0>(index), get<0>(start), get<0>(window_size), get<0>(length)));
  } else if constexpr (rank(index) == 2) {
    return make_tuple(
        get_win_end<get<0>(Causal{})>(
            get<0>(index), get<0>(start), get<0>(window_size), get<0>(length)),
        get_win_end<get<1>(Causal{})>(
            get<1>(index), get<1>(start), get<1>(window_size), get<1>(length)));
  } else {
    return make_tuple(
        get_win_end<get<0>(Causal{})>(
            get<0>(index), get<0>(start), get<0>(window_size), get<0>(length)),
        get_win_end<get<1>(Causal{})>(
            get<1>(index), get<1>(start), get<1>(window_size), get<1>(length)),
        get_win_end<get<2>(Causal{})>(
            get<2>(index), get<2>(start), get<2>(window_size), get<2>(length)));
  }
}

template <class Coord, class NADim>
CUTE_DEVICE bool is_neighbor(Coord kv_coord, NADim kv_start, NADim kv_end) {
  static_assert(rank(kv_coord) > 0 && rank(kv_coord) < 4);
  static_assert(rank(kv_coord) == rank(kv_start));
  if constexpr (rank(kv_coord) == 1) {
    return get<0>(kv_coord) >= get<0>(kv_start) &&
        get<0>(kv_coord) < get<0>(kv_end);
  } else if constexpr (rank(kv_coord) == 2) {
    return get<0>(kv_coord) >= get<0>(kv_start) &&
        get<0>(kv_coord) < get<0>(kv_end) &&
        get<1>(kv_coord) >= get<1>(kv_start) &&
        get<1>(kv_coord) < get<1>(kv_end);
  } else {
    return get<0>(kv_coord) >= get<0>(kv_start) &&
        get<0>(kv_coord) < get<0>(kv_end) &&
        get<1>(kv_coord) >= get<1>(kv_start) &&
        get<1>(kv_coord) < get<1>(kv_end) &&
        get<2>(kv_coord) >= get<2>(kv_start) &&
        get<2>(kv_coord) < get<2>(kv_end);
  }
}

template <class T0, class T1>
CUTE_DEVICE constexpr auto floor_div_tuple(T0 const& t0, T1 const& t1) {
  return transform_leaf(
      t0, t1, [&](auto const& a, auto const& b) { return a / b; });
}

template <class T0, class T1>
CUTE_DEVICE constexpr auto mod_tuple(T0 const& t0, T1 const& t1) {
  return transform_leaf(
      t0, t1, [&](auto const& a, auto const& b) { return a % b; });
}

template <class Coord>
CUTE_DEVICE bool is_equal(Coord a, Coord b) {
  static_assert(rank(a) > 0 && rank(a) < 4);
  static_assert(rank(a) == rank(b));
  if constexpr (rank(a) == 1) {
    return get<0>(a) == get<0>(b);
  } else if constexpr (rank(a) == 2) {
    return get<0>(a) == get<0>(b) && get<1>(a) == get<1>(b);
  } else {
    return get<0>(a) == get<0>(b) && get<1>(a) == get<1>(b) &&
        get<2>(a) == get<2>(b);
  }
}

CUTE_HOST_DEVICE
int qkv_fix_dilation(int qkv_shape, int dilation, int dilation_group) {
  auto padding =
      1 - ((dilation_group + (dilation - (qkv_shape % dilation))) / dilation);
  return (qkv_shape / dilation) + padding;
}

template <class NADim, class Coord>
CUTE_HOST_DEVICE auto correct_qkv_shape_wrt_dilation(
    NADim qkv_shape,
    NADim dilation,
    Coord dilation_group) {
  static_assert(rank(qkv_shape) > 0 && rank(qkv_shape) < 4);
  static_assert(rank(qkv_shape) == rank(dilation_group));
  if constexpr (rank(qkv_shape) == 1) {
    return make_tuple(qkv_fix_dilation(
        get<0>(qkv_shape), get<0>(dilation), get<0>(dilation_group)));
  } else if constexpr (rank(qkv_shape) == 2) {
    return make_tuple(
        qkv_fix_dilation(
            get<0>(qkv_shape), get<0>(dilation), get<0>(dilation_group)),
        qkv_fix_dilation(
            get<1>(qkv_shape), get<1>(dilation), get<1>(dilation_group)));
  } else {
    return make_tuple(
        qkv_fix_dilation(
            get<0>(qkv_shape), get<0>(dilation), get<0>(dilation_group)),
        qkv_fix_dilation(
            get<1>(qkv_shape), get<1>(dilation), get<1>(dilation_group)),
        qkv_fix_dilation(
            get<2>(qkv_shape), get<2>(dilation), get<2>(dilation_group)));
  }
}

// Not using CuTe tilers here because we'd need to make dilation a layout
// as well since CuTe and torch identity layouts aren't the same, and
// that creeps into weird places that we just don't want for a reference
// kernel.
template <class NADim, class QKVLayout>
CUTE_DEVICE auto map_idx_to_di_coords(
    int idx,
    NADim dilation,
    QKVLayout qkv_layout) {
  auto coord_global = idx2crd(idx, qkv_layout.shape(), qkv_layout.stride());

  auto coord_within_di_group = floor_div_tuple(coord_global, dilation);
  auto dilation_group_crd = mod_tuple(coord_global, dilation);

  return cute::make_tuple(coord_within_di_group, dilation_group_crd);
}

// Reference Mask
// Handles all parameters (window size, stride, causal, AND dilation)
// in the same place.
template <class NADim_, class Causal_, class QKVLayout>
struct NeighborhoodAttentionReferenceMask {
  using NADim = NADim_;
  using Causal = Causal_;
  static_assert(rank(NADim{}) >= 1 && rank(NADim{}) < 4);
  static_assert(rank(Causal{}) >= 1 && rank(Causal{}) < 4);

  NADim window_size;
  NADim window_left;
  NADim window_right;
  NADim stride;
  NADim dilation;
  QKVLayout qkv_layout;
  int num_additional_kv;
  int additional_kv_offset;

  CUTE_DEVICE NeighborhoodAttentionReferenceMask(
      NADim window_size_,
      NADim stride_,
      NADim dilation_,
      QKVLayout qkv_layout_,
      int num_additional_kv_)
      : window_size(window_size_),
        stride(stride_),
        dilation(dilation_),
        qkv_layout(qkv_layout_),
        num_additional_kv(num_additional_kv_),
        additional_kv_offset(size(qkv_layout_.shape())) {
    window_left = get_window_left(window_size_);
    window_right = get_window_right(window_size_);
  }

  template <class AccQK, class IndexQK>
  CUTE_DEVICE void apply_mask(AccQK& acc_qk, IndexQK const& index_qk) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      // NOTE: this is very inefficient if there's multiple queries in the
      // accumulator. But in the reference kernel we know accum size is 1x1.
      auto [q_idx, kv_idx] = index_qk(i);

      // If kv_idx > size(qkv_shape), then it's part of the cross attention
      // tokens. Rule is: allow all of them up to num_additional_kv, mask out
      // the rest.
      if (kv_idx >= additional_kv_offset &&
          kv_idx - additional_kv_offset < num_additional_kv) {
        continue;
      } else if (kv_idx >= additional_kv_offset) {
        acc_qk(i) = -INFINITY;
        continue;
      }

      auto [q_coord, q_di_coord] =
          map_idx_to_di_coords(q_idx, dilation, qkv_layout);
      auto [kv_coord, kv_di_coord] =
          map_idx_to_di_coords(kv_idx, dilation, qkv_layout);

      // Fixup input shape according to dilation group
      auto qkv_shape = correct_qkv_shape_wrt_dilation(
          qkv_layout.shape(), dilation, q_di_coord);

      auto kv_start = get_window_start<Causal>(
          q_coord, window_left, window_right, stride, qkv_shape);
      auto kv_end =
          get_window_end<Causal>(q_coord, kv_start, window_size, qkv_shape);

      if (not is_equal(q_di_coord, kv_di_coord) or
          not is_neighbor(kv_coord, kv_start, kv_end)) {
        acc_qk(i) = -INFINITY;
      }
    }
  }
};

} // namespace mask

} // namespace reference
} // namespace cuda
} // namespace natten

/////////////////////////////////////////////////////////////////////////////////////////////////
