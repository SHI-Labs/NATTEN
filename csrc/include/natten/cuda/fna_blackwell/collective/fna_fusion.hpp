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

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

namespace cutlass::fna::collective {

using namespace cute;

// Tuple utils

template <class T0, class T1>
CUTE_HOST_DEVICE constexpr auto floor_tuple(T0 const& t0, T1 const& t1) {
  return transform_leaf(
      t0, t1, [&](auto const& a, auto const& b) { return (a / b) * b; });
}

template <class T0, class T1>
CUTE_HOST_DEVICE constexpr auto ceil_tuple(T0 const& t0, T1 const& t1) {
  return transform_leaf(
      t0, t1, [&](auto const& a, auto const& b) { return ceil_div(a, b) * b; });
}

template <class T0, class T1>
CUTE_HOST_DEVICE constexpr auto tuple_sub(T0 const& t0, T1 const& t1) {
  return transform_leaf(
      t0, t1, [&](auto const& a, auto const& b) { return a - b; });
}

template <class T0, class T1>
CUTE_HOST_DEVICE constexpr auto tuple_add(T0 const& t0, T1 const& t1) {
  return transform_leaf(
      t0, t1, [&](auto const& a, auto const& b) { return a + b; });
}

template <class T0, class T1>
CUTE_HOST_DEVICE constexpr auto tuple_mul(T0 const& t0, T1 const& t1) {
  return transform_leaf(
      t0, t1, [&](auto const& a, auto const& b) { return a * b; });
}

template <class T>
CUTE_HOST_DEVICE bool tuple_leq(T t0, T t1) {
  static_assert(rank(T{}) > 0 && rank(T{}) < 4);
  if constexpr (rank(T{}) == 1) {
    return get<0>(t0) <= get<0>(t1);
  } else if constexpr (rank(T{}) == 2) {
    return get<0>(t0) <= get<0>(t1) && get<1>(t0) <= get<1>(t1);
  } else {
    return get<0>(t0) <= get<0>(t1) && get<1>(t0) <= get<1>(t1) &&
        get<2>(t0) <= get<2>(t1);
  }
}

// NA mask utils
template <bool IsCausal>
CUTE_HOST_DEVICE int get_win_start(
    int index,
    int window_left,
    int window_right,
    int stride,
    int length) {
  if constexpr (IsCausal) {
    int32_t stride_group_leader_idx =
        cutlass::fast_min((index / stride) * stride + stride - 1, length - 1);
    // window_size == window_left + window_right + 1
    // return cutlass::fast_max(stride_group_leader_idx - window_size + 1, 0);
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
CUTE_HOST_DEVICE int get_win_end(
    int index,
    int start,
    int window_size,
    int length) {
  if constexpr (IsCausal) {
    return cutlass::fast_min(index + 1, length);
  } else {
    return start + window_size;
  }
}

template <class Causal, class NADim, class Coord>
CUTE_HOST_DEVICE auto get_window_start(
    Coord index,
    NADim window_left,
    NADim window_right,
    NADim stride,
    NADim length) {
  static_assert(rank(index) > 0 && rank(index) < 4);
  static_assert(rank(index) == rank(Causal{}));
  static_assert(rank(index) == rank(length));
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

template <class Causal, class NADim, class Coord>
CUTE_HOST_DEVICE auto get_window_end(
    Coord index,
    NADim start,
    NADim window_size,
    NADim length) {
  static_assert(rank(index) > 0 && rank(index) < 4);
  static_assert(rank(index) == rank(Causal{}));
  static_assert(rank(index) == rank(length));
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
CUTE_HOST_DEVICE bool is_neighbor(
    Coord kv_coord,
    NADim kv_start,
    NADim kv_end) {
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

template <class Coord, class NADim>
CUTE_HOST_DEVICE bool is_within_bounds(Coord kv_coord, NADim kv_shape) {
  static_assert(rank(kv_coord) > 0 && rank(kv_coord) < 4);
  static_assert(rank(kv_coord) == rank(kv_shape));
  if constexpr (rank(kv_coord) == 1) {
    return get<0>(kv_coord) < get<0>(kv_shape);
  } else if constexpr (rank(kv_coord) == 2) {
    return get<0>(kv_coord) < get<0>(kv_shape) &&
        get<1>(kv_coord) < get<1>(kv_shape);
  } else {
    return get<0>(kv_coord) < get<0>(kv_shape) &&
        get<1>(kv_coord) < get<1>(kv_shape) &&
        get<2>(kv_coord) < get<2>(kv_shape);
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

template <class Causal_>
struct NeighborhoodAttentionMask {
  using Causal = Causal_;
  static_assert(rank(Causal{}) >= 1 && rank(Causal{}) < 4);

  // QKV shape Correction
  // Only if dilated and input size % dilation != 0
  // NOTE: every warp role has to execute this before using the mask!!
  template <class ProblemShape, class BlkCoord, class QKVShape, class Dilation>
  CUTLASS_DEVICE auto correct_qkv_shape(
      ProblemShape const& problem_shape,
      QKVShape const& qkv_shape, // this is pre-padding, pre-token permute, just
                                 // the original shape of the sequence mode in
                                 // the self attention
      BlkCoord const& blk_coord,
      Dilation const& dilation,
      int num_heads_actual) {
    auto head_idx = get<2, 0>(blk_coord);

    auto dilation_group_idx = head_idx / num_heads_actual;
    auto dilation_group_crd = idx2crd(dilation_group_idx, dilation);

    return correct_qkv_shape_wrt_dilation(
        qkv_shape, dilation, dilation_group_crd);
  }

  template <
      class BlkCoord,
      class MultiDimTileShape,
      class QKVShape,
      class NAParams>
  CUTLASS_DEVICE auto get_trip_count(
      BlkCoord const& blk_coord,
      MultiDimTileShape const& multi_dim_tile_shapes,
      QKVShape const& q_shape,
      QKVShape const& qkv_shape,
      NAParams const& na_params) {
    auto [q_tile_shape, kv_tile_shape] = multi_dim_tile_shapes;

    auto [window_size, window_left, window_right, stride] = na_params;

    auto q_tiled = ceil_div(q_shape, q_tile_shape);

    // Map query index back to the "original" coord
    auto q_tile_coord = idx2crd(static_cast<int>(get<0>(blk_coord)), q_tiled);
    auto q_coord = tuple_mul(q_tile_coord, q_tile_shape);

    auto q_tile_offset_last = idx2crd(size(q_tile_shape) - 1, q_tile_shape);
    auto q_coord_last = tuple_add(q_coord, q_tile_offset_last);

    auto kv_start_actual = get_window_start<Causal>(
        q_coord, window_left, window_right, stride, qkv_shape);

    auto last_kv_start_actual = get_window_start<Causal>(
        q_coord_last, window_left, window_right, stride, qkv_shape);
    auto kv_end_actual = get_window_end<Causal>(
        q_coord_last, last_kv_start_actual, window_size, qkv_shape);

    auto kv_start = floor_tuple(kv_start_actual, kv_tile_shape);
    auto kv_end = ceil_tuple(kv_end_actual, kv_tile_shape);

    auto kv_diff = tuple_sub(kv_end, kv_start);
    auto kv_diff_tiles = ceil_div(kv_diff, kv_tile_shape);

    return make_tuple(kv_start, kv_diff_tiles);
  }

  template <class MultiDimTileShape>
  CUTLASS_DEVICE auto get_extra_trip_count(
      MultiDimTileShape const& multi_dim_tile_shapes,
      int num_extra_kv) {
    auto [q_tile_shape, kv_tile_shape] = multi_dim_tile_shapes;

    int num_unmasked_trips = num_extra_kv / size(kv_tile_shape);
    bool has_masked_trips = num_extra_kv % size(kv_tile_shape) != 0;

    return make_tuple(num_unmasked_trips, has_masked_trips);
  }

  template <class MultiDimTileShape>
  CUTLASS_DEVICE auto get_extra_trip_info(
      MultiDimTileShape const& multi_dim_tile_shapes,
      int num_extra_kv) {
    auto [q_tile_shape, kv_tile_shape] = multi_dim_tile_shapes;

    auto [num_trips, has_masked_trips] =
        get_extra_trip_count(multi_dim_tile_shapes, num_extra_kv);
    if (has_masked_trips) {
      num_trips++;
    }

    return num_trips;
  }

  template <
      class AccQK,
      class IndexQK,
      class MultiDimTileShape,
      class QKVShape,
      class NAParams>
  CUTLASS_DEVICE void apply_mask(
      AccQK& acc_qk,
      IndexQK const& index_qk,
      MultiDimTileShape const& multi_dim_tile_shapes,
      QKVShape const& q_shape,
      QKVShape const& qkv_shape,
      NAParams const& na_params,
      QKVShape const& blk_kv_offset,
      QKVShape const& kv_diff_tiles) {
    auto [q_tile_shape, kv_tile_shape] = multi_dim_tile_shapes;
    auto [window_size, window_left, window_right, stride] = na_params;

    auto q_tiled = ceil_div(q_shape, q_tile_shape);

    // NOTE: the following assumes each thread owns exactly 1 row, and
    // contiguous columns. This is an assumption made in the FMHA fwd kernel. It
    // also assumes the first kv_idx always evenly divides KV tile --size--.
    // Otherwise we'd need a modulo like we have for q_idx.
    auto [q_idx, kv_idx] = index_qk(0);

    int q_tile_idx = q_idx / size(q_tile_shape);
    int q_tile_res = q_idx % size(q_tile_shape);

    auto q_tile_coord = idx2crd(q_tile_idx, q_tiled);
    auto q_tile_offset = idx2crd(q_tile_res, q_tile_shape);
    auto q_coord =
        tuple_add(tuple_mul(q_tile_coord, q_tile_shape), q_tile_offset);

    auto kv_start = get_window_start<Causal>(
        q_coord, window_left, window_right, stride, qkv_shape);
    auto kv_end =
        get_window_end<Causal>(q_coord, kv_start, window_size, qkv_shape);

    int kv_tile_idx = kv_idx / size(kv_tile_shape);

    auto kv_tile_coord = idx2crd(kv_tile_idx, kv_diff_tiles);
    auto kv_tile_offset =
        tuple_add(blk_kv_offset, tuple_mul(kv_tile_coord, kv_tile_shape));

    auto kv_ctr = make_identity_tensor(kv_tile_shape);
    auto kv_ctr_offset = domain_offset(kv_tile_offset, kv_ctr);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      auto kv_coord = kv_ctr_offset(i);

      if (not is_neighbor(kv_coord, kv_start, kv_end)) {
        acc_qk(i) = -INFINITY;
      }
    }
  }

  template <
      class AccQK,
      class IndexQK,
      class MultiDimTileShape,
      class QKVShape,
      class NAParams>
  CUTLASS_DEVICE void apply_padded_mask(
      AccQK& acc_qk,
      IndexQK const& index_qk,
      MultiDimTileShape const& multi_dim_tile_shapes,
      QKVShape const& qkv_shape,
      NAParams const& na_params,
      QKVShape const& blk_kv_offset,
      QKVShape const& kv_diff_tiles) {
    auto [q_tile_shape, kv_tile_shape] = multi_dim_tile_shapes;

    // NOTE: the following assumes each contiguous columns.
    // This is an assumption made in the FMHA fwd kernel.
    // It also assumes the first kv_idx always evenly divides KV tile --size--.
    // Otherwise we'd need a modulo like we have for q_idx.
    auto [q_idx, kv_idx] = index_qk(0);

    int kv_tile_idx = kv_idx / size(kv_tile_shape);

    auto kv_tile_coord = idx2crd(kv_tile_idx, kv_diff_tiles);
    auto kv_tile_offset =
        tuple_add(blk_kv_offset, tuple_mul(kv_tile_coord, kv_tile_shape));

    auto kv_ctr = make_identity_tensor(kv_tile_shape);
    auto kv_ctr_offset = domain_offset(kv_tile_offset, kv_ctr);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      auto kv_coord = kv_ctr_offset(i);

      if (not is_within_bounds(kv_coord, qkv_shape)) {
        acc_qk(i) = -INFINITY;
      }
    }
  }

  template <class AccQK, class IndexQK>
  CUTLASS_DEVICE void apply_extra_kv_mask(
      AccQK& acc_qk,
      IndexQK const& index_qk,
      int num_extra_kv) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      if (get<1>(index_qk(i)) >= num_extra_kv) {
        acc_qk(i) = -INFINITY;
      }
    }
  }
};

// Misc
template <class NADim>
CUTE_HOST_DEVICE constexpr auto get_window_left(NADim const& window_size) {
  return transform_leaf(window_size, [&](auto const& w) { return w / 2; });
}

template <class NADim>
CUTE_HOST_DEVICE constexpr auto get_window_right(NADim const& window_size) {
  return transform_leaf(
      window_size, [&](auto const& w) { return (w / 2) + ((w % 2) - 1); });
}

CUTE_HOST_DEVICE
bool is_fully_block_sparse_int(
    int input_size,
    int window_size,
    int stride,
    int q_tile_size,
    int kv_tile_size) {
  int num_stride_groups = ceil_div(input_size, stride);
  int window_left = window_size / 2;
  int window_right = (window_size / 2) + ((window_size % 2) - 1);
  int stride_group_center = stride / 2;

  int last_stride_group_center =
      min((input_size / stride) * stride + stride_group_center, input_size - 1);
  int last_stride_group_window_start = get_win_start<false>(
      last_stride_group_center, window_left, window_right, stride, input_size);

  return input_size == window_size ||
      (stride % q_tile_size == 0 && window_size % kv_tile_size == 0 &&
       (input_size % stride == 0 ||
        last_stride_group_window_start % kv_tile_size == 0) &&
       (((window_left - stride_group_center) % kv_tile_size == 0) ||
        (
            // input_size % stride == 0 &&
            input_size % kv_tile_size == 0 && num_stride_groups == 2 &&
            stride + stride_group_center >= input_size - window_size)));
}

template <class Causal, class NADim, class QTile, class KVTile>
CUTE_HOST_DEVICE bool fully_block_sparse(
    NADim input_size,
    NADim window_size,
    NADim stride,
    QTile q_tile_shape,
    KVTile kv_tile_shape) {
#ifdef NATTEN_DISABLE_FULLY_BLOCK_SPARSE_FAST_PATH
  return false;
#else
  // Causal masking can never be fully block-sparse (unless tile/block size is
  // 1)
  // TODO?
  if (cute::any_of(Causal{}, [&](auto const& a) { return a; })) {
    return false;
  }

  static_assert(rank(input_size) > 0 && rank(input_size) < 4);
  static_assert(rank(q_tile_shape) == rank(input_size));
  static_assert(rank(kv_tile_shape) == rank(input_size));

  if constexpr (rank(input_size) == 1) {
    return is_fully_block_sparse_int(
        get<0>(input_size),
        get<0>(window_size),
        get<0>(stride),
        get<0>(q_tile_shape),
        get<0>(kv_tile_shape));
  } else if constexpr (rank(input_size) == 2) {
    return is_fully_block_sparse_int(
               get<0>(input_size),
               get<0>(window_size),
               get<0>(stride),
               get<0>(q_tile_shape),
               get<0>(kv_tile_shape)) &&
        is_fully_block_sparse_int(
               get<1>(input_size),
               get<1>(window_size),
               get<1>(stride),
               get<1>(q_tile_shape),
               get<1>(kv_tile_shape));
  } else {
    return is_fully_block_sparse_int(
               get<0>(input_size),
               get<0>(window_size),
               get<0>(stride),
               get<0>(q_tile_shape),
               get<0>(kv_tile_shape)) &&
        is_fully_block_sparse_int(
               get<1>(input_size),
               get<1>(window_size),
               get<1>(stride),
               get<1>(q_tile_shape),
               get<1>(kv_tile_shape)) &&
        is_fully_block_sparse_int(
               get<2>(input_size),
               get<2>(window_size),
               get<2>(stride),
               get<2>(q_tile_shape),
               get<2>(kv_tile_shape));
  }
#endif
}

template <class NADim>
CUTE_HOST_DEVICE bool is_dilated(NADim dilation) {
  static_assert(rank(dilation) > 0 && rank(dilation) < 4);

  if constexpr (rank(dilation) == 1) {
    return get<0>(dilation) != 1;
  } else if constexpr (rank(dilation) == 2) {
    return get<0>(dilation) != 1 || get<1>(dilation) != 1;
  } else {
    return get<0>(dilation) != 1 || get<1>(dilation) != 1 ||
        get<2>(dilation) != 1;
  }
}

} // namespace cutlass::fna::collective
