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

#include "natten/cuda/fna_blackwell/collective/fna_fusion.hpp"

namespace cutlass::fna::collective {

using namespace cute;

// BWD

// BWD mask
template <bool IsCausal>
CUTE_HOST_DEVICE int get_bwd_win_start(
    int index,
    int stride_group_offset,
    int window_left,
    int window_right,
    int window_size,
    int stride,
    int length) {
  if constexpr (IsCausal) {
    return index;
  } else {
    auto window_start_pre_trunc =
        (index >= window_size) * (index - window_right + stride_group_offset);
    return (window_start_pre_trunc / stride) * stride;
  }
}

template <bool IsCausal>
CUTE_HOST_DEVICE int get_bwd_win_end(
    int index,
    int stride_group_offset,
    int window_left,
    int window_right,
    int window_size,
    int stride,
    int length) {
  if constexpr (IsCausal) {
    // Window end is always the last query that attends to this key/value + 1
    // In the strided case, it will always be the last query in a stride
    // group + 1, which means it will always be a multiple of stride.
    // In the case of incomplete stridegroups (input size % stride != 0),
    // we must correct backward window end by incrementing first non-attending
    // stridegroup by 1.
    auto first_non_attending_stride_group_idx =
        (index + window_size) / stride + (index + window_size >= length);
    return cutlass::fast_min(
        first_non_attending_stride_group_idx * stride, length);

  } else {
    auto bound_check = (index >= (length - window_size));
    auto window_end =
        ((index + window_left + stride_group_offset + 1) / stride) * stride;
    return (bound_check * length) + ((1 - bound_check) * window_end);
  }
}

template <class Causal, class NADim>
CUTE_HOST_DEVICE auto get_bwd_window_start(
    NADim index,
    NADim stride_group_offset,
    NADim window_left,
    NADim window_right,
    NADim window_size,
    NADim stride,
    NADim length) {
  static_assert(rank(index) > 0 && rank(index) < 4);
  static_assert(rank(index) == rank(Causal{}));
  if constexpr (rank(index) == 1) {
    return make_tuple(get_bwd_win_start<get<0>(Causal{})>(
        get<0>(index),
        get<0>(stride_group_offset),
        get<0>(window_left),
        get<0>(window_right),
        get<0>(window_size),
        get<0>(stride),
        get<0>(length)));
  } else if constexpr (rank(index) == 2) {
    return make_tuple(
        get_bwd_win_start<get<0>(Causal{})>(
            get<0>(index),
            get<0>(stride_group_offset),
            get<0>(window_left),
            get<0>(window_right),
            get<0>(window_size),
            get<0>(stride),
            get<0>(length)),
        get_bwd_win_start<get<1>(Causal{})>(
            get<1>(index),
            get<1>(stride_group_offset),
            get<1>(window_left),
            get<1>(window_right),
            get<1>(window_size),
            get<1>(stride),
            get<1>(length)));
  } else {
    return make_tuple(
        get_bwd_win_start<get<0>(Causal{})>(
            get<0>(index),
            get<0>(stride_group_offset),
            get<0>(window_left),
            get<0>(window_right),
            get<0>(window_size),
            get<0>(stride),
            get<0>(length)),
        get_bwd_win_start<get<1>(Causal{})>(
            get<1>(index),
            get<1>(stride_group_offset),
            get<1>(window_left),
            get<1>(window_right),
            get<1>(window_size),
            get<1>(stride),
            get<1>(length)),
        get_bwd_win_start<get<2>(Causal{})>(
            get<2>(index),
            get<2>(stride_group_offset),
            get<2>(window_left),
            get<2>(window_right),
            get<2>(window_size),
            get<2>(stride),
            get<2>(length)));
  }
}

template <class Causal, class NADim>
CUTE_HOST_DEVICE auto get_bwd_window_end(
    NADim index,
    NADim stride_group_offset,
    NADim window_left,
    NADim window_right,
    NADim window_size,
    NADim stride,
    NADim length) {
  static_assert(rank(index) > 0 && rank(index) < 4);
  static_assert(rank(index) == rank(Causal{}));
  if constexpr (rank(index) == 1) {
    return make_tuple(get_bwd_win_end<get<0>(Causal{})>(
        get<0>(index),
        get<0>(stride_group_offset),
        get<0>(window_left),
        get<0>(window_right),
        get<0>(window_size),
        get<0>(stride),
        get<0>(length)));
  } else if constexpr (rank(index) == 2) {
    return make_tuple(
        get_bwd_win_end<get<0>(Causal{})>(
            get<0>(index),
            get<0>(stride_group_offset),
            get<0>(window_left),
            get<0>(window_right),
            get<0>(window_size),
            get<0>(stride),
            get<0>(length)),
        get_bwd_win_end<get<1>(Causal{})>(
            get<1>(index),
            get<1>(stride_group_offset),
            get<1>(window_left),
            get<1>(window_right),
            get<1>(window_size),
            get<1>(stride),
            get<1>(length)));
  } else {
    return make_tuple(
        get_bwd_win_end<get<0>(Causal{})>(
            get<0>(index),
            get<0>(stride_group_offset),
            get<0>(window_left),
            get<0>(window_right),
            get<0>(window_size),
            get<0>(stride),
            get<0>(length)),
        get_bwd_win_end<get<1>(Causal{})>(
            get<1>(index),
            get<1>(stride_group_offset),
            get<1>(window_left),
            get<1>(window_right),
            get<1>(window_size),
            get<1>(stride),
            get<1>(length)),
        get_bwd_win_end<get<2>(Causal{})>(
            get<2>(index),
            get<2>(stride_group_offset),
            get<2>(window_left),
            get<2>(window_right),
            get<2>(window_size),
            get<2>(stride),
            get<2>(length)));
  }
}

// Backward pass mask
// Does not allow extra KV fusion
template <class Causal_>
struct NeighborhoodAttentionBackwardMask {
  using Causal = Causal_;
  static_assert(rank(Causal{}) >= 1 && rank(Causal{}) < 4);

  // Identical to NeighborhoodAttentionMask::correct_qkv_shape
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

  // Unlike in forward pass, trip counts are over Q tiles and not KV tiles.
  template <
      class BlkCoord,
      class MultiDimTileShape,
      class QKVShape,
      class NAParams>
  CUTLASS_DEVICE auto get_trip_count(
      BlkCoord const& blk_coord,
      MultiDimTileShape const& multi_dim_tile_shapes,
      QKVShape const& kv_shape,
      QKVShape const& qkv_shape,
      NAParams const& na_params) {
    auto [q_tile_shape, kv_tile_shape] = multi_dim_tile_shapes;

    auto [window_size, window_left, window_right, stride, stride_group_offset] =
        na_params;

    auto kv_tiled = ceil_div(kv_shape, kv_tile_shape);

    // Map KV index back to coord
    auto kv_tile_coord = idx2crd(static_cast<int>(get<1>(blk_coord)), kv_tiled);
    auto kv_coord = tuple_mul(kv_tile_coord, kv_tile_shape);

    auto kv_tile_offset_last = idx2crd(size(kv_tile_shape) - 1, kv_tile_shape);
    auto kv_coord_last = tuple_add(kv_coord, kv_tile_offset_last);

    // q start and end instead of kv like in forward pass
    auto q_start_actual = get_bwd_window_start<Causal>(
        kv_coord,
        stride_group_offset,
        window_left,
        window_right,
        window_size,
        stride,
        qkv_shape);

    auto last_q_start_actual = get_bwd_window_start<Causal>(
        kv_coord_last,
        stride_group_offset,
        window_left,
        window_right,
        window_size,
        stride,
        qkv_shape);
    auto q_end_actual = get_bwd_window_end<Causal>(
        kv_coord_last,
        stride_group_offset,
        window_left,
        window_right,
        window_size,
        stride,
        qkv_shape);

    auto q_start = floor_tuple(q_start_actual, q_tile_shape);
    auto q_end = ceil_tuple(q_end_actual, q_tile_shape);

    auto q_diff = tuple_sub(q_end, q_start);
    auto q_diff_tiles = ceil_div(q_diff, q_tile_shape);

    return make_tuple(q_start, q_diff_tiles);
  }

  template <
      class AccQK,
      class IndexQ,
      class IndexQK,
      class MultiDimTileShape,
      class QKVShape,
      class NAParams>
  CUTLASS_DEVICE void apply_mask(
      AccQK& acc_qk,
      IndexQ const& index_q,
      IndexQK const& index_qk,
      MultiDimTileShape const& multi_dim_tile_shapes,
      QKVShape const& kv_shape,
      QKVShape const& qkv_shape,
      NAParams const& na_params,
      QKVShape const& blk_q_offset,
      QKVShape const& q_diff_tiles) {
    auto [q_tile_shape, kv_tile_shape] = multi_dim_tile_shapes;
    auto [window_size, window_left, window_right, stride, stride_group_offset] =
        na_params;

    auto kv_tiled = ceil_div(kv_shape, kv_tile_shape);

    // NOTE: Unlike the forward pass, there is no guarantee that each thread
    // owns entire rows, and starts with the first dot product in the tile
    // (which eliminated the need to do an extra modulo). However, each thread
    // visits only one row (one key, many queries), which simplifies things
    // again like in the forward pass, compared to Hopper. We can just compute
    // window start and end once, and just do bound checks for each dot product.
    auto [q_idx, kv_idx] = index_qk(0);

    int kv_tile_idx = kv_idx / size(kv_tile_shape);
    int kv_tile_res = kv_idx % size(kv_tile_shape);

    auto kv_tile_coord = idx2crd(kv_tile_idx, kv_tiled);
    auto kv_tile_offset = idx2crd(kv_tile_res, kv_tile_shape);
    auto kv_coord =
        tuple_add(tuple_mul(kv_tile_coord, kv_tile_shape), kv_tile_offset);

    auto q_start = get_bwd_window_start<Causal>(
        kv_coord,
        stride_group_offset,
        window_left,
        window_right,
        window_size,
        stride,
        qkv_shape);
    auto q_end = get_bwd_window_end<Causal>(
        kv_coord,
        stride_group_offset,
        window_left,
        window_right,
        window_size,
        stride,
        qkv_shape);

    int q_tile_idx = q_idx / size(q_tile_shape);
    int q_tile_res = q_idx % size(q_tile_shape);

    auto q_tile_coord = idx2crd(q_tile_idx, q_diff_tiles);
    auto q_iter_offset = idx2crd(q_tile_res, q_tile_shape);
    auto q_tile_offset = tuple_add(
        q_iter_offset,
        tuple_add(blk_q_offset, tuple_mul(q_tile_coord, q_tile_shape)));

    auto q_ctr = make_identity_tensor(q_tile_shape);
    auto q_ctr_offset = domain_offset(q_tile_offset, q_ctr);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      auto q_coord = q_ctr_offset(index_q(i) - q_idx);

      if (not is_neighbor(q_coord, q_start, q_end)) {
        acc_qk(i) = -INFINITY;
      }
    }
  }

  template <
      class AccQK,
      class IndexQ,
      class MultiDimTileShape,
      class QKVShape,
      class NAParams>
  CUTLASS_DEVICE void apply_padded_mask(
      AccQK& acc_qk,
      IndexQ const& index_q,
      MultiDimTileShape const& multi_dim_tile_shapes,
      QKVShape const& qkv_shape,
      NAParams const& na_params,
      QKVShape const& blk_q_offset,
      QKVShape const& q_diff_tiles) {
    auto [q_tile_shape, kv_tile_shape] = multi_dim_tile_shapes;

    // NOTE: Unlike the forward pass, there is no guarantee that each thread
    // owns entire rows, and starts with the first dot product in the tile
    // (which eliminated the need to do an extra modulo). However, each thread
    // visits only one row (one key, many queries), which simplifies things
    // again like in the forward pass, compared to Hopper. We can just compute
    // window start and end once, and just do bound checks for each dot product.
    auto q_idx = index_q(0);

    int q_tile_idx = q_idx / size(q_tile_shape);
    int q_tile_res = q_idx % size(q_tile_shape);

    auto q_tile_coord = idx2crd(q_tile_idx, q_diff_tiles);
    auto q_iter_offset = idx2crd(q_tile_res, q_tile_shape);
    auto q_tile_offset = tuple_add(
        q_iter_offset,
        tuple_add(blk_q_offset, tuple_mul(q_tile_coord, q_tile_shape)));

    auto q_ctr = make_identity_tensor(q_tile_shape);
    auto q_ctr_offset = domain_offset(q_tile_offset, q_ctr);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(acc_qk); i++) {
      auto q_coord = q_ctr_offset(index_q(i) - q_idx);

      if (not is_within_bounds(q_coord, qkv_shape)) {
        acc_qk(i) = -INFINITY;
      }
    }
  }
};

// Misc

template <class NADim>
CUTE_HOST_DEVICE constexpr auto get_bwd_stride_offset(NADim const& stride) {
  return transform_leaf(
      stride, [&](auto const& s) { return (s - (s / 2) - 1); });
}

} // namespace cutlass::fna::collective
