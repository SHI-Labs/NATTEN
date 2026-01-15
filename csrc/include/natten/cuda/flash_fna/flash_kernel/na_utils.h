#pragma once
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

namespace natten {
namespace cuda {
namespace flash_fna {

using namespace cute;

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

template <class Causal, class Coord, class NADim>
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

template <class Causal, class NADim, class Coord>
CUTE_HOST_DEVICE auto get_bwd_window_start(
    Coord index,
    NADim stride_group_offset,
    NADim window_left,
    NADim window_right,
    NADim window_size,
    NADim stride,
    NADim length) {
  static_assert(rank(index) > 0 && rank(index) < 4);
  static_assert(rank(index) == rank(Causal{}));
  static_assert(rank(index) == rank(length));
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

template <class Causal, class NADim, class Coord>
CUTE_HOST_DEVICE auto get_bwd_window_end(
    Coord index,
    NADim stride_group_offset,
    NADim window_left,
    NADim window_right,
    NADim window_size,
    NADim stride,
    NADim length) {
  static_assert(rank(index) > 0 && rank(index) < 4);
  static_assert(rank(index) == rank(Causal{}));
  static_assert(rank(index) == rank(length));
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

template <class NADim>
CUTE_HOST_DEVICE auto correct_qkv_shape_wrt_dilation(
    NADim qkv_shape,
    NADim dilation,
    NADim dilation_group) {
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

template <class NADim>
CUTLASS_DEVICE auto correct_qkv_shape(
    NADim const& qkv_shape, // this is pre-padding, pre-token permute, just
                               // the original shape of the sequence mode in
                               // the self attention
    int head_idx,
    NADim const& dilation,
    int num_heads_actual) {

  auto dilation_group_idx = head_idx / num_heads_actual;
  auto dilation_group_crd = idx2crd(dilation_group_idx, dilation);

  return correct_qkv_shape_wrt_dilation(
      qkv_shape, dilation, dilation_group_crd);
}

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

template <class NADim>
CUTE_HOST_DEVICE constexpr auto get_bwd_stride_offset(NADim const& stride) {
  return transform_leaf(
      stride, [&](auto const& s) { return (s - (s / 2) - 1); });
}

} // namespace flash_fna
} // namespace cuda
} // namespace natten
