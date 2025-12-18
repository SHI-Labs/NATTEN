/***************************************************************************************************
 * Copyright (c) 2022-2025 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 **************************************************************************************************/

#pragma once

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>

#include <natten/cuda/utils/cutlass.cuh>

#include <natten/cuda/tokperm/utils/permute.cuh>
#include <natten/cuda/tokperm/utils/stride.cuh>

namespace natten::tokperm {

using namespace cute;
using namespace natten::tokperm::utils;

template <class CuteTuple>
CUTE_HOST_DEVICE constexpr auto make_token_permuted_layout(
    CuteTuple const& rest,
    CuteTuple const& tile_shape,
    CuteTuple const& dilation,
    int batch,
    int heads,
    int dim,
    bool flip_tiled_dims) {
  static_assert(tuple_size_v<CuteTuple> >= 1 && tuple_size_v<CuteTuple> <= 3);

  if constexpr (tuple_size_v<CuteTuple> == 1) {
    auto [R0] = rest;
    auto [T0] = tile_shape;
    auto [D0] = dilation;

    auto problem_shape_out =
        cute::make_tuple(batch, cute::make_tuple(R0, T0, D0), heads, dim);

    // flip_tiled_dims doesn't make a difference for 1D
    // (R0 T0 D0) -> (D0 R0 T0)
    auto problem_shape_out_view = permute_tokens<2, 0, 1>(problem_shape_out);

    auto stride_out_view =
        utils::make_torch_contiguous_stride(problem_shape_out_view);

    // (D0 R0 T0) -> (R0 T0 D0)
    auto stride_out = permute_tokens<1, 2, 0>(stride_out_view);

    return make_layout(problem_shape_out, stride_out);

  } else if constexpr (tuple_size_v<CuteTuple> == 2) {
    auto [R0, R1] = rest;
    auto [T0, T1] = tile_shape;
    auto [D0, D1] = dilation;

    auto problem_shape_out = cute::make_tuple(
        batch, cute::make_tuple(R0, T0, D0, R1, T1, D1), heads, dim);

    if (flip_tiled_dims) {
      // (R0 T0 D0 R1 T1 D1) -> (D1 D0 R1 R0 T1 T0)
      auto problem_shape_out_view =
          permute_tokens<5, 2, 3, 0, 4, 1>(problem_shape_out);

      auto stride_out_view =
          utils::make_torch_contiguous_stride(problem_shape_out_view);

      // (D1 D0 R1 R0 T1 T0) -> (R0 T0 D0 R1 T1 D1)
      auto stride_out = permute_tokens<3, 5, 1, 2, 4, 0>(stride_out_view);

      return make_layout(problem_shape_out, stride_out);

    } else {
      // (R0 T0 D0 R1 T1 D1) -> (D0 D1 R0 R1 T0 T1)
      auto problem_shape_out_view =
          permute_tokens<2, 5, 0, 3, 1, 4>(problem_shape_out);

      auto stride_out_view =
          utils::make_torch_contiguous_stride(problem_shape_out_view);

      // (D0 D1 R0 R1 T0 T1) -> (R0 T0 D0 R1 T1 D1)
      auto stride_out = permute_tokens<2, 4, 0, 3, 5, 1>(stride_out_view);

      return make_layout(problem_shape_out, stride_out);
    }

  } else if constexpr (tuple_size_v<CuteTuple> == 3) {
    auto [R0, R1, R2] = rest;
    auto [T0, T1, T2] = tile_shape;
    auto [D0, D1, D2] = dilation;

    auto problem_shape_out = cute::make_tuple(
        batch,
        cute::make_tuple(R0, T0, D0, R1, T1, D1, R2, T2, D2),
        heads,
        dim);

    if (flip_tiled_dims) {
      // (R0 T0 D0 R1 T1 D1 R2 T2 D2) -> (D2 D1 D0 R2 R1 R0 T2 T1 T0)
      auto problem_shape_out_view =
          permute_tokens<8, 5, 2, 6, 3, 0, 7, 4, 1>(problem_shape_out);

      auto stride_out_view =
          utils::make_torch_contiguous_stride(problem_shape_out_view);

      // (D2 D1 D0 R2 R1 R0 T2 T1 T0) -> (R0 T0 D0 R1 T1 D1 R2 T2 D2)
      auto stride_out =
          permute_tokens<5, 8, 2, 4, 7, 1, 3, 6, 0>(stride_out_view);

      return make_layout(problem_shape_out, stride_out);

    } else {
      // (R0 T0 D0 R1 T1 D1 R2 T2 D2) -> (D0 D1 D2 R0 R1 R2 T0 T1 T2)
      auto problem_shape_out_view =
          permute_tokens<2, 5, 8, 0, 3, 6, 1, 4, 7>(problem_shape_out);

      auto stride_out_view =
          utils::make_torch_contiguous_stride(problem_shape_out_view);

      // (D0 D1 D2 R0 R1 R2 T0 T1 T2) -> (R0 T0 D0 R1 T1 D1 R2 T2 D2)
      auto stride_out =
          permute_tokens<3, 6, 0, 4, 7, 1, 5, 8, 2>(stride_out_view);

      return make_layout(problem_shape_out, stride_out);
    }
  }
}

// Identical to make_token_permuted_layout, just without batch, and heads and
// head_dim are merged
template <class CuteTuple>
CUTE_HOST_DEVICE constexpr auto make_token_permuted_layout_varlen(
    CuteTuple const& rest,
    CuteTuple const& tile_shape,
    CuteTuple const& dilation,
    bool flip_tiled_dims,
    int heads_dims) {
  static_assert(tuple_size_v<CuteTuple> >= 1 && tuple_size_v<CuteTuple> <= 3);

  if constexpr (tuple_size_v<CuteTuple> == 1) {
    auto [R0] = rest;
    auto [T0] = tile_shape;
    auto [D0] = dilation;

    auto problem_shape_out =
        cute::make_tuple(cute::make_tuple(R0, T0, D0), heads_dims);

    // flip_tiled_dims doesn't make a difference for 1D
    // (R0 T0 D0) -> (D0 R0 T0)
    auto problem_shape_out_view =
        permute_tokens_varlen<2, 0, 1>(problem_shape_out);

    auto stride_out_view =
        utils::make_torch_contiguous_stride(problem_shape_out_view);

    // (D0 R0 T0) -> (R0 T0 D0)
    auto stride_out = permute_tokens_varlen<1, 2, 0>(stride_out_view);

    return make_layout(problem_shape_out, stride_out);

  } else if constexpr (tuple_size_v<CuteTuple> == 2) {
    auto [R0, R1] = rest;
    auto [T0, T1] = tile_shape;
    auto [D0, D1] = dilation;

    auto problem_shape_out =
        cute::make_tuple(cute::make_tuple(R0, T0, D0, R1, T1, D1), heads_dims);

    if (flip_tiled_dims) {
      // (R0 T0 D0 R1 T1 D1) -> (D1 D0 R1 R0 T1 T0)
      auto problem_shape_out_view =
          permute_tokens_varlen<5, 2, 3, 0, 4, 1>(problem_shape_out);

      auto stride_out_view =
          utils::make_torch_contiguous_stride(problem_shape_out_view);

      // (D1 D0 R1 R0 T1 T0) -> (R0 T0 D0 R1 T1 D1)
      auto stride_out =
          permute_tokens_varlen<3, 5, 1, 2, 4, 0>(stride_out_view);

      return make_layout(problem_shape_out, stride_out);

    } else {
      // (R0 T0 D0 R1 T1 D1) -> (D0 D1 R0 R1 T0 T1)
      auto problem_shape_out_view =
          permute_tokens_varlen<2, 5, 0, 3, 1, 4>(problem_shape_out);

      auto stride_out_view =
          utils::make_torch_contiguous_stride(problem_shape_out_view);

      // (D0 D1 R0 R1 T0 T1) -> (R0 T0 D0 R1 T1 D1)
      auto stride_out =
          permute_tokens_varlen<2, 4, 0, 3, 5, 1>(stride_out_view);

      return make_layout(problem_shape_out, stride_out);
    }

  } else if constexpr (tuple_size_v<CuteTuple> == 3) {
    auto [R0, R1, R2] = rest;
    auto [T0, T1, T2] = tile_shape;
    auto [D0, D1, D2] = dilation;

    auto problem_shape_out = cute::make_tuple(
        cute::make_tuple(R0, T0, D0, R1, T1, D1, R2, T2, D2), heads_dims);

    if (flip_tiled_dims) {
      // (R0 T0 D0 R1 T1 D1 R2 T2 D2) -> (D2 D1 D0 R2 R1 R0 T2 T1 T0)
      auto problem_shape_out_view =
          permute_tokens_varlen<8, 5, 2, 6, 3, 0, 7, 4, 1>(problem_shape_out);

      auto stride_out_view =
          utils::make_torch_contiguous_stride(problem_shape_out_view);

      // (D2 D1 D0 R2 R1 R0 T2 T1 T0) -> (R0 T0 D0 R1 T1 D1 R2 T2 D2)
      auto stride_out =
          permute_tokens_varlen<5, 8, 2, 4, 7, 1, 3, 6, 0>(stride_out_view);

      return make_layout(problem_shape_out, stride_out);

    } else {
      // (R0 T0 D0 R1 T1 D1 R2 T2 D2) -> (D0 D1 D2 R0 R1 R2 T0 T1 T2)
      auto problem_shape_out_view =
          permute_tokens_varlen<2, 5, 8, 0, 3, 6, 1, 4, 7>(problem_shape_out);

      auto stride_out_view =
          utils::make_torch_contiguous_stride(problem_shape_out_view);

      // (D0 D1 D2 R0 R1 R2 T0 T1 T2) -> (R0 T0 D0 R1 T1 D1 R2 T2 D2)
      auto stride_out =
          permute_tokens_varlen<3, 6, 0, 4, 7, 1, 5, 8, 2>(stride_out_view);

      return make_layout(problem_shape_out, stride_out);
    }
  }
}

} // namespace natten::tokperm
