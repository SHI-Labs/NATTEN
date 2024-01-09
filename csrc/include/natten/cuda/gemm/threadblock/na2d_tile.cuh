/***************************************************************************************************
 * Copyright (c) 2022-2024 Ali Hassani.
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
/*! \file
    \brief 2D NA ``Tile`` information.
*/

#pragma once

#include <cutlass/cutlass.h>

#include "natten/cuda/gemm/na2d_problem_size.cuh"
#include "natten/cuda/gemm/neighborhood_attention.cuh"

namespace natten {
namespace cuda {
namespace gemm {
namespace threadblock {

/// Parameters structure
struct NA2dTileInfoParams {
  cutlass::MatrixCoord hw_tiled_shape;
  int block_x_multiplier;
  int out_stride_0, out_stride_1, out_stride_2, out_stride_3;
  bool has_bias;

  CUTLASS_HOST_DEVICE
  NA2dTileInfoParams() {}

  CUTLASS_HOST_DEVICE
  NA2dTileInfoParams(
      cutlass::MatrixCoord hw_tiled_shape,
      int block_x_multiplier,
      int out_stride_0,
      int out_stride_1,
      int out_stride_2,
      int out_stride_3,
      bool has_bias)
      : hw_tiled_shape(hw_tiled_shape),
        block_x_multiplier(block_x_multiplier),
        has_bias(has_bias),
        out_stride_0(out_stride_0),
        out_stride_1(out_stride_1),
        out_stride_2(out_stride_2),
        out_stride_3(out_stride_3) {}
};

template <typename NAShape_, Operator NAOperator>
struct NA2dTileInfo;

template <typename NAShape_>
struct NA2dTileInfo<NAShape_, Operator::kPN> {
  using NAShape = NAShape_;

  int gemm_m_offset, tile_h_length, tile_w_length, out_offset, bias_offset;
  int b, n;
  int h, w, h_ext, w_ext, h_abs, w_abs, h_ext_abs, w_ext_abs;
  bool valid_batch;

 public:
  CUTLASS_HOST_DEVICE
  NA2dTileInfo(
      NA2dTileInfoParams const& params,
      NA2dProblemSize const& problem_size) {
    gemm_m_offset = blockIdx.x % params.block_x_multiplier;
    int na_tile_idx = blockIdx.x / params.block_x_multiplier;
    int offset_bn_ = na_tile_idx /
        (params.hw_tiled_shape.row() * params.hw_tiled_shape.column());
    int offset_hw_ = na_tile_idx %
        (params.hw_tiled_shape.row() * params.hw_tiled_shape.column());
    int cta_x = (offset_hw_ / params.hw_tiled_shape.column());
    int cta_y = (offset_hw_ % params.hw_tiled_shape.column());
    n = offset_bn_ % problem_size.N;
    b = offset_bn_ / problem_size.N;

    int cta_x_mod_di = cta_x % problem_size.dilation_h;
    int cta_y_mod_di = cta_y % problem_size.dilation_w;

    tile_h_length = problem_size.Hdiv + (cta_x_mod_di < problem_size.Hmod);
    tile_w_length = problem_size.Wdiv + (cta_y_mod_di < problem_size.Wmod);
    h = int(cta_x / problem_size.dilation_h) * NAShape::kTile;
    w = int(cta_y / problem_size.dilation_w) * NAShape::kTile;

    h_ext = window_start(h, tile_h_length, problem_size.R, problem_size.RH);
    w_ext = window_start(w, tile_w_length, problem_size.S, problem_size.SH);

    h_ext_abs = h_ext * problem_size.dilation_h + cta_x_mod_di;
    w_ext_abs = w_ext * problem_size.dilation_w + cta_y_mod_di;
    h_abs = h * problem_size.dilation_h + cta_x_mod_di;
    w_abs = w * problem_size.dilation_w + cta_y_mod_di;

    // TODO: check height and width coordinates here as well.
    // Because of the extended tiles, it won't be the same as NN and IN.
    valid_batch = b < problem_size.B && n < problem_size.N && b >= 0 && n >= 0;

    out_offset = b * (params.out_stride_0) + n * (params.out_stride_1) +
        h_ext_abs * (params.out_stride_2) + w_ext_abs * (params.out_stride_3);
    bias_offset =
        (params.has_bias) ? n * (problem_size.RPB_R * problem_size.RPB_S) : 0;
  }

  // Returns the neighborhood start coordinate
  CUTLASS_HOST_DEVICE
  int window_start(
      const int index,
      const int length,
      const int window_size,
      const int window_radius) const {
    return cutlass::fast_max<int>(index - window_radius, 0) +
        (index + window_radius >= length) *
        (length - index - window_radius - 1);
  }
};

template <typename NAShape_>
struct NA2dTileInfo<NAShape_, Operator::kNN> {
  using NAShape = NAShape_;

  int gemm_m_offset, tile_h_length, tile_w_length, out_offset, bias_offset;
  int b, n;
  int h, w, h_ext, w_ext, h_abs, w_abs, h_ext_abs, w_ext_abs;
  bool valid_batch;

 public:
  CUTLASS_HOST_DEVICE
  NA2dTileInfo(
      NA2dTileInfoParams const& params,
      NA2dProblemSize const& problem_size) {
    gemm_m_offset = blockIdx.x % params.block_x_multiplier;
    int na_tile_idx = blockIdx.x / params.block_x_multiplier;
    int offset_bn_ = na_tile_idx /
        (params.hw_tiled_shape.row() * params.hw_tiled_shape.column());
    int offset_hw_ = na_tile_idx %
        (params.hw_tiled_shape.row() * params.hw_tiled_shape.column());
    int cta_x = (offset_hw_ / params.hw_tiled_shape.column());
    int cta_y = (offset_hw_ % params.hw_tiled_shape.column());
    n = offset_bn_ % problem_size.N;
    b = offset_bn_ / problem_size.N;

    int cta_x_mod_di = cta_x % problem_size.dilation_h;
    int cta_y_mod_di = cta_y % problem_size.dilation_w;

    tile_h_length = problem_size.Hdiv + (cta_x_mod_di < problem_size.Hmod);
    tile_w_length = problem_size.Wdiv + (cta_y_mod_di < problem_size.Wmod);

    h = int(cta_x / problem_size.dilation_h) * NAShape::kTile;
    w = int(cta_y / problem_size.dilation_w) * NAShape::kTile;

    h_ext = window_start(h, tile_h_length, problem_size.R, problem_size.RH);
    w_ext = window_start(w, tile_w_length, problem_size.S, problem_size.SH);

    h_ext_abs = h_ext * problem_size.dilation_h + cta_x_mod_di;
    w_ext_abs = w_ext * problem_size.dilation_w + cta_y_mod_di;
    h_abs = h * problem_size.dilation_h + cta_x_mod_di;
    w_abs = w * problem_size.dilation_w + cta_y_mod_di;

    valid_batch = b < problem_size.B && n < problem_size.N && b >= 0 && n >= 0;

    out_offset = b * (params.out_stride_0) + n * (params.out_stride_1) +
        h_abs * (params.out_stride_2) + w_abs * (params.out_stride_3);
    bias_offset = 0;
  }

  // Returns the neighborhood start coordinate
  CUTLASS_HOST_DEVICE
  int window_start(
      const int index,
      const int length,
      const int window_size,
      const int window_radius) const {
    return cutlass::fast_max<int>(index - window_radius, 0) +
        (index + window_radius >= length) *
        (length - index - window_radius - 1);
  }
};

template <typename NAShape_>
struct NA2dTileInfo<NAShape_, Operator::kIN> {
  using NAShape = NAShape_;

  int gemm_m_offset, tile_h_length, tile_w_length, out_offset, bias_offset;
  int b, n;
  int h, w, h_ext, w_ext, h_abs, w_abs, h_ext_abs, w_ext_abs;
  bool valid_batch;

 public:
  CUTLASS_HOST_DEVICE
  NA2dTileInfo(
      NA2dTileInfoParams const& params,
      NA2dProblemSize const& problem_size) {
    gemm_m_offset = blockIdx.x % params.block_x_multiplier;
    int na_tile_idx = blockIdx.x / params.block_x_multiplier;
    int offset_bn_ = na_tile_idx /
        (params.hw_tiled_shape.row() * params.hw_tiled_shape.column());
    int offset_hw_ = na_tile_idx %
        (params.hw_tiled_shape.row() * params.hw_tiled_shape.column());
    int cta_x = (offset_hw_ / params.hw_tiled_shape.column());
    int cta_y = (offset_hw_ % params.hw_tiled_shape.column());
    n = offset_bn_ % problem_size.N;
    b = offset_bn_ / problem_size.N;

    int cta_x_mod_di = cta_x % problem_size.dilation_h;
    int cta_y_mod_di = cta_y % problem_size.dilation_w;

    tile_h_length = problem_size.Hdiv + (cta_x_mod_di < problem_size.Hmod);
    tile_w_length = problem_size.Wdiv + (cta_y_mod_di < problem_size.Wmod);
    h = int(cta_x / problem_size.dilation_h) * NAShape::kTile;
    w = int(cta_y / problem_size.dilation_w) * NAShape::kTile;

    h_ext =
        window_inverse_start(h, tile_h_length, problem_size.R, problem_size.RH);
    w_ext =
        window_inverse_start(w, tile_w_length, problem_size.S, problem_size.SH);

    h_ext_abs = h_ext * problem_size.dilation_h + cta_x_mod_di;
    w_ext_abs = w_ext * problem_size.dilation_w + cta_y_mod_di;
    h_abs = h * problem_size.dilation_h + cta_x_mod_di;
    w_abs = w * problem_size.dilation_w + cta_y_mod_di;

    valid_batch = b < problem_size.B && n < problem_size.N && b >= 0 && n >= 0;

    out_offset = b * (params.out_stride_0) + n * (params.out_stride_1) +
        h_abs * (params.out_stride_2) + w_abs * (params.out_stride_3);
    bias_offset = 0;
  }

  // Returns the inverse neighborhood start coordinate
  CUTLASS_HOST_DEVICE
  int window_inverse_start(
      const int index,
      const int length,
      const int window_size,
      const int window_radius) const {
    return (index >= window_size) * (index - window_radius);
  }
};

} // namespace threadblock
} // namespace gemm
} // namespace cuda
} // namespace natten
