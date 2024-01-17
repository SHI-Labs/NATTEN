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
    \brief Pointwise-Neighborhood CPU kernel for 3D data.
           Computes attention weights between query points and their
   corresponding key neighborhood. Extra kernel with fused bias (relative
   positional bias.)
*/

#pragma once
// TODO: these kernels should be independent of torch api.
// But for now, we do need vectorized reads.
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <torch/extension.h>
#include <vector>

#if defined(AVX_INT)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif

#include <natten/cpu/naive/natten_cpu_commons.h>
#include <natten/natten.h>

namespace natten {
namespace cpu {
namespace naive {

#define GRAIN_SIZE 0

template <typename scalar_t>
struct PointwiseNeighborhood3D {
  void operator()(
      bool is_grad,
      void* query_ptr,
      void* key_ptr,
      void* attn_ptr,
      int32_t batch_size,
      int32_t heads,
      int32_t depth,
      int32_t height,
      int32_t width,
      int32_t dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int64_t attn_stride_3,
      int64_t attn_stride_4,
      const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
      const std::tuple<int32_t, int32_t, int32_t>& dilation,
      const std::tuple<bool, bool, bool>& is_causal) {
    launch(
        is_grad,
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
        depth,
        height,
        width,
        heads,
        std::get<0>(kernel_size),
        std::get<1>(kernel_size),
        std::get<2>(kernel_size),
        std::get<0>(dilation),
        std::get<1>(dilation),
        std::get<2>(dilation),
        dim,
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        attn_stride_4,
        is_causal);
  }

  void launch( // QK    / A-grad
      bool is_grad,
      scalar_t* query, // query / d_out
      scalar_t* key, // key   / value
      scalar_t* attn, // attn  / d_attn
      int32_t depth,
      int32_t height,
      int32_t width,
      int32_t heads,
      int32_t kernel_size_0,
      int32_t kernel_size_1,
      int32_t kernel_size_2,
      int32_t dilation_0,
      int32_t dilation_1,
      int32_t dilation_2,
      int32_t dim,
      int32_t batch_size,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int64_t attn_stride_3,
      int64_t attn_stride_4,
      const std::tuple<bool, bool, bool>& is_causal) {
    auto is_causal_0 = std::get<0>(is_causal);
    auto is_causal_1 = std::get<1>(is_causal);
    auto is_causal_2 = std::get<2>(is_causal);
    auto neighborhood_size_0 = kernel_size_0 / 2;
    auto neighborhood_size_1 = kernel_size_1 / 2;
    auto neighborhood_size_2 = kernel_size_2 / 2;
    int64_t query_stride_4 = dim;
    int64_t query_stride_3 = width * query_stride_4;
    int64_t query_stride_2 = height * query_stride_3;
    int64_t query_stride_1 = depth * query_stride_2;
    int64_t query_stride_0 = heads * query_stride_1;
    auto mask_value = AttnMask<scalar_t>::value(is_grad);
    at::parallel_for(
        0,
        batch_size * heads * depth * height * width,
        GRAIN_SIZE,
        [&](int32_t start, int32_t end) {
          for (int32_t x = start; x < end; x++) {
            int32_t indtmp1 = x / width;
            auto j = x - indtmp1 * width;
            int32_t indtmp2 = indtmp1 / height;
            auto i = indtmp1 - indtmp2 * height;
            indtmp1 = indtmp2;
            indtmp2 = indtmp1 / depth;
            auto k = indtmp1 - indtmp2 * depth;
            indtmp1 = indtmp2;
            indtmp2 = indtmp1 / heads;
            auto h = indtmp1 - indtmp2 * heads;
            auto& b = indtmp2;
            auto nk = get_window_start(
                k,
                depth,
                kernel_size_0,
                neighborhood_size_0,
                dilation_0,
                is_causal_0);
            auto ni = get_window_start(
                i,
                height,
                kernel_size_1,
                neighborhood_size_1,
                dilation_1,
                is_causal_1);
            auto nj = get_window_start(
                j,
                width,
                kernel_size_2,
                neighborhood_size_2,
                dilation_2,
                is_causal_2);
            auto ek = get_window_end(
                k,
                nk,
                depth,
                kernel_size_0,
                neighborhood_size_0,
                dilation_0,
                is_causal_0);
            auto ei = get_window_end(
                i,
                ni,
                height,
                kernel_size_1,
                neighborhood_size_1,
                dilation_1,
                is_causal_1);
            auto ej = get_window_end(
                j,
                nj,
                width,
                kernel_size_2,
                neighborhood_size_2,
                dilation_2,
                is_causal_2);
            for (int32_t kk = 0; kk < kernel_size_0; kk++) {
              auto key_idx_k = kk * dilation_0 + nk;
              for (int32_t ki = 0; ki < kernel_size_1; ki++) {
                auto key_idx_i = ki * dilation_1 + ni;
                for (int32_t kj = 0; kj < kernel_size_2; kj++) {
                  auto key_idx_j = kj * dilation_2 + nj;
                  scalar_t updt = scalar_t(0);
                  int64_t batchHeadOffset =
                      b * query_stride_0 + h * query_stride_1;
                  int64_t queryOffset = batchHeadOffset + k * query_stride_2 +
                      i * query_stride_3 + j * query_stride_4;
                  if (key_idx_k < ek && key_idx_i < ei && key_idx_j < ej) {
                    int64_t keyOffset = batchHeadOffset +
                        key_idx_k * query_stride_2 +
                        key_idx_i * query_stride_3 + key_idx_j * query_stride_4;
                    for (int64_t dimOffset = 0; dimOffset < dim; ++dimOffset)
                      updt += query[queryOffset + dimOffset] *
                          key[keyOffset + dimOffset];
                  } else {
                    updt = mask_value;
                  }
                  int64_t index = b * attn_stride_0 + h * attn_stride_1 +
                      k * attn_stride_2 + i * attn_stride_3 +
                      j * attn_stride_4 + kk * (kernel_size_1 * kernel_size_2) +
                      ki * kernel_size_2 + kj;
                  attn[index] = updt;
                }
              }
            }
          }
        });
  }
};

template <typename scalar_t>
struct PointwiseNeighborhood3DWithBias {
  void operator()(
      void* query_ptr,
      void* key_ptr,
      void* bias_ptr,
      void* attn_ptr,
      int32_t batch_size,
      int32_t heads,
      int32_t depth,
      int32_t height,
      int32_t width,
      int32_t dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int64_t attn_stride_3,
      int64_t attn_stride_4,
      const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
      const std::tuple<int32_t, int32_t, int32_t>& dilation,
      const std::tuple<bool, bool, bool>& is_causal) {
    NATTEN_CHECK(
        !any_true(is_causal),
        "Neighborhood attention with causal masking does not support positional biases yet.");
    launch(
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(bias_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
        depth,
        height,
        width,
        heads,
        std::get<0>(kernel_size),
        std::get<1>(kernel_size),
        std::get<2>(kernel_size),
        std::get<0>(dilation),
        std::get<1>(dilation),
        std::get<2>(dilation),
        dim,
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        attn_stride_4);
  }

  void launch( // QK
      scalar_t* query, // query / d_out
      scalar_t* key, // key   / value
      scalar_t* bias, // relative positional bias tensor
      scalar_t* attn, // attn  / d_attn
      int32_t depth,
      int32_t height,
      int32_t width,
      int32_t heads,
      int32_t kernel_size_0,
      int32_t kernel_size_1,
      int32_t kernel_size_2,
      int32_t dilation_0,
      int32_t dilation_1,
      int32_t dilation_2,
      int32_t dim,
      int32_t batch_size,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int64_t attn_stride_3,
      int64_t attn_stride_4) {
    auto neighborhood_size_0 = kernel_size_0 / 2;
    auto neighborhood_size_1 = kernel_size_1 / 2;
    auto neighborhood_size_2 = kernel_size_2 / 2;
    int64_t bias_stride_2 = (2 * kernel_size_2 - 1);
    int64_t bias_stride_1 = (2 * kernel_size_1 - 1) * bias_stride_2;
    int64_t bias_stride_0 = (2 * kernel_size_0 - 1) * bias_stride_1;
    int64_t query_stride_4 = dim;
    int64_t query_stride_3 = width * query_stride_4;
    int64_t query_stride_2 = height * query_stride_3;
    int64_t query_stride_1 = depth * query_stride_2;
    int64_t query_stride_0 = heads * query_stride_1;
    at::parallel_for(
        0,
        batch_size * heads * depth * height * width,
        GRAIN_SIZE,
        [&](int32_t start, int32_t end) {
          for (int32_t x = start; x < end; x++) {
            int32_t indtmp1 = x / width;
            auto j = x - indtmp1 * width;
            int32_t indtmp2 = indtmp1 / height;
            auto i = indtmp1 - indtmp2 * height;
            indtmp1 = indtmp2;
            indtmp2 = indtmp1 / depth;
            auto k = indtmp1 - indtmp2 * depth;
            indtmp1 = indtmp2;
            indtmp2 = indtmp1 / heads;
            auto h = indtmp1 - indtmp2 * heads;
            auto& b = indtmp2;
            auto ni = get_window_start(
                i,
                height,
                kernel_size_1,
                neighborhood_size_1,
                dilation_1,
                false);
            auto nj = get_window_start(
                j,
                width,
                kernel_size_2,
                neighborhood_size_2,
                dilation_2,
                false);
            auto nk = get_window_start(
                k,
                depth,
                kernel_size_0,
                neighborhood_size_0,
                dilation_0,
                false);
            auto pi = get_pb_start(
                i, height, kernel_size_1, neighborhood_size_1, dilation_1);
            auto pj = get_pb_start(
                j, width, kernel_size_2, neighborhood_size_2, dilation_2);
            auto pk = get_pb_start(
                k, depth, kernel_size_0, neighborhood_size_0, dilation_0);
            for (int32_t kk = 0; kk < kernel_size_0; kk++) {
              for (int32_t ki = 0; ki < kernel_size_1; ki++) {
                for (int32_t kj = 0; kj < kernel_size_2; kj++) {
                  scalar_t updt = scalar_t(0);
                  int64_t batchHeadOffset =
                      b * query_stride_0 + h * query_stride_1;
                  int64_t queryOffset = batchHeadOffset + k * query_stride_2 +
                      i * query_stride_3 + j * query_stride_4;
                  int64_t keyOffset = batchHeadOffset +
                      (kk * dilation_0 + nk) * query_stride_2 +
                      (ki * dilation_1 + ni) * query_stride_3 +
                      (kj * dilation_2 + nj) * query_stride_4;
                  for (int64_t dimOffset = 0; dimOffset < dim; ++dimOffset)
                    updt += query[queryOffset + dimOffset] *
                        key[keyOffset + dimOffset];
                  int64_t index = b * attn_stride_0 + h * attn_stride_1 +
                      k * attn_stride_2 + i * attn_stride_3 +
                      j * attn_stride_4 + kk * (kernel_size_1 * kernel_size_2) +
                      ki * kernel_size_2 + kj;
                  int64_t biasIndex = h * bias_stride_0 +
                      (pk + kk) * bias_stride_1 + (pi + ki) * bias_stride_2 +
                      (pj + kj);
                  updt += bias[biasIndex];
                  attn[index] = updt;
                }
              }
            }
          }
        });
  }
};

} // namespace naive
} // namespace cpu
} // namespace natten
