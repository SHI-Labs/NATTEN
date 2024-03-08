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
    \brief Pointwise-Neighborhood CPU kernel for 2D data.
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
struct PointwiseNeighborhood2D {
  void operator()(
      bool is_grad,
      void* query_ptr,
      void* key_ptr,
      void* attn_ptr,
      int32_t batch_size,
      int32_t heads,
      int32_t height,
      int32_t width,
      int32_t dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int64_t attn_stride_3,
      const std::tuple<int32_t, int32_t>& kernel_size,
      const std::tuple<int32_t, int32_t>& dilation,
      const std::tuple<bool, bool>& is_causal) {
    launch(
        is_grad,
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
        height,
        width,
        heads,
        std::get<0>(kernel_size),
        std::get<1>(kernel_size),
        std::get<0>(dilation),
        std::get<1>(dilation),
        dim,
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        is_causal);
  }

  void launch( // QK    / A-grad
      bool is_grad,
      scalar_t* query, // query / d_out
      scalar_t* key, // key   / value
      scalar_t* attn, // attn  / d_attn
      int32_t height,
      int32_t width,
      int32_t heads,
      int32_t kernel_size_0,
      int32_t kernel_size_1,
      int32_t dilation_0,
      int32_t dilation_1,
      int32_t dim,
      int32_t batch_size,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int64_t attn_stride_3,
      const std::tuple<bool, bool>& is_causal) {
    auto is_causal_0 = std::get<0>(is_causal);
    auto is_causal_1 = std::get<1>(is_causal);
    auto neighborhood_size_0 = kernel_size_0 / 2;
    auto neighborhood_size_1 = kernel_size_1 / 2;
    int64_t query_stride_3 = dim;
    int64_t query_stride_2 = width * query_stride_3;
    int64_t query_stride_1 = height * query_stride_2;
    int64_t query_stride_0 = heads * query_stride_1;
    auto mask_value = AttnMask<scalar_t>::value(is_grad);
#if defined(AVX_INT)
    using Vec = at::vec::Vectorized<scalar_t>;
    at::parallel_for(
        0,
        batch_size * heads * height * width,
        GRAIN_SIZE,
        [&](int32_t start, int32_t end) {
          for (int32_t x = start; x < end; x++) {
            int32_t indtmp1 = x / width;
            auto j = x - indtmp1 * width;
            int32_t indtmp2 = indtmp1 / height;
            auto i = indtmp1 - indtmp2 * height;
            indtmp1 = indtmp2;
            indtmp2 = indtmp1 / heads;
            auto h = indtmp1 - indtmp2 * heads;
            auto& b = indtmp2;
            auto ni = get_window_start(
                i,
                height,
                kernel_size_0,
                neighborhood_size_0,
                dilation_0,
                is_causal_0);
            auto nj = get_window_start(
                j,
                width,
                kernel_size_1,
                neighborhood_size_1,
                dilation_1,
                is_causal_1);
            auto ei = get_window_end(
                i,
                ni,
                height,
                kernel_size_0,
                neighborhood_size_0,
                dilation_0,
                is_causal_0);
            auto ej = get_window_end(
                j,
                nj,
                width,
                kernel_size_1,
                neighborhood_size_1,
                dilation_1,
                is_causal_1);
            int64_t batchHeadOffset = b * query_stride_0 + h * query_stride_1;
            int64_t queryOffset =
                batchHeadOffset + i * query_stride_2 + j * query_stride_3;
            int64_t index = b * attn_stride_0 + h * attn_stride_1 +
                i * attn_stride_2 + j * attn_stride_3;
            scalar_t* _qaddr = query + queryOffset;
            for (int32_t ki = 0; ki < kernel_size_0; ki++) {
              auto key_idx_i = ki * dilation_0 + ni;
              for (int32_t kj = 0; kj < kernel_size_1; kj++) {
                auto key_idx_j = kj * dilation_1 + nj;
                Vec updt = Vec(scalar_t(0));
                scalar_t sum_val;
                if (key_idx_i < ei && key_idx_j < ej) {
                  int64_t keyOffset = batchHeadOffset +
                      key_idx_i * query_stride_2 + key_idx_j * query_stride_3;
                  scalar_t* _kaddr = key + keyOffset;
                  int64_t d1 = 0;
                  for (; d1 < dim - (dim % Vec::size()); d1 += Vec::size())
                    updt = at::vec::fmadd(
                        Vec::loadu(_qaddr + d1), Vec::loadu(_kaddr + d1), updt);
                  sum_val = at::vec::vec_reduce_all(
                      [](Vec& x, Vec& y) { return x + y; }, updt, Vec::size());
                  for (; d1 < dim; ++d1)
                    sum_val += _qaddr[d1] * _kaddr[d1];
                } else {
                  sum_val = mask_value;
                }
                attn[index] = sum_val;
                index++;
              }
            }
          }
        });
#else
    at::parallel_for(
        0,
        batch_size * heads * height * width,
        GRAIN_SIZE,
        [&](int32_t start, int32_t end) {
          for (int32_t x = start; x < end; x++) {
            int32_t indtmp1 = x / width;
            auto j = x - indtmp1 * width;
            int32_t indtmp2 = indtmp1 / height;
            auto i = indtmp1 - indtmp2 * height;
            indtmp1 = indtmp2;
            indtmp2 = indtmp1 / heads;
            auto h = indtmp1 - indtmp2 * heads;
            auto& b = indtmp2;
            auto ni = get_window_start(
                i,
                height,
                kernel_size_0,
                neighborhood_size_0,
                dilation_0,
                is_causal_0);
            auto nj = get_window_start(
                j,
                width,
                kernel_size_1,
                neighborhood_size_1,
                dilation_1,
                is_causal_1);
            auto ni = get_window_end(
                i,
                ni,
                height,
                kernel_size_0,
                neighborhood_size_0,
                dilation_0,
                is_causal_0);
            auto nj = get_window_end(
                j,
                nj,
                width,
                kernel_size_1,
                neighborhood_size_1,
                dilation_1,
                is_causal_1);
            for (int32_t ki = 0; ki < kernel_size_0; ki++) {
              auto key_idx_i = ki * dilation_0 + ni;
              for (int32_t kj = 0; kj < kernel_size_1; kj++) {
                auto key_idx_j = kj * dilation_1 + nj;
                scalar_t updt = scalar_t(0);
                int64_t batchHeadOffset =
                    b * query_stride_0 + h * query_stride_1;
                int64_t queryOffset =
                    batchHeadOffset + i * query_stride_2 + j * query_stride_3;
                if (key_idx_i < ei && key_idx_j < ej) {
                  int64_t keyOffset = batchHeadOffset +
                      (ki * dilation_0 + ni) * query_stride_2 +
                      (kj * dilation_1 + nj) * query_stride_3;
                  for (int64_t dimOffset = 0; dimOffset < dim; ++dimOffset)
                    updt += query[queryOffset + dimOffset] *
                        key[keyOffset + dimOffset];
                } else {
                  updt = mask_value;
                }
                int64_t index = b * attn_stride_0 + h * attn_stride_1 +
                    i * attn_stride_2 + j * attn_stride_3 + ki * kernel_size_1 +
                    kj;
                attn[index] = updt;
              }
            }
          }
        });
#endif
  }
};

template <typename scalar_t>
struct PointwiseNeighborhood2DWithBias {
  void operator()(
      void* query_ptr,
      void* key_ptr,
      void* bias_ptr,
      void* attn_ptr,
      int32_t batch_size,
      int32_t heads,
      int32_t height,
      int32_t width,
      int32_t dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int64_t attn_stride_3,
      const std::tuple<int32_t, int32_t>& kernel_size,
      const std::tuple<int32_t, int32_t>& dilation,
      const std::tuple<bool, bool>& is_causal) {
    NATTEN_CHECK(
        !any_true(is_causal),
        "Neighborhood attention with causal masking does not support positional biases yet.");
    launch(
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(bias_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
        height,
        width,
        heads,
        std::get<0>(kernel_size),
        std::get<1>(kernel_size),
        std::get<0>(dilation),
        std::get<1>(dilation),
        dim,
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3);
  }

  void launch( // QK
      scalar_t* query, // query / d_out
      scalar_t* key, // key   / value
      scalar_t* bias, // relative positional bias tensor
      scalar_t* attn, // attn  / d_attn
      int32_t height,
      int32_t width,
      int32_t heads,
      int32_t kernel_size_0,
      int32_t kernel_size_1,
      int32_t dilation_0,
      int32_t dilation_1,
      int32_t dim,
      int32_t batch_size,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int64_t attn_stride_3) {
    auto neighborhood_size_0 = kernel_size_0 / 2;
    auto neighborhood_size_1 = kernel_size_1 / 2;
    int64_t bias_stride_1 = (2 * kernel_size_1 - 1);
    int64_t bias_stride_0 = (2 * kernel_size_0 - 1) * bias_stride_1;
    int64_t query_stride_3 = dim;
    int64_t query_stride_2 = width * query_stride_3;
    int64_t query_stride_1 = height * query_stride_2;
    int64_t query_stride_0 = heads * query_stride_1;
#if defined(AVX_INT)
    using Vec = at::vec::Vectorized<scalar_t>;
    at::parallel_for(
        0,
        batch_size * heads * height * width,
        GRAIN_SIZE,
        [&](int32_t start, int32_t end) {
          for (int32_t x = start; x < end; x++) {
            int32_t indtmp1 = x / width;
            auto j = x - indtmp1 * width;
            int32_t indtmp2 = indtmp1 / height;
            auto i = indtmp1 - indtmp2 * height;
            indtmp1 = indtmp2;
            indtmp2 = indtmp1 / heads;
            auto h = indtmp1 - indtmp2 * heads;
            auto& b = indtmp2;
            auto ni = get_window_start(
                i,
                height,
                kernel_size_0,
                neighborhood_size_0,
                dilation_0,
                false);
            int32_t pi = get_pb_start(
                i, height, kernel_size_0, neighborhood_size_0, dilation_0);
            auto nj = get_window_start(
                j,
                width,
                kernel_size_1,
                neighborhood_size_1,
                dilation_1,
                false);
            int32_t pj = get_pb_start(
                j, width, kernel_size_1, neighborhood_size_1, dilation_1);
            int64_t batchHeadOffset = b * query_stride_0 + h * query_stride_1;
            int64_t queryOffset =
                batchHeadOffset + i * query_stride_2 + j * query_stride_3;
            int64_t index = b * attn_stride_0 + h * attn_stride_1 +
                i * attn_stride_2 + j * attn_stride_3;
            scalar_t* _qaddr = query + queryOffset;
            for (int32_t ki = 0; ki < kernel_size_0; ki++) {
              for (int32_t kj = 0; kj < kernel_size_1; kj++) {
                Vec updt = Vec(scalar_t(0));
                int64_t keyOffset = batchHeadOffset +
                    (ki * dilation_0 + ni) * query_stride_2 +
                    (kj * dilation_1 + nj) * query_stride_3;
                scalar_t* _kaddr = key + keyOffset;
                int64_t d1 = 0;
                for (; d1 < dim - (dim % Vec::size()); d1 += Vec::size())
                  updt = at::vec::fmadd(
                      Vec::loadu(_qaddr + d1), Vec::loadu(_kaddr + d1), updt);
                scalar_t sum_val = at::vec::vec_reduce_all(
                    [](Vec& x, Vec& y) { return x + y; }, updt, Vec::size());
                for (; d1 < dim; ++d1)
                  sum_val += _qaddr[d1] * _kaddr[d1];
                int64_t biasIndex =
                    h * bias_stride_0 + (pi + ki) * bias_stride_1 + (pj + kj);
                attn[index] = bias[biasIndex] + sum_val;
                index++;
              }
            }
          }
        });
#else
    at::parallel_for(
        0,
        batch_size * heads * height * width,
        GRAIN_SIZE,
        [&](int32_t start, int32_t end) {
          for (int32_t x = start; x < end; x++) {
            int32_t indtmp1 = x / width;
            auto j = x - indtmp1 * width;
            int32_t indtmp2 = indtmp1 / height;
            auto i = indtmp1 - indtmp2 * height;
            indtmp1 = indtmp2;
            indtmp2 = indtmp1 / heads;
            auto h = indtmp1 - indtmp2 * heads;
            auto& b = indtmp2;
            auto ni = get_window_start(
                i,
                height,
                kernel_size_0,
                neighborhood_size_0,
                dilation_0,
                false);
            auto pi = get_pb_start(
                i, height, kernel_size_0, neighborhood_size_0, dilation_0);
            auto nj = get_window_start(
                j,
                width,
                kernel_size_1,
                neighborhood_size_1,
                dilation_1,
                false);
            auto pj = get_pb_start(
                j, width, kernel_size_1, neighborhood_size_1, dilation_1);
            for (int32_t ki = 0; ki < kernel_size_0; ki++) {
              for (int32_t kj = 0; kj < kernel_size_1; kj++) {
                scalar_t updt = scalar_t(0);
                int64_t batchHeadOffset =
                    b * query_stride_0 + h * query_stride_1;
                int64_t queryOffset =
                    batchHeadOffset + i * query_stride_2 + j * query_stride_3;
                int64_t keyOffset = batchHeadOffset +
                    (ki * dilation_0 + ni) * query_stride_2 +
                    (kj * dilation_1 + nj) * query_stride_3;
                for (int64_t dimOffset = 0; dimOffset < dim; ++dimOffset)
                  updt += query[queryOffset + dimOffset] *
                      key[keyOffset + dimOffset];
                int64_t index = b * attn_stride_0 + h * attn_stride_1 +
                    i * attn_stride_2 + j * attn_stride_3 + ki * kernel_size_1 +
                    kj;
                int64_t biasIndex =
                    h * bias_stride_0 + (pi + ki) * bias_stride_1 + (pj + kj);
                updt += bias[biasIndex];
                attn[index] = updt;
              }
            }
          }
        });
#endif
  }
};

} // namespace naive
} // namespace cpu
} // namespace natten
