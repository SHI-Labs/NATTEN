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

#include "natten/cpu/naive/natten_cpu_commons.h"

namespace natten {
namespace cpu {
namespace naive {

#define GRAIN_SIZE 0

template <typename scalar_t>
struct PointwiseNeighborhood3D {
  void operator()(
      void* query_ptr,
      void* key_ptr,
      void* attn_ptr,
      int batch_size,
      int heads,
      int depth,
      int height,
      int width,
      int dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int64_t attn_stride_3,
      int64_t attn_stride_4,
      int kernel_size,
      int kernel_size_depth,
      int dilation,
      int dilation_depth) {
    launch(
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
        depth,
        height,
        width,
        heads,
        kernel_size,
        kernel_size_depth,
        dilation,
        dilation_depth,
        dim,
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        attn_stride_4);
  }

  void launch( // QK    / A-grad
      scalar_t* query, // query / d_out
      scalar_t* key, // key   / value
      scalar_t* attn, // attn  / d_attn
      const int depth,
      const int height,
      const int width,
      const int heads,
      const int kernel_size,
      const int kernel_size_d,
      const int dilation,
      const int dilation_d,
      const int dim,
      const int batch_size,
      const int64_t attn_stride_0,
      const int64_t attn_stride_1,
      const int64_t attn_stride_2,
      const int64_t attn_stride_3,
      const int64_t attn_stride_4) {
    const int neighborhood_size = kernel_size / 2;
    const int neighborhood_size_d = kernel_size_d / 2;
    const int query_stride_4 = dim;
    const int query_stride_3 = width * query_stride_4;
    const int query_stride_2 = height * query_stride_3;
    const int query_stride_1 = depth * query_stride_2;
    const int query_stride_0 = heads * query_stride_1;
    at::parallel_for(
        0,
        batch_size * heads * depth * height * width,
        GRAIN_SIZE,
        [&](int start, int end) {
          for (int x = start; x < end; x++) {
            int indtmp1 = x / width;
            const int j = x - indtmp1 * width;
            int indtmp2 = indtmp1 / height;
            const int i = indtmp1 - indtmp2 * height;
            indtmp1 = indtmp2;
            indtmp2 = indtmp1 / depth;
            const int k = indtmp1 - indtmp2 * depth;
            indtmp1 = indtmp2;
            indtmp2 = indtmp1 / heads;
            const int h = indtmp1 - indtmp2 * heads;
            const int b = indtmp2;
            const int ni = get_window_start(
                i, height, kernel_size, neighborhood_size, dilation);
            const int nj = get_window_start(
                j, width, kernel_size, neighborhood_size, dilation);
            const int nk = get_window_start(
                k, depth, kernel_size_d, neighborhood_size_d, dilation_d);
            for (int kk = 0; kk < kernel_size_d; kk++) {
              for (int ki = 0; ki < kernel_size; ki++) {
                for (int kj = 0; kj < kernel_size; kj++) {
                  scalar_t updt = scalar_t(0);
                  const int64_t batchHeadOffset =
                      b * query_stride_0 + h * query_stride_1;
                  const int64_t queryOffset = batchHeadOffset + k * query_stride_2 +
                      i * query_stride_3 + j * query_stride_4;
                  const int64_t keyOffset = batchHeadOffset +
                      (kk * dilation_d + nk) * query_stride_2 +
                      (ki * dilation + ni) * query_stride_3 +
                      (kj * dilation + nj) * query_stride_4;
                  for (int64_t dimOffset = 0; dimOffset < dim; ++dimOffset)
                    updt += query[queryOffset + dimOffset] *
                        key[keyOffset + dimOffset];
                  const int64_t index = b * attn_stride_0 + h * attn_stride_1 +
                      k * attn_stride_2 + i * attn_stride_3 +
                      j * attn_stride_4 + kk * (kernel_size * kernel_size) +
                      ki * kernel_size + kj;
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
      int batch_size,
      int heads,
      int depth,
      int height,
      int width,
      int dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int64_t attn_stride_3,
      int64_t attn_stride_4,
      int kernel_size,
      int kernel_size_depth,
      int dilation,
      int dilation_depth) {
    launch(
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(bias_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
        depth,
        height,
        width,
        heads,
        kernel_size,
        kernel_size_depth,
        dilation,
        dilation_depth,
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
      const int depth,
      const int height,
      const int width,
      const int heads,
      const int kernel_size,
      const int kernel_size_d,
      const int dilation,
      const int dilation_d,
      const int dim,
      const int batch_size,
      const int64_t attn_stride_0,
      const int64_t attn_stride_1,
      const int64_t attn_stride_2,
      const int64_t attn_stride_3,
      const int64_t attn_stride_4) {
    const int neighborhood_size = kernel_size / 2;
    const int neighborhood_size_d = kernel_size_d / 2;
    const int bias_stride_2 = (2 * kernel_size - 1);
    const int bias_stride_1 = (2 * kernel_size - 1) * bias_stride_2;
    const int bias_stride_0 = (2 * kernel_size_d - 1) * bias_stride_1;
    const int query_stride_4 = dim;
    const int query_stride_3 = width * query_stride_4;
    const int query_stride_2 = height * query_stride_3;
    const int query_stride_1 = depth * query_stride_2;
    const int query_stride_0 = heads * query_stride_1;
    at::parallel_for(
        0,
        batch_size * heads * depth * height * width,
        GRAIN_SIZE,
        [&](int start, int end) {
          for (int x = start; x < end; x++) {
            int indtmp1 = x / width;
            const int j = x - indtmp1 * width;
            int indtmp2 = indtmp1 / height;
            const int i = indtmp1 - indtmp2 * height;
            indtmp1 = indtmp2;
            indtmp2 = indtmp1 / depth;
            const int k = indtmp1 - indtmp2 * depth;
            indtmp1 = indtmp2;
            indtmp2 = indtmp1 / heads;
            const int h = indtmp1 - indtmp2 * heads;
            const int b = indtmp2;
            const int ni = get_window_start(
                i, height, kernel_size, neighborhood_size, dilation);
            const int nj = get_window_start(
                j, width, kernel_size, neighborhood_size, dilation);
            const int nk = get_window_start(
                k, depth, kernel_size_d, neighborhood_size_d, dilation_d);
            const int pi = get_pb_start(
                i, height, kernel_size, neighborhood_size, dilation);
            const int pj = get_pb_start(
                j, width, kernel_size, neighborhood_size, dilation);
            const int pk = get_pb_start(
                k, depth, kernel_size_d, neighborhood_size_d, dilation_d);
            for (int kk = 0; kk < kernel_size_d; kk++) {
              for (int ki = 0; ki < kernel_size; ki++) {
                for (int kj = 0; kj < kernel_size; kj++) {
                  scalar_t updt = scalar_t(0);
                  const int64_t batchHeadOffset =
                      b * query_stride_0 + h * query_stride_1;
                  const int64_t queryOffset = batchHeadOffset + k * query_stride_2 +
                      i * query_stride_3 + j * query_stride_4;
                  const int64_t keyOffset = batchHeadOffset +
                      (kk * dilation_d + nk) * query_stride_2 +
                      (ki * dilation + ni) * query_stride_3 +
                      (kj * dilation + nj) * query_stride_4;
                  for (int64_t dimOffset = 0; dimOffset < dim; ++dimOffset)
                    updt += query[queryOffset + dimOffset] *
                        key[keyOffset + dimOffset];
                  const int64_t index = b * attn_stride_0 + h * attn_stride_1 +
                      k * attn_stride_2 + i * attn_stride_3 +
                      j * attn_stride_4 + kk * (kernel_size * kernel_size) +
                      ki * kernel_size + kj;
                  const int64_t biasIndex = h * bias_stride_0 +
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
