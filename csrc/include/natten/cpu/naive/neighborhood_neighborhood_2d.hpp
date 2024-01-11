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
    \brief Neighborhood-Neighborhood CPU kernel for 2D data.
           Applies neighborhood attention weights to neighborhood values.

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
struct NeighborhoodNeighborhood2D {
  void operator()(
      void* attn_ptr,
      void* value_ptr,
      void* output_ptr,
      int batch_size,
      int heads,
      int height,
      int width,
      int dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int64_t attn_stride_3,
      int kernel_size,
      int dilation) {
    launch(
        reinterpret_cast<scalar_t*>(attn_ptr),
        reinterpret_cast<scalar_t*>(value_ptr),
        reinterpret_cast<scalar_t*>(output_ptr),
        height,
        width,
        heads,
        kernel_size,
        dilation,
        dim,
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3);
  }

  void launch( // AV     / Q-grad
      scalar_t* weights, // attn   / d_attn
      scalar_t* values, // value  / key
      scalar_t* output, // output / d_query
      const int height,
      const int width,
      const int heads,
      const int kernel_size,
      const int dilation,
      const int dim,
      const int batch_size,
      const int64_t weights_stride_0,
      const int64_t weights_stride_1,
      const int64_t weights_stride_2,
      const int64_t weights_stride_3) {
    const int neighborhood_size = kernel_size / 2;
    const int values_stride_3 = dim;
    const int values_stride_2 = width * values_stride_3;
    const int values_stride_1 = height * values_stride_2;
    const int values_stride_0 = heads * values_stride_1;
    // NOTE: this function originally had an AVX impl,
    // but it was removed when migrating to the new NATTEN api
    // Unsure what the issue was; I wrote this well over a year ago so...
    // But, these kernels should be re-written from scratch (that is if
    // I find more time by some miracle.)
    at::parallel_for(
        0,
        batch_size * heads * height * width,
        GRAIN_SIZE,
        [&](int start, int end) {
          for (int x = start; x < end; x++) {
            int indtmp1 = x / width;
            const int j = x - indtmp1 * width;
            int indtmp2 = indtmp1 / height;
            const int i = indtmp1 - indtmp2 * height;
            indtmp1 = indtmp2;
            indtmp2 = indtmp1 / heads;
            const int h = indtmp1 - indtmp2 * heads;
            const int b = indtmp2;
            const int ni = get_window_start(
                i, height, kernel_size, neighborhood_size, dilation);
            const int nj = get_window_start(
                j, width, kernel_size, neighborhood_size, dilation);
            for (int d = 0; d < dim; d++) {
              scalar_t updt = scalar_t(0);
              int64_t weightsOffset = b * weights_stride_0 + h * weights_stride_1 +
                  i * weights_stride_2 + j * weights_stride_3;
              const int64_t valuesOffset =
                  b * values_stride_0 + h * values_stride_1 + d;
              for (int xi = ni; xi < ni + kernel_size * dilation;
                   xi += dilation) {
                for (int xj = nj; xj < nj + kernel_size * dilation;
                     xj += dilation) {
                  const int64_t valuesIndex = valuesOffset + xi * values_stride_2 +
                      xj * values_stride_3;
                  updt += weights[weightsOffset] * values[valuesIndex];
                  ++weightsOffset;
                }
              }
              const int64_t linearIndex = b * values_stride_0 +
                  h * values_stride_1 + i * values_stride_2 +
                  j * values_stride_3 + d;
              output[linearIndex] = updt;
            }
          }
        });
  }
};

} // namespace naive
} // namespace cpu
} // namespace natten
