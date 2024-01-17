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
    \brief Inverse-Neighborhood-Neighborhood CPU kernel for 1D data.
           Applies inverse neighborhood attention weights to inverse
   neighborhood values. Used to compute key and value grads.
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

// TODO: AVX

template <typename scalar_t>
struct InverseNeighborhood1D {
  void operator()(
      void* attn_ptr,
      void* d_output_ptr,
      void* d_value_ptr,
      int32_t batch_size,
      int32_t heads,
      int32_t length,
      int32_t dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      const std::tuple<int32_t>& kernel_size,
      const std::tuple<int32_t>& dilation,
      const std::tuple<bool>& is_causal) {
    launch(
        reinterpret_cast<scalar_t*>(attn_ptr),
        reinterpret_cast<scalar_t*>(d_output_ptr),
        reinterpret_cast<scalar_t*>(d_value_ptr),
        length,
        heads,
        std::get<0>(kernel_size),
        std::get<0>(dilation),
        dim,
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        is_causal);
  }

  void launch( // K-grad / V-grad
      scalar_t* weights, // d_attn / attn
      scalar_t* values, // query  / d_out
      scalar_t* output, // d_key  / d_value
      int32_t length,
      int32_t heads,
      int32_t kernel_size,
      int32_t dilation,
      int32_t dim,
      int32_t batch_size,
      int64_t weights_stride_0,
      int64_t weights_stride_1,
      int64_t weights_stride_2,
      const std::tuple<bool>& is_causal) {
    auto is_causal_0 = std::get<0>(is_causal);
    auto neighborhood_size = kernel_size / 2;
    int64_t values_stride_2 = dim;
    int64_t values_stride_1 = length * values_stride_2;
    int64_t values_stride_0 = heads * values_stride_1;
    for (int32_t b = 0; b < batch_size; b++) {
      at::parallel_for(0, heads, GRAIN_SIZE, [&](int32_t start, int32_t end) {
        for (int32_t h = start; h < end; h++) {
          for (int32_t i = 0; i < length; i++) {
            auto ni = get_backward_window_start(
                i, kernel_size, neighborhood_size, dilation, is_causal_0);
            auto ei = get_backward_window_end(
                i,
                length,
                kernel_size,
                neighborhood_size,
                dilation,
                is_causal_0);
            for (int32_t d = 0; d < dim; d++) {
              int64_t weightsOffset =
                  b * weights_stride_0 + h * weights_stride_1;
              int64_t valuesOffset =
                  b * values_stride_0 + h * values_stride_1 + d;
              scalar_t output_update = scalar_t(0);
              for (int32_t xi = ni; xi < ei; xi += dilation) {
                auto oni = get_window_start(
                    xi,
                    length,
                    kernel_size,
                    neighborhood_size,
                    dilation,
                    is_causal_0);
                int64_t valuesIndex = valuesOffset + xi * values_stride_2;
                int64_t weightsIndex = weightsOffset + xi * weights_stride_2 +
                    int32_t((i - oni) / dilation);
                output_update += values[valuesIndex] * weights[weightsIndex];
              }
              int64_t linearIndex = b * values_stride_0 + h * values_stride_1 +
                  i * values_stride_2 + d;
              output[linearIndex] = output_update;
            }
          }
        }
      });
    }
  }
};

} // namespace naive
} // namespace cpu
} // namespace natten
