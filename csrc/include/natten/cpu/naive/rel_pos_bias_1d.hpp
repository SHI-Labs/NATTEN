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
    \brief Relative positional bias backward pass CPU kernel for 1D data.
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
struct RelPosBiasGradient1D {
  void operator()(
      void* d_bias_ptr,
      void* d_attn_ptr,
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
    NATTEN_CHECK(
        !any_true(is_causal),
        "Neighborhood attention with causal masking does not support positional biases yet.");
    launch(
        reinterpret_cast<scalar_t*>(d_bias_ptr),
        reinterpret_cast<scalar_t*>(d_attn_ptr),
        length,
        heads,
        std::get<0>(kernel_size),
        std::get<0>(dilation),
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2);
  }

  void launch(
      scalar_t* d_bias,
      scalar_t* d_attn,
      int32_t length,
      int32_t heads,
      int32_t kernel_size,
      int32_t dilation,
      int32_t batch_size,
      int64_t d_attn_stride_0,
      int64_t d_attn_stride_1,
      int64_t d_attn_stride_2) {
    auto neighborhood_size = kernel_size / 2;
    int64_t d_bias_stride_0 = 2 * kernel_size - 1;
    at::parallel_for(0, heads, GRAIN_SIZE, [&](int32_t start, int32_t end) {
      for (int32_t h = start; h < end; h++) {
        for (int32_t i = 0; i < length; i++) {
          auto pi =
              get_pb_start(i, length, kernel_size, neighborhood_size, dilation);
          for (int32_t ki = 0; ki < kernel_size; ki++) {
            scalar_t d_bias_update = scalar_t(0);
            int64_t attnOffset = h * d_attn_stride_1 + i * d_attn_stride_2 + ki;
            for (int32_t b = 0; b < batch_size; ++b) {
              d_bias_update += d_attn[attnOffset];
              attnOffset += d_attn_stride_0;
            }
            int64_t index = h * d_bias_stride_0 + (pi + ki);
            d_bias[index] += d_bias_update;
          }
        }
      }
    });
  }
};

} // namespace naive
} // namespace cpu
} // namespace natten
