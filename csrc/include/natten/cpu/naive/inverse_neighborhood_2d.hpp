/***************************************************************************************************
 * Copyright (c) 2023 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
    \brief Inverse-Neighborhood-Neighborhood CPU kernel for 2D data.
           Applies inverse neighborhood attention weights to inverse neighborhood values.
           Used to compute key and value grads.
*/

#pragma once
// TODO: these kernels should be independent of torch api.
// But for now, we do need vectorized reads.
#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#if defined(AVX_INT)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif

#include "natten/cpu/naive/natten_cpu_commons.h"

namespace natten {
namespace cpu {
namespace naive {

#define GRAIN_SIZE 0

// TODO: AVX

template <typename scalar_t>
struct InverseNeighborhood2D {

  void operator()(
    void * attn_ptr,
    void * d_output_ptr,
    void * d_value_ptr,
    int batch_size,
    int heads,
    int height,
    int width,
    int dim,
    int kernel_size,
    int dilation) {
    launch(
      reinterpret_cast<scalar_t*>(attn_ptr),
      reinterpret_cast<scalar_t*>(d_output_ptr),
      reinterpret_cast<scalar_t*>(d_value_ptr),
      height, width, heads, kernel_size, dilation, dim, batch_size);
  }

  void launch(                // K-grad / V-grad
    scalar_t* weights,        // d_attn / attn
    scalar_t* values,         // query  / d_out
    scalar_t* output,         // d_key  / d_value
    const int height, 
    const int width,
    const int heads,
    const int kernel_size,
    const int dilation,
    const int dim,
    const int batch_size) {
  const int neighborhood_size = kernel_size / 2;
  const int weights_stride_3 = kernel_size * kernel_size;
  const int weights_stride_2 = width * weights_stride_3;
  const int weights_stride_1 = height * weights_stride_2;
  const int weights_stride_0 = heads * weights_stride_1;
  const int values_stride_3  = dim;
  const int values_stride_2  = width * values_stride_3;
  const int values_stride_1  = height * values_stride_2;
  const int values_stride_0  = heads * values_stride_1;
  for (int b = 0; b < batch_size; b++) {
      at::parallel_for(0, heads, GRAIN_SIZE, [&](int start, int end) {
      for (int h = start; h < end; h++) {
          for (int i = 0; i < height; i++) {
          const int ni = get_backward_window_start(i, kernel_size, neighborhood_size, dilation);
          const int ei = get_backward_window_end(i, height, kernel_size, neighborhood_size, dilation);
          for (int j = 0; j < width; j++) {
              const int nj = get_backward_window_start(j, kernel_size, neighborhood_size, dilation);
              const int ej = get_backward_window_end(j, width, kernel_size, neighborhood_size, dilation);
              for (int d = 0; d < dim; d++) {
                  const int weightsOffset = b * weights_stride_0 + h * weights_stride_1;
                  const int outOffset = b * values_stride_0 + h * values_stride_1 + d;
                  scalar_t output_update = scalar_t(0);
                  for (int xi=ni; xi < ei; xi+=dilation){
                  const int oni = get_window_start(xi, height, kernel_size, neighborhood_size, dilation);
                  for (int xj=nj; xj < ej; xj+=dilation){
                      const int onj = get_window_start(xj, width, kernel_size, neighborhood_size, dilation);
                      const int outIndex = outOffset + xi * values_stride_2 + xj * values_stride_3;
                      const int weightsIndex = weightsOffset + xi * weights_stride_2 + xj * weights_stride_3 + int((i-oni)/dilation)*kernel_size + int((j-onj)/dilation);
                        output_update += values[outIndex] * weights[weightsIndex];
                    }}
                    const int linearIndex = b*values_stride_0 + h*values_stride_1 + i*values_stride_2 + j*values_stride_3 + d;
                    output[linearIndex] = output_update;
                }
            }}
        }});
    }
  }
};

} // namespace naive
} // namespace cpu 
} // namespace natten
