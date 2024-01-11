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
    \brief Neighborhood Attention 3D
*/

#pragma once
#include <natten_autogen/cpu/naive/interface.h>

namespace natten {
namespace cpu {

template <typename T>
void na3d_qk_forward(
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
    int dilation,
    int depth_kernel_size,
    int depth_dilation) {
  if (bias_ptr == nullptr) {
    DISPATCH_DTYPE_na3d_pn_cpu_naive(
        T,
        query_ptr,
        key_ptr,
        attn_ptr,
        batch_size,
        heads,
        depth,
        height,
        width,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        attn_stride_4,
        kernel_size,
        depth_kernel_size,
        dilation,
        depth_dilation);
  } else {
    DISPATCH_DTYPE_na3d_pn_bias_cpu_naive(
        T,
        query_ptr,
        key_ptr,
        bias_ptr,
        attn_ptr,
        batch_size,
        heads,
        depth,
        height,
        width,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        attn_stride_4,
        kernel_size,
        depth_kernel_size,
        dilation,
        depth_dilation);
  }
}

template <typename T>
void na3d_qk_backward(
    void* query_ptr,
    void* key_ptr,
    void* d_attn_ptr,
    void* d_query_ptr,
    void* d_key_ptr,
    void* d_bias_ptr,
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
    int dilation,
    int depth_kernel_size,
    int depth_dilation) {
  DISPATCH_DTYPE_na3d_nn_cpu_naive(
      T,
      d_attn_ptr,
      key_ptr,
      d_query_ptr,
      batch_size,
      heads,
      depth,
      height,
      width,
      dim,
      attn_stride_0,
      attn_stride_1,
      attn_stride_2,
      attn_stride_3,
      attn_stride_4,
      kernel_size,
      depth_kernel_size,
      dilation,
      depth_dilation);
  DISPATCH_DTYPE_na3d_in_cpu_naive(
      T,
      d_attn_ptr,
      query_ptr,
      d_key_ptr,
      batch_size,
      heads,
      depth,
      height,
      width,
      dim,
      attn_stride_0,
      attn_stride_1,
      attn_stride_2,
      attn_stride_3,
      attn_stride_4,
      kernel_size,
      depth_kernel_size,
      dilation,
      depth_dilation);
  if (d_bias_ptr != nullptr) {
    DISPATCH_DTYPE_na3d_rpbgrad_cpu_naive(
        T,
        d_bias_ptr,
        d_attn_ptr,
        batch_size,
        heads,
        depth,
        height,
        width,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        attn_stride_4,
        kernel_size,
        depth_kernel_size,
        dilation,
        depth_dilation);
  }
}

template <typename T>
void na3d_av_forward(
    void* attn_ptr,
    void* value_ptr,
    void* output_ptr,
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
    int dilation,
    int depth_kernel_size,
    int depth_dilation) {
  DISPATCH_DTYPE_na3d_nn_cpu_naive(
      T,
      attn_ptr,
      value_ptr,
      output_ptr,
      batch_size,
      heads,
      depth,
      height,
      width,
      dim,
      attn_stride_0,
      attn_stride_1,
      attn_stride_2,
      attn_stride_3,
      attn_stride_4,
      kernel_size,
      depth_kernel_size,
      dilation,
      depth_dilation);
}

template <typename T>
void na3d_av_backward(
    void* attn_ptr,
    void* value_ptr,
    void* d_output_ptr,
    void* d_attn_ptr,
    void* d_value_ptr,
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
    int dilation,
    int depth_kernel_size,
    int depth_dilation) {
  DISPATCH_DTYPE_na3d_pn_cpu_naive(
      T,
      d_output_ptr,
      value_ptr,
      d_attn_ptr,
      batch_size,
      heads,
      depth,
      height,
      width,
      dim,
      attn_stride_0,
      attn_stride_1,
      attn_stride_2,
      attn_stride_3,
      attn_stride_4,
      kernel_size,
      depth_kernel_size,
      dilation,
      depth_dilation);
  DISPATCH_DTYPE_na3d_in_cpu_naive(
      T,
      attn_ptr,
      d_output_ptr,
      d_value_ptr,
      batch_size,
      heads,
      depth,
      height,
      width,
      dim,
      attn_stride_0,
      attn_stride_1,
      attn_stride_2,
      attn_stride_3,
      attn_stride_4,
      kernel_size,
      depth_kernel_size,
      dilation,
      depth_dilation);
}

} // namespace cpu
} // namespace natten
