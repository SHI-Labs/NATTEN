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
    \brief Neighborhood Attention 1D
*/

#pragma once
#include <natten/natten.h>
#include <natten_autogen/cpu/naive/interface.h>

namespace natten {
namespace cpu {

template <typename T>
void na1d_qk_forward(
    void* query_ptr,
    void* key_ptr,
    void* bias_ptr,
    void* attn_ptr,
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
  if (bias_ptr == nullptr) {
    DISPATCH_DTYPE_na1d_pn_cpu_naive(
        T,
        /* is_grad = */ false,
        query_ptr,
        key_ptr,
        attn_ptr,
        batch_size,
        heads,
        length,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        kernel_size,
        dilation,
        is_causal);
  } else {
    NATTEN_CHECK(
        !any_true(is_causal),
        "Neighborhood attention with causal masking does not support positional biases yet.");
    DISPATCH_DTYPE_na1d_pn_bias_cpu_naive(
        T,
        query_ptr,
        key_ptr,
        bias_ptr,
        attn_ptr,
        batch_size,
        heads,
        length,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        kernel_size,
        dilation,
        is_causal);
  }
}

template <typename T>
void na1d_qk_backward(
    void* query_ptr,
    void* key_ptr,
    void* d_attn_ptr,
    void* d_query_ptr,
    void* d_key_ptr,
    void* d_bias_ptr,
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
  DISPATCH_DTYPE_na1d_nn_cpu_naive(
      T,
      d_attn_ptr,
      key_ptr,
      d_query_ptr,
      batch_size,
      heads,
      length,
      dim,
      attn_stride_0,
      attn_stride_1,
      attn_stride_2,
      kernel_size,
      dilation,
      is_causal);
  DISPATCH_DTYPE_na1d_in_cpu_naive(
      T,
      d_attn_ptr,
      query_ptr,
      d_key_ptr,
      batch_size,
      heads,
      length,
      dim,
      attn_stride_0,
      attn_stride_1,
      attn_stride_2,
      kernel_size,
      dilation,
      is_causal);
  if (d_bias_ptr != nullptr) {
    NATTEN_CHECK(
        !any_true(is_causal),
        "Neighborhood attention with causal masking does not support positional biases yet.");
    DISPATCH_DTYPE_na1d_rpbgrad_cpu_naive(
        T,
        d_bias_ptr,
        d_attn_ptr,
        batch_size,
        heads,
        length,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        kernel_size,
        dilation,
        is_causal);
  }
}

template <typename T>
void na1d_av_forward(
    void* attn_ptr,
    void* value_ptr,
    void* output_ptr,
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
  DISPATCH_DTYPE_na1d_nn_cpu_naive(
      T,
      attn_ptr,
      value_ptr,
      output_ptr,
      batch_size,
      heads,
      length,
      dim,
      attn_stride_0,
      attn_stride_1,
      attn_stride_2,
      kernel_size,
      dilation,
      is_causal);
}

template <typename T>
void na1d_av_backward(
    void* attn_ptr,
    void* value_ptr,
    void* d_output_ptr,
    void* d_attn_ptr,
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
  DISPATCH_DTYPE_na1d_pn_cpu_naive(
      T,
      /* is_grad = */ true,
      d_output_ptr,
      value_ptr,
      d_attn_ptr,
      batch_size,
      heads,
      length,
      dim,
      attn_stride_0,
      attn_stride_1,
      attn_stride_2,
      kernel_size,
      dilation,
      is_causal);
  DISPATCH_DTYPE_na1d_in_cpu_naive(
      T,
      attn_ptr,
      d_output_ptr,
      d_value_ptr,
      batch_size,
      heads,
      length,
      dim,
      attn_stride_0,
      attn_stride_1,
      attn_stride_2,
      kernel_size,
      dilation,
      is_causal);
}

} // namespace cpu
} // namespace natten
