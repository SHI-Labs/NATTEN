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
    \brief Neighborhood Attention 1D
*/

#pragma once

#include "natten/config.h"
#include <natten_autogen/cuda/naive/interface.h> 
#ifdef NATTEN_WITH_CUTLASS
#if (NATTEN_CUTLASS_TARGET_SM >= 80)
#include <natten_autogen/cuda/gemm/1d/sm80/interface.h> 
#elif (NATTEN_CUTLASS_TARGET_SM >= 75)
#include <natten_autogen/cuda/gemm/1d/sm75/interface.h> 
#elif (NATTEN_CUTLASS_TARGET_SM >= 70)
#include <natten_autogen/cuda/gemm/1d/sm70/interface.h> 
#endif
#endif

namespace natten {
namespace cuda {

template<typename T>
void na1d_qk_forward(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  #ifdef NATTEN_WITH_CUTLASS
  #if (NATTEN_CUTLASS_TARGET_SM >= 80)
  if (natten::kEnableGemmNA) {
  #elif (NATTEN_CUTLASS_TARGET_SM >= 70)
  if (natten::kEnableGemmNA && std::is_same<T, natten::float16>::value) {
  #endif
    DISPATCH_DTYPE_na1d_pn_cuda_gemm(T, dim,
      query_ptr, key_ptr, attn_ptr, bias_ptr,
      batch_size, heads, length, dim, 
      kernel_size, dilation, 1.0);
  }
  else {
  #endif
    if (bias_ptr == nullptr) {
      DISPATCH_DTYPE_na1d_pn_cuda_naive(T, kernel_size, dilation,
        query_ptr, key_ptr, attn_ptr,
        batch_size, heads, length, dim, 
        kernel_size, dilation);
    }
    else {
      DISPATCH_DTYPE_na1d_pn_bias_cuda_naive(T, kernel_size, dilation,
        query_ptr, key_ptr, bias_ptr, attn_ptr,
        batch_size, heads, length, dim, 
        kernel_size, dilation);
    }
  #ifdef NATTEN_WITH_CUTLASS
  }
  #endif
}

template<typename T>
void na1d_qk_backward(
  void * query_ptr,
  void * key_ptr,
  void * d_attn_ptr,
  void * d_query_ptr,
  void * d_key_ptr,
  void * d_bias_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  #ifdef NATTEN_WITH_CUTLASS
  #if (NATTEN_CUTLASS_TARGET_SM >= 80)
  if (natten::kEnableGemmNA) {
  #elif (NATTEN_CUTLASS_TARGET_SM >= 70)
  if (natten::kEnableGemmNA && std::is_same<T, natten::float16>::value) {
  #endif
    DISPATCH_DTYPE_na1d_nn_cuda_gemm(T, dim,
      d_attn_ptr, key_ptr, d_query_ptr,
      batch_size, heads, length, dim, 
      kernel_size, dilation, 1.0);
    DISPATCH_DTYPE_na1d_in_cuda_gemm(T, dim,
      d_attn_ptr, query_ptr, d_key_ptr,
      batch_size, heads, length, dim, 
      kernel_size, dilation, 1.0);
  }
  else {
  #endif
    DISPATCH_DTYPE_na1d_nn_cuda_naive(T, kernel_size, dilation,
      d_attn_ptr, key_ptr, d_query_ptr,
      batch_size, heads, length, dim, 
      kernel_size, dilation);
    DISPATCH_DTYPE_na1d_in_cuda_naive(T, kernel_size, dilation,
      d_attn_ptr, query_ptr, d_key_ptr,
      batch_size, heads, length, dim, 
      kernel_size, dilation);
  #ifdef NATTEN_WITH_CUTLASS
  }
  #endif
  if (d_bias_ptr != nullptr) {
    DISPATCH_DTYPE_na1d_rpbgrad_cuda_naive(T, kernel_size, dilation,
      d_bias_ptr, d_attn_ptr,
      batch_size, heads, length, dim, 
      kernel_size, dilation);
  }
}

template<typename T>
void na1d_av_forward(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  #ifdef NATTEN_WITH_CUTLASS
  #if (NATTEN_CUTLASS_TARGET_SM >= 80)
  if (natten::kEnableGemmNA) {
  #elif (NATTEN_CUTLASS_TARGET_SM >= 70)
  if (natten::kEnableGemmNA && std::is_same<T, natten::float16>::value) {
  #endif
    DISPATCH_DTYPE_na1d_nn_cuda_gemm(T, dim,
      attn_ptr, value_ptr, output_ptr,
      batch_size, heads, length, dim, 
      kernel_size, dilation, 1.0);
  }
  else {
  #endif
    DISPATCH_DTYPE_na1d_nn_cuda_naive(T, kernel_size, dilation,
      attn_ptr, value_ptr, output_ptr,
      batch_size, heads, length, dim, 
      kernel_size, dilation);
  #ifdef NATTEN_WITH_CUTLASS
  }
  #endif
}

template<typename T>
void na1d_av_backward(
  void * attn_ptr,
  void * value_ptr,
  void * d_output_ptr,
  void * d_attn_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  #ifdef NATTEN_WITH_CUTLASS
  #if (NATTEN_CUTLASS_TARGET_SM >= 80)
  if (natten::kEnableGemmNA) {
  #elif (NATTEN_CUTLASS_TARGET_SM >= 70)
  if (natten::kEnableGemmNA && std::is_same<T, natten::float16>::value) {
  #endif
    DISPATCH_DTYPE_na1d_pn_cuda_gemm(T, dim,
      d_output_ptr, value_ptr, d_attn_ptr, nullptr,
      batch_size, heads, length, dim, 
      kernel_size, dilation, 1.0);
    DISPATCH_DTYPE_na1d_in_cuda_gemm(T, dim,
      attn_ptr, d_output_ptr, d_value_ptr,
      batch_size, heads, length, dim, 
      kernel_size, dilation, 1.0);
  }
  else {
  #endif
    DISPATCH_DTYPE_na1d_pn_cuda_naive(T, kernel_size, dilation,
      d_output_ptr, value_ptr, d_attn_ptr,
      batch_size, heads, length, dim, 
      kernel_size, dilation);
    DISPATCH_DTYPE_na1d_in_cuda_naive(T, kernel_size, dilation,
      attn_ptr, d_output_ptr, d_value_ptr,
      batch_size, heads, length, dim, 
      kernel_size, dilation);
  #ifdef NATTEN_WITH_CUTLASS
  }
  #endif
}

} // namespace cuda
} // namespace natten

