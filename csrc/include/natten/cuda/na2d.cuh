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
    \brief Neighborhood Attention 2D
*/

#pragma once

#include <natten/natten.h>
#include <natten_autogen/cuda/naive/interface.h>
#ifdef NATTEN_WITH_CUTLASS
#include <natten_autogen/cuda/gemm/2d/interface.h>
#include <natten/cuda/fna/fna_backward.cuh>
#include <natten/cuda/fna/fna_forward.cuh>
#endif

namespace natten {
namespace cuda {

template <typename T, typename MemoryAllocator>
void na2d_forward(
    int32_t cc,
    size_t max_smem,
    cudaStream_t stream,
    MemoryAllocator alloc_bytes,
    void* query_ptr,
    void* key_ptr,
    void* value_ptr,
    void* out_ptr,
    void* rpb_ptr,
    void* logsumexp_ptr,
    int32_t batch_size,
    int32_t height,
    int32_t width,
    int32_t heads,
    int32_t dim,
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& dilation,
    const std::tuple<bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t>& query_tile_size,
    const std::tuple<int32_t, int32_t>& key_tile_size) {
#ifdef NATTEN_WITH_CUTLASS
  if (cc >= 80 || (cc >= 50 && !std::is_same<T, natten::bfloat16>::value)) {
    natten::cuda::fna::fna_forward_generic<T>(
        cc,
        max_smem,
        stream,
        alloc_bytes,
        query_ptr,
        key_ptr,
        value_ptr,
        out_ptr,
        rpb_ptr,
        batch_size,
        {height, width},
        heads,
        dim,
        dim, // dim_value
        kernel_size,
        dilation,
        is_causal,
        attn_scale,
        logsumexp_ptr,
        query_tile_size,
        key_tile_size);
  } else {
#endif
    NATTEN_FAILURE(
        "Fused kernels are only available on devices with "
        "compute capability >= 50 for FP16/FP32 inputs, and devices with "
        "compute capability >= 80 for FP32, BF16, and FP16 inputs.");
#ifdef NATTEN_WITH_CUTLASS
  }
#endif
}

template <typename T, typename MemoryAllocator>
void na2d_backward(
    int32_t cc,
    size_t max_smem,
    cudaStream_t stream,
    MemoryAllocator alloc_bytes,
    void* grad_out_ptr,
    void* query_ptr,
    void* key_ptr,
    void* value_ptr,
    // bias is not supported!
    void* logsumexp_ptr,
    void* delta_ptr,
    void* out_ptr,
    // Outputs:
    void* grad_query_ptr,
    void* grad_key_ptr,
    void* grad_value_ptr,
    int32_t batch_size,
    int32_t height,
    int32_t width,
    int32_t heads,
    int32_t dim,
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& dilation,
    const std::tuple<bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t>& query_tile_size,
    const std::tuple<int32_t, int32_t>& key_tile_size,
    const std::tuple<int32_t, int32_t>& num_splits_key) {
#ifdef NATTEN_WITH_CUTLASS
  if (cc >= 80 || (cc >= 50 && !std::is_same<T, natten::bfloat16>::value)) {
    natten::cuda::fna::fna_backward_generic<T>(
        cc,
        max_smem,
        stream,
        alloc_bytes,
        grad_out_ptr,
        query_ptr,
        key_ptr,
        value_ptr,
        logsumexp_ptr,
        delta_ptr,
        out_ptr,
        grad_query_ptr,
        grad_key_ptr,
        grad_value_ptr,
        batch_size,
        {height, width},
        heads,
        dim,
        dim, // dim_value
        kernel_size,
        dilation,
        is_causal,
        attn_scale,
        query_tile_size,
        key_tile_size,
        num_splits_key);
  } else {
#endif
    NATTEN_FAILURE(
        "Fused kernels are only available on devices with "
        "compute capability >= 50 for FP16/FP32 inputs, and devices with "
        "compute capability >= 80 for FP32, BF16, and FP16 inputs.");
#ifdef NATTEN_WITH_CUTLASS
  }
#endif
}

template <typename T>
void na2d_qk_forward(
    int32_t cc,
    size_t max_smem,
    cudaStream_t stream,
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
#ifdef NATTEN_WITH_CUTLASS
  if (natten::kEnableGemmNA &&
      !any_true(is_causal) && // TODO: remove when GEMM supports causal masking
      all_dims_match(kernel_size) && // TODO: remove when GEMM supports varying
                                     // kernel sizes
      all_dims_match(
          dilation) && // TODO: remove when GEMM supports varying dilations
      (cc >= 80 || (cc >= 70 && std::is_same<T, natten::float16>::value))) {
    LAUNCH_na2d_pn_cuda_gemm(
        cc,
        T,
        std::get<0>(kernel_size), // TODO: remove when GEMM supports varying
                                  // kernel sizes
        dim,
        query_ptr,
        key_ptr,
        attn_ptr,
        bias_ptr,
        batch_size,
        heads,
        height,
        width,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        std::get<0>(kernel_size), // TODO: remove when GEMM supports varying
                                  // kernel sizes
        std::get<0>(
            dilation), // TODO: remove when GEMM supports varying dilations
        1.0,
        stream);
  } else {
#endif
    if (bias_ptr == nullptr) {
      DISPATCH_DTYPE_na2d_pn_cuda_naive(
          T,
          is_causal,
          // kernel_size,
          // dilation,
          cc,
          stream,
          /* is_grad = */ false,
          query_ptr,
          key_ptr,
          attn_ptr,
          batch_size,
          heads,
          height,
          width,
          dim,
          attn_stride_0,
          attn_stride_1,
          attn_stride_2,
          attn_stride_3,
          kernel_size,
          dilation);
    } else {
      NATTEN_CHECK(
          !any_true(is_causal),
          "Neighborhood attention with causal masking does not support positional biases yet.");
      DISPATCH_DTYPE_na2d_pn_bias_cuda_naive(
          T,
          is_causal,
          // kernel_size,
          // dilation,
          cc,
          stream,
          query_ptr,
          key_ptr,
          bias_ptr,
          attn_ptr,
          batch_size,
          heads,
          height,
          width,
          dim,
          attn_stride_0,
          attn_stride_1,
          attn_stride_2,
          attn_stride_3,
          kernel_size,
          dilation);
    }
#ifdef NATTEN_WITH_CUTLASS
  }
#endif
}

template <typename T>
void na2d_qk_backward(
    int32_t cc,
    size_t max_smem,
    cudaStream_t stream,
    void* query_ptr,
    void* key_ptr,
    void* d_attn_ptr,
    void* d_query_ptr,
    void* d_key_ptr,
    void* d_bias_ptr,
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
#ifdef NATTEN_WITH_CUTLASS
  if (natten::kEnableGemmNA &&
      !any_true(is_causal) && // TODO: remove when GEMM supports causal masking
      all_dims_match(kernel_size) && // TODO: remove when GEMM supports varying
                                     // kernel sizes
      all_dims_match(
          dilation) && // TODO: remove when GEMM supports varying dilations
      (cc >= 80 || (cc >= 70 && std::is_same<T, natten::float16>::value))) {
    LAUNCH_na2d_nn_cuda_gemm(
        cc,
        T,
        std::get<0>(kernel_size), // TODO: remove when GEMM supports varying
                                  // kernel sizes
        dim,
        d_attn_ptr,
        key_ptr,
        d_query_ptr,
        batch_size,
        heads,
        height,
        width,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        std::get<0>(kernel_size), // TODO: remove when GEMM supports varying
                                  // kernel sizes
        std::get<0>(
            dilation), // TODO: remove when GEMM supports varying dilations
        1.0,
        stream);
    LAUNCH_na2d_in_cuda_gemm(
        cc,
        T,
        std::get<0>(kernel_size), // TODO: remove when GEMM supports varying
                                  // kernel sizes
        dim,
        d_attn_ptr,
        query_ptr,
        d_key_ptr,
        batch_size,
        heads,
        height,
        width,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        std::get<0>(kernel_size), // TODO: remove when GEMM supports varying
                                  // kernel sizes
        std::get<0>(
            dilation), // TODO: remove when GEMM supports varying dilations
        1.0,
        stream);
  } else {
#endif
    DISPATCH_DTYPE_na2d_nn_cuda_naive(
        T,
        is_causal,
        // kernel_size,
        // dilation,
        cc,
        stream,
        d_attn_ptr,
        key_ptr,
        d_query_ptr,
        batch_size,
        heads,
        height,
        width,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        kernel_size,
        dilation);
    DISPATCH_DTYPE_na2d_in_cuda_naive(
        T,
        is_causal,
        // kernel_size,
        // dilation,
        cc,
        stream,
        d_attn_ptr,
        query_ptr,
        d_key_ptr,
        batch_size,
        heads,
        height,
        width,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        kernel_size,
        dilation);
#ifdef NATTEN_WITH_CUTLASS
  }
#endif
  if (d_bias_ptr != nullptr) {
    NATTEN_CHECK(
        !any_true(is_causal),
        "Neighborhood attention with causal masking does not support positional biases yet.");
    DISPATCH_DTYPE_na2d_rpbgrad_cuda_naive(
        T,
        is_causal,
        // kernel_size,
        // dilation,
        cc,
        stream,
        d_bias_ptr,
        d_attn_ptr,
        batch_size,
        heads,
        height,
        width,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        kernel_size,
        dilation);
  }
}

template <typename T>
void na2d_av_forward(
    int32_t cc,
    size_t max_smem,
    cudaStream_t stream,
    void* attn_ptr,
    void* value_ptr,
    void* output_ptr,
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
#ifdef NATTEN_WITH_CUTLASS
  if (natten::kEnableGemmNA &&
      !any_true(is_causal) && // TODO: remove when GEMM supports causal masking
      all_dims_match(kernel_size) && // TODO: remove when GEMM supports varying
                                     // kernel sizes
      all_dims_match(
          dilation) && // TODO: remove when GEMM supports varying dilations
      (cc >= 80 || (cc >= 70 && std::is_same<T, natten::float16>::value))) {
    LAUNCH_na2d_nn_cuda_gemm(
        cc,
        T,
        std::get<0>(kernel_size), // TODO: remove when GEMM supports varying
                                  // kernel sizes
        dim,
        attn_ptr,
        value_ptr,
        output_ptr,
        batch_size,
        heads,
        height,
        width,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        std::get<0>(kernel_size), // TODO: remove when GEMM supports varying
                                  // kernel sizes
        std::get<0>(
            dilation), // TODO: remove when GEMM supports varying dilations
        1.0,
        stream);
  } else {
#endif
    DISPATCH_DTYPE_na2d_nn_cuda_naive(
        T,
        is_causal,
        // kernel_size,
        // dilation,
        cc,
        stream,
        attn_ptr,
        value_ptr,
        output_ptr,
        batch_size,
        heads,
        height,
        width,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        kernel_size,
        dilation);
#ifdef NATTEN_WITH_CUTLASS
  }
#endif
}

template <typename T>
void na2d_av_backward(
    int32_t cc,
    size_t max_smem,
    cudaStream_t stream,
    void* attn_ptr,
    void* value_ptr,
    void* d_output_ptr,
    void* d_attn_ptr,
    void* d_value_ptr,
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
#ifdef NATTEN_WITH_CUTLASS
  if (natten::kEnableGemmNA &&
      !any_true(is_causal) && // TODO: remove when GEMM supports causal masking
      all_dims_match(kernel_size) && // TODO: remove when GEMM supports varying
                                     // kernel sizes
      all_dims_match(
          dilation) && // TODO: remove when GEMM supports varying dilations
      (cc >= 80 || (cc >= 70 && std::is_same<T, natten::float16>::value))) {
    LAUNCH_na2d_pn_cuda_gemm(
        cc,
        T,
        std::get<0>(kernel_size), // TODO: remove when GEMM supports varying
                                  // kernel sizes
        dim,
        d_output_ptr,
        value_ptr,
        d_attn_ptr,
        nullptr,
        batch_size,
        heads,
        height,
        width,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        std::get<0>(kernel_size), // TODO: remove when GEMM supports varying
                                  // kernel sizes
        std::get<0>(
            dilation), // TODO: remove when GEMM supports varying dilations
        1.0,
        stream);
    LAUNCH_na2d_in_cuda_gemm(
        cc,
        T,
        std::get<0>(kernel_size), // TODO: remove when GEMM supports varying
                                  // kernel sizes
        dim,
        attn_ptr,
        d_output_ptr,
        d_value_ptr,
        batch_size,
        heads,
        height,
        width,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        std::get<0>(kernel_size), // TODO: remove when GEMM supports varying
                                  // kernel sizes
        std::get<0>(
            dilation), // TODO: remove when GEMM supports varying dilations
        1.0,
        stream);
  } else {
#endif
    DISPATCH_DTYPE_na2d_pn_cuda_naive(
        T,
        is_causal,
        // kernel_size,
        // dilation,
        cc,
        stream,
        /* is_grad = */ true,
        d_output_ptr,
        value_ptr,
        d_attn_ptr,
        batch_size,
        heads,
        height,
        width,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        kernel_size,
        dilation);
    DISPATCH_DTYPE_na2d_in_cuda_naive(
        T,
        is_causal,
        // kernel_size,
        // dilation,
        cc,
        stream,
        attn_ptr,
        d_output_ptr,
        d_value_ptr,
        batch_size,
        heads,
        height,
        width,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        kernel_size,
        dilation);
#ifdef NATTEN_WITH_CUTLASS
  }
#endif
}

} // namespace cuda
} // namespace natten
