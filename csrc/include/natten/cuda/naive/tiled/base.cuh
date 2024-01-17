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
    \brief Pointwise-Neighborhood kernel for 2D data.
           Computes attention weights between query points and their
   corresponding key neighborhood. Extra kernel with fused bias (relative
   positional bias.)
           + Tiled kernels for NA with window size 3, 5, 7, 9, 11, and 13 (only
   32 dim per head supported, and these kernels will not be updated anymore in
   favor of the cutlass kernels.)
*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <natten/cuda/naive/natten_commons.cuh>
#include <natten/cuda/naive/natten_tiled_macros.cuh>

namespace natten {
namespace cuda {
namespace naive {

template <typename scalar_t>
struct PointwiseNeighborhood2DBase {
  struct Params {
    bool is_grad;
    scalar_t* query; // query / d_out
    scalar_t* key; // key   / value
    scalar_t* bias = nullptr; // optional: bias
    scalar_t* attn; // attn  / d_attn
    int32_t height;
    int32_t width;
    int32_t heads;
    int32_t kernel_size_0, kernel_size_1;
    int32_t dilation_0, dilation_1;
    int32_t dim;
    int32_t batch_size;
    int64_t attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3;
    int64_t query_stride_0, query_stride_1, query_stride_2, query_stride_3;
    int64_t bias_stride_0, bias_stride_1;

    __device__ __host__ Params() {}

    __device__ __host__ Params(
        bool is_grad,
        scalar_t* query,
        scalar_t* key,
        scalar_t* attn,
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
        int64_t attn_stride_3)
        : is_grad(is_grad),
          query(query),
          key(key),
          attn(attn),
          height(height),
          width(width),
          heads(heads),
          kernel_size_0(kernel_size_0),
          kernel_size_1(kernel_size_1),
          dilation_0(dilation_0),
          dilation_1(dilation_1),
          dim(dim),
          batch_size(batch_size),
          bias_stride_1(0),
          bias_stride_0(0),
          attn_stride_3(attn_stride_3),
          attn_stride_2(attn_stride_2),
          attn_stride_1(attn_stride_1),
          attn_stride_0(attn_stride_0),
          query_stride_3(dim),
          query_stride_2(dim * width),
          query_stride_1(dim * width * height),
          query_stride_0(dim * width * height * heads) {}

    // CTOR with bias
    __device__ __host__ Params( // AV     / Q-grad
        scalar_t* query, // attn   / d_attn
        scalar_t* key, // value  / key
        scalar_t* bias, // relative positional bias tensor
        scalar_t* attn, // output / d_query
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
        int64_t attn_stride_3)
        : is_grad(false),
          query(query),
          key(key),
          bias(bias),
          attn(attn),
          height(height),
          width(width),
          heads(heads),
          kernel_size_0(kernel_size_0),
          kernel_size_1(kernel_size_1),
          dilation_0(dilation_0),
          dilation_1(dilation_1),
          dim(dim),
          batch_size(batch_size),
          bias_stride_1(2 * kernel_size_1 - 1),
          bias_stride_0((2 * kernel_size_0 - 1) * (2 * kernel_size_1 - 1)),
          attn_stride_3(attn_stride_3),
          attn_stride_2(attn_stride_2),
          attn_stride_1(attn_stride_1),
          attn_stride_0(attn_stride_0),
          query_stride_3(dim),
          query_stride_2(dim * width),
          query_stride_1(dim * width * height),
          query_stride_0(dim * width * height * heads) {}
  };

  __device__ __host__ PointwiseNeighborhood2DBase() {}

  static LaunchParams get_launch_params(
      int32_t batch_dim,
      int32_t spatial_size,
      int32_t attention_span) {
    int32_t KERNELTHREADS =
        min(CUDA_NUM_THREADS, attention_span /* == kernel_size^2 */);
    int32_t PIXELTHREADS =
        min(int32_t(CUDA_NUM_THREADS / KERNELTHREADS), spatial_size);
    int32_t BATCHTHREADS =
        min(64, max(1, CUDA_NUM_THREADS / (PIXELTHREADS * KERNELTHREADS)));
    dim3 grid(
        (spatial_size + PIXELTHREADS - 1) / PIXELTHREADS,
        (attention_span + KERNELTHREADS - 1) / KERNELTHREADS,
        (batch_dim + BATCHTHREADS - 1) / BATCHTHREADS);
    dim3 block(PIXELTHREADS, KERNELTHREADS, BATCHTHREADS);
    return LaunchParams(grid, block);
  }
};

} // namespace naive
} // namespace cuda
} // namespace natten
