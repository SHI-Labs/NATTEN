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
    \brief Relative positional bias backward pass kernel for 2D data.
*/

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

// We're still using ATen's atomic add!
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>

#include "natten/cuda/naive/natten_commons.cuh"

namespace natten {
namespace cuda {
namespace naive {

template <typename scalar_t, typename acc_t>
struct RelPosBiasGradient2DBase {

  struct Params {
    acc_t* d_bias;
    scalar_t* d_attn;
    const int height;
    const int width;
    const int heads;
    const int kernel_size_in;
    const int dilation_in;
    const int batch_size;
    const int problem_size, num_threads;
    const int attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3;
    const int bias_stride_0, bias_stride_1;

    __device__  __host__ Params() {}

    __device__  __host__ Params(
      acc_t* d_bias,
      scalar_t* d_attn,
      const int height,
      const int width,
      const int heads,
      const int kernel_size_in,
      const int dilation_in,
      const int batch_size,
      const int problem_size,
      const int num_threads): 
      d_bias(d_bias),
      d_attn(d_attn),
      height(height),
      width(width),
      heads(heads),
      kernel_size_in(kernel_size_in),
      dilation_in(dilation_in),
      batch_size(batch_size),
      problem_size(problem_size),
      num_threads(num_threads),
      bias_stride_1(2 * kernel_size_in - 1),
      bias_stride_0((2 * kernel_size_in - 1) * (2 * kernel_size_in - 1)),
      attn_stride_3(kernel_size_in * kernel_size_in),
      attn_stride_2(kernel_size_in * kernel_size_in * width),
      attn_stride_1(kernel_size_in * kernel_size_in * width * height),
      attn_stride_0(kernel_size_in * kernel_size_in * width * height * heads) {}
  };

  __device__ __host__ RelPosBiasGradient2DBase() {}

  static LaunchParams get_launch_params(int n_threads) {
    dim3 grid(GET_BLOCKS(n_threads, 64));
    dim3 block(64);
    return LaunchParams(grid, block);
  }
};


template <typename scalar_t, typename acc_t, int KS, int NS, int DILATION>
struct RelPosBiasGradient2DFull: RelPosBiasGradient2DBase<scalar_t, acc_t> {
  using Base   = RelPosBiasGradient2DBase<scalar_t, acc_t>;
  using Params = typename Base::Params;

  __device__ __host__ RelPosBiasGradient2DFull(): Base() {}

  __device__ void launch(Params p) {
    const int KERNEL_SIZE = (KS>1) ? KS : p.kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : p.dilation_in;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < p.num_threads){
      int indtmp1 = linearIndex/KERNEL_SIZE;
      const int kj = linearIndex - indtmp1 * KERNEL_SIZE;
      int indtmp2 = indtmp1/KERNEL_SIZE;
      const int ki = indtmp1 - indtmp2 * KERNEL_SIZE;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1/p.width;
      const int j = indtmp1 - indtmp2 * p.width;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1/p.height;
      const int i = indtmp1 - indtmp2 * p.height;
      const int h = indtmp2;
      const int pi = get_pb_start(i, p.height, KERNEL_SIZE,   NEIGHBORHOOD_SIZE,   dilation);
      const int pj = get_pb_start(j, p.width,  KERNEL_SIZE,   NEIGHBORHOOD_SIZE,   dilation);
      acc_t d_rpb_update = acc_t(0);
      int attnOffset = h * p.attn_stride_1 + i * p.attn_stride_2 + j * p.attn_stride_3 + (ki*KERNEL_SIZE+kj);
      #pragma unroll
      for (int b=0; b < p.batch_size; ++b){
        d_rpb_update += static_cast<acc_t>(p.d_attn[attnOffset]);
        attnOffset += p.attn_stride_0;
      }
      const int index = h * p.bias_stride_0 + (pi+ki) * p.bias_stride_1 + (pj+kj);
      at::native::fastAtomicAdd(p.d_bias, index, p.problem_size, d_rpb_update, true);
    }
  }
};

template <typename scalar_t, typename acc_t, int KS, int NS, int DILATION>
struct RelPosBiasGradient2DHalf: RelPosBiasGradient2DBase<scalar_t, acc_t> {
  using Base   = RelPosBiasGradient2DBase<scalar_t, acc_t>;
  using Params = typename Base::Params;

  __device__  __host__ RelPosBiasGradient2DHalf(): Base() {}

  using HalfHelper = typename HalfArray<scalar_t>::Base;

  __device__ void launch(Params p) {
    const int KERNEL_SIZE = (KS>1) ? KS : p.kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : p.dilation_in;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < p.num_threads){
      int indtmp1 = linearIndex/KERNEL_SIZE;
      const int kj = linearIndex - indtmp1 * KERNEL_SIZE;
      int indtmp2 = indtmp1/KERNEL_SIZE;
      const int ki = indtmp1 - indtmp2 * KERNEL_SIZE;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1/p.width;
      const int j = indtmp1 - indtmp2 * p.width;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1/p.height;
      const int i = indtmp1 - indtmp2 * p.height;
      const int h = indtmp2;
      const int pi = get_pb_start(i, p.height, KERNEL_SIZE,   NEIGHBORHOOD_SIZE,   dilation);
      const int pj = get_pb_start(j, p.width,  KERNEL_SIZE,   NEIGHBORHOOD_SIZE,   dilation);
      acc_t d_rpb_update = acc_t(0);
      int attnOffset = h * p.attn_stride_1 + i * p.attn_stride_2 + j * p.attn_stride_3 + (ki*KERNEL_SIZE+kj);
      #pragma unroll
      for (int b=0; b < p.batch_size; ++b){
        d_rpb_update += HalfHelper::to_float(p.d_attn[attnOffset]);
        attnOffset += p.attn_stride_0;
      }
      const int index = h * p.bias_stride_0 + (pi+ki) * p.bias_stride_1 + (pj+kj);
      at::native::fastAtomicAdd(p.d_bias, index, p.problem_size, d_rpb_update, true);
    }
  }
};

template <typename Args_>
struct RelPosBiasGradient2D {
  using Args = Args_;
  static constexpr int KS = Args::KernelSize;
  static constexpr int NS = Args::NeighborhoodSize;
  static constexpr int DILATION = Args::Dilation;
  using scalar_t = typename Args::Dtype;
  using acc_t = typename std::conditional<
    sizeof(scalar_t) >= 4, 
    scalar_t, 
    float
  >::type;
  using Kernel = typename std::conditional<
    sizeof(scalar_t) >= 4, 
    RelPosBiasGradient2DFull<scalar_t, acc_t, KS, NS, DILATION>, 
    RelPosBiasGradient2DHalf<scalar_t, acc_t, KS, NS, DILATION>
  >::type;
  using Params = typename Kernel::Params;

  void operator()(
    void * d_bias_ptr,
    void * d_attn_ptr,
    int batch_size,
    int heads,
    int height,
    int width,
    int dim,
    int kernel_size,
    int dilation) {
    int num_threads = heads * height * width * kernel_size * kernel_size;
    int problem_size = heads * (2 * kernel_size - 1);
    LaunchParams lp = Kernel::Base::get_launch_params(num_threads);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    auto params = Params(
      reinterpret_cast<acc_t*>(d_bias_ptr),
      reinterpret_cast<scalar_t*>(d_attn_ptr),
      height, width, heads, kernel_size, dilation,
      batch_size, problem_size, num_threads);
    launch_cuda_kernel<Kernel><<<lp.grid, lp.block, 0, stream>>>(params);
  }
};


} // namespace naive
} // namespace cuda
} // namespace natten
