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
    \brief Relative positional bias backward pass kernel for 1D data.
*/

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <natten/cuda/naive/natten_commons.cuh>

// TODO: We're still using ATen's atomic add!
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <ATen/native/cuda/KernelUtils.cuh>

namespace natten {
namespace cuda {
namespace naive {

template <typename scalar_t, typename acc_t>
struct RelPosBiasGradient1DBase {
  struct Params {
    acc_t* d_bias;
    scalar_t* d_attn;
    const int length;
    const int heads;
    const int kernel_size_in;
    const int dilation_in;
    const int batch_size;
    const int num_threads;
    const int64_t problem_size;
    const int64_t attn_stride_0, attn_stride_1, attn_stride_2;
    const int64_t bias_stride_0;

    __device__ __host__ Params() {}

    __device__ __host__ Params(
        acc_t* d_bias,
        scalar_t* d_attn,
        const int length,
        const int heads,
        const int kernel_size_in,
        const int dilation_in,
        const int batch_size,
        const int64_t attn_stride_0,
        const int64_t attn_stride_1,
        const int64_t attn_stride_2,
        const int64_t problem_size,
        const int num_threads)
        : d_bias(d_bias),
          d_attn(d_attn),
          length(length),
          heads(heads),
          kernel_size_in(kernel_size_in),
          dilation_in(dilation_in),
          batch_size(batch_size),
          problem_size(problem_size),
          num_threads(num_threads),
          bias_stride_0(2 * kernel_size_in - 1),
          attn_stride_2(attn_stride_2),
          attn_stride_1(attn_stride_1),
          attn_stride_0(attn_stride_0) {}
  };

  __device__ __host__ RelPosBiasGradient1DBase() {}

  static LaunchParams get_launch_params(int n_threads) {
    dim3 grid(GET_BLOCKS(n_threads, 64));
    dim3 block(64);
    return LaunchParams(grid, block);
  }
};

template <typename scalar_t, typename acc_t, int KS, int NS, int DILATION>
struct RelPosBiasGradient1DFull : RelPosBiasGradient1DBase<scalar_t, acc_t> {
  using Base = RelPosBiasGradient1DBase<scalar_t, acc_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = false;
  static constexpr bool IsHalfKernel = false;
  static constexpr bool UsesSmem = false;

  __device__ __host__ RelPosBiasGradient1DFull() : Base() {}

  __device__ void launch(Params p) {
    const int KERNEL_SIZE = (KS > 1) ? KS : p.kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS > 0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION > 0) ? DILATION : p.dilation_in;
    const int64_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < p.num_threads) {
      int indtmp1 = linearIndex / KERNEL_SIZE;
      const int ki = linearIndex - indtmp1 * KERNEL_SIZE;
      const int h = indtmp1 / p.length;
      const int i = indtmp1 - h * p.length;
      const int pi =
          get_pb_start(i, p.length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      acc_t d_rpb_update = acc_t(0);
      int64_t attnOffset = h * p.attn_stride_1 + i * p.attn_stride_2 + ki;
#pragma unroll
      for (int b = 0; b < p.batch_size; ++b) {
        d_rpb_update += static_cast<acc_t>(p.d_attn[attnOffset]);
        attnOffset += p.attn_stride_0;
      }
      const int64_t index = h * p.bias_stride_0 + (pi + ki);
      at::native::fastAtomicAdd(
          p.d_bias, index, p.problem_size, d_rpb_update, true);
    }
  }
};

template <typename scalar_t, typename acc_t, int KS, int NS, int DILATION>
struct RelPosBiasGradient1DHalf : RelPosBiasGradient1DBase<scalar_t, acc_t> {
  using Base = RelPosBiasGradient1DBase<scalar_t, acc_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = IsBF16<scalar_t>::value;
  static constexpr bool IsHalfKernel = true;
  static constexpr bool UsesSmem = false;

  __device__ __host__ RelPosBiasGradient1DHalf() : Base() {}

  using HalfHelper = typename HalfArray<scalar_t>::Base;

  __device__ void launch(Params p) {
    const int KERNEL_SIZE = (KS > 1) ? KS : p.kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS > 0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION > 0) ? DILATION : p.dilation_in;
    const int64_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < p.num_threads) {
      int indtmp1 = linearIndex / KERNEL_SIZE;
      const int ki = linearIndex - indtmp1 * KERNEL_SIZE;
      const int h = indtmp1 / p.length;
      const int i = indtmp1 - h * p.length;
      const int pi =
          get_pb_start(i, p.length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      acc_t d_rpb_update = acc_t(0);
      int64_t attnOffset = h * p.attn_stride_1 + i * p.attn_stride_2 + ki;
#pragma unroll
      for (int b = 0; b < p.batch_size; ++b) {
        d_rpb_update += HalfHelper::to_float(p.d_attn[attnOffset]);
        attnOffset += p.attn_stride_0;
      }
      const int64_t index = h * p.bias_stride_0 + (pi + ki);
      at::native::fastAtomicAdd(
          p.d_bias, index, p.problem_size, d_rpb_update, true);
    }
  }
};

template <typename Args_>
struct RelPosBiasGradient1D {
  using Args = Args_;
  static constexpr int KS = Args::KernelSize;
  static constexpr int NS = Args::NeighborhoodSize;
  static constexpr int DILATION = Args::Dilation;
  using scalar_t = typename Args::Dtype;
  using acc_t =
      typename std::conditional<sizeof(scalar_t) >= 4, scalar_t, float>::type;
  using Kernel = typename std::conditional<
      sizeof(scalar_t) >= 4,
      RelPosBiasGradient1DFull<scalar_t, acc_t, KS, NS, DILATION>,
      RelPosBiasGradient1DHalf<scalar_t, acc_t, KS, NS, DILATION>>::type;
  using Params = typename Kernel::Params;

  void operator()(
      const int cc,
      cudaStream_t stream,
      void* d_bias_ptr,
      void* d_attn_ptr,
      int batch_size,
      int heads,
      int length,
      int dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int kernel_size,
      int dilation) {
    int num_threads = heads * length * kernel_size;
    int64_t problem_size = heads * (2 * kernel_size - 1);
    LaunchParams lp = Kernel::Base::get_launch_params(num_threads);
    auto params = Params(
        reinterpret_cast<acc_t*>(d_bias_ptr),
        reinterpret_cast<scalar_t*>(d_attn_ptr),
        length,
        heads,
        kernel_size,
        dilation,
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        problem_size,
        num_threads);
    launch_cuda_kernel<Kernel><<<lp.grid, lp.block, 0, stream>>>(params);
  }
};

} // namespace naive
} // namespace cuda
} // namespace natten
