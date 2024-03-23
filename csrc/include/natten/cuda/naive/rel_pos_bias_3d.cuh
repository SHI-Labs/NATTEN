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
    \brief Relative positional bias backward pass kernel for 3D data.
*/

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <natten/natten.h>
#include <natten/cuda/naive/natten_commons.cuh>

namespace natten {
namespace cuda {
namespace naive {

template <typename scalar_t, typename acc_t>
struct RelPosBiasGradient3DBase {
  struct Params {
    acc_t* d_bias;
    scalar_t* d_attn;
    int32_t depth;
    int32_t height;
    int32_t width;
    int32_t heads;
    int32_t kernel_size_0, kernel_size_1, kernel_size_2;
    int32_t dilation_0, dilation_1, dilation_2;
    int32_t batch_size;
    int32_t num_threads;
    int64_t problem_size;
    int64_t attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3,
        attn_stride_4;
    int64_t bias_stride_0, bias_stride_1, bias_stride_2;

    __device__ __host__ Params() {}

    __device__ __host__ Params(
        acc_t* d_bias,
        scalar_t* d_attn,
        int32_t depth,
        int32_t height,
        int32_t width,
        int32_t heads,
        int32_t kernel_size_0,
        int32_t kernel_size_1,
        int32_t kernel_size_2,
        int32_t dilation_0,
        int32_t dilation_1,
        int32_t dilation_2,
        int32_t batch_size,
        int64_t attn_stride_0,
        int64_t attn_stride_1,
        int64_t attn_stride_2,
        int64_t attn_stride_3,
        int64_t attn_stride_4,
        int64_t problem_size,
        int32_t num_threads)
        : d_bias(d_bias),
          d_attn(d_attn),
          depth(depth),
          height(height),
          width(width),
          heads(heads),
          kernel_size_0(kernel_size_0),
          kernel_size_1(kernel_size_1),
          kernel_size_2(kernel_size_2),
          dilation_0(dilation_0),
          dilation_1(dilation_1),
          dilation_2(dilation_2),
          batch_size(batch_size),
          problem_size(problem_size),
          num_threads(num_threads),
          bias_stride_2(2 * kernel_size_2 - 1),
          bias_stride_1((2 * kernel_size_1 - 1) * (2 * kernel_size_2 - 1)),
          bias_stride_0(
              (2 * kernel_size_0 - 1) * (2 * kernel_size_1 - 1) *
              (2 * kernel_size_2 - 1)),
          attn_stride_4(attn_stride_4),
          attn_stride_3(attn_stride_3),
          attn_stride_2(attn_stride_2),
          attn_stride_1(attn_stride_1),
          attn_stride_0(attn_stride_0) {}
  };

  __device__ __host__ RelPosBiasGradient3DBase() {}

  static LaunchParams get_launch_params(int32_t n_threads) {
    dim3 grid(GET_BLOCKS(n_threads, 128));
    dim3 block(128);
    return LaunchParams(grid, block);
  }
};

template <typename scalar_t, typename acc_t>
struct RelPosBiasGradient3DFull : RelPosBiasGradient3DBase<scalar_t, acc_t> {
  using Base = RelPosBiasGradient3DBase<scalar_t, acc_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = false;
  static constexpr bool IsHalfKernel = false;
  static constexpr bool UsesSmem = false;

  __device__ __host__ RelPosBiasGradient3DFull() : Base() {}

  __device__ void launch(Params p) {
    auto kernel_size_half_0 = p.kernel_size_0 / 2;
    auto kernel_size_half_1 = p.kernel_size_1 / 2;
    auto kernel_size_half_2 = p.kernel_size_2 / 2;
    int64_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < p.num_threads) {
      int32_t indtmp1 = linearIndex / p.kernel_size_2;
      auto kj = linearIndex - indtmp1 * p.kernel_size_2;
      int32_t indtmp2 = indtmp1 / p.kernel_size_1;
      auto ki = indtmp1 - indtmp2 * p.kernel_size_1;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.kernel_size_0;
      auto kk = indtmp1 - indtmp2 * p.kernel_size_0;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.width;
      auto j = indtmp1 - indtmp2 * p.width;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.height;
      auto i = indtmp1 - indtmp2 * p.height;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.depth;
      auto k = indtmp1 - indtmp2 * p.depth;
      auto& h = indtmp2;
      auto pk = get_pb_start(
          k, p.depth, p.kernel_size_0, kernel_size_half_0, p.dilation_0);
      auto pi = get_pb_start(
          i, p.height, p.kernel_size_1, kernel_size_half_1, p.dilation_1);
      auto pj = get_pb_start(
          j, p.width, p.kernel_size_2, kernel_size_half_2, p.dilation_2);
      acc_t d_rpb_update = acc_t(0);
      int64_t attnOffset = h * p.attn_stride_1 + k * p.attn_stride_2 +
          i * p.attn_stride_3 + j * p.attn_stride_4 +
          ((kk * p.kernel_size_1 * p.kernel_size_2) + (ki * p.kernel_size_2) +
           kj);
#pragma unroll
      for (int32_t b = 0; b < p.batch_size; ++b) {
        d_rpb_update += static_cast<acc_t>(p.d_attn[attnOffset]);
        attnOffset += p.attn_stride_0;
      }
      int64_t index = h * p.bias_stride_0 + (pk + kk) * p.bias_stride_1 +
          (pi + ki) * p.bias_stride_2 + (pj + kj);
      floatOrDoubleAtomicAdd(p.d_bias + index, d_rpb_update);
    }
  }
};

template <typename scalar_t, typename acc_t>
struct RelPosBiasGradient3DHalf : RelPosBiasGradient3DBase<scalar_t, acc_t> {
  using Base = RelPosBiasGradient3DBase<scalar_t, acc_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = IsBF16<scalar_t>::value;
  static constexpr bool IsHalfKernel = true;
  static constexpr bool UsesSmem = false;

  __device__ __host__ RelPosBiasGradient3DHalf() : Base() {}

  using HalfHelper = typename HalfArray<scalar_t>::Base;

  __device__ void launch(Params p) {
    auto kernel_size_half_0 = p.kernel_size_0 / 2;
    auto kernel_size_half_1 = p.kernel_size_1 / 2;
    auto kernel_size_half_2 = p.kernel_size_2 / 2;
    int64_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < p.num_threads) {
      int32_t indtmp1 = linearIndex / p.kernel_size_2;
      auto kj = linearIndex - indtmp1 * p.kernel_size_2;
      int32_t indtmp2 = indtmp1 / p.kernel_size_1;
      auto ki = indtmp1 - indtmp2 * p.kernel_size_1;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.kernel_size_0;
      auto kk = indtmp1 - indtmp2 * p.kernel_size_0;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.width;
      auto j = indtmp1 - indtmp2 * p.width;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.height;
      auto i = indtmp1 - indtmp2 * p.height;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.depth;
      auto k = indtmp1 - indtmp2 * p.depth;
      auto& h = indtmp2;
      auto pk = get_pb_start(
          k, p.depth, p.kernel_size_0, kernel_size_half_0, p.dilation_0);
      auto pi = get_pb_start(
          i, p.height, p.kernel_size_1, kernel_size_half_1, p.dilation_1);
      auto pj = get_pb_start(
          j, p.width, p.kernel_size_2, kernel_size_half_2, p.dilation_2);
      acc_t d_rpb_update = acc_t(0);
      int64_t attnOffset = h * p.attn_stride_1 + k * p.attn_stride_2 +
          i * p.attn_stride_3 + j * p.attn_stride_4 +
          ((kk * p.kernel_size_1 * p.kernel_size_2) + (ki * p.kernel_size_2) +
           kj);
#pragma unroll
      for (int32_t b = 0; b < p.batch_size; ++b) {
        d_rpb_update += HalfHelper::to_float(p.d_attn[attnOffset]);
        attnOffset += p.attn_stride_0;
      }
      int64_t index = h * p.bias_stride_0 + (pk + kk) * p.bias_stride_1 +
          (pi + ki) * p.bias_stride_2 + (pj + kj);
      atomicAdd(p.d_bias + index, d_rpb_update);
    }
  }
};

template <typename Args_>
struct RelPosBiasGradient3D {
  using Args = Args_;
  using scalar_t = typename Args::Dtype;
  using CausalMask = typename Args::CausalMask;
  using acc_t =
      typename std::conditional<sizeof(scalar_t) >= 4, scalar_t, float>::type;
  using Kernel = typename std::conditional<
      sizeof(scalar_t) >= 4,
      RelPosBiasGradient3DFull<scalar_t, acc_t>,
      RelPosBiasGradient3DHalf<scalar_t, acc_t>>::type;
  using Params = typename Kernel::Params;

  static_assert(
      !CausalMask::Dim0 && !CausalMask::Dim1 && !CausalMask::Dim2,
      "PN+Bias does not support causal masking.");

  void operator()(
      int32_t cc,
      cudaStream_t stream,
      void* d_bias_ptr,
      void* d_attn_ptr,
      int32_t batch_size,
      int32_t heads,
      int32_t depth,
      int32_t height,
      int32_t width,
      int32_t dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int64_t attn_stride_3,
      int64_t attn_stride_4,
      const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
      const std::tuple<int32_t, int32_t, int32_t>& dilation) {
    int32_t num_threads =
        heads * depth * height * width * natten::flatten(kernel_size);
    int64_t problem_size = heads * (2 * std::get<0>(kernel_size) - 1) *
        (2 * std::get<1>(kernel_size) - 1) * (2 * std::get<2>(kernel_size) - 1);
    LaunchParams lp = Kernel::Base::get_launch_params(num_threads);
    auto params = Params(
        reinterpret_cast<acc_t*>(d_bias_ptr),
        reinterpret_cast<scalar_t*>(d_attn_ptr),
        depth,
        height,
        width,
        heads,
        std::get<0>(kernel_size),
        std::get<1>(kernel_size),
        std::get<2>(kernel_size),
        std::get<0>(dilation),
        std::get<1>(dilation),
        std::get<2>(dilation),
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        attn_stride_4,
        problem_size,
        num_threads);
    launch_cuda_kernel<Kernel><<<lp.grid, lp.block, 0, stream>>>(params);
  }
};

} // namespace naive
} // namespace cuda
} // namespace natten
