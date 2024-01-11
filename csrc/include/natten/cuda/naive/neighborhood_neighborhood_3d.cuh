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
    \brief Neighborhood-Neighborhood kernel for 3D data.
           Applies neighborhood attention weights to neighborhood values.
*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <natten/cuda/naive/natten_commons.cuh>

namespace natten {
namespace cuda {
namespace naive {

template <typename scalar_t>
struct NeighborhoodNeighborhood3DBase {
  struct Params {
    scalar_t* weights;
    scalar_t* values;
    scalar_t* output;
    const int depth;
    const int height;
    const int width;
    const int heads;
    const int kernel_size_in;
    const int dilation_in;
    const int depth_kernel_size_in;
    const int depth_dilation_in;
    const int dim;
    const int64_t problem_size;
    const int64_t weights_stride_0, weights_stride_1, weights_stride_2,
        weights_stride_3, weights_stride_4;
    const int64_t values_stride_0, values_stride_1, values_stride_2,
        values_stride_3, values_stride_4;

    __device__ __host__ Params() {}

    __device__ __host__ Params( // AV     / Q-grad
        scalar_t* weights, // attn   / d_attn
        scalar_t* values, // value  / key
        scalar_t* output, // output / d_query
        const int depth,
        const int height,
        const int width,
        const int heads,
        const int kernel_size_in,
        const int dilation_in,
        const int depth_kernel_size_in,
        const int depth_dilation_in,
        const int dim,
        const int64_t weights_stride_0,
        const int64_t weights_stride_1,
        const int64_t weights_stride_2,
        const int64_t weights_stride_3,
        const int64_t weights_stride_4,
        const int problem_size)
        : weights(weights),
          values(values),
          output(output),
          depth(depth),
          height(height),
          width(width),
          heads(heads),
          kernel_size_in(kernel_size_in),
          dilation_in(dilation_in),
          depth_kernel_size_in(depth_kernel_size_in),
          depth_dilation_in(depth_dilation_in),
          dim(dim),
          problem_size(problem_size),
          weights_stride_4(weights_stride_4),
          weights_stride_3(weights_stride_3),
          weights_stride_2(weights_stride_2),
          weights_stride_1(weights_stride_1),
          weights_stride_0(weights_stride_0),
          values_stride_4(dim),
          values_stride_3(dim * width),
          values_stride_2(dim * width * height),
          values_stride_1(dim * width * height * depth),
          values_stride_0(dim * width * height * depth * heads) {}
  };

  __device__ __host__ NeighborhoodNeighborhood3DBase() {}

  static dim3 get_grid(int64_t problem_size_) {
    dim3 grid(GET_BLOCKS(problem_size_, /* CUDA_NUM_THREADS = */ 512));
    return grid;
  }

  static dim3 get_block() {
    dim3 block(/* CUDA_NUM_THREADS = */ 512);
    return block;
  }
};

template <
    typename scalar_t,
    int KS,
    int NS,
    int DILATION,
    int DKS,
    int DNS,
    int DDILATION>
struct NeighborhoodNeighborhood3DFull
    : NeighborhoodNeighborhood3DBase<scalar_t> {
  using Base = NeighborhoodNeighborhood3DBase<scalar_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = false;
  static constexpr bool IsHalfKernel = false;
  static constexpr bool UsesSmem = false;

  __device__ __host__ NeighborhoodNeighborhood3DFull() : Base() {}

  static __host__ int get_dim(int dim) {
    return dim;
  }

  __device__ void launch(Params p) {
    const int KERNEL_SIZE = (KS > 1) ? KS : p.kernel_size_in;
    const int KERNEL_SIZE_D = (DKS > 1) ? DKS : p.depth_kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS > 0) ? NS : KERNEL_SIZE / 2;
    const int NEIGHBORHOOD_SIZE_D = (DNS > 0) ? DNS : KERNEL_SIZE_D / 2;
    const int dilation = (DILATION > 0) ? DILATION : p.dilation_in;
    const int dilation_d = (DDILATION > 0) ? DDILATION : p.depth_dilation_in;
    const int64_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < p.problem_size) {
      int indtmp1 = linearIndex / p.dim;
      const int d = linearIndex - indtmp1 * p.dim;
      int indtmp2 = indtmp1 / p.width;
      const int j = indtmp1 - indtmp2 * p.width;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.height;
      const int i = indtmp1 - indtmp2 * p.height;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.depth;
      const int k = indtmp1 - indtmp2 * p.depth;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.heads;
      const int h = indtmp1 - indtmp2 * p.heads;
      const int b = indtmp2;

      const int ni = get_window_start(
          i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int nj = get_window_start(
          j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int nk = get_window_start(
          k, p.depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
      scalar_t updt = scalar_t(0);
      int weightsOffset = b * p.weights_stride_0 + h * p.weights_stride_1 +
          k * p.weights_stride_2 + i * p.weights_stride_3 +
          j * p.weights_stride_4;
      const int64_t valuesOffset =
          b * p.values_stride_0 + h * p.values_stride_1 + d;
#pragma unroll
      for (int xk = nk; xk < nk + KERNEL_SIZE_D * dilation_d; xk += dilation_d)
#pragma unroll
        for (int xi = ni; xi < ni + KERNEL_SIZE * dilation; xi += dilation)
#pragma unroll
          for (int xj = nj; xj < nj + KERNEL_SIZE * dilation; xj += dilation) {
            const int64_t valuesIndex = valuesOffset + xk * p.values_stride_2 +
                xi * p.values_stride_3 + xj * p.values_stride_4;
            updt += p.weights[weightsOffset] * p.values[valuesIndex];
            ++weightsOffset;
          }
      p.output[linearIndex] = updt;
    }
  }
};

template <
    typename scalar_t,
    int KS,
    int NS,
    int DILATION,
    int DKS,
    int DNS,
    int DDILATION>
struct NeighborhoodNeighborhood3DHalf
    : NeighborhoodNeighborhood3DBase<scalar_t> {
  using Base = NeighborhoodNeighborhood3DBase<scalar_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = IsBF16<scalar_t>::value;
  static constexpr bool IsHalfKernel = true;
  static constexpr bool UsesSmem = false;

  __device__ __host__ NeighborhoodNeighborhood3DHalf() : Base() {}

  using HalfHelper = typename HalfArray<scalar_t>::Base;

  static __host__ int get_dim(int dim) {
    if (dim % 2 != 0) {
      std::cerr
          << "Naive NATTEN half-precision kernels only support 32-bit alignment. "
          << "Hint: Make sure dimensions per head are multiples of 2."
          << std::endl;
      exit(EXIT_FAILURE);
    }
    return dim / 2;
  }

  __device__ void launch(Params p) {
    const int KERNEL_SIZE = (KS > 1) ? KS : p.kernel_size_in;
    const int KERNEL_SIZE_D = (DKS > 1) ? DKS : p.depth_kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS > 0) ? NS : KERNEL_SIZE / 2;
    const int NEIGHBORHOOD_SIZE_D = (DNS > 0) ? DNS : KERNEL_SIZE_D / 2;
    const int dilation = (DILATION > 0) ? DILATION : p.dilation_in;
    const int dilation_d = (DDILATION > 0) ? DDILATION : p.depth_dilation_in;
    const int64_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < p.problem_size) {
      auto values2 = HalfHelper::typecast(p.values);
      auto output2 = HalfHelper::typecast(p.output);
      int indtmp1 = linearIndex / p.dim;
      const int d = linearIndex - indtmp1 * p.dim;
      int indtmp2 = indtmp1 / p.width;
      const int j = indtmp1 - indtmp2 * p.width;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.height;
      const int i = indtmp1 - indtmp2 * p.height;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.depth;
      const int k = indtmp1 - indtmp2 * p.depth;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.heads;
      const int h = indtmp1 - indtmp2 * p.heads;
      const int b = indtmp2;

      const int ni = get_window_start(
          i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int nj = get_window_start(
          j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int nk = get_window_start(
          k, p.depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
      auto updt = HalfHelper::zero();
      int weightsOffset = b * p.weights_stride_0 + h * p.weights_stride_1 +
          k * p.weights_stride_2 + i * p.weights_stride_3 +
          j * p.weights_stride_4;
      const int64_t valuesOffset =
          b * p.values_stride_0 + h * p.values_stride_1 + d;
#pragma unroll
      for (int xk = nk; xk < nk + KERNEL_SIZE_D * dilation_d; xk += dilation_d)
#pragma unroll
        for (int xi = ni; xi < ni + KERNEL_SIZE * dilation; xi += dilation)
#pragma unroll
          for (int xj = nj; xj < nj + KERNEL_SIZE * dilation; xj += dilation) {
            const int64_t valuesIndex = valuesOffset + xk * p.values_stride_2 +
                xi * p.values_stride_3 + xj * p.values_stride_4;
            updt = HalfHelper::fma(
                values2[valuesIndex], p.weights[weightsOffset], updt);
            ++weightsOffset;
          }
      output2[linearIndex] = updt;
    }
  }
};

template <typename Args_>
struct NeighborhoodNeighborhood3D {
  using Args = Args_;
  static constexpr int KS = Args::KernelSize;
  static constexpr int NS = Args::NeighborhoodSize;
  static constexpr int DKS = Args::DepthKernelSize;
  static constexpr int DNS = Args::DepthNeighborhoodSize;
  static constexpr int DILATION = Args::Dilation;
  static constexpr int DDILATION = Args::DepthDilation;
  using scalar_t = typename Args::Dtype;
  using Kernel = typename std::conditional<
      sizeof(scalar_t) >= 4,
      NeighborhoodNeighborhood3DFull<
          scalar_t,
          KS,
          NS,
          DILATION,
          DKS,
          DNS,
          DDILATION>,
      NeighborhoodNeighborhood3DHalf<
          scalar_t,
          KS,
          NS,
          DILATION,
          DKS,
          DNS,
          DDILATION>>::type;
  using Params = typename Kernel::Params;

  void operator()(
      const int cc,
      cudaStream_t stream,
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
      int kernel_size_depth,
      int dilation,
      int dilation_depth) {
    dim = Kernel::get_dim(dim);
    int64_t problem_size = batch_size * heads * depth * height * width * dim;
    auto grid = Kernel::Base::get_grid(problem_size);
    auto block = Kernel::Base::get_block();
    auto params = Params(
        reinterpret_cast<scalar_t*>(attn_ptr),
        reinterpret_cast<scalar_t*>(value_ptr),
        reinterpret_cast<scalar_t*>(output_ptr),
        depth,
        height,
        width,
        heads,
        kernel_size,
        dilation,
        kernel_size_depth,
        dilation_depth,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        attn_stride_4,
        problem_size);
    launch_cuda_kernel<Kernel><<<grid, block, 0, stream>>>(params);
  }
};

} // namespace naive
} // namespace cuda
} // namespace natten
