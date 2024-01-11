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
    \brief Inverse-Neighborhood-Neighborhood kernel for 2D data.
           Applies inverse neighborhood attention weights to inverse
   neighborhood values. Used to compute key and value grads.
*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <natten/cuda/naive/natten_commons.cuh>

namespace natten {
namespace cuda {
namespace naive {

template <typename scalar_t>
struct InverseNeighborhood2DBase {
  struct Params {
    scalar_t* weights;
    scalar_t* values;
    scalar_t* output;
    const int height;
    const int width;
    const int heads;
    const int kernel_size_in;
    const int dilation_in;
    const int dim;
    const int64_t problem_size;
    const int64_t weights_stride_0, weights_stride_1, weights_stride_2,
        weights_stride_3;
    const int64_t values_stride_0, values_stride_1, values_stride_2,
        values_stride_3;

    __device__ __host__ Params() {}

    __device__ __host__ Params( // AV     / Q-grad
        scalar_t* weights, // attn   / d_attn
        scalar_t* values, // value  / key
        scalar_t* output, // output / d_query
        const int height,
        const int width,
        const int heads,
        const int kernel_size_in,
        const int dilation_in,
        const int dim,
        const int64_t weights_stride_0,
        const int64_t weights_stride_1,
        const int64_t weights_stride_2,
        const int64_t weights_stride_3,
        const int problem_size)
        : weights(weights),
          values(values),
          output(output),
          height(height),
          width(width),
          heads(heads),
          kernel_size_in(kernel_size_in),
          dilation_in(dilation_in),
          dim(dim),
          problem_size(problem_size),
          weights_stride_3(weights_stride_3),
          weights_stride_2(weights_stride_2),
          weights_stride_1(weights_stride_1),
          weights_stride_0(weights_stride_0),
          values_stride_3(dim),
          values_stride_2(dim * width),
          values_stride_1(dim * width * height),
          values_stride_0(dim * width * height * heads) {}
  };

  __device__ __host__ InverseNeighborhood2DBase() {}

  static dim3 get_grid(int64_t problem_size_) {
    dim3 grid(GET_BLOCKS(problem_size_, /* CUDA_NUM_THREADS = */ 512));
    return grid;
  }

  static dim3 get_block() {
    dim3 block(/* CUDA_NUM_THREADS = */ 512);
    return block;
  }
};

template <typename scalar_t, int KS, int NS, int DILATION>
struct InverseNeighborhood2DFull : InverseNeighborhood2DBase<scalar_t> {
  using Base = InverseNeighborhood2DBase<scalar_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = false;
  static constexpr bool IsHalfKernel = false;
  static constexpr bool UsesSmem = false;

  __device__ __host__ InverseNeighborhood2DFull() : Base() {}

  static __host__ int get_dim(int dim) {
    return dim;
  }

  __device__ void launch(Params p) {
    const int KERNEL_SIZE = (KS > 1) ? KS : p.kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS > 0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION > 0) ? DILATION : p.dilation_in;
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
      indtmp2 = indtmp1 / p.heads;
      const int h = indtmp1 - indtmp2 * p.heads;
      const int b = indtmp2;
      const int ni = get_backward_window_start(
          i, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int nj = get_backward_window_start(
          j, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int ei = get_backward_window_end(
          i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int ej = get_backward_window_end(
          j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int64_t weightsOffset = b * p.weights_stride_0 + h * p.weights_stride_1;
      const int64_t valuesOffset =
          b * p.values_stride_0 + h * p.values_stride_1 + d;
      scalar_t output_update = scalar_t(0);
#pragma unroll
      for (int xi = ni; xi < ei; xi += dilation) {
        const int oni = get_window_start(
            xi, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
#pragma unroll
        for (int xj = nj; xj < ej; xj += dilation) {
          const int onj = get_window_start(
              xj, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
          const int64_t valuesIndex =
              valuesOffset + xi * p.values_stride_2 + xj * p.values_stride_3;
          const int64_t weightsIndex = weightsOffset + xi * p.weights_stride_2 +
              xj * p.weights_stride_3 +
              int((i - oni) / dilation) * KERNEL_SIZE +
              int((j - onj) / dilation);
          output_update += p.values[valuesIndex] * p.weights[weightsIndex];
        }
      }
      p.output[linearIndex] = output_update;
    }
  }
};

template <typename scalar_t, int KS, int NS, int DILATION>
struct InverseNeighborhood2DHalf : InverseNeighborhood2DBase<scalar_t> {
  using Base = InverseNeighborhood2DBase<scalar_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = IsBF16<scalar_t>::value;
  static constexpr bool IsHalfKernel = true;
  static constexpr bool UsesSmem = false;

  __device__ __host__ InverseNeighborhood2DHalf() : Base() {}

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
    const int NEIGHBORHOOD_SIZE = (NS > 0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION > 0) ? DILATION : p.dilation_in;
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
      indtmp2 = indtmp1 / p.heads;
      const int h = indtmp1 - indtmp2 * p.heads;
      const int b = indtmp2;
      const int ni = get_backward_window_start(
          i, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int nj = get_backward_window_start(
          j, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int ei = get_backward_window_end(
          i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int ej = get_backward_window_end(
          j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int64_t weightsOffset = b * p.weights_stride_0 + h * p.weights_stride_1;
      const int64_t valuesOffset =
          b * p.values_stride_0 + h * p.values_stride_1 + d;
      auto output_update = HalfHelper::zero();
#pragma unroll
      for (int xi = ni; xi < ei; xi += dilation) {
        const int oni = get_window_start(
            xi, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
#pragma unroll
        for (int xj = nj; xj < ej; xj += dilation) {
          const int onj = get_window_start(
              xj, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
          const int64_t valuesIndex =
              valuesOffset + xi * p.values_stride_2 + xj * p.values_stride_3;
          const int64_t weightsIndex = weightsOffset + xi * p.weights_stride_2 +
              xj * p.weights_stride_3 +
              int((i - oni) / dilation) * KERNEL_SIZE +
              int((j - onj) / dilation);
          output_update = HalfHelper::fma(
              values2[valuesIndex], p.weights[weightsOffset], output_update);
        }
      }
      output2[linearIndex] = output_update;
    }
  }
};

template <typename Args_>
struct InverseNeighborhood2D {
  using Args = Args_;
  static constexpr int KS = Args::KernelSize;
  static constexpr int NS = Args::NeighborhoodSize;
  static constexpr int DILATION = Args::Dilation;
  using scalar_t = typename Args::Dtype;
  using Kernel = typename std::conditional<
      sizeof(scalar_t) >= 4,
      InverseNeighborhood2DFull<scalar_t, KS, NS, DILATION>,
      InverseNeighborhood2DHalf<scalar_t, KS, NS, DILATION>>::type;
  using Params = typename Kernel::Params;

  void operator()(
      const int cc,
      cudaStream_t stream,
      void* attn_ptr,
      void* d_output_ptr,
      void* d_value_ptr,
      int batch_size,
      int heads,
      int height,
      int width,
      int dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int64_t attn_stride_3,
      int kernel_size,
      int dilation) {
    dim = Kernel::get_dim(dim);
    int64_t problem_size = batch_size * heads * height * width * dim;
    auto grid = Kernel::Base::get_grid(problem_size);
    auto block = Kernel::Base::get_block();
    auto params = Params(
        reinterpret_cast<scalar_t*>(attn_ptr),
        reinterpret_cast<scalar_t*>(d_output_ptr),
        reinterpret_cast<scalar_t*>(d_value_ptr),
        height,
        width,
        heads,
        kernel_size,
        dilation,
        dim,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        problem_size);
    launch_cuda_kernel<Kernel><<<grid, block, 0, stream>>>(params);
  }
};

} // namespace naive
} // namespace cuda
} // namespace natten
