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

#include <natten/naive_argpack.h>
#include <natten/natten.h>
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
    int32_t depth;
    int32_t height;
    int32_t width;
    int32_t heads;
    int32_t kernel_size_0, kernel_size_1, kernel_size_2;
    int32_t dilation_0, dilation_1, dilation_2;
    int32_t dim;
    int64_t problem_size;
    int64_t weights_stride_0, weights_stride_1, weights_stride_2,
        weights_stride_3, weights_stride_4;
    int64_t values_stride_0, values_stride_1, values_stride_2, values_stride_3,
        values_stride_4;

    __device__ __host__ Params() {}

    __device__ __host__ Params( // AV     / Q-grad
        scalar_t* weights, // attn   / d_attn
        scalar_t* values, // value  / key
        scalar_t* output, // output / d_query
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
        int32_t dim,
        int64_t weights_stride_0,
        int64_t weights_stride_1,
        int64_t weights_stride_2,
        int64_t weights_stride_3,
        int64_t weights_stride_4,
        int32_t problem_size)
        : weights(weights),
          values(values),
          output(output),
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

template <typename scalar_t, typename CausalMask_>
struct NeighborhoodNeighborhood3DFull
    : NeighborhoodNeighborhood3DBase<scalar_t> {
  using Base = NeighborhoodNeighborhood3DBase<scalar_t>;
  using Params = typename Base::Params;
  using CausalMask = CausalMask_;
  static constexpr bool IsBF16Kernel = false;
  static constexpr bool IsHalfKernel = false;
  static constexpr bool UsesSmem = false;

  __device__ __host__ NeighborhoodNeighborhood3DFull() : Base() {}

  static __host__ int32_t get_dim(int32_t dim) {
    return dim;
  }

  __device__ void launch(Params p) {
    auto mask_0 = NeighborhoodMask<CausalMask::Dim0>(
        p.depth, p.kernel_size_0, p.dilation_0);
    auto mask_1 = NeighborhoodMask<CausalMask::Dim1>(
        p.height, p.kernel_size_1, p.dilation_1);
    auto mask_2 = NeighborhoodMask<CausalMask::Dim2>(
        p.width, p.kernel_size_2, p.dilation_2);
    int64_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < p.problem_size) {
      int32_t indtmp1 = linearIndex / p.dim;
      auto d = linearIndex - indtmp1 * p.dim;
      int32_t indtmp2 = indtmp1 / p.width;
      auto j = indtmp1 - indtmp2 * p.width;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.height;
      auto i = indtmp1 - indtmp2 * p.height;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.depth;
      auto k = indtmp1 - indtmp2 * p.depth;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.heads;
      auto h = indtmp1 - indtmp2 * p.heads;
      auto& b = indtmp2;

      auto nk = mask_0.get_window_start(k);
      auto ni = mask_1.get_window_start(i);
      auto nj = mask_2.get_window_start(j);
      auto ek = mask_0.get_window_end(k, nk);
      auto ei = mask_1.get_window_end(i, ni);
      auto ej = mask_2.get_window_end(j, nj);

      scalar_t updt = scalar_t(0);
      int64_t weightsOffset = b * p.weights_stride_0 + h * p.weights_stride_1 +
          k * p.weights_stride_2 + i * p.weights_stride_3 +
          j * p.weights_stride_4;
      int64_t valuesOffset = b * p.values_stride_0 + h * p.values_stride_1 + d;
      for (int32_t xk = nk; xk < ek; xk += p.dilation_0)
        for (int32_t xi = ni; xi < ei; xi += p.dilation_1)
          for (int32_t xj = nj; xj < ej; xj += p.dilation_2) {
            int64_t valuesIndex = valuesOffset + xk * p.values_stride_2 +
                xi * p.values_stride_3 + xj * p.values_stride_4;
            int64_t weightsIndex = weightsOffset +
                int32_t((xk - nk) / p.dilation_0) *
                    (p.kernel_size_1 * p.kernel_size_2) +
                int32_t((xi - ni) / p.dilation_1) * p.kernel_size_2 +
                int32_t((xj - nj) / p.dilation_2);
            updt += p.weights[weightsIndex] * p.values[valuesIndex];
          }
      p.output[linearIndex] = updt;
    }
  }
};

template <typename scalar_t, typename CausalMask_>
struct NeighborhoodNeighborhood3DHalf
    : NeighborhoodNeighborhood3DBase<scalar_t> {
  using Base = NeighborhoodNeighborhood3DBase<scalar_t>;
  using Params = typename Base::Params;
  using CausalMask = CausalMask_;
  static constexpr bool IsBF16Kernel = IsBF16<scalar_t>::value;
  static constexpr bool IsHalfKernel = true;
  static constexpr bool UsesSmem = false;

  __device__ __host__ NeighborhoodNeighborhood3DHalf() : Base() {}

  using HalfHelper = typename HalfArray<scalar_t>::Base;

  static __host__ int32_t get_dim(int32_t dim) {
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
    auto mask_0 = NeighborhoodMask<CausalMask::Dim0>(
        p.depth, p.kernel_size_0, p.dilation_0);
    auto mask_1 = NeighborhoodMask<CausalMask::Dim1>(
        p.height, p.kernel_size_1, p.dilation_1);
    auto mask_2 = NeighborhoodMask<CausalMask::Dim2>(
        p.width, p.kernel_size_2, p.dilation_2);
    int64_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < p.problem_size) {
      auto values2 = HalfHelper::typecast(p.values);
      auto output2 = HalfHelper::typecast(p.output);
      int32_t indtmp1 = linearIndex / p.dim;
      auto d = linearIndex - indtmp1 * p.dim;
      int32_t indtmp2 = indtmp1 / p.width;
      auto j = indtmp1 - indtmp2 * p.width;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.height;
      auto i = indtmp1 - indtmp2 * p.height;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.depth;
      auto k = indtmp1 - indtmp2 * p.depth;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1 / p.heads;
      auto h = indtmp1 - indtmp2 * p.heads;
      auto& b = indtmp2;

      auto nk = mask_0.get_window_start(k);
      auto ni = mask_1.get_window_start(i);
      auto nj = mask_2.get_window_start(j);
      auto ek = mask_0.get_window_end(k, nk);
      auto ei = mask_1.get_window_end(i, ni);
      auto ej = mask_2.get_window_end(j, nj);

      auto updt = HalfHelper::zeros();
      int64_t weightsOffset = b * p.weights_stride_0 + h * p.weights_stride_1 +
          k * p.weights_stride_2 + i * p.weights_stride_3 +
          j * p.weights_stride_4;
      int64_t valuesOffset = b * p.values_stride_0 + h * p.values_stride_1 + d;
      for (int32_t xk = nk; xk < ek; xk += p.dilation_0)
        for (int32_t xi = ni; xi < ei; xi += p.dilation_1)
          for (int32_t xj = nj; xj < ej; xj += p.dilation_2) {
            int64_t valuesIndex = valuesOffset + xk * p.values_stride_2 +
                xi * p.values_stride_3 + xj * p.values_stride_4;
            int64_t weightsIndex = weightsOffset +
                int32_t((xk - nk) / p.dilation_0) *
                    (p.kernel_size_1 * p.kernel_size_2) +
                int32_t((xi - ni) / p.dilation_1) * p.kernel_size_2 +
                int32_t((xj - nj) / p.dilation_2);
            updt = HalfHelper::fma(
                values2[valuesIndex], p.weights[weightsIndex], updt);
          }
      output2[linearIndex] = updt;
    }
  }
};

template <typename Args_>
struct NeighborhoodNeighborhood3D {
  using Args = Args_;
  using scalar_t = typename Args::Dtype;
  using CausalMask = typename Args::CausalMask;
  using Kernel = typename std::conditional<
      sizeof(scalar_t) >= 4,
      NeighborhoodNeighborhood3DFull<scalar_t, CausalMask>,
      NeighborhoodNeighborhood3DHalf<scalar_t, CausalMask>>::type;
  using Params = typename Kernel::Params;

  void operator()(
      int32_t cc,
      cudaStream_t stream,
      void* attn_ptr,
      void* value_ptr,
      void* output_ptr,
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
        std::get<0>(kernel_size),
        std::get<1>(kernel_size),
        std::get<2>(kernel_size),
        std::get<0>(dilation),
        std::get<1>(dilation),
        std::get<2>(dilation),
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
