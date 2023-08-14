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
    \brief Inverse-Neighborhood-Neighborhood kernel for 3D data.
           Applies inverse neighborhood attention weights to inverse neighborhood values.
           Used to compute key and value grads.
*/

#pragma once
// TODO: remaining dependency to torch: getCurrentCUDAStream
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "natten/cuda/naive/natten_commons.cuh"

namespace natten {
namespace cuda {
namespace naive {

template <typename scalar_t>
struct InverseNeighborhood3DBase {

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
    const int weights_stride_0, weights_stride_1, weights_stride_2, weights_stride_3, weights_stride_4;
    const int values_stride_0, values_stride_1, values_stride_2, values_stride_3, values_stride_4;

    __device__  __host__ Params() {}

    __device__  __host__ Params(// AV     / Q-grad
      scalar_t* weights,        // attn   / d_attn
      scalar_t* values,         // value  / key
      scalar_t* output,         // output / d_query
      const int depth,
      const int height,
      const int width,
      const int heads,
      const int kernel_size_in,
      const int dilation_in,
      const int depth_kernel_size_in,
      const int depth_dilation_in,
      const int dim,
      const int problem_size): 
      weights(weights),
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
      weights_stride_4(kernel_size_in * kernel_size_in * depth_kernel_size_in),
      weights_stride_3(kernel_size_in * kernel_size_in * depth_kernel_size_in * width),
      weights_stride_2(kernel_size_in * kernel_size_in * depth_kernel_size_in * width * height),
      weights_stride_1(kernel_size_in * kernel_size_in * depth_kernel_size_in * width * height * depth),
      weights_stride_0(kernel_size_in * kernel_size_in * depth_kernel_size_in * width * height * depth * heads),
      values_stride_4(dim),
      values_stride_3(dim * width),
      values_stride_2(dim * width * height),
      values_stride_1(dim * width * height * depth),
      values_stride_0(dim * width * height * depth * heads) {}
  };

  __device__ __host__ InverseNeighborhood3DBase() {}

  static dim3 get_grid(int64_t problem_size_) {
    dim3 grid(GET_BLOCKS(problem_size_, /* CUDA_NUM_THREADS = */ 512));
    return grid;
  }

  static dim3 get_block() {
    dim3 block(/* CUDA_NUM_THREADS = */ 512);
    return block;
  }
};


template <typename scalar_t, int KS, int NS, int DILATION, int DKS, int DNS, int DDILATION>
struct InverseNeighborhood3DFull: InverseNeighborhood3DBase<scalar_t> {
  using Base   = InverseNeighborhood3DBase<scalar_t>;
  using Params = typename Base::Params;

  __device__ __host__ InverseNeighborhood3DFull(): Base() {}

  static __host__ int get_dim(int dim) {
    return dim;
  }

  __device__ void launch(Params p) {
    const int KERNEL_SIZE = (KS>1) ? KS : p.kernel_size_in;
    const int KERNEL_SIZE_D = (DKS>1) ? DKS : p.depth_kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int NEIGHBORHOOD_SIZE_D = (DNS>0) ? DNS : KERNEL_SIZE_D / 2;
    const int dilation = (DILATION>0) ? DILATION : p.dilation_in;
    const int dilation_d = (DDILATION>0) ? DDILATION : p.depth_dilation_in;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < p.problem_size){
      int indtmp1 = linearIndex/p.dim;
      const int d = linearIndex - indtmp1 * p.dim;
      int indtmp2 = indtmp1/p.width;
      const int j = indtmp1 - indtmp2 * p.width;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1/p.height;
      const int i = indtmp1 - indtmp2 * p.height;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1/p.depth;
      const int k = indtmp1 - indtmp2 * p.depth;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1/p.heads;
      const int h = indtmp1 - indtmp2 * p.heads;
      const int b = indtmp2;
      const int ni = get_backward_window_start(i,   KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int nj = get_backward_window_start(j,   KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int nk = get_backward_window_start(k, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
      const int ei = get_backward_window_end(i, p.height, KERNEL_SIZE,   NEIGHBORHOOD_SIZE,   dilation);
      const int ej = get_backward_window_end(j, p.width,  KERNEL_SIZE,   NEIGHBORHOOD_SIZE,   dilation);
      const int ek = get_backward_window_end(k, p.depth,  KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
      const int weightsOffset = b * p.weights_stride_0 + h * p.weights_stride_1;
      const int valuesOffset = b * p.values_stride_0 + h * p.values_stride_1 + d;
      scalar_t output_update = scalar_t(0);
      #pragma unroll
      for (int xk=nk; xk < ek; xk+=dilation_d){
        const int onk = get_window_start(xk, p.depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
        #pragma unroll
        for (int xi=ni; xi < ei; xi+=dilation){
          const int oni = get_window_start(xi, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
          #pragma unroll
          for (int xj=nj; xj < ej; xj+=dilation){
            const int onj = get_window_start(xj, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            const int valuesIndex = valuesOffset + xk * p.values_stride_2 + xi * p.values_stride_3 + xj * p.values_stride_4;
            const int weightsIndex = weightsOffset + xk * p.weights_stride_2 + xi * p.weights_stride_3 + xj * p.weights_stride_4 + (int((k-onk)/dilation_d)*KERNEL_SIZE*KERNEL_SIZE)+int((i-oni)/dilation)*KERNEL_SIZE+int((j-onj)/dilation);
            output_update += p.values[valuesIndex] * p.weights[weightsIndex];
          }
        }
      }
      p.output[linearIndex] = output_update;
    }
  }
};

template <typename scalar_t, int KS, int NS, int DILATION, int DKS, int DNS, int DDILATION>
struct InverseNeighborhood3DHalf: InverseNeighborhood3DBase<scalar_t> {
  using Base   = InverseNeighborhood3DBase<scalar_t>;
  using Params = typename Base::Params;

  __device__  __host__ InverseNeighborhood3DHalf(): Base() {}

  using HalfHelper = typename HalfArray<scalar_t>::Base;

  static __host__ int get_dim(int dim) {
    if (dim % 2 != 0) {
      std::cerr << "Naive NATTEN half-precision kernels only support 32-bit alignment. "
                << "Hint: Make sure dimensions per head are multiples of 2."
                << std::endl;
      exit(EXIT_FAILURE);
    }
    return dim / 2;
  }

  __device__ void launch(Params p) {
    const int KERNEL_SIZE = (KS>1) ? KS : p.kernel_size_in;
    const int KERNEL_SIZE_D = (DKS>1) ? DKS : p.depth_kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int NEIGHBORHOOD_SIZE_D = (DNS>0) ? DNS : KERNEL_SIZE_D / 2;
    const int dilation = (DILATION>0) ? DILATION : p.dilation_in;
    const int dilation_d = (DDILATION>0) ? DDILATION : p.depth_dilation_in;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < p.problem_size){
      auto values2 = HalfHelper::typecast(p.values);
      auto output2 = HalfHelper::typecast(p.output);
      int indtmp1 = linearIndex/p.dim;
      const int d = linearIndex - indtmp1 * p.dim;
      int indtmp2 = indtmp1/p.width;
      const int j = indtmp1 - indtmp2 * p.width;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1/p.height;
      const int i = indtmp1 - indtmp2 * p.height;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1/p.depth;
      const int k = indtmp1 - indtmp2 * p.depth;
      indtmp1 = indtmp2;
      indtmp2 = indtmp1/p.heads;
      const int h = indtmp1 - indtmp2 * p.heads;
      const int b = indtmp2;
      const int ni = get_backward_window_start(i,   KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int nj = get_backward_window_start(j,   KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int nk = get_backward_window_start(k, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
      const int ei = get_backward_window_end(i, p.height, KERNEL_SIZE,   NEIGHBORHOOD_SIZE,   dilation);
      const int ej = get_backward_window_end(j, p.width,  KERNEL_SIZE,   NEIGHBORHOOD_SIZE,   dilation);
      const int ek = get_backward_window_end(k, p.depth,  KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
      const int weightsOffset = b * p.weights_stride_0 + h * p.weights_stride_1;
      const int valuesOffset = b * p.values_stride_0 + h * p.values_stride_1 + d;
      auto output_update = HalfHelper::zero();
      #pragma unroll
      for (int xk=nk; xk < ek; xk+=dilation_d){
        const int onk = get_window_start(xk, p.depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
        #pragma unroll
        for (int xi=ni; xi < ei; xi+=dilation){
          const int oni = get_window_start(xi, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
          #pragma unroll
          for (int xj=nj; xj < ej; xj+=dilation){
            const int onj = get_window_start(xj, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            const int valuesIndex = valuesOffset + xk * p.values_stride_2 + xi * p.values_stride_3 + xj * p.values_stride_4;
            const int weightsIndex = weightsOffset + xk * p.weights_stride_2 + xi * p.weights_stride_3 + xj * p.weights_stride_4 + (int((k-onk)/dilation_d)*KERNEL_SIZE*KERNEL_SIZE)+int((i-oni)/dilation)*KERNEL_SIZE+int((j-onj)/dilation);
            output_update = HalfHelper::fma(values2[valuesIndex], p.weights[weightsOffset], output_update);
          }
        }
      }
      output2[linearIndex] = output_update;
    }
  }
};

template <typename Args_>
struct InverseNeighborhood3D {
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
    InverseNeighborhood3DFull<scalar_t, KS, NS, DILATION, DKS, DNS, DDILATION>, 
    InverseNeighborhood3DHalf<scalar_t, KS, NS, DILATION, DKS, DNS, DDILATION>
  >::type;
  using Params = typename Kernel::Params;

  void operator()(
    void * attn_ptr,
    void * d_output_ptr,
    void * d_value_ptr,
    int batch_size,
    int heads,
    int depth,
    int height,
    int width,
    int dim,
    int kernel_size,
    int kernel_size_depth,
    int dilation,
    int dilation_depth) {
    dim = Kernel::get_dim(dim);
    int64_t problem_size = batch_size * heads * depth * height * width * dim;
    auto grid  = Kernel::Base::get_grid(problem_size);
    auto block = Kernel::Base::get_block();
    const auto stream = c10::cuda::getCurrentCUDAStream();
    auto params = Params(
      reinterpret_cast<scalar_t*>(attn_ptr),
      reinterpret_cast<scalar_t*>(d_output_ptr),
      reinterpret_cast<scalar_t*>(d_value_ptr),
      depth, height, width,
      heads, 
      kernel_size, dilation, 
      kernel_size_depth, dilation_depth, 
      dim, problem_size);
    launch_cuda_kernel<Kernel><<<grid, block, 0, stream>>>(params);
  }
};


} // namespace naive
} // namespace cuda
} // namespace natten
