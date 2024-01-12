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
    \brief Pointwise-Neighborhood kernel for 1D data.
           Computes attention weights between query points and their
   corresponding key neighborhood. Extra kernel with fused bias (relative
   positional bias.)
*/

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <natten/cuda/naive/natten_commons.cuh>

namespace natten {
namespace cuda {
namespace naive {

template <typename scalar_t>
struct PointwiseNeighborhood1DBase {
  struct Params {
    scalar_t* query; // query / d_out
    scalar_t* key; // key   / value
    scalar_t* bias = nullptr; // optional: bias
    scalar_t* attn; // attn  / d_attn
    const int length;
    const int heads;
    const int kernel_size_in;
    const int dilation_in;
    const int dim;
    const int batch_size;
    const int64_t attn_stride_0, attn_stride_1, attn_stride_2;
    const int64_t query_stride_0, query_stride_1, query_stride_2;
    const int64_t bias_stride_0;

    __device__ __host__ Params() {}

    __device__ __host__ Params(
        scalar_t* query,
        scalar_t* key,
        scalar_t* attn,
        const int length,
        const int heads,
        const int kernel_size_in,
        const int dilation_in,
        const int dim,
        const int batch_size,
        const int64_t attn_stride_0,
        const int64_t attn_stride_1,
        const int64_t attn_stride_2)
        : query(query),
          key(key),
          attn(attn),
          length(length),
          heads(heads),
          kernel_size_in(kernel_size_in),
          dilation_in(dilation_in),
          dim(dim),
          batch_size(batch_size),
          bias_stride_0(0),
          attn_stride_2(attn_stride_2),
          attn_stride_1(attn_stride_1),
          attn_stride_0(attn_stride_0),
          query_stride_2(dim),
          query_stride_1(dim * length),
          query_stride_0(dim * length * heads) {}

    // CTOR with bias
    __device__ __host__ Params( // AV     / Q-grad
        scalar_t* query, // attn   / d_attn
        scalar_t* key, // value  / key
        scalar_t* bias, // relative positional bias tensor
        scalar_t* attn, // output / d_query
        const int length,
        const int heads,
        const int kernel_size_in,
        const int dilation_in,
        const int dim,
        const int batch_size,
        const int64_t attn_stride_0,
        const int64_t attn_stride_1,
        const int64_t attn_stride_2)
        : query(query),
          key(key),
          bias(bias),
          attn(attn),
          length(length),
          heads(heads),
          kernel_size_in(kernel_size_in),
          dilation_in(dilation_in),
          dim(dim),
          batch_size(batch_size),
          bias_stride_0(2 * kernel_size_in - 1),
          attn_stride_2(attn_stride_2),
          attn_stride_1(attn_stride_1),
          attn_stride_0(attn_stride_0),
          query_stride_2(dim),
          query_stride_1(dim * length),
          query_stride_0(dim * length * heads) {}
  };

  __device__ __host__ PointwiseNeighborhood1DBase() {}

  static LaunchParams get_launch_params(
      int batch_dim,
      int length,
      int kernel_size) {
    int KERNELTHREADS = min(CUDA_NUM_THREADS, kernel_size);
    int TOKENTHREADS = min(CUDA_NUM_THREADS / KERNELTHREADS, length);
    int BATCHTHREADS =
        min(64, max(1, CUDA_NUM_THREADS / (TOKENTHREADS * KERNELTHREADS)));
    dim3 grid(
        (length + TOKENTHREADS - 1) / TOKENTHREADS,
        (kernel_size + KERNELTHREADS - 1) / KERNELTHREADS,
        (batch_dim + BATCHTHREADS - 1) / BATCHTHREADS);
    dim3 block(TOKENTHREADS, KERNELTHREADS, BATCHTHREADS);
    return LaunchParams(grid, block);
  }
};

template <typename scalar_t, int KS, int NS, int DILATION>
struct PointwiseNeighborhood1DFull : PointwiseNeighborhood1DBase<scalar_t> {
  using Base = PointwiseNeighborhood1DBase<scalar_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = false;
  static constexpr bool IsHalfKernel = false;
  static constexpr bool UsesSmem = false;

  __device__ __host__ PointwiseNeighborhood1DFull() : Base() {}

  static __host__ int get_dim(int dim) {
    return dim;
  }

  __device__ void launch(Params p) {
    const int KERNEL_SIZE = (KS > 1) ? KS : p.kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS > 0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION > 0) ? DILATION : p.dilation_in;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < p.batch_size * p.heads) {
      const int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < p.length) {
        const int ki = blockIdx.y * blockDim.y + threadIdx.y;
        if (ki < KERNEL_SIZE) {
          const int b = z / p.heads;
          const int h = z - b * p.heads;
          const int ni = get_window_start(
              i, p.length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
          scalar_t updt = scalar_t(0);
          const int64_t batchHeadOffset =
              b * p.query_stride_0 + h * p.query_stride_1;
          const int64_t queryOffset = batchHeadOffset + i * p.query_stride_2;
          const int64_t keyOffset =
              batchHeadOffset + (ki * dilation + ni) * p.query_stride_2;
#pragma unroll
          for (int dimOffset = 0; dimOffset < p.dim; ++dimOffset)
            updt +=
                p.query[queryOffset + dimOffset] * p.key[keyOffset + dimOffset];
          if (p.bias) {
            const int pi = get_pb_start(
                i, p.length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            updt += p.bias[h * p.bias_stride_0 + (pi + ki)];
          }
          p.attn
              [b * p.attn_stride_0 + h * p.attn_stride_1 + i * p.attn_stride_2 +
               ki] = updt;
        }
      }
    }
  }
};

template <typename scalar_t, int KS, int NS, int DILATION>
struct PointwiseNeighborhood1DHalf : PointwiseNeighborhood1DBase<scalar_t> {
  using Base = PointwiseNeighborhood1DBase<scalar_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = IsBF16<scalar_t>::value;
  static constexpr bool IsHalfKernel = true;
  static constexpr bool UsesSmem = false;

  __device__ __host__ PointwiseNeighborhood1DHalf() : Base() {}

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
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < p.batch_size * p.heads) {
      const int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < p.length) {
        const int ki = blockIdx.y * blockDim.y + threadIdx.y;
        if (ki < KERNEL_SIZE) {
          auto query2 = HalfHelper::typecast(p.query);
          auto key2 = HalfHelper::typecast(p.key);
          const int b = z / p.heads;
          const int h = z - b * p.heads;
          const int ni = get_window_start(
              i, p.length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
          auto updt = HalfHelper::zero();
          const int64_t batchHeadOffset =
              b * p.query_stride_0 + h * p.query_stride_1;
          const int64_t queryOffset = batchHeadOffset + i * p.query_stride_2;
          const int64_t keyOffset =
              batchHeadOffset + (ki * dilation + ni) * p.query_stride_2;
#pragma unroll
          for (int dimOffset = 0; dimOffset < p.dim; ++dimOffset)
            updt = HalfHelper::fma(
                query2[queryOffset + dimOffset],
                key2[keyOffset + dimOffset],
                updt);
          scalar_t acc = HalfHelper::cast_back(HalfHelper::add(updt.x, updt.y));
          if (p.bias) {
            const int pi = get_pb_start(
                i, p.length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            acc = HalfHelper::add(acc, p.bias[h * p.bias_stride_0 + (pi + ki)]);
          }
          p.attn
              [b * p.attn_stride_0 + h * p.attn_stride_1 + i * p.attn_stride_2 +
               ki] = acc;
        }
      }
    }
  }
};

template <typename Args_>
struct PointwiseNeighborhood1D {
  using Args = Args_;
  static constexpr int KS = Args::KernelSize;
  static constexpr int NS = Args::NeighborhoodSize;
  static constexpr int DILATION = Args::Dilation;
  using scalar_t = typename Args::Dtype;
  using Kernel = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood1DFull<scalar_t, KS, NS, DILATION>,
      PointwiseNeighborhood1DHalf<scalar_t, KS, NS, DILATION>>::type;
  using Params = typename Kernel::Params;

  void operator()(
      const int cc,
      cudaStream_t stream,
      void* query_ptr,
      void* key_ptr,
      void* attn_ptr,
      int batch_size,
      int heads,
      int length,
      int dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int kernel_size,
      int dilation) {
    dim = Kernel::get_dim(dim);
    LaunchParams lp = Kernel::Base::get_launch_params(
        batch_size * heads, length, kernel_size);
    auto params = Params(
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
        length,
        heads,
        kernel_size,
        dilation,
        dim,
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2);
    launch_cuda_kernel<Kernel><<<lp.grid, lp.block, 0, stream>>>(params);
  }
};

template <typename Args_>
struct PointwiseNeighborhood1DWithBias {
  using Args = Args_;
  static constexpr int KS = Args::KernelSize;
  static constexpr int NS = Args::NeighborhoodSize;
  static constexpr int DILATION = Args::Dilation;
  using scalar_t = typename Args::Dtype;
  using Kernel = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood1DFull<scalar_t, KS, NS, DILATION>,
      PointwiseNeighborhood1DHalf<scalar_t, KS, NS, DILATION>>::type;
  using Params = typename Kernel::Params;

  void operator()(
      const int cc,
      cudaStream_t stream,
      void* query_ptr,
      void* key_ptr,
      void* bias_ptr,
      void* attn_ptr,
      int batch_size,
      int heads,
      int length,
      int dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int kernel_size,
      int dilation) {
    dim = Kernel::get_dim(dim);
    LaunchParams lp = Kernel::Base::get_launch_params(
        batch_size * heads, length, kernel_size);
    auto params = Params(
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(bias_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
        length,
        heads,
        kernel_size,
        dilation,
        dim,
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2);
    launch_cuda_kernel<Kernel><<<lp.grid, lp.block, 0, stream>>>(params);
  }
};

} // namespace naive
} // namespace cuda
} // namespace natten
