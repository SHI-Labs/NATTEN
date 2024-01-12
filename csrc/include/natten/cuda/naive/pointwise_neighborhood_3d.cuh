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
    \brief Pointwise-Neighborhood kernel for 3D data.
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
struct PointwiseNeighborhood3DBase {
  struct Params {
    scalar_t* query; // query / d_out
    scalar_t* key; // key   / value
    scalar_t* bias = nullptr; // optional: bias
    scalar_t* attn; // attn  / d_attn
    const int depth;
    const int height;
    const int width;
    const int heads;
    const int kernel_size_in;
    const int dilation_in;
    const int depth_kernel_size_in;
    const int depth_dilation_in;
    const int dim;
    const int batch_size;
    const int64_t attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3,
        attn_stride_4;
    const int64_t query_stride_0, query_stride_1, query_stride_2, query_stride_3,
        query_stride_4;
    const int64_t bias_stride_0, bias_stride_1, bias_stride_2;

    __device__ __host__ Params() {}

    __device__ __host__ Params(
        scalar_t* query,
        scalar_t* key,
        scalar_t* attn,
        const int depth,
        const int height,
        const int width,
        const int heads,
        const int kernel_size_in,
        const int dilation_in,
        const int depth_kernel_size_in,
        const int depth_dilation_in,
        const int dim,
        const int batch_size,
        const int64_t attn_stride_0,
        const int64_t attn_stride_1,
        const int64_t attn_stride_2,
        const int64_t attn_stride_3,
        const int64_t attn_stride_4)
        : query(query),
          key(key),
          attn(attn),
          depth(depth),
          height(height),
          width(width),
          heads(heads),
          kernel_size_in(kernel_size_in),
          dilation_in(dilation_in),
          depth_kernel_size_in(depth_kernel_size_in),
          depth_dilation_in(depth_dilation_in),
          dim(dim),
          batch_size(batch_size),
          bias_stride_2(0),
          bias_stride_1(0),
          bias_stride_0(0),
          attn_stride_4(attn_stride_4),
          attn_stride_3(attn_stride_3),
          attn_stride_2(attn_stride_2),
          attn_stride_1(attn_stride_1),
          attn_stride_0(attn_stride_0),
          query_stride_4(dim),
          query_stride_3(dim * width),
          query_stride_2(dim * width * height),
          query_stride_1(dim * width * height * depth),
          query_stride_0(dim * width * height * depth * heads) {}

    // CTOR with bias
    __device__ __host__ Params( // AV     / Q-grad
        scalar_t* query, // attn   / d_attn
        scalar_t* key, // value  / key
        scalar_t* bias, // relative positional bias tensor
        scalar_t* attn, // output / d_query
        const int depth,
        const int height,
        const int width,
        const int heads,
        const int kernel_size_in,
        const int dilation_in,
        const int depth_kernel_size_in,
        const int depth_dilation_in,
        const int dim,
        const int batch_size,
        const int64_t attn_stride_0,
        const int64_t attn_stride_1,
        const int64_t attn_stride_2,
        const int64_t attn_stride_3,
        const int64_t attn_stride_4)
        : query(query),
          key(key),
          bias(bias),
          attn(attn),
          depth(depth),
          height(height),
          width(width),
          heads(heads),
          kernel_size_in(kernel_size_in),
          dilation_in(dilation_in),
          depth_kernel_size_in(depth_kernel_size_in),
          depth_dilation_in(depth_dilation_in),
          dim(dim),
          batch_size(batch_size),
          bias_stride_2(2 * kernel_size_in - 1),
          bias_stride_1((2 * kernel_size_in - 1) * (2 * kernel_size_in - 1)),
          bias_stride_0(
              (2 * kernel_size_in - 1) * (2 * kernel_size_in - 1) *
              (2 * depth_kernel_size_in - 1)),
          attn_stride_4(attn_stride_4),
          attn_stride_3(attn_stride_3),
          attn_stride_2(attn_stride_2),
          attn_stride_1(attn_stride_1),
          attn_stride_0(attn_stride_0),
          query_stride_4(dim),
          query_stride_3(dim * width),
          query_stride_2(dim * width * height),
          query_stride_1(dim * width * height * depth),
          query_stride_0(dim * width * height * depth * heads) {}
  };

  __device__ __host__ PointwiseNeighborhood3DBase() {}

  static LaunchParams get_launch_params(
      int batch_dim,
      int spatial_size,
      int attention_span) {
    int KERNELTHREADS =
        min(CUDA_NUM_THREADS, attention_span /* == kernel_size^3 */);
    int PIXELTHREADS = min(int(CUDA_NUM_THREADS / KERNELTHREADS), spatial_size);
    int BATCHTHREADS =
        min(64, max(1, CUDA_NUM_THREADS / (PIXELTHREADS * KERNELTHREADS)));
    dim3 grid(
        (spatial_size + PIXELTHREADS - 1) / PIXELTHREADS,
        (attention_span + KERNELTHREADS - 1) / KERNELTHREADS,
        (batch_dim + BATCHTHREADS - 1) / BATCHTHREADS);
    dim3 block(PIXELTHREADS, KERNELTHREADS, BATCHTHREADS);
    return LaunchParams(grid, block);
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
struct PointwiseNeighborhood3DFull : PointwiseNeighborhood3DBase<scalar_t> {
  using Base = PointwiseNeighborhood3DBase<scalar_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = false;
  static constexpr bool IsHalfKernel = false;
  static constexpr bool UsesSmem = false;

  __device__ __host__ PointwiseNeighborhood3DFull() : Base() {}

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
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < p.batch_size * p.heads) {
      const int x = blockIdx.x * blockDim.x + threadIdx.x;
      if (x < p.depth * p.height * p.width) {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < KERNEL_SIZE_D * KERNEL_SIZE * KERNEL_SIZE) {
          int indtmp1 = y / KERNEL_SIZE;
          const int kk = indtmp1 / KERNEL_SIZE;
          const int kj = y - indtmp1 * KERNEL_SIZE;
          const int ki = indtmp1 - kk * KERNEL_SIZE;

          indtmp1 = x / p.width;
          const int k = indtmp1 / p.height;
          const int j = x - indtmp1 * p.width;
          const int i = indtmp1 - k * p.height;

          const int b = z / p.heads;
          const int h = z - b * p.heads;

          const int ni = get_window_start(
              i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
          const int nj = get_window_start(
              j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
          const int nk = get_window_start(
              k, p.depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);

          scalar_t updt = scalar_t(0);
          const int64_t batchHeadOffset =
              b * p.query_stride_0 + h * p.query_stride_1;
          const int64_t queryOffset = batchHeadOffset + k * p.query_stride_2 +
              i * p.query_stride_3 + j * p.query_stride_4;
          const int64_t keyOffset = batchHeadOffset +
              (kk * dilation_d + nk) * p.query_stride_2 +
              (ki * dilation + ni) * p.query_stride_3 +
              (kj * dilation + nj) * p.query_stride_4;
#pragma unroll
          for (int dimOffset = 0; dimOffset < p.dim; ++dimOffset)
            updt +=
                p.query[queryOffset + dimOffset] * p.key[keyOffset + dimOffset];
          const int64_t index = b * p.attn_stride_0 + h * p.attn_stride_1 +
              k * p.attn_stride_2 + i * p.attn_stride_3 + j * p.attn_stride_4 +
              y;
          if (p.bias) {
            const int pi = get_pb_start(
                i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            const int pj = get_pb_start(
                j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            const int pk = get_pb_start(
                k, p.depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
            const int64_t biasIndex = h * p.bias_stride_0 +
                (pk + kk) * p.bias_stride_1 + (pi + ki) * p.bias_stride_2 +
                (pj + kj);
            updt += p.bias[biasIndex];
          }
          p.attn[index] = updt;
        }
      }
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
struct PointwiseNeighborhood3DHalf : PointwiseNeighborhood3DBase<scalar_t> {
  using Base = PointwiseNeighborhood3DBase<scalar_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = IsBF16<scalar_t>::value;
  static constexpr bool IsHalfKernel = true;
  static constexpr bool UsesSmem = false;

  __device__ __host__ PointwiseNeighborhood3DHalf() : Base() {}

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
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < p.batch_size * p.heads) {
      const int x = blockIdx.x * blockDim.x + threadIdx.x;
      if (x < p.depth * p.height * p.width) {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < KERNEL_SIZE_D * KERNEL_SIZE * KERNEL_SIZE) {
          auto query2 = HalfHelper::typecast(p.query);
          auto key2 = HalfHelper::typecast(p.key);
          int indtmp1 = y / KERNEL_SIZE;
          const int kk = indtmp1 / KERNEL_SIZE;
          const int kj = y - indtmp1 * KERNEL_SIZE;
          const int ki = indtmp1 - kk * KERNEL_SIZE;

          indtmp1 = x / p.width;
          const int k = indtmp1 / p.height;
          const int j = x - indtmp1 * p.width;
          const int i = indtmp1 - k * p.height;

          const int b = z / p.heads;
          const int h = z - b * p.heads;

          const int ni = get_window_start(
              i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
          const int nj = get_window_start(
              j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
          const int nk = get_window_start(
              k, p.depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);

          auto updt = HalfHelper::zero();
          const int64_t batchHeadOffset =
              b * p.query_stride_0 + h * p.query_stride_1;
          const int64_t queryOffset = batchHeadOffset + k * p.query_stride_2 +
              i * p.query_stride_3 + j * p.query_stride_4;
          const int64_t keyOffset = batchHeadOffset +
              (kk * dilation_d + nk) * p.query_stride_2 +
              (ki * dilation + ni) * p.query_stride_3 +
              (kj * dilation + nj) * p.query_stride_4;
#pragma unroll
          for (int dimOffset = 0; dimOffset < p.dim; ++dimOffset)
            updt = HalfHelper::fma(
                query2[queryOffset + dimOffset],
                key2[keyOffset + dimOffset],
                updt);
          const int64_t index = b * p.attn_stride_0 + h * p.attn_stride_1 +
              k * p.attn_stride_2 + i * p.attn_stride_3 + j * p.attn_stride_4 +
              y;
          scalar_t acc = HalfHelper::cast_back(HalfHelper::add(updt.x, updt.y));
          if (p.bias) {
            const int pi = get_pb_start(
                i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            const int pj = get_pb_start(
                j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            const int pk = get_pb_start(
                k, p.depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
            const int64_t biasIndex = h * p.bias_stride_0 +
                (pk + kk) * p.bias_stride_1 + (pi + ki) * p.bias_stride_2 +
                (pj + kj);
            acc = HalfHelper::add(acc, p.bias[biasIndex]);
          }
          p.attn[index] = acc;
        }
      }
    }
  }
};

template <typename Args_>
struct PointwiseNeighborhood3D {
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
      PointwiseNeighborhood3DFull<
          scalar_t,
          KS,
          NS,
          DILATION,
          DKS,
          DNS,
          DDILATION>,
      PointwiseNeighborhood3DHalf<
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
      void* query_ptr,
      void* key_ptr,
      void* attn_ptr,
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
    LaunchParams lp = Kernel::Base::get_launch_params(
        batch_size * heads,
        depth * height * width,
        kernel_size_depth * kernel_size * kernel_size);
    auto params = Params(
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
        depth,
        height,
        width,
        heads,
        kernel_size,
        dilation,
        kernel_size_depth,
        dilation_depth,
        dim,
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        attn_stride_4);
    launch_cuda_kernel<Kernel><<<lp.grid, lp.block, 0, stream>>>(params);
  }
};

template <typename Args_>
struct PointwiseNeighborhood3DWithBias {
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
      PointwiseNeighborhood3DFull<
          scalar_t,
          KS,
          NS,
          DILATION,
          DKS,
          DNS,
          DDILATION>,
      PointwiseNeighborhood3DHalf<
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
      void* query_ptr,
      void* key_ptr,
      void* bias_ptr,
      void* attn_ptr,
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
    LaunchParams lp = Kernel::Base::get_launch_params(
        batch_size * heads,
        depth * height * width,
        kernel_size_depth * kernel_size * kernel_size);
    auto params = Params(
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(bias_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
        depth,
        height,
        width,
        heads,
        kernel_size,
        dilation,
        kernel_size_depth,
        dilation_depth,
        dim,
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3,
        attn_stride_4);
    launch_cuda_kernel<Kernel><<<lp.grid, lp.block, 0, stream>>>(params);
  }
};

} // namespace naive
} // namespace cuda
} // namespace natten
