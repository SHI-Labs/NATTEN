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

#include <natten/config.h>
#include <natten/cuda/naive/natten_commons.cuh>
#include <natten/cuda/naive/natten_tiled_macros.cuh>
#include <natten/cuda/naive/tiled/base.cuh>
#include <natten/cuda/naive/tiled/pointwise_neighborhood_2d_tiled_11x11_13x13.cuh>
#include <natten/cuda/naive/tiled/pointwise_neighborhood_2d_tiled_3x3.cuh>
#include <natten/cuda/naive/tiled/pointwise_neighborhood_2d_tiled_5x5.cuh>
#include <natten/cuda/naive/tiled/pointwise_neighborhood_2d_tiled_7x7_9x9.cuh>

namespace natten {
namespace cuda {
namespace naive {

///////////////////////////////////////////////////////////////////////////////
///////////////////////////// Main kernels ////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

template <typename scalar_t, int KS, int NS, int DILATION>
struct PointwiseNeighborhood2DFull : PointwiseNeighborhood2DBase<scalar_t> {
  using Base = PointwiseNeighborhood2DBase<scalar_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = false;
  static constexpr bool IsHalfKernel = false;
  static constexpr bool UsesSmem = false;

  __device__ __host__ PointwiseNeighborhood2DFull() : Base() {}

  static __host__ int get_dim(int dim) {
    return dim;
  }

  __device__ void launch(Params p) {
    const int KERNEL_SIZE = (KS > 1) ? KS : p.kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS > 0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION > 0) ? DILATION : p.dilation_in;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < p.batch_size * p.heads) {
      const int x = blockIdx.x * blockDim.x + threadIdx.x;
      if (x < p.height * p.width) {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < KERNEL_SIZE * KERNEL_SIZE) {
          const int b = z / p.heads;
          const int h = z - b * p.heads;
          const int ki = y / KERNEL_SIZE;
          const int kj = y - ki * KERNEL_SIZE;
          const int i = x / p.width;
          const int j = x - i * p.width;

          const int ni = get_window_start(
              i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
          const int nj = get_window_start(
              j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);

          scalar_t updt = scalar_t(0);
          const int64_t batchHeadOffset =
              b * p.query_stride_0 + h * p.query_stride_1;
          const int64_t queryOffset =
              batchHeadOffset + i * p.query_stride_2 + j * p.query_stride_3;
          const int64_t keyOffset = batchHeadOffset +
              (ki * dilation + ni) * p.query_stride_2 +
              (kj * dilation + nj) * p.query_stride_3;
          for (int dimOffset = 0; dimOffset < p.dim; ++dimOffset)
            updt +=
                p.query[queryOffset + dimOffset] * p.key[keyOffset + dimOffset];
          if (p.bias) {
            const int pi = get_pb_start(
                i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            const int pj = get_pb_start(
                j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            const int64_t biasIndex =
                h * p.bias_stride_0 + (pi + ki) * p.bias_stride_1 + (pj + kj);
            updt += p.bias[biasIndex];
          }
          const int64_t index = b * p.attn_stride_0 + h * p.attn_stride_1 +
              i * p.attn_stride_2 + j * p.attn_stride_3 + y;
          p.attn[index] = updt;
        }
      }
    }
  }
};

template <typename scalar_t, int KS, int NS, int DILATION>
struct PointwiseNeighborhood2DHalf : PointwiseNeighborhood2DBase<scalar_t> {
  using Base = PointwiseNeighborhood2DBase<scalar_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = IsBF16<scalar_t>::value;
  static constexpr bool IsHalfKernel = true;
  static constexpr bool UsesSmem = false;

  __device__ __host__ PointwiseNeighborhood2DHalf() : Base() {}

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
      const int x = blockIdx.x * blockDim.x + threadIdx.x;
      if (x < p.height * p.width) {
        const int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < KERNEL_SIZE * KERNEL_SIZE) {
          auto query2 = HalfHelper::typecast(p.query);
          auto key2 = HalfHelper::typecast(p.key);
          const int b = z / p.heads;
          const int h = z - b * p.heads;
          const int ki = y / KERNEL_SIZE;
          const int kj = y - ki * KERNEL_SIZE;
          const int i = x / p.width;
          const int j = x - i * p.width;

          const int ni = get_window_start(
              i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
          const int nj = get_window_start(
              j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);

          auto updt = HalfHelper::zero();
          const int64_t batchHeadOffset =
              b * p.query_stride_0 + h * p.query_stride_1;
          const int64_t queryOffset =
              batchHeadOffset + i * p.query_stride_2 + j * p.query_stride_3;
          const int64_t keyOffset = batchHeadOffset +
              (ki * dilation + ni) * p.query_stride_2 +
              (kj * dilation + nj) * p.query_stride_3;
          for (int dimOffset = 0; dimOffset < p.dim; ++dimOffset)
            updt = HalfHelper::fma(
                query2[queryOffset + dimOffset],
                key2[keyOffset + dimOffset],
                updt);
          const int64_t index = b * p.attn_stride_0 + h * p.attn_stride_1 +
              i * p.attn_stride_2 + j * p.attn_stride_3 + y;
          scalar_t acc = HalfHelper::cast_back(HalfHelper::add(updt.x, updt.y));
          if (p.bias) {
            const int pi = get_pb_start(
                i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            const int pj = get_pb_start(
                j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            const int64_t biasIndex =
                h * p.bias_stride_0 + (pi + ki) * p.bias_stride_1 + (pj + kj);
            acc = HalfHelper::add(acc, p.bias[biasIndex]);
          }
          p.attn[index] = acc;
        }
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

template <typename Args_>
struct PointwiseNeighborhood2D {
  using Args = Args_;
  static constexpr int KS = Args::KernelSize;
  static constexpr int NS = Args::NeighborhoodSize;
  static constexpr int DILATION = Args::Dilation;
  using scalar_t = typename Args::Dtype;
  using Kernel = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull<scalar_t, KS, NS, DILATION>,
      PointwiseNeighborhood2DHalf<scalar_t, KS, NS, DILATION>>::type;
  using Kernel3x3 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull3x3<scalar_t, DILATION>,
      PointwiseNeighborhood2DHalf3x3<scalar_t, DILATION>>::type;
  using Kernel5x5 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull5x5<scalar_t, DILATION>,
      PointwiseNeighborhood2DHalf5x5<scalar_t, DILATION>>::type;
  using Kernel7x7 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull7x7<scalar_t, DILATION>,
      PointwiseNeighborhood2DHalf7x7<scalar_t, DILATION>>::type;
  using Kernel9x9 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull9x9<scalar_t, DILATION>,
      PointwiseNeighborhood2DHalf9x9<scalar_t, DILATION>>::type;
  using Kernel11x11 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull11x11<scalar_t, DILATION>,
      PointwiseNeighborhood2DHalf11x11<scalar_t, DILATION>>::type;
  using Kernel13x13 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull13x13<scalar_t, DILATION>,
      PointwiseNeighborhood2DHalf13x13<scalar_t, DILATION>>::type;
  using Params = typename Kernel::Params;

  void operator()(
      const int cc,
      cudaStream_t stream,
      void* query_ptr,
      void* key_ptr,
      void* attn_ptr,
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
    if (dim == 32 && natten::kEnableTiledNA && cc >= 60) {
      if (kernel_size == 3) {
        LaunchParams lp = Kernel3x3::get_launch_params(
            batch_size * heads,
            height,
            width,
            kernel_size * kernel_size,
            dilation);
        dim = Kernel3x3::get_dim(dim);
        auto params = Params(
            reinterpret_cast<scalar_t*>(query_ptr),
            reinterpret_cast<scalar_t*>(key_ptr),
            reinterpret_cast<scalar_t*>(attn_ptr),
            height,
            width,
            heads,
            kernel_size,
            dilation,
            dim,
            batch_size,
            attn_stride_0,
            attn_stride_1,
            attn_stride_2,
            attn_stride_3);
        launch_cuda_kernel<Kernel3x3><<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      } else if (kernel_size == 5) {
        LaunchParams lp = Kernel5x5::get_launch_params(
            batch_size * heads,
            height,
            width,
            kernel_size * kernel_size,
            dilation);
        dim = Kernel5x5::get_dim(dim);
        auto params = Params(
            reinterpret_cast<scalar_t*>(query_ptr),
            reinterpret_cast<scalar_t*>(key_ptr),
            reinterpret_cast<scalar_t*>(attn_ptr),
            height,
            width,
            heads,
            kernel_size,
            dilation,
            dim,
            batch_size,
            attn_stride_0,
            attn_stride_1,
            attn_stride_2,
            attn_stride_3);
        launch_cuda_kernel<Kernel5x5><<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      } else if (kernel_size == 7) {
        LaunchParams lp = Kernel7x7::get_launch_params(
            batch_size * heads,
            height,
            width,
            kernel_size * kernel_size,
            dilation);
        dim = Kernel7x7::get_dim(dim);
        auto params = Params(
            reinterpret_cast<scalar_t*>(query_ptr),
            reinterpret_cast<scalar_t*>(key_ptr),
            reinterpret_cast<scalar_t*>(attn_ptr),
            height,
            width,
            heads,
            kernel_size,
            dilation,
            dim,
            batch_size,
            attn_stride_0,
            attn_stride_1,
            attn_stride_2,
            attn_stride_3);
        launch_cuda_kernel<Kernel7x7><<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      } else if (kernel_size == 9) {
        LaunchParams lp = Kernel9x9::get_launch_params(
            batch_size * heads,
            height,
            width,
            kernel_size * kernel_size,
            dilation);
        dim = Kernel9x9::get_dim(dim);
        auto params = Params(
            reinterpret_cast<scalar_t*>(query_ptr),
            reinterpret_cast<scalar_t*>(key_ptr),
            reinterpret_cast<scalar_t*>(attn_ptr),
            height,
            width,
            heads,
            kernel_size,
            dilation,
            dim,
            batch_size,
            attn_stride_0,
            attn_stride_1,
            attn_stride_2,
            attn_stride_3);
        launch_cuda_kernel<Kernel9x9><<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      } else if (kernel_size == 11) {
        LaunchParams lp = Kernel11x11::get_launch_params(
            batch_size * heads,
            height,
            width,
            kernel_size * kernel_size,
            dilation);
        dim = Kernel11x11::get_dim(dim);
        auto params = Params(
            reinterpret_cast<scalar_t*>(query_ptr),
            reinterpret_cast<scalar_t*>(key_ptr),
            reinterpret_cast<scalar_t*>(attn_ptr),
            height,
            width,
            heads,
            kernel_size,
            dilation,
            dim,
            batch_size,
            attn_stride_0,
            attn_stride_1,
            attn_stride_2,
            attn_stride_3);
        launch_cuda_kernel<Kernel11x11>
            <<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      } else if (kernel_size == 13) {
        LaunchParams lp = Kernel13x13::get_launch_params(
            batch_size * heads,
            height,
            width,
            kernel_size * kernel_size,
            dilation);
        dim = Kernel13x13::get_dim(dim);
        auto params = Params(
            reinterpret_cast<scalar_t*>(query_ptr),
            reinterpret_cast<scalar_t*>(key_ptr),
            reinterpret_cast<scalar_t*>(attn_ptr),
            height,
            width,
            heads,
            kernel_size,
            dilation,
            dim,
            batch_size,
            attn_stride_0,
            attn_stride_1,
            attn_stride_2,
            attn_stride_3);
        launch_cuda_kernel<Kernel13x13>
            <<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      }
    }
    LaunchParams lp = Kernel::get_launch_params(
        batch_size * heads, height * width, kernel_size * kernel_size);
    dim = Kernel::get_dim(dim);
    auto params = Params(
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
        height,
        width,
        heads,
        kernel_size,
        dilation,
        dim,
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3);
    launch_cuda_kernel<Kernel><<<lp.grid, lp.block, 0, stream>>>(params);
  }
};

template <typename Args_>
struct PointwiseNeighborhood2DWithBias {
  using Args = Args_;
  static constexpr int KS = Args::KernelSize;
  static constexpr int NS = Args::NeighborhoodSize;
  static constexpr int DILATION = Args::Dilation;
  using scalar_t = typename Args::Dtype;
  using Kernel = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull<scalar_t, KS, NS, DILATION>,
      PointwiseNeighborhood2DHalf<scalar_t, KS, NS, DILATION>>::type;
  using Kernel3x3 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull3x3<scalar_t, DILATION>,
      PointwiseNeighborhood2DHalf3x3<scalar_t, DILATION>>::type;
  using Kernel5x5 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull5x5<scalar_t, DILATION>,
      PointwiseNeighborhood2DHalf5x5<scalar_t, DILATION>>::type;
  using Kernel7x7 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull7x7<scalar_t, DILATION>,
      PointwiseNeighborhood2DHalf7x7<scalar_t, DILATION>>::type;
  using Kernel9x9 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull9x9<scalar_t, DILATION>,
      PointwiseNeighborhood2DHalf9x9<scalar_t, DILATION>>::type;
  using Kernel11x11 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull11x11<scalar_t, DILATION>,
      PointwiseNeighborhood2DHalf11x11<scalar_t, DILATION>>::type;
  using Kernel13x13 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull13x13<scalar_t, DILATION>,
      PointwiseNeighborhood2DHalf13x13<scalar_t, DILATION>>::type;
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
      int height,
      int width,
      int dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int64_t attn_stride_3,
      int kernel_size,
      int dilation) {
    if (dim == 32 && natten::kEnableTiledNA && cc >= 60) {
      if (kernel_size == 3) {
        LaunchParams lp = Kernel3x3::get_launch_params(
            batch_size * heads,
            height,
            width,
            kernel_size * kernel_size,
            dilation);
        dim = Kernel3x3::get_dim(dim);
        auto params = Params(
            reinterpret_cast<scalar_t*>(query_ptr),
            reinterpret_cast<scalar_t*>(key_ptr),
            reinterpret_cast<scalar_t*>(bias_ptr),
            reinterpret_cast<scalar_t*>(attn_ptr),
            height,
            width,
            heads,
            kernel_size,
            dilation,
            dim,
            batch_size,
            attn_stride_0,
            attn_stride_1,
            attn_stride_2,
            attn_stride_3);
        launch_cuda_kernel<Kernel3x3><<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      } else if (kernel_size == 5) {
        LaunchParams lp = Kernel5x5::get_launch_params(
            batch_size * heads,
            height,
            width,
            kernel_size * kernel_size,
            dilation);
        dim = Kernel5x5::get_dim(dim);
        auto params = Params(
            reinterpret_cast<scalar_t*>(query_ptr),
            reinterpret_cast<scalar_t*>(key_ptr),
            reinterpret_cast<scalar_t*>(bias_ptr),
            reinterpret_cast<scalar_t*>(attn_ptr),
            height,
            width,
            heads,
            kernel_size,
            dilation,
            dim,
            batch_size,
            attn_stride_0,
            attn_stride_1,
            attn_stride_2,
            attn_stride_3);
        launch_cuda_kernel<Kernel5x5><<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      } else if (kernel_size == 7) {
        LaunchParams lp = Kernel7x7::get_launch_params(
            batch_size * heads,
            height,
            width,
            kernel_size * kernel_size,
            dilation);
        dim = Kernel7x7::get_dim(dim);
        auto params = Params(
            reinterpret_cast<scalar_t*>(query_ptr),
            reinterpret_cast<scalar_t*>(key_ptr),
            reinterpret_cast<scalar_t*>(bias_ptr),
            reinterpret_cast<scalar_t*>(attn_ptr),
            height,
            width,
            heads,
            kernel_size,
            dilation,
            dim,
            batch_size,
            attn_stride_0,
            attn_stride_1,
            attn_stride_2,
            attn_stride_3);
        launch_cuda_kernel<Kernel7x7><<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      } else if (kernel_size == 9) {
        LaunchParams lp = Kernel9x9::get_launch_params(
            batch_size * heads,
            height,
            width,
            kernel_size * kernel_size,
            dilation);
        dim = Kernel9x9::get_dim(dim);
        auto params = Params(
            reinterpret_cast<scalar_t*>(query_ptr),
            reinterpret_cast<scalar_t*>(key_ptr),
            reinterpret_cast<scalar_t*>(bias_ptr),
            reinterpret_cast<scalar_t*>(attn_ptr),
            height,
            width,
            heads,
            kernel_size,
            dilation,
            dim,
            batch_size,
            attn_stride_0,
            attn_stride_1,
            attn_stride_2,
            attn_stride_3);
        launch_cuda_kernel<Kernel9x9><<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      } else if (kernel_size == 11) {
        LaunchParams lp = Kernel11x11::get_launch_params(
            batch_size * heads,
            height,
            width,
            kernel_size * kernel_size,
            dilation);
        dim = Kernel11x11::get_dim(dim);
        auto params = Params(
            reinterpret_cast<scalar_t*>(query_ptr),
            reinterpret_cast<scalar_t*>(key_ptr),
            reinterpret_cast<scalar_t*>(bias_ptr),
            reinterpret_cast<scalar_t*>(attn_ptr),
            height,
            width,
            heads,
            kernel_size,
            dilation,
            dim,
            batch_size,
            attn_stride_0,
            attn_stride_1,
            attn_stride_2,
            attn_stride_3);
        launch_cuda_kernel<Kernel11x11>
            <<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      } else if (kernel_size == 13) {
        LaunchParams lp = Kernel13x13::get_launch_params(
            batch_size * heads,
            height,
            width,
            kernel_size * kernel_size,
            dilation);
        dim = Kernel13x13::get_dim(dim);
        auto params = Params(
            reinterpret_cast<scalar_t*>(query_ptr),
            reinterpret_cast<scalar_t*>(key_ptr),
            reinterpret_cast<scalar_t*>(bias_ptr),
            reinterpret_cast<scalar_t*>(attn_ptr),
            height,
            width,
            heads,
            kernel_size,
            dilation,
            dim,
            batch_size,
            attn_stride_0,
            attn_stride_1,
            attn_stride_2,
            attn_stride_3);
        launch_cuda_kernel<Kernel13x13>
            <<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      }
    }
    LaunchParams lp = Kernel::get_launch_params(
        batch_size * heads, height * width, kernel_size * kernel_size);
    dim = Kernel::get_dim(dim);
    auto params = Params(
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(bias_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
        height,
        width,
        heads,
        kernel_size,
        dilation,
        dim,
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3);
    launch_cuda_kernel<Kernel><<<lp.grid, lp.block, 0, stream>>>(params);
  }
};

} // namespace naive
} // namespace cuda
} // namespace natten
