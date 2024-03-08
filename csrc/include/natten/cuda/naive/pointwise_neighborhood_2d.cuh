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
#include <natten/naive_argpack.h>
#include <natten/natten.h>
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

template <typename scalar_t, typename CausalMask_>
struct PointwiseNeighborhood2DFull : PointwiseNeighborhood2DBase<scalar_t> {
  using Base = PointwiseNeighborhood2DBase<scalar_t>;
  using Params = typename Base::Params;
  using CausalMask = CausalMask_;
  static constexpr bool IsBF16Kernel = false;
  static constexpr bool IsHalfKernel = false;
  static constexpr bool UsesSmem = false;

  __device__ __host__ PointwiseNeighborhood2DFull() : Base() {}

  static __host__ int32_t get_dim(int32_t dim) {
    return dim;
  }

  __device__ void launch(Params p) {
    auto mask_value = AttnMask<scalar_t>::value(p.is_grad);
    auto mask_0 = NeighborhoodMask<CausalMask::Dim0>(
        p.height, p.kernel_size_0, p.dilation_0);
    auto mask_1 = NeighborhoodMask<CausalMask::Dim1>(
        p.width, p.kernel_size_1, p.dilation_1);
    int32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < p.batch_size * p.heads) {
      int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
      if (x < p.height * p.width) {
        int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < p.kernel_size_0 * p.kernel_size_1) {
          auto b = z / p.heads;
          auto h = z - b * p.heads;
          auto ki = y / p.kernel_size_1;
          auto kj = y - ki * p.kernel_size_1;
          auto i = x / p.width;
          auto j = x - i * p.width;

          auto ni = mask_0.get_window_start(i);
          auto nj = mask_1.get_window_start(j);
          auto ei = mask_0.get_window_end(i, ni);
          auto ej = mask_1.get_window_end(j, nj);

          auto key_idx_i = ki * p.dilation_0 + ni;
          auto key_idx_j = kj * p.dilation_1 + nj;
          scalar_t updt = scalar_t(0.0);
          if (key_idx_i < ei && key_idx_j < ej) {
            int64_t batchHeadOffset =
                b * p.query_stride_0 + h * p.query_stride_1;
            int64_t queryOffset =
                batchHeadOffset + i * p.query_stride_2 + j * p.query_stride_3;
            int64_t keyOffset = batchHeadOffset + key_idx_i * p.query_stride_2 +
                key_idx_j * p.query_stride_3;
            for (int32_t dimOffset = 0; dimOffset < p.dim; ++dimOffset)
              updt += p.query[queryOffset + dimOffset] *
                  p.key[keyOffset + dimOffset];
            if (p.bias) {
              auto pi = mask_0.get_pb_start(i);
              auto pj = mask_1.get_pb_start(j);
              int64_t biasIndex =
                  h * p.bias_stride_0 + (pi + ki) * p.bias_stride_1 + (pj + kj);
              updt += p.bias[biasIndex];
            }
          } else {
            updt = mask_value;
          }
          int64_t index = b * p.attn_stride_0 + h * p.attn_stride_1 +
              i * p.attn_stride_2 + j * p.attn_stride_3 + y;
          p.attn[index] = updt;
        }
      }
    }
  }
};

template <typename scalar_t, typename CausalMask_>
struct PointwiseNeighborhood2DHalf : PointwiseNeighborhood2DBase<scalar_t> {
  using Base = PointwiseNeighborhood2DBase<scalar_t>;
  using Params = typename Base::Params;
  using CausalMask = CausalMask_;
  static constexpr bool IsBF16Kernel = IsBF16<scalar_t>::value;
  static constexpr bool IsHalfKernel = true;
  static constexpr bool UsesSmem = false;

  __device__ __host__ PointwiseNeighborhood2DHalf() : Base() {}

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
    auto mask_value = AttnMask<scalar_t>::value(p.is_grad);
    auto mask_0 = NeighborhoodMask<CausalMask::Dim0>(
        p.height, p.kernel_size_0, p.dilation_0);
    auto mask_1 = NeighborhoodMask<CausalMask::Dim1>(
        p.width, p.kernel_size_1, p.dilation_1);
    int32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < p.batch_size * p.heads) {
      int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
      if (x < p.height * p.width) {
        int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < p.kernel_size_0 * p.kernel_size_1) {
          auto query2 = HalfHelper::typecast(p.query);
          auto key2 = HalfHelper::typecast(p.key);
          auto b = z / p.heads;
          auto h = z - b * p.heads;
          auto ki = y / p.kernel_size_1;
          auto kj = y - ki * p.kernel_size_1;
          auto i = x / p.width;
          auto j = x - i * p.width;

          auto ni = mask_0.get_window_start(i);
          auto nj = mask_1.get_window_start(j);
          auto ei = mask_0.get_window_end(i, ni);
          auto ej = mask_1.get_window_end(j, nj);

          auto key_idx_i = ki * p.dilation_0 + ni;
          auto key_idx_j = kj * p.dilation_1 + nj;
          scalar_t acc = HalfHelper::zero();

          if (key_idx_i < ei && key_idx_j < ej) {
            auto updt = HalfHelper::zeros();
            int64_t batchHeadOffset =
                b * p.query_stride_0 + h * p.query_stride_1;
            int64_t queryOffset =
                batchHeadOffset + i * p.query_stride_2 + j * p.query_stride_3;
            int64_t keyOffset = batchHeadOffset + key_idx_i * p.query_stride_2 +
                key_idx_j * p.query_stride_3;
            for (int32_t dimOffset = 0; dimOffset < p.dim; ++dimOffset)
              updt = HalfHelper::fma(
                  query2[queryOffset + dimOffset],
                  key2[keyOffset + dimOffset],
                  updt);
            acc = HalfHelper::cast_back(HalfHelper::add(updt.x, updt.y));
            if (p.bias) {
              auto pi = mask_0.get_pb_start(i);
              auto pj = mask_1.get_pb_start(j);
              int64_t biasIndex =
                  h * p.bias_stride_0 + (pi + ki) * p.bias_stride_1 + (pj + kj);
              acc = HalfHelper::add(acc, p.bias[biasIndex]);
            }
          } else {
            acc = mask_value;
          }
          int64_t index = b * p.attn_stride_0 + h * p.attn_stride_1 +
              i * p.attn_stride_2 + j * p.attn_stride_3 + y;
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
  using scalar_t = typename Args::Dtype;
  using CausalMask = typename Args::CausalMask;
  using Kernel = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull<scalar_t, CausalMask>,
      PointwiseNeighborhood2DHalf<scalar_t, CausalMask>>::type;

  using Kernel3x3 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull3x3<scalar_t>,
      PointwiseNeighborhood2DHalf3x3<scalar_t>>::type;
  using Kernel5x5 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull5x5<scalar_t>,
      PointwiseNeighborhood2DHalf5x5<scalar_t>>::type;
  using Kernel7x7 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull7x7<scalar_t>,
      PointwiseNeighborhood2DHalf7x7<scalar_t>>::type;
  using Kernel9x9 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull9x9<scalar_t>,
      PointwiseNeighborhood2DHalf9x9<scalar_t>>::type;
  using Kernel11x11 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull11x11<scalar_t>,
      PointwiseNeighborhood2DHalf11x11<scalar_t>>::type;
  using Kernel13x13 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull13x13<scalar_t>,
      PointwiseNeighborhood2DHalf13x13<scalar_t>>::type;
  using Params = typename Kernel::Params;

  void operator()(
      int32_t cc,
      cudaStream_t stream,
      bool is_grad,
      void* query_ptr,
      void* key_ptr,
      void* attn_ptr,
      int32_t batch_size,
      int32_t heads,
      int32_t height,
      int32_t width,
      int32_t original_dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int64_t attn_stride_3,
      const std::tuple<int32_t, int32_t>& kernel_size,
      const std::tuple<int32_t, int32_t>& dilation) {
    auto dim = Kernel::get_dim(original_dim);
    auto params = Params(
        is_grad,
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
        height,
        width,
        heads,
        std::get<0>(kernel_size),
        std::get<1>(kernel_size),
        std::get<0>(dilation),
        std::get<1>(dilation),
        dim,
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3);
    if constexpr (!CausalMask::Dim0 && !CausalMask::Dim1) {
      if (natten::kEnableTiledNA && all_dims_match(kernel_size) &&
          all_dims_match(dilation) && original_dim == 32 && cc >= 50) {
        auto kernel_size_ = std::get<0>(kernel_size);
        auto dilation_ = std::get<0>(dilation);
        if (kernel_size_ == 3) {
          LaunchParams lp = Kernel3x3::get_launch_params(
              batch_size * heads,
              height,
              width,
              natten::flatten(kernel_size),
              dilation_);
          launch_cuda_kernel<Kernel3x3>
              <<<lp.grid, lp.block, 0, stream>>>(params);
          return;
        } else if (kernel_size_ == 5) {
          LaunchParams lp = Kernel5x5::get_launch_params(
              batch_size * heads,
              height,
              width,
              natten::flatten(kernel_size),
              dilation_);
          launch_cuda_kernel<Kernel5x5>
              <<<lp.grid, lp.block, 0, stream>>>(params);
          return;
        } else if (kernel_size_ == 7) {
          LaunchParams lp = Kernel7x7::get_launch_params(
              batch_size * heads,
              height,
              width,
              natten::flatten(kernel_size),
              dilation_);
          launch_cuda_kernel<Kernel7x7>
              <<<lp.grid, lp.block, 0, stream>>>(params);
          return;
        } else if (kernel_size_ == 9) {
          LaunchParams lp = Kernel9x9::get_launch_params(
              batch_size * heads,
              height,
              width,
              natten::flatten(kernel_size),
              dilation_);
          launch_cuda_kernel<Kernel9x9>
              <<<lp.grid, lp.block, 0, stream>>>(params);
          return;
        } else if (kernel_size_ == 11) {
          LaunchParams lp = Kernel11x11::get_launch_params(
              batch_size * heads,
              height,
              width,
              natten::flatten(kernel_size),
              dilation_);
          launch_cuda_kernel<Kernel11x11>
              <<<lp.grid, lp.block, 0, stream>>>(params);
          return;
        } else if (kernel_size_ == 13) {
          LaunchParams lp = Kernel13x13::get_launch_params(
              batch_size * heads,
              height,
              width,
              natten::flatten(kernel_size),
              dilation_);
          launch_cuda_kernel<Kernel13x13>
              <<<lp.grid, lp.block, 0, stream>>>(params);
          return;
        }
      }
      LaunchParams lp = Kernel::get_launch_params(
          batch_size * heads, height * width, natten::flatten(kernel_size));
      launch_cuda_kernel<Kernel><<<lp.grid, lp.block, 0, stream>>>(params);
    } else {
      LaunchParams lp = Kernel::get_launch_params(
          batch_size * heads, height * width, natten::flatten(kernel_size));
      launch_cuda_kernel<Kernel><<<lp.grid, lp.block, 0, stream>>>(params);
    }
  }
};

template <typename Args_>
struct PointwiseNeighborhood2DWithBias {
  using Args = Args_;
  using scalar_t = typename Args::Dtype;
  using CausalMask = typename Args::CausalMask;
  using Kernel = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull<scalar_t, CausalMask>,
      PointwiseNeighborhood2DHalf<scalar_t, CausalMask>>::type;
  using Kernel3x3 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull3x3<scalar_t>,
      PointwiseNeighborhood2DHalf3x3<scalar_t>>::type;
  using Kernel5x5 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull5x5<scalar_t>,
      PointwiseNeighborhood2DHalf5x5<scalar_t>>::type;
  using Kernel7x7 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull7x7<scalar_t>,
      PointwiseNeighborhood2DHalf7x7<scalar_t>>::type;
  using Kernel9x9 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull9x9<scalar_t>,
      PointwiseNeighborhood2DHalf9x9<scalar_t>>::type;
  using Kernel11x11 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull11x11<scalar_t>,
      PointwiseNeighborhood2DHalf11x11<scalar_t>>::type;
  using Kernel13x13 = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood2DFull13x13<scalar_t>,
      PointwiseNeighborhood2DHalf13x13<scalar_t>>::type;
  using Params = typename Kernel::Params;

  static_assert(
      !CausalMask::Dim0 && !CausalMask::Dim1,
      "PN+Bias does not support causal masking.");

  void operator()(
      int32_t cc,
      cudaStream_t stream,
      void* query_ptr,
      void* key_ptr,
      void* bias_ptr,
      void* attn_ptr,
      int32_t batch_size,
      int32_t heads,
      int32_t height,
      int32_t width,
      int32_t original_dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      int64_t attn_stride_3,
      const std::tuple<int32_t, int32_t>& kernel_size,
      const std::tuple<int32_t, int32_t>& dilation) {
    auto dim = Kernel::get_dim(original_dim);
    auto params = Params(
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(bias_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
        height,
        width,
        heads,
        std::get<0>(kernel_size),
        std::get<1>(kernel_size),
        std::get<0>(dilation),
        std::get<1>(dilation),
        dim,
        batch_size,
        attn_stride_0,
        attn_stride_1,
        attn_stride_2,
        attn_stride_3);
    if (natten::kEnableTiledNA && all_dims_match(kernel_size) &&
        all_dims_match(dilation) && original_dim == 32 && cc >= 50) {
      auto kernel_size_ = std::get<0>(kernel_size);
      auto dilation_ = std::get<0>(dilation);
      if (kernel_size_ == 3) {
        LaunchParams lp = Kernel3x3::get_launch_params(
            batch_size * heads,
            height,
            width,
            kernel_size_ * kernel_size_,
            dilation_);
        launch_cuda_kernel<Kernel3x3><<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      } else if (kernel_size_ == 5) {
        LaunchParams lp = Kernel5x5::get_launch_params(
            batch_size * heads,
            height,
            width,
            natten::flatten(kernel_size),
            dilation_);
        launch_cuda_kernel<Kernel5x5><<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      } else if (kernel_size_ == 7) {
        LaunchParams lp = Kernel7x7::get_launch_params(
            batch_size * heads,
            height,
            width,
            natten::flatten(kernel_size),
            dilation_);
        launch_cuda_kernel<Kernel7x7><<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      } else if (kernel_size_ == 9) {
        LaunchParams lp = Kernel9x9::get_launch_params(
            batch_size * heads,
            height,
            width,
            natten::flatten(kernel_size),
            dilation_);
        launch_cuda_kernel<Kernel9x9><<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      } else if (kernel_size_ == 11) {
        LaunchParams lp = Kernel11x11::get_launch_params(
            batch_size * heads,
            height,
            width,
            natten::flatten(kernel_size),
            dilation_);
        launch_cuda_kernel<Kernel11x11>
            <<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      } else if (kernel_size_ == 13) {
        LaunchParams lp = Kernel13x13::get_launch_params(
            batch_size * heads,
            height,
            width,
            natten::flatten(kernel_size),
            dilation_);
        launch_cuda_kernel<Kernel13x13>
            <<<lp.grid, lp.block, 0, stream>>>(params);
        return;
      }
    }
    LaunchParams lp = Kernel::get_launch_params(
        batch_size * heads, height * width, natten::flatten(kernel_size));
    launch_cuda_kernel<Kernel><<<lp.grid, lp.block, 0, stream>>>(params);
  }
};

} // namespace naive
} // namespace cuda
} // namespace natten
