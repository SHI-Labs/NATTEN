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

#include <natten/naive_argpack.h>
#include <natten/natten.h>
#include <natten/cuda/naive/natten_commons.cuh>

namespace natten {
namespace cuda {
namespace naive {

template <typename scalar_t>
struct PointwiseNeighborhood3DBase {
  struct Params {
    bool is_grad;
    scalar_t* query; // query / d_out
    scalar_t* key; // key   / value
    scalar_t* bias = nullptr; // optional: bias
    scalar_t* attn; // attn  / d_attn
    int32_t depth;
    int32_t height;
    int32_t width;
    int32_t heads;
    int32_t kernel_size_0, kernel_size_1, kernel_size_2;
    int32_t dilation_0, dilation_1, dilation_2;
    int32_t dim;
    int32_t batch_size;
    int64_t attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3,
        attn_stride_4;
    int64_t query_stride_0, query_stride_1, query_stride_2, query_stride_3,
        query_stride_4;
    int64_t bias_stride_0, bias_stride_1, bias_stride_2;

    __device__ __host__ Params() {}

    __device__ __host__ Params(
        bool is_grad,
        scalar_t* query,
        scalar_t* key,
        scalar_t* attn,
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
        int32_t batch_size,
        int64_t attn_stride_0,
        int64_t attn_stride_1,
        int64_t attn_stride_2,
        int64_t attn_stride_3,
        int64_t attn_stride_4)
        : is_grad(is_grad),
          query(query),
          key(key),
          attn(attn),
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
        int32_t batch_size,
        int64_t attn_stride_0,
        int64_t attn_stride_1,
        int64_t attn_stride_2,
        int64_t attn_stride_3,
        int64_t attn_stride_4)
        : is_grad(false),
          query(query),
          key(key),
          bias(bias),
          attn(attn),
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
          batch_size(batch_size),
          bias_stride_2(2 * kernel_size_2 - 1),
          bias_stride_1((2 * kernel_size_1 - 1) * (2 * kernel_size_2 - 1)),
          bias_stride_0(
              (2 * kernel_size_0 - 1) * (2 * kernel_size_1 - 1) *
              (2 * kernel_size_2 - 1)),
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
      int32_t batch_dim,
      int32_t spatial_size,
      int32_t attention_span) {
    // NOTE: was 1024 as of 01/30/2024, but since adding causal masks,
    // runs out of registers with certain causal masks (F, T, T) with
    // 1024 threads per CTA.
    int32_t num_threads = 512;
    int32_t KERNELTHREADS =
        min(num_threads, attention_span /* == kernel_size^3 */);
    int32_t PIXELTHREADS =
        min(int32_t(num_threads / KERNELTHREADS), spatial_size);
    int32_t BATCHTHREADS =
        min(64, max(1, num_threads / (PIXELTHREADS * KERNELTHREADS)));
    dim3 grid(
        (spatial_size + PIXELTHREADS - 1) / PIXELTHREADS,
        (attention_span + KERNELTHREADS - 1) / KERNELTHREADS,
        (batch_dim + BATCHTHREADS - 1) / BATCHTHREADS);
    dim3 block(PIXELTHREADS, KERNELTHREADS, BATCHTHREADS);
    return LaunchParams(grid, block);
  }
};

template <typename scalar_t, typename CausalMask_>
struct PointwiseNeighborhood3DFull : PointwiseNeighborhood3DBase<scalar_t> {
  using Base = PointwiseNeighborhood3DBase<scalar_t>;
  using Params = typename Base::Params;
  using CausalMask = CausalMask_;
  static constexpr bool IsBF16Kernel = false;
  static constexpr bool IsHalfKernel = false;
  static constexpr bool UsesSmem = false;

  __device__ __host__ PointwiseNeighborhood3DFull() : Base() {}

  static __host__ int32_t get_dim(int32_t dim) {
    return dim;
  }

  __device__ void launch(Params p) {
    auto mask_value = AttnMask<scalar_t>::value(p.is_grad);
    auto mask_0 = NeighborhoodMask<CausalMask::Dim0>(
        p.depth, p.kernel_size_0, p.dilation_0);
    auto mask_1 = NeighborhoodMask<CausalMask::Dim1>(
        p.height, p.kernel_size_1, p.dilation_1);
    auto mask_2 = NeighborhoodMask<CausalMask::Dim2>(
        p.width, p.kernel_size_2, p.dilation_2);
    int32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < p.batch_size * p.heads) {
      int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
      if (x < p.depth * p.height * p.width) {
        int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < p.kernel_size_0 * p.kernel_size_1 * p.kernel_size_2) {
          int32_t indtmp1 = y / p.kernel_size_2;
          auto kk = indtmp1 / p.kernel_size_1;
          auto kj = y - indtmp1 * p.kernel_size_2;
          auto ki = indtmp1 - kk * p.kernel_size_1;

          indtmp1 = x / p.width;
          auto k = indtmp1 / p.height;
          auto j = x - indtmp1 * p.width;
          auto i = indtmp1 - k * p.height;

          auto b = z / p.heads;
          auto h = z - b * p.heads;

          auto nk = mask_0.get_window_start(k);
          auto ni = mask_1.get_window_start(i);
          auto nj = mask_2.get_window_start(j);
          auto ek = mask_0.get_window_end(k, nk);
          auto ei = mask_1.get_window_end(i, ni);
          auto ej = mask_2.get_window_end(j, nj);

          auto key_idx_k = kk * p.dilation_0 + nk;
          auto key_idx_i = ki * p.dilation_1 + ni;
          auto key_idx_j = kj * p.dilation_2 + nj;
          scalar_t updt = scalar_t(0);
          if (key_idx_i < ei && key_idx_j < ej && key_idx_k < ek) {
            int64_t batchHeadOffset =
                b * p.query_stride_0 + h * p.query_stride_1;
            int64_t queryOffset = batchHeadOffset + k * p.query_stride_2 +
                i * p.query_stride_3 + j * p.query_stride_4;
            int64_t keyOffset = batchHeadOffset + key_idx_k * p.query_stride_2 +
                key_idx_i * p.query_stride_3 + key_idx_j * p.query_stride_4;
            for (int32_t dimOffset = 0; dimOffset < p.dim; ++dimOffset)
              updt += p.query[queryOffset + dimOffset] *
                  p.key[keyOffset + dimOffset];
            if (p.bias) {
              auto pk = mask_0.get_pb_start(k);
              auto pi = mask_1.get_pb_start(i);
              auto pj = mask_2.get_pb_start(j);
              int64_t biasIndex = h * p.bias_stride_0 +
                  (pk + kk) * p.bias_stride_1 + (pi + ki) * p.bias_stride_2 +
                  (pj + kj);
              updt += p.bias[biasIndex];
            }
          } else {
            updt = mask_value;
          }
          int64_t index = b * p.attn_stride_0 + h * p.attn_stride_1 +
              k * p.attn_stride_2 + i * p.attn_stride_3 + j * p.attn_stride_4 +
              y;
          p.attn[index] = updt;
        }
      }
    }
  }
};

template <typename scalar_t, typename CausalMask_>
struct PointwiseNeighborhood3DHalf : PointwiseNeighborhood3DBase<scalar_t> {
  using Base = PointwiseNeighborhood3DBase<scalar_t>;
  using Params = typename Base::Params;
  using CausalMask = CausalMask_;
  static constexpr bool IsBF16Kernel = IsBF16<scalar_t>::value;
  static constexpr bool IsHalfKernel = true;
  static constexpr bool UsesSmem = false;

  __device__ __host__ PointwiseNeighborhood3DHalf() : Base() {}

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
        p.depth, p.kernel_size_0, p.dilation_0);
    auto mask_1 = NeighborhoodMask<CausalMask::Dim1>(
        p.height, p.kernel_size_1, p.dilation_1);
    auto mask_2 = NeighborhoodMask<CausalMask::Dim2>(
        p.width, p.kernel_size_2, p.dilation_2);
    int32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < p.batch_size * p.heads) {
      int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
      if (x < p.depth * p.height * p.width) {
        int32_t y = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < p.kernel_size_0 * p.kernel_size_1 * p.kernel_size_2) {
          auto query2 = HalfHelper::typecast(p.query);
          auto key2 = HalfHelper::typecast(p.key);
          int32_t indtmp1 = y / p.kernel_size_2;
          auto kk = indtmp1 / p.kernel_size_1;
          auto kj = y - indtmp1 * p.kernel_size_2;
          auto ki = indtmp1 - kk * p.kernel_size_1;

          indtmp1 = x / p.width;
          auto k = indtmp1 / p.height;
          auto j = x - indtmp1 * p.width;
          auto i = indtmp1 - k * p.height;

          auto b = z / p.heads;
          auto h = z - b * p.heads;

          auto nk = mask_0.get_window_start(k);
          auto ni = mask_1.get_window_start(i);
          auto nj = mask_2.get_window_start(j);
          auto ek = mask_0.get_window_end(k, nk);
          auto ei = mask_1.get_window_end(i, ni);
          auto ej = mask_2.get_window_end(j, nj);

          auto key_idx_k = kk * p.dilation_0 + nk;
          auto key_idx_i = ki * p.dilation_1 + ni;
          auto key_idx_j = kj * p.dilation_2 + nj;
          scalar_t acc = HalfHelper::zero();

          if (key_idx_i < ei && key_idx_j < ej && key_idx_k < ek) {
            auto updt = HalfHelper::zeros();
            int64_t batchHeadOffset =
                b * p.query_stride_0 + h * p.query_stride_1;
            int64_t queryOffset = batchHeadOffset + k * p.query_stride_2 +
                i * p.query_stride_3 + j * p.query_stride_4;
            int64_t keyOffset = batchHeadOffset + key_idx_k * p.query_stride_2 +
                key_idx_i * p.query_stride_3 + key_idx_j * p.query_stride_4;
            for (int32_t dimOffset = 0; dimOffset < p.dim; ++dimOffset)
              updt = HalfHelper::fma(
                  query2[queryOffset + dimOffset],
                  key2[keyOffset + dimOffset],
                  updt);
            acc = HalfHelper::cast_back(HalfHelper::add(updt.x, updt.y));
            if (p.bias) {
              auto pk = mask_0.get_pb_start(k);
              auto pi = mask_1.get_pb_start(i);
              auto pj = mask_2.get_pb_start(j);
              int64_t biasIndex = h * p.bias_stride_0 +
                  (pk + kk) * p.bias_stride_1 + (pi + ki) * p.bias_stride_2 +
                  (pj + kj);
              acc = HalfHelper::add(acc, p.bias[biasIndex]);
            }
          } else {
            acc = mask_value;
          }
          int64_t index = b * p.attn_stride_0 + h * p.attn_stride_1 +
              k * p.attn_stride_2 + i * p.attn_stride_3 + j * p.attn_stride_4 +
              y;
          p.attn[index] = acc;
        }
      }
    }
  }
};

template <typename Args_>
struct PointwiseNeighborhood3D {
  using Args = Args_;
  using scalar_t = typename Args::Dtype;
  using CausalMask = typename Args::CausalMask;
  using Kernel = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood3DFull<scalar_t, CausalMask>,
      PointwiseNeighborhood3DHalf<scalar_t, CausalMask>>::type;
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
    LaunchParams lp = Kernel::Base::get_launch_params(
        batch_size * heads,
        depth * height * width,
        natten::flatten(kernel_size));
    auto params = Params(
        is_grad,
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
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
  using scalar_t = typename Args::Dtype;
  using CausalMask = typename Args::CausalMask;
  using Kernel = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood3DFull<scalar_t, CausalMask>,
      PointwiseNeighborhood3DHalf<scalar_t, CausalMask>>::type;
  using Params = typename Kernel::Params;

  static_assert(
      !CausalMask::Dim0 && !CausalMask::Dim1 && !CausalMask::Dim2,
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
    LaunchParams lp = Kernel::Base::get_launch_params(
        batch_size * heads,
        depth * height * width,
        natten::flatten(kernel_size));
    auto params = Params(
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(bias_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
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
