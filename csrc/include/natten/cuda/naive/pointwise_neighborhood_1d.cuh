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

#include <natten/naive_argpack.h>
#include <natten/natten.h>
#include <natten/cuda/naive/natten_commons.cuh>

namespace natten {
namespace cuda {
namespace naive {

template <typename scalar_t>
struct PointwiseNeighborhood1DBase {
  struct Params {
    bool is_grad;
    scalar_t* query; // query / d_out
    scalar_t* key; // key   / value
    scalar_t* bias = nullptr; // optional: bias
    scalar_t* attn; // attn  / d_attn
    int32_t length;
    int32_t heads;
    int32_t kernel_size;
    int32_t dilation;
    int32_t dim;
    int32_t batch_size;
    int64_t attn_stride_0, attn_stride_1, attn_stride_2;
    int64_t query_stride_0, query_stride_1, query_stride_2;
    int64_t bias_stride_0;

    __device__ __host__ Params() {}

    __device__ __host__ Params(
        bool is_grad,
        scalar_t* query,
        scalar_t* key,
        scalar_t* attn,
        int32_t length,
        int32_t heads,
        int32_t kernel_size,
        int32_t dilation,
        int32_t dim,
        int32_t batch_size,
        int64_t attn_stride_0,
        int64_t attn_stride_1,
        int64_t attn_stride_2)
        : is_grad(is_grad),
          query(query),
          key(key),
          attn(attn),
          length(length),
          heads(heads),
          kernel_size(kernel_size),
          dilation(dilation),
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
        int32_t length,
        int32_t heads,
        int32_t kernel_size,
        int32_t dilation,
        int32_t dim,
        int32_t batch_size,
        int64_t attn_stride_0,
        int64_t attn_stride_1,
        int64_t attn_stride_2)
        : is_grad(false),
          query(query),
          key(key),
          bias(bias),
          attn(attn),
          length(length),
          heads(heads),
          kernel_size(kernel_size),
          dilation(dilation),
          dim(dim),
          batch_size(batch_size),
          bias_stride_0(2 * kernel_size - 1),
          attn_stride_2(attn_stride_2),
          attn_stride_1(attn_stride_1),
          attn_stride_0(attn_stride_0),
          query_stride_2(dim),
          query_stride_1(dim * length),
          query_stride_0(dim * length * heads) {}
  };

  __device__ __host__ PointwiseNeighborhood1DBase() {}

  static LaunchParams get_launch_params(
      int32_t batch_dim,
      int32_t length,
      int32_t kernel_size) {
    int32_t KERNELTHREADS = min(CUDA_NUM_THREADS, kernel_size);
    int32_t TOKENTHREADS = min(CUDA_NUM_THREADS / KERNELTHREADS, length);
    int32_t BATCHTHREADS =
        min(64, max(1, CUDA_NUM_THREADS / (TOKENTHREADS * KERNELTHREADS)));
    dim3 grid(
        (length + TOKENTHREADS - 1) / TOKENTHREADS,
        (kernel_size + KERNELTHREADS - 1) / KERNELTHREADS,
        (batch_dim + BATCHTHREADS - 1) / BATCHTHREADS);
    dim3 block(TOKENTHREADS, KERNELTHREADS, BATCHTHREADS);
    return LaunchParams(grid, block);
  }
};

template <typename scalar_t, typename CausalMask_>
struct PointwiseNeighborhood1DFull : PointwiseNeighborhood1DBase<scalar_t> {
  using Base = PointwiseNeighborhood1DBase<scalar_t>;
  using Params = typename Base::Params;
  using CausalMask = CausalMask_;
  static constexpr bool IsBF16Kernel = false;
  static constexpr bool IsHalfKernel = false;
  static constexpr bool UsesSmem = false;

  __device__ __host__ PointwiseNeighborhood1DFull() : Base() {}

  static __host__ int32_t get_dim(int32_t dim) {
    return dim;
  }

  __device__ void launch(Params p) {
    auto mask_value = AttnMask<scalar_t>::value(p.is_grad);
    auto mask_0 =
        NeighborhoodMask<CausalMask::Dim0>(p.length, p.kernel_size, p.dilation);
    int32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < p.batch_size * p.heads) {
      int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < p.length) {
        int32_t ki = blockIdx.y * blockDim.y + threadIdx.y;
        if (ki < p.kernel_size) {
          auto b = z / p.heads;
          auto h = z - b * p.heads;

          auto ni = mask_0.get_window_start(i);
          auto ei = mask_0.get_window_end(i, ni);

          scalar_t updt = scalar_t(0.0);
          auto key_idx = ki * p.dilation + ni;
          if (key_idx < ei) {
            int64_t batchHeadOffset =
                b * p.query_stride_0 + h * p.query_stride_1;
            int64_t queryOffset = batchHeadOffset + i * p.query_stride_2;
            int64_t keyOffset = batchHeadOffset + key_idx * p.query_stride_2;
            for (int32_t dimOffset = 0; dimOffset < p.dim; ++dimOffset)
              updt += p.query[queryOffset + dimOffset] *
                  p.key[keyOffset + dimOffset];
            if (p.bias) {
              auto pi = mask_0.get_pb_start(i);
              updt += p.bias[h * p.bias_stride_0 + (pi + ki)];
            }
          } else {
            updt = mask_value;
          }
          p.attn
              [b * p.attn_stride_0 + h * p.attn_stride_1 + i * p.attn_stride_2 +
               ki] = updt;
        }
      }
    }
  }
};

template <typename scalar_t, typename CausalMask_>
struct PointwiseNeighborhood1DHalf : PointwiseNeighborhood1DBase<scalar_t> {
  using Base = PointwiseNeighborhood1DBase<scalar_t>;
  using Params = typename Base::Params;
  using CausalMask = CausalMask_;
  static constexpr bool IsBF16Kernel = IsBF16<scalar_t>::value;
  static constexpr bool IsHalfKernel = true;
  static constexpr bool UsesSmem = false;

  __device__ __host__ PointwiseNeighborhood1DHalf() : Base() {}

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
    auto mask_0 =
        NeighborhoodMask<CausalMask::Dim0>(p.length, p.kernel_size, p.dilation);
    int32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < p.batch_size * p.heads) {
      int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < p.length) {
        int32_t ki = blockIdx.y * blockDim.y + threadIdx.y;
        if (ki < p.kernel_size) {
          auto query2 = HalfHelper::typecast(p.query);
          auto key2 = HalfHelper::typecast(p.key);
          auto b = z / p.heads;
          auto h = z - b * p.heads;

          auto ni = mask_0.get_window_start(i);
          auto ei = mask_0.get_window_end(i, ni);

          scalar_t acc = HalfHelper::zero();
          auto key_idx = ki * p.dilation + ni;
          if (key_idx < ei) {
            auto updt = HalfHelper::zeros();
            int64_t batchHeadOffset =
                b * p.query_stride_0 + h * p.query_stride_1;
            int64_t queryOffset = batchHeadOffset + i * p.query_stride_2;
            int64_t keyOffset = batchHeadOffset + key_idx * p.query_stride_2;
            for (int32_t dimOffset = 0; dimOffset < p.dim; ++dimOffset)
              updt = HalfHelper::fma(
                  query2[queryOffset + dimOffset],
                  key2[keyOffset + dimOffset],
                  updt);
            acc = HalfHelper::cast_back(HalfHelper::add(updt.x, updt.y));
            if (p.bias) {
              auto pi = mask_0.get_pb_start(i);
              acc =
                  HalfHelper::add(acc, p.bias[h * p.bias_stride_0 + (pi + ki)]);
            }
          } else {
            acc = mask_value;
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
  using scalar_t = typename Args::Dtype;
  using CausalMask = typename Args::CausalMask;
  using Kernel = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood1DFull<scalar_t, CausalMask>,
      PointwiseNeighborhood1DHalf<scalar_t, CausalMask>>::type;
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
      int32_t length,
      int32_t dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      const std::tuple<int32_t>& kernel_size,
      const std::tuple<int32_t>& dilation) {
    dim = Kernel::get_dim(dim);
    LaunchParams lp = Kernel::Base::get_launch_params(
        batch_size * heads, length, natten::flatten(kernel_size));
    auto params = Params(
        is_grad,
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
        length,
        heads,
        std::get<0>(kernel_size),
        std::get<0>(dilation),
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
  using scalar_t = typename Args::Dtype;
  using CausalMask = typename Args::CausalMask;
  using Kernel = typename std::conditional<
      sizeof(scalar_t) >= 4,
      PointwiseNeighborhood1DFull<scalar_t, CausalMask>,
      PointwiseNeighborhood1DHalf<scalar_t, CausalMask>>::type;
  using Params = typename Kernel::Params;

  static_assert(!CausalMask::Dim0, "PN+Bias does not support causal masking.");

  void operator()(
      int32_t cc,
      cudaStream_t stream,
      void* query_ptr,
      void* key_ptr,
      void* bias_ptr,
      void* attn_ptr,
      int32_t batch_size,
      int32_t heads,
      int32_t length,
      int32_t dim,
      int64_t attn_stride_0,
      int64_t attn_stride_1,
      int64_t attn_stride_2,
      const std::tuple<int32_t>& kernel_size,
      const std::tuple<int32_t>& dilation) {
    dim = Kernel::get_dim(dim);
    LaunchParams lp = Kernel::Base::get_launch_params(
        batch_size * heads, length, natten::flatten(kernel_size));
    auto params = Params(
        reinterpret_cast<scalar_t*>(query_ptr),
        reinterpret_cast<scalar_t*>(key_ptr),
        reinterpret_cast<scalar_t*>(bias_ptr),
        reinterpret_cast<scalar_t*>(attn_ptr),
        length,
        heads,
        std::get<0>(kernel_size),
        std::get<0>(dilation),
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
