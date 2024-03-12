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

#include <natten/cuda/naive/natten_commons.cuh>
#include <natten/cuda/naive/natten_tiled_macros.cuh>
#include <natten/cuda/naive/tiled/base.cuh>

namespace natten {
namespace cuda {
namespace naive {

///////////////////////////////////////////////////////////////////////////////
/////////////////////////////// Tiled NA //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

/////////////// 3x3
template <typename scalar_t>
struct PointwiseNeighborhood2DFull3x3 : PointwiseNeighborhood2DBase<scalar_t> {
  using Base = PointwiseNeighborhood2DBase<scalar_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = false;
  static constexpr bool IsHalfKernel = false;
  static constexpr bool UsesSmem = true;

  __device__ __host__ PointwiseNeighborhood2DFull3x3() : Base() {}

  __device__ void launch(Params p) {
    // Because batch heads have stride 1 per threadblock, we can just use
    // blockIdx since blockDim will be 1 and threadIdx will always be 0. const
    // int32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    // TODO: remove when/if tiled kernels allow varying dilations
    auto dilation = p.dilation_0;
    int32_t z = blockIdx.z;
    int32_t b = z / p.heads;
    int32_t h = z - b * p.heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    int32_t lti = threadIdx.y * (TILE_3 * KERNEL_SIZE_3) + threadIdx.x;
    int64_t batchHeadOffset = b * p.query_stride_0 + h * p.query_stride_1;
    int32_t si = int32_t(blockIdx.y / dilation) * (TILE_3 * dilation) +
        (blockIdx.y % dilation);
    int32_t sj = int32_t(blockIdx.x / dilation) * (TILE_3 * dilation) +
        (blockIdx.x % dilation);
    int32_t sni = get_window_start(
        si, p.height, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
    int32_t snj = get_window_start(
        sj, p.width, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
    __shared__ scalar_t tile[TILE_3 * TILE_3][DIM_32 + 3];
    __shared__ scalar_t kTile[KTILE_3 * KTILE_3][DIM_32 + 3];

    /* query tile */
    int32_t qtx = lti / QSTRIDE_3;
    int32_t qty = (lti - qtx * QSTRIDE_3) * QITERS_3;
    if (qtx < TILE_3 * TILE_3) {
      int32_t qi = qtx / TILE_3;
      int32_t qj = (qtx - qi * TILE_3) * dilation + sj;
      qi = qi * dilation + si;
      if (qi < p.height && qj < p.width) {
#pragma unroll
        for (int32_t ti = 0; ti < QITERS_3; ++ti)
          tile[qtx][qty + ti] = p.query
                                    [batchHeadOffset + qi * p.query_stride_2 +
                                     qj * p.query_stride_3 + qty + ti];
      }
    }
    /* key tile */
    int32_t ktx = lti / KSTRIDE_32;
    int32_t kty = (lti - ktx * KSTRIDE_32) * KITERS_32;
    if (ktx < KTILE_3 * KTILE_3) {
      int32_t bi = ktx / KTILE_3;
      int32_t bj = (ktx - bi * KTILE_3) * dilation + snj;
      bi = bi * dilation + sni;
      if (bi < p.height && bj < p.width) {
        int64_t keyOffset = batchHeadOffset + bi * p.query_stride_2 +
            bj * p.query_stride_3 + kty;
#pragma unroll
        for (int32_t ti = 0; ti < KITERS_32; ++ti)
          kTile[ktx][kty + ti] = p.key[keyOffset + ti];
      }
    }
    __syncthreads();
    int32_t ii = threadIdx.y / KERNEL_SIZE_3;
    int32_t ki = threadIdx.y - ii * KERNEL_SIZE_3;
    int32_t jj = threadIdx.x / KERNEL_SIZE_3;
    int32_t kj = threadIdx.x - jj * KERNEL_SIZE_3;
    int32_t i = si + ii * dilation, j = sj + jj * dilation;
    if (i < p.height && j < p.width) {
      int32_t ni = get_window_start(
          i, p.height, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
      int32_t nj = get_window_start(
          j, p.width, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
      scalar_t updt = scalar_t(0);
      int32_t queryIdx = ii * TILE_3 + jj;
      int32_t keyIdx =
          int32_t((ni + ki * dilation - sni) / dilation) * KTILE_3 +
          int32_t((nj + kj * dilation - snj) / dilation);

#pragma unroll
      for (int32_t dimOffset = 0; dimOffset < DIM_32; ++dimOffset)
        updt += tile[queryIdx][dimOffset] * kTile[keyIdx][dimOffset];

      int64_t index = b * p.attn_stride_0 + h * p.attn_stride_1 +
          i * p.attn_stride_2 + j * p.attn_stride_3 + ki * KERNEL_SIZE_3 + kj;
      if (p.bias) {
        int32_t pi = get_pb_start(
            i, p.height, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
        int32_t pj = get_pb_start(
            j, p.width, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
        int64_t biasIndex =
            h * p.bias_stride_0 + (pi + ki) * p.bias_stride_1 + (pj + kj);
        updt += p.bias[biasIndex];
      }
      p.attn[index] = updt;
    }
    //}
  }

  static LaunchParams get_launch_params(
      int32_t batch_dim,
      int32_t height,
      int32_t width,
      int32_t attention_span,
      int32_t dilation) {
    int32_t xsize = width * KERNEL_SIZE_3;
    int32_t ysize = height * KERNEL_SIZE_3;
    int32_t XTHREADS = XYTHREADS_3;
    int32_t YTHREADS = XYTHREADS_3;
    int32_t BATCHTHREADS = BATCHTHREADS_3;
    const dim3 grid(
        (xsize + XTHREADS * dilation - 1) / XTHREADS,
        (ysize + YTHREADS * dilation - 1) / YTHREADS,
        (batch_dim + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 block(XTHREADS, YTHREADS, BATCHTHREADS);
    return LaunchParams(grid, block);
  }
};

template <typename scalar_t>
struct PointwiseNeighborhood2DHalf3x3 : PointwiseNeighborhood2DBase<scalar_t> {
  using Base = PointwiseNeighborhood2DBase<scalar_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = IsBF16<scalar_t>::value;
  static constexpr bool IsHalfKernel = true;
  static constexpr bool UsesSmem = true;

  __device__ __host__ PointwiseNeighborhood2DHalf3x3() : Base() {}

  using HalfHelper = typename HalfArray<scalar_t>::Base;
  using HalfType = typename HalfHelper::ElementVector;

  __device__ void launch(Params p) {
    // Because batch heads have stride 1 per threadblock, we can just use
    // blockIdx since blockDim will be 1 and threadIdx will always be 0. const
    // int32_t z = blockIdx.z * blockDim.z + threadIdx.z;
    // TODO: remove when/if tiled kernels allow varying dilations
    auto dilation = p.dilation_0;
    int32_t z = blockIdx.z;
    int32_t b = z / p.heads;
    int32_t h = z - b * p.heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    int32_t lti = threadIdx.y * (TILE_3 * KERNEL_SIZE_3) + threadIdx.x;
    int32_t stride2 = DIMHALF_32 * p.width;
    int64_t batchHeadOffset = b * p.query_stride_0 + h * p.query_stride_1;
    int32_t si = int32_t(blockIdx.y / dilation) * (TILE_3 * dilation) +
        (blockIdx.y % dilation);
    int32_t sj = int32_t(blockIdx.x / dilation) * (TILE_3 * dilation) +
        (blockIdx.x % dilation);
    int32_t sni = get_window_start(
        si, p.height, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
    int32_t snj = get_window_start(
        sj, p.width, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
    __shared__ HalfType tile[TILE_3 * TILE_3][DIM_32 + 3];
    __shared__ HalfType kTile[KTILE_3 * KTILE_3][DIM_32 + 3];
    auto query2 = HalfHelper::typecast(p.query);
    auto key2 = HalfHelper::typecast(p.key);

    /* query tile */
    int32_t qtx = lti / QSTRIDE_3_HALF;
    int32_t qty = (lti - qtx * QSTRIDE_3_HALF) * QITERS_3_HALF;
    if (qtx < TILE_3 * TILE_3) {
      int32_t qi = qtx / TILE_3;
      int32_t qj = (qtx - qi * TILE_3) * dilation + sj;
      qi = qi * dilation + si;
      if (qi < p.height && qj < p.width) {
#pragma unroll
        for (int32_t ti = 0; ti < QITERS_3_HALF; ++ti)
          tile[qtx][qty + ti] = query2
              [batchHeadOffset + qi * stride2 + qj * DIMHALF_32 + qty + ti];
      }
    }
    /* key tile */
    int32_t ktx = lti / KSTRIDE_32;
    int32_t kty = (lti - ktx * KSTRIDE_32) * KHALFITERS_32;
    if (ktx < KTILE_3 * KTILE_3) {
      int32_t bi = ktx / KTILE_3;
      int32_t bj = (ktx - bi * KTILE_3) * dilation + snj;
      bi = bi * dilation + sni;
      if (bi < p.height && bj < p.width) {
        int64_t keyOffset =
            batchHeadOffset + bi * stride2 + bj * DIMHALF_32 + kty;
#pragma unroll
        for (int32_t ti = 0; ti < KHALFITERS_32; ++ti)
          kTile[ktx][kty + ti] = key2[keyOffset + ti];
      }
    }
    __syncthreads();
    int32_t ii = threadIdx.y / KERNEL_SIZE_3;
    int32_t ki = threadIdx.y - ii * KERNEL_SIZE_3;
    int32_t jj = threadIdx.x / KERNEL_SIZE_3;
    int32_t kj = threadIdx.x - jj * KERNEL_SIZE_3;
    int32_t i = si + ii * dilation, j = sj + jj * dilation;
    if (i < p.height && j < p.width) {
      int32_t ni = get_window_start(
          i, p.height, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
      int32_t nj = get_window_start(
          j, p.width, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
      auto updt = HalfHelper::zeros();
      int32_t queryIdx = ii * TILE_3 + jj;
      int32_t keyIdx =
          int32_t((ni + ki * dilation - sni) / dilation) * KTILE_3 +
          int32_t((nj + kj * dilation - snj) / dilation);

#pragma unroll
      for (int32_t dimOffset = 0; dimOffset < DIMHALF_32; ++dimOffset)
        updt = HalfHelper::fma(
            tile[queryIdx][dimOffset], kTile[keyIdx][dimOffset], updt);

      int64_t index = b * p.attn_stride_0 + h * p.attn_stride_1 +
          i * p.attn_stride_2 + j * p.attn_stride_3 + ki * KERNEL_SIZE_3 + kj;
      scalar_t acc = HalfHelper::cast_back(HalfHelper::add(updt.x, updt.y));
      if (p.bias) {
        int32_t pi = get_pb_start(
            i, p.height, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
        int32_t pj = get_pb_start(
            j, p.width, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
        int64_t biasIndex =
            h * p.bias_stride_0 + (pi + ki) * p.bias_stride_1 + (pj + kj);
        acc = HalfHelper::add(acc, p.bias[biasIndex]);
      }
      p.attn[index] = acc;
    }
    //}
  }

  static LaunchParams get_launch_params(
      int32_t batch_dim,
      int32_t height,
      int32_t width,
      int32_t attention_span,
      int32_t dilation) {
    int32_t xsize = width * KERNEL_SIZE_3;
    int32_t ysize = height * KERNEL_SIZE_3;
    int32_t XTHREADS = XYTHREADS_3;
    int32_t YTHREADS = XYTHREADS_3;
    int32_t BATCHTHREADS = BATCHTHREADS_3;
    const dim3 grid(
        (xsize + XTHREADS * dilation - 1) / XTHREADS,
        (ysize + YTHREADS * dilation - 1) / YTHREADS,
        (batch_dim + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 block(XTHREADS, YTHREADS, BATCHTHREADS);
    return LaunchParams(grid, block);
  }
};

} // namespace naive
} // namespace cuda
} // namespace natten
