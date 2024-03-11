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

/////////////// 11x11 and 13x13
template <
    typename scalar_t,
    typename memscalar_t,
    int32_t TILEX,
    int32_t TILEY,
    int32_t KTILEX,
    int32_t KTILEY,
    int32_t XTHREADS,
    int32_t YTHREADS,
    int32_t BATCHTHREADS,
    int32_t KERNEL_SIZE,
    int32_t NEIGHBORHOOD_SIZE>
struct PointwiseNeighborhood2DFull11x11_13x13
    : PointwiseNeighborhood2DBase<scalar_t> {
  using Base = PointwiseNeighborhood2DBase<scalar_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = false;
  static constexpr bool IsHalfKernel = false;
  static constexpr bool UsesSmem = true;

  __device__ __host__ PointwiseNeighborhood2DFull11x11_13x13() : Base() {}

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
    int32_t lti = threadIdx.y * (TILEY * KERNEL_SIZE) + threadIdx.x;
    int64_t batchHeadOffset = b * p.query_stride_0 + h * p.query_stride_1;
    int32_t si = int32_t(blockIdx.y / dilation) * (TILEX * dilation) +
        (blockIdx.y % dilation);
    int32_t sj = int32_t(blockIdx.x / dilation) * (TILEY * dilation) +
        (blockIdx.x % dilation);
    int32_t sni = get_window_start(
        si, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    int32_t snj =
        get_window_start(sj, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    __shared__ memscalar_t tile[TILEX * TILEY][DIM_32 + 3];
    __shared__ memscalar_t kTile[KTILEX * KTILEY][DIM_32 + 3];

    /* query tile */
    int32_t qtx = lti / DIM_32;
    int32_t qty = lti - qtx * DIM_32;
    if (qtx < TILEX * TILEY) {
      int32_t qi = qtx / TILEY;
      int32_t qj = (qtx - qi * TILEY) * dilation + sj;
      qi = qi * dilation + si;
      if (qi < p.height && qj < p.width)
        tile[qtx][qty] = p.query
                             [batchHeadOffset + qi * p.query_stride_2 +
                              qj * p.query_stride_3 + qty];
    }
    /* key tile */
    int32_t ktx = lti / KSTRIDE_32;
    int32_t kty = (lti - ktx * KSTRIDE_32) * KITERS_32;
    if (ktx < KTILEX * KTILEY) {
      int32_t bi = ktx / KTILEY;
      int32_t bj = (ktx - bi * KTILEY) * dilation + snj;
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
    int32_t ii = threadIdx.y / KERNEL_SIZE;
    int32_t ki = threadIdx.y - ii * KERNEL_SIZE;
    int32_t jj = threadIdx.x / KERNEL_SIZE;
    int32_t kj = threadIdx.x - jj * KERNEL_SIZE;
    int32_t i = si + ii * dilation, j = sj + jj * dilation;
    if (i < p.height && j < p.width) {
      int32_t ni = get_window_start(
          i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      int32_t nj = get_window_start(
          j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      scalar_t updt = scalar_t(0);
      int32_t queryIdx = ii * TILEY + jj;
      int32_t keyIdx = int32_t((ni + ki * dilation - sni) / dilation) * KTILEY +
          int32_t((nj + kj * dilation - snj) / dilation);

#pragma unroll
      for (int32_t dimOffset = 0; dimOffset < DIM_32; ++dimOffset)
        updt += tile[queryIdx][dimOffset] * kTile[keyIdx][dimOffset];

      int64_t index = b * p.attn_stride_0 + h * p.attn_stride_1 +
          i * p.attn_stride_2 + j * p.attn_stride_3 + ki * KERNEL_SIZE + kj;
      if (p.bias) {
        int32_t pi =
            get_pb_start(i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        int32_t pj =
            get_pb_start(j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
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
    int32_t xsize = width * KERNEL_SIZE;
    int32_t ysize = height * KERNEL_SIZE;
    const dim3 grid(
        (xsize + XTHREADS * dilation - 1) / XTHREADS,
        (ysize + YTHREADS * dilation - 1) / YTHREADS,
        (batch_dim + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 block(XTHREADS, YTHREADS, BATCHTHREADS);
    return LaunchParams(grid, block);
  }
};

template <
    typename scalar_t,
    typename memscalar_t,
    int32_t TILEX,
    int32_t TILEY,
    int32_t KTILEX,
    int32_t KTILEY,
    int32_t XTHREADS,
    int32_t YTHREADS,
    int32_t BATCHTHREADS,
    int32_t KERNEL_SIZE,
    int32_t NEIGHBORHOOD_SIZE>
struct PointwiseNeighborhood2DHalf11x11_13x13
    : PointwiseNeighborhood2DBase<scalar_t> {
  using Base = PointwiseNeighborhood2DBase<scalar_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = IsBF16<scalar_t>::value;
  static constexpr bool IsHalfKernel = true;
  static constexpr bool UsesSmem = true;

  __device__ __host__ PointwiseNeighborhood2DHalf11x11_13x13() : Base() {}

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
    int32_t lti = threadIdx.y * (TILEY * KERNEL_SIZE) + threadIdx.x;
    int32_t stride2 = DIMHALF_32 * p.width;
    int64_t batchHeadOffset = b * p.query_stride_0 + h * p.query_stride_1;
    int32_t si = int32_t(blockIdx.y / dilation) * (TILEX * dilation) +
        (blockIdx.y % dilation);
    int32_t sj = int32_t(blockIdx.x / dilation) * (TILEY * dilation) +
        (blockIdx.x % dilation);
    int32_t sni = get_window_start(
        si, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    int32_t snj =
        get_window_start(sj, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    __shared__ HalfType tile[TILEX * TILEY][DIM_32 + 3];
    __shared__ HalfType kTile[KTILEX * KTILEY][DIM_32 + 3];
    auto query2 = HalfHelper::typecast(p.query);
    auto key2 = HalfHelper::typecast(p.key);

    /* query tile */
    int32_t qtx = lti / DIM_32;
    int32_t qtyp = lti - qtx * DIM_32;
    int32_t qdi = qtyp / KHALFITERS_32;
    int32_t qdj = qtyp - qdi * KHALFITERS_32;
    int32_t qty = qdi * KITERS_32 + qdj;
    if (qtx < TILEX * TILEY && qtyp < DIMHALF_32) {
      int32_t qi = qtx / TILEY;
      int32_t qj = (qtx - qi * TILEY) * dilation + sj;
      qi = qi * dilation + si;
      if (qi < p.height && qj < p.width)
        tile[qtx][qty] =
            query2[batchHeadOffset + qi * stride2 + qj * DIMHALF_32 + qtyp];
    }
    /* key tile */
    int32_t ktx = lti / KSTRIDE_32;
    int32_t kty = (lti - ktx * KSTRIDE_32) * KHALFITERS_32;
    if (ktx < KTILEX * KTILEY) {
      int32_t bi = ktx / KTILEY;
      int32_t bj = (ktx - bi * KTILEY) * dilation + snj;
      bi = bi * dilation + sni;
      if (bi < p.height && bj < p.width) {
        int64_t keyOffset =
            batchHeadOffset + bi * stride2 + bj * DIMHALF_32 + kty;
#pragma unroll
        for (int32_t ti = 0; ti < KHALFITERS_32; ++ti)
          kTile[ktx][kty * 2 + ti] = key2[keyOffset + ti];
      }
    }
    __syncthreads();
    int32_t ii = threadIdx.y / KERNEL_SIZE;
    int32_t ki = threadIdx.y - ii * KERNEL_SIZE;
    int32_t jj = threadIdx.x / KERNEL_SIZE;
    int32_t kj = threadIdx.x - jj * KERNEL_SIZE;
    int32_t i = si + ii * dilation, j = sj + jj * dilation;
    if (i < p.height && j < p.width) {
      int32_t ni = get_window_start(
          i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      int32_t nj = get_window_start(
          j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      auto updt = HalfHelper::zeros();
      int32_t queryIdx = ii * TILEY + jj;
      int32_t keyIdx = int32_t((ni + ki * dilation - sni) / dilation) * KTILEY +
          int32_t((nj + kj * dilation - snj) / dilation);

#pragma unroll
      for (int32_t di = 0; di < KSTRIDE_32; ++di)
#pragma unroll
        for (int32_t dj = 0; dj < KHALFITERS_32; ++dj)
          updt = HalfHelper::fma(
              tile[queryIdx][di * KITERS_32 + dj],
              kTile[keyIdx][di * KITERS_32 + dj],
              updt);

      int64_t index = b * p.attn_stride_0 + h * p.attn_stride_1 +
          i * p.attn_stride_2 + j * p.attn_stride_3 + ki * KERNEL_SIZE + kj;
      scalar_t acc = HalfHelper::cast_back(HalfHelper::add(updt.x, updt.y));
      if (p.bias) {
        int32_t pi =
            get_pb_start(i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        int32_t pj =
            get_pb_start(j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
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
    int32_t xsize = width * KERNEL_SIZE;
    int32_t ysize = height * KERNEL_SIZE;
    const dim3 grid(
        (xsize + XTHREADS * dilation - 1) / XTHREADS,
        (ysize + YTHREADS * dilation - 1) / YTHREADS,
        (batch_dim + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 block(XTHREADS, YTHREADS, BATCHTHREADS);
    return LaunchParams(grid, block);
  }
};

template <typename scalar_t>
struct PointwiseNeighborhood2DFull11x11
    : PointwiseNeighborhood2DFull11x11_13x13<
          scalar_t,
          scalar_t,
          TILE_11_X,
          TILE_11_Y,
          KTILE_11_X,
          KTILE_11_Y,
          XTHREADS_11,
          YTHREADS_11,
          BATCHTHREADS_11,
          11,
          5> {
  using Base = PointwiseNeighborhood2DFull11x11_13x13<
      scalar_t,
      scalar_t,
      TILE_11_X,
      TILE_11_Y,
      KTILE_11_X,
      KTILE_11_Y,
      XTHREADS_11,
      YTHREADS_11,
      BATCHTHREADS_11,
      11,
      5>;

  __device__ __host__ PointwiseNeighborhood2DFull11x11() : Base() {}
};

template <typename scalar_t>
struct PointwiseNeighborhood2DHalf11x11
    : PointwiseNeighborhood2DHalf11x11_13x13<
          scalar_t,
          scalar_t,
          TILE_11_X,
          TILE_11_Y,
          KTILE_11_X,
          KTILE_11_Y,
          XTHREADS_11,
          YTHREADS_11,
          BATCHTHREADS_11,
          11,
          5> {
  using Base = PointwiseNeighborhood2DHalf11x11_13x13<
      scalar_t,
      scalar_t,
      TILE_11_X,
      TILE_11_Y,
      KTILE_11_X,
      KTILE_11_Y,
      XTHREADS_11,
      YTHREADS_11,
      BATCHTHREADS_11,
      11,
      5>;

  __device__ __host__ PointwiseNeighborhood2DHalf11x11() : Base() {}
};

template <typename scalar_t>
struct PointwiseNeighborhood2DFull13x13
    : PointwiseNeighborhood2DFull11x11_13x13<
          scalar_t,
          float,
          TILE_13_X,
          TILE_13_Y,
          KTILE_13_X,
          KTILE_13_Y,
          XTHREADS_13,
          YTHREADS_13,
          BATCHTHREADS_13,
          13,
          6> {
  using Base = PointwiseNeighborhood2DFull11x11_13x13<
      scalar_t,
      float,
      TILE_13_X,
      TILE_13_Y,
      KTILE_13_X,
      KTILE_13_Y,
      XTHREADS_13,
      YTHREADS_13,
      BATCHTHREADS_13,
      13,
      6>;

  __device__ __host__ PointwiseNeighborhood2DFull13x13() : Base() {}
};

template <typename scalar_t>
struct PointwiseNeighborhood2DHalf13x13
    : PointwiseNeighborhood2DHalf11x11_13x13<
          scalar_t,
          float,
          TILE_13_X,
          TILE_13_Y,
          KTILE_13_X,
          KTILE_13_Y,
          XTHREADS_13,
          YTHREADS_13,
          BATCHTHREADS_13,
          13,
          6> {
  using Base = PointwiseNeighborhood2DHalf11x11_13x13<
      scalar_t,
      float,
      TILE_13_X,
      TILE_13_Y,
      KTILE_13_X,
      KTILE_13_Y,
      XTHREADS_13,
      YTHREADS_13,
      BATCHTHREADS_13,
      13,
      6>;

  __device__ __host__ PointwiseNeighborhood2DHalf13x13() : Base() {}
};

} // namespace naive
} // namespace cuda
} // namespace natten
