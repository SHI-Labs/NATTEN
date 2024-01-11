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
// TODO: remaining dependency to torch: getCurrentCUDAStream
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "natten/cuda/naive/natten_commons.cuh"
#include "natten/cuda/naive/natten_tiled_macros.cuh"
#include "natten/cuda/naive/tiled/base.cuh"

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
    int TILEX,
    int TILEY,
    int KTILEX,
    int KTILEY,
    int XTHREADS,
    int YTHREADS,
    int BATCHTHREADS,
    int KERNEL_SIZE,
    int NEIGHBORHOOD_SIZE,
    int DILATION>
struct PointwiseNeighborhood2DFull11x11_13x13
    : PointwiseNeighborhood2DBase<scalar_t> {
  using Base = PointwiseNeighborhood2DBase<scalar_t>;
  using Params = typename Base::Params;
  static constexpr bool IsBF16Kernel = false;
  static constexpr bool IsHalfKernel = false;
  static constexpr bool UsesSmem = true;

  __device__ __host__ PointwiseNeighborhood2DFull11x11_13x13() : Base() {}

  static __host__ int get_dim(int dim) {
    return 32;
  }

  __device__ void launch(Params p) {
    const int dilation = (DILATION > 0) ? DILATION : p.dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use
    // blockIdx since blockDim will be 1 and threadIdx will always be 0. const
    // int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / p.heads;
    const int h = z - b * p.heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILEY * KERNEL_SIZE) + threadIdx.x;
    const int64_t batchHeadOffset = b * p.query_stride_0 + h * p.query_stride_1;
    const int si = int(blockIdx.y / dilation) * (TILEX * dilation) +
        (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILEY * dilation) +
        (blockIdx.x % dilation);
    const int sni = get_window_start(
        si, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    const int snj =
        get_window_start(sj, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    __shared__ memscalar_t tile[TILEX * TILEY][DIM_32 + 3];
    __shared__ memscalar_t kTile[KTILEX * KTILEY][DIM_32 + 3];

    /* query tile */
    const int qtx = lti / DIM_32;
    const int qty = lti - qtx * DIM_32;
    if (qtx < TILEX * TILEY) {
      int qi = qtx / TILEY;
      const int qj = (qtx - qi * TILEY) * dilation + sj;
      qi = qi * dilation + si;
      if (qi < p.height && qj < p.width)
        tile[qtx][qty] = p.query
                             [batchHeadOffset + qi * p.query_stride_2 +
                              qj * p.query_stride_3 + qty];
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KITERS_32;
    if (ktx < KTILEX * KTILEY) {
      int bi = ktx / KTILEY;
      const int bj = (ktx - bi * KTILEY) * dilation + snj;
      bi = bi * dilation + sni;
      if (bi < p.height && bj < p.width) {
        const int64_t keyOffset = batchHeadOffset + bi * p.query_stride_2 +
            bj * p.query_stride_3 + kty;
#pragma unroll
        for (int ti = 0; ti < KITERS_32; ++ti)
          kTile[ktx][kty + ti] = p.key[keyOffset + ti];
      }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE;
    const int ki = threadIdx.y - ii * KERNEL_SIZE;
    const int jj = threadIdx.x / KERNEL_SIZE;
    const int kj = threadIdx.x - jj * KERNEL_SIZE;
    const int i = si + ii * dilation, j = sj + jj * dilation;
    if (i < p.height && j < p.width) {
      const int ni = get_window_start(
          i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int nj = get_window_start(
          j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      scalar_t updt = scalar_t(0);
      const int queryIdx = ii * TILEY + jj;
      const int keyIdx = int((ni + ki * dilation - sni) / dilation) * KTILEY +
          int((nj + kj * dilation - snj) / dilation);

#pragma unroll
      for (int dimOffset = 0; dimOffset < DIM_32; ++dimOffset)
        updt += tile[queryIdx][dimOffset] * kTile[keyIdx][dimOffset];

      const int64_t index = b * p.attn_stride_0 + h * p.attn_stride_1 +
          i * p.attn_stride_2 + j * p.attn_stride_3 + ki * KERNEL_SIZE + kj;
      if (p.bias) {
        const int pi =
            get_pb_start(i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pj =
            get_pb_start(j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int64_t biasIndex =
            h * p.bias_stride_0 + (pi + ki) * p.bias_stride_1 + (pj + kj);
        updt += p.bias[biasIndex];
      }
      p.attn[index] = updt;
    }
    //}
  }

  static LaunchParams get_launch_params(
      int batch_dim,
      int height,
      int width,
      int attention_span,
      int dilation) {
    int xsize = width * KERNEL_SIZE;
    int ysize = height * KERNEL_SIZE;
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
    int TILEX,
    int TILEY,
    int KTILEX,
    int KTILEY,
    int XTHREADS,
    int YTHREADS,
    int BATCHTHREADS,
    int KERNEL_SIZE,
    int NEIGHBORHOOD_SIZE,
    int DILATION>
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

  static __host__ int get_dim(int dim) {
    return 16;
  }

  __device__ void launch(Params p) {
    const int dilation = (DILATION > 0) ? DILATION : p.dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use
    // blockIdx since blockDim will be 1 and threadIdx will always be 0. const
    // int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / p.heads;
    const int h = z - b * p.heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILEY * KERNEL_SIZE) + threadIdx.x;
    const int stride2 = DIMHALF_32 * p.width;
    const int64_t batchHeadOffset = b * p.query_stride_0 + h * p.query_stride_1;
    const int si = int(blockIdx.y / dilation) * (TILEX * dilation) +
        (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILEY * dilation) +
        (blockIdx.x % dilation);
    const int sni = get_window_start(
        si, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    const int snj =
        get_window_start(sj, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    __shared__ HalfType tile[TILEX * TILEY][DIM_32 + 3];
    __shared__ HalfType kTile[KTILEX * KTILEY][DIM_32 + 3];
    auto query2 = HalfHelper::typecast(p.query);
    auto key2 = HalfHelper::typecast(p.key);

    /* query tile */
    const int qtx = lti / DIM_32;
    const int qtyp = lti - qtx * DIM_32;
    const int qdi = qtyp / KHALFITERS_32;
    const int qdj = qtyp - qdi * KHALFITERS_32;
    const int qty = qdi * KITERS_32 + qdj;
    if (qtx < TILEX * TILEY && qtyp < DIMHALF_32) {
      int qi = qtx / TILEY;
      const int qj = (qtx - qi * TILEY) * dilation + sj;
      qi = qi * dilation + si;
      if (qi < p.height && qj < p.width)
        tile[qtx][qty] =
            query2[batchHeadOffset + qi * stride2 + qj * DIMHALF_32 + qtyp];
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KHALFITERS_32;
    if (ktx < KTILEX * KTILEY) {
      int bi = ktx / KTILEY;
      const int bj = (ktx - bi * KTILEY) * dilation + snj;
      bi = bi * dilation + sni;
      if (bi < p.height && bj < p.width) {
        const int64_t keyOffset =
            batchHeadOffset + bi * stride2 + bj * DIMHALF_32 + kty;
#pragma unroll
        for (int ti = 0; ti < KHALFITERS_32; ++ti)
          kTile[ktx][kty * 2 + ti] = key2[keyOffset + ti];
      }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE;
    const int ki = threadIdx.y - ii * KERNEL_SIZE;
    const int jj = threadIdx.x / KERNEL_SIZE;
    const int kj = threadIdx.x - jj * KERNEL_SIZE;
    const int i = si + ii * dilation, j = sj + jj * dilation;
    if (i < p.height && j < p.width) {
      const int ni = get_window_start(
          i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      const int nj = get_window_start(
          j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
      auto updt = HalfHelper::zero();
      const int queryIdx = ii * TILEY + jj;
      const int keyIdx = int((ni + ki * dilation - sni) / dilation) * KTILEY +
          int((nj + kj * dilation - snj) / dilation);

#pragma unroll
      for (int di = 0; di < KSTRIDE_32; ++di)
#pragma unroll
        for (int dj = 0; dj < KHALFITERS_32; ++dj)
          updt = HalfHelper::fma(
              tile[queryIdx][di * KITERS_32 + dj],
              kTile[keyIdx][di * KITERS_32 + dj],
              updt);

      const int64_t index = b * p.attn_stride_0 + h * p.attn_stride_1 +
          i * p.attn_stride_2 + j * p.attn_stride_3 + ki * KERNEL_SIZE + kj;
      scalar_t acc = HalfHelper::cast_back(HalfHelper::add(updt.x, updt.y));
      if (p.bias) {
        const int pi =
            get_pb_start(i, p.height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pj =
            get_pb_start(j, p.width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int64_t biasIndex =
            h * p.bias_stride_0 + (pi + ki) * p.bias_stride_1 + (pj + kj);
        acc = HalfHelper::add(acc, p.bias[biasIndex]);
      }
      p.attn[index] = acc;
    }
    //}
  }

  static LaunchParams get_launch_params(
      int batch_dim,
      int height,
      int width,
      int attention_span,
      int dilation) {
    int xsize = width * KERNEL_SIZE;
    int ysize = height * KERNEL_SIZE;
    const dim3 grid(
        (xsize + XTHREADS * dilation - 1) / XTHREADS,
        (ysize + YTHREADS * dilation - 1) / YTHREADS,
        (batch_dim + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 block(XTHREADS, YTHREADS, BATCHTHREADS);
    return LaunchParams(grid, block);
  }
};

template <typename scalar_t, int DILATION>
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
          5,
          DILATION> {
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
      5,
      DILATION>;

  __device__ __host__ PointwiseNeighborhood2DFull11x11() : Base() {}
};

template <typename scalar_t, int DILATION>
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
          5,
          DILATION> {
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
      5,
      DILATION>;

  __device__ __host__ PointwiseNeighborhood2DHalf11x11() : Base() {}
};

template <typename scalar_t, int DILATION>
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
          6,
          DILATION> {
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
      6,
      DILATION>;

  __device__ __host__ PointwiseNeighborhood2DFull13x13() : Base() {}
};

template <typename scalar_t, int DILATION>
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
          6,
          DILATION> {
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
      6,
      DILATION>;

  __device__ __host__ PointwiseNeighborhood2DHalf13x13() : Base() {}
};

} // namespace naive
} // namespace cuda
} // namespace natten
