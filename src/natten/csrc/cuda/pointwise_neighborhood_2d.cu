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
    \brief Pointwise-Neighborhood kernel for 1D data.
           Computes attention weights between query points and their corresponding
           key neighborhood.
           Extra kernel with fused bias (relative positional bias.)
           + Tiled kernels for NA with window size 3, 5, 7, 9, 11, and 13 (only 32 dim per head 
           supported at the moment.)
*/

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <ATen/native/cuda/KernelUtils.cuh>
#include <ATen/AccumulateType.h>
#include <cuda_fp16.h>

#include "natten_commons.cuh"
#include "natten_tiled_macros.cuh"

namespace natten {

template<class scalar_t>
using Tensor3D = typename torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits>;
template<class scalar_t>
using Tensor5D = typename torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits>;

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_bias_fp16( // QK
    const Tensor5D<scalar_t> query,                  // query
    const Tensor5D<scalar_t> key,                    // key  
    const Tensor3D<scalar_t> bias,                   // relative positional bias tensor
    Tensor5D<scalar_t> attn,                         // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in,
    const int dimhalf) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < height * width){
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (y < KERNEL_SIZE * KERNEL_SIZE){
                __half2* query2 = reinterpret_cast<__half2*>(query.data());
                __half2* key2 = reinterpret_cast<__half2*>(key.data());
                const int b = z / heads;
                const int h = z - b * heads;
                const int ki = y / KERNEL_SIZE;
                const int kj = y - ki * KERNEL_SIZE;
                const int i = x / width;
                const int j = x - i * width;
                const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                __half2 updt = __float2half2_rn(0.f);
                const int stride2 = dimhalf * width;
                const int batchHeadOffset = b * (stride2*height*heads) + h * (stride2*height);
                const int queryOffset = batchHeadOffset + i * stride2 + j * dimhalf;
                const int keyOffset = batchHeadOffset + (ki*dilation+ni) * stride2 + (kj*dilation+nj) * dimhalf;
                #pragma unroll
                for (int dimOffset=0; dimOffset < dimhalf; ++dimOffset)
                    updt = __hfma2(query2[queryOffset+dimOffset], key2[keyOffset+dimOffset], updt);
                const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + y * attn.stride(4);
                const int biasIndex = h * bias.stride(0) + (pi+ki) * bias.stride(1) + (pj+kj) * bias.stride(2);
                attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y)) + bias.data()[biasIndex];
            }
        }
    }
}


template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_bias( // QK
    const Tensor5D<scalar_t> query,             // query
    const Tensor5D<scalar_t> key,               // key  
    const Tensor3D<scalar_t> bias,              // relative positional bias tensor
    Tensor5D<scalar_t> attn,                    // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in,
    const int dim) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < height * width){
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (y < KERNEL_SIZE * KERNEL_SIZE){
                const int b = z / heads;
                const int h = z - b * heads;
                const int ki = y / KERNEL_SIZE;
                const int kj = y - ki * KERNEL_SIZE;
                const int i = x / width;
                const int j = x - i * width;
                const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                scalar_t updt = scalar_t(0);
                const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
                const int queryOffset = batchHeadOffset + i * query.stride(2) + j * query.stride(3);
                const int keyOffset = batchHeadOffset + (ki*dilation+ni) * key.stride(2) + (kj*dilation+nj) * key.stride(3);
                #pragma unroll
                for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                    updt += query.data()[queryOffset+dimOffset] * key.data()[keyOffset+dimOffset];
                const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + y * attn.stride(4);
                const int biasIndex = h * bias.stride(0) + (pi+ki) * bias.stride(1) + (pj+kj) * bias.stride(2);
                updt += bias.data()[biasIndex];
                attn.data()[index] = updt;
            }
        }
    }
}


/* TODO: FIX BANK CONFLICTS */
template <int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_bias_fp16_3x3_32( // QK
    const Tensor5D<scalar_t> query,                         // query
    const Tensor5D<scalar_t> key,                           // key  
    const Tensor3D<scalar_t> bias,                          // relative positional bias tensor
    Tensor5D<scalar_t> attn,                                // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use blockIdx since blockDim will be 1 and threadIdx will
    // always be 0.
    // const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / heads;
    const int h = z - b * heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILE_3*KERNEL_SIZE_3) + threadIdx.x;
    const int stride2 = DIMHALF_32 * width;
    const int batchHeadOffset = b * (stride2*height*heads) + h * (stride2*height);
    const int si = int(blockIdx.y / dilation) * (TILE_3 * dilation) + (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILE_3 * dilation) + (blockIdx.x % dilation);
    const int sni = get_window_start(si, height, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
    const int snj = get_window_start(sj, width, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
    __shared__ __half2 tile[TILE_3*TILE_3][DIM_32+3];
    __shared__ __half2 kTile[KTILE_3*KTILE_3][DIM_32+3];
    __half2* query2 = reinterpret_cast<__half2*>(query.data());
    __half2* key2 = reinterpret_cast<__half2*>(key.data());

    /* query tile */
    const int qtx = lti / QSTRIDE_3_HALF;
    const int qty = (lti - qtx * QSTRIDE_3_HALF) * QITERS_3_HALF;
    if (qtx < TILE_3*TILE_3)
    {
        int qi = qtx / TILE_3;
        const int qj = (qtx - qi * TILE_3) * dilation + sj;
        qi = qi * dilation + si;
        if (qi < height && qj < width){
            #pragma unroll
            for (int ti=0; ti < QITERS_3_HALF; ++ti)
                tile[qtx][qty+ti] = query2[batchHeadOffset + qi * stride2 + qj * DIMHALF_32 + qty+ti];
        }
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KHALFITERS_32;
    if (ktx < KTILE_3*KTILE_3)
    {
        int bi = ktx / KTILE_3;
        const int bj = (ktx - bi * KTILE_3) * dilation + snj;
        bi = bi * dilation + sni;
        if (bi < height && bj < width){
            const int keyOffset = batchHeadOffset + bi * stride2 + bj * DIMHALF_32 + kty;
            #pragma unroll
            for (int ti=0; ti < KHALFITERS_32; ++ti)
                kTile[ktx][kty + ti] = key2[keyOffset + ti];
        }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE_3;
    const int ki = threadIdx.y - ii * KERNEL_SIZE_3;
    const int jj = threadIdx.x / KERNEL_SIZE_3;
    const int kj = threadIdx.x - jj * KERNEL_SIZE_3;
    const int i = si + ii*dilation, j = sj + jj*dilation;
    if (i < height && j < width){
        const int ni = get_window_start(i, height, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
        const int pi = get_pb_start(i, height, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
        const int pj = get_pb_start(j, width, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
        __half2 updt = __float2half2_rn(0.f);
        const int queryIdx = ii*TILE_3 + jj;
        const int keyIdx = int((ni+ki*dilation - sni)/dilation)*KTILE_3 + int((nj+kj*dilation - snj)/dilation);

        #pragma unroll
        for (int dimOffset=0; dimOffset < DIMHALF_32; ++dimOffset)
            updt = __hfma2(tile[queryIdx][dimOffset], kTile[keyIdx][dimOffset], updt);
        const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE_3+kj;
        const int biasIndex = h * bias.stride(0) + (pi+ki) * bias.stride(1) + (pj+kj) * bias.stride(2);
        attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y)) + bias.data()[biasIndex];
    }
    //}
}

/* TODO: CHECK BANK CONFLICTS */
template <int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_bias_3x3_32( // QK
    const Tensor5D<scalar_t> query,                    // query
    const Tensor5D<scalar_t> key,                      // key  
    const Tensor3D<scalar_t> bias,                     // relative positional bias tensor
    Tensor5D<scalar_t> attn,                           // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use blockIdx since blockDim will be 1 and threadIdx will
    // always be 0.
    // const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / heads;
    const int h = z - b * heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILE_3*KERNEL_SIZE_3) + threadIdx.x;
    const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
    const int si = int(blockIdx.y / dilation) * (TILE_3 * dilation) + (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILE_3 * dilation) + (blockIdx.x % dilation);
    const int sni = get_window_start(si, height, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
    const int snj = get_window_start(sj, width, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
    __shared__ scalar_t tile[TILE_3*TILE_3][DIM_32+3];
    __shared__ scalar_t kTile[KTILE_3*KTILE_3][DIM_32+3];

    /* query tile */
    const int qtx = lti / QSTRIDE_3;
    const int qty = (lti - qtx * QSTRIDE_3) * QITERS_3;
    if (qtx < TILE_3*TILE_3)
    {
        int qi = qtx / TILE_3;
        const int qj = (qtx - qi * TILE_3) * dilation + sj;
        qi = qi * dilation + si;
        if (qi < height && qj < width){
            #pragma unroll
            for (int ti=0; ti < QITERS_3; ++ti)
                tile[qtx][qty+ti] = query.data()[batchHeadOffset + qi * query.stride(2) + qj * query.stride(3) + qty+ti];
        }
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KITERS_32;
    if (ktx < KTILE_3*KTILE_3)
    {
        int bi = ktx / KTILE_3;
        const int bj = (ktx - bi * KTILE_3) * dilation + snj;
        bi = bi * dilation + sni;
        if (bi < height && bj < width){
            const int keyOffset = batchHeadOffset + bi * query.stride(2) + bj * query.stride(3) + kty;
            #pragma unroll
            for (int ti=0; ti < KITERS_32; ++ti)
                kTile[ktx][kty + ti] = key.data()[keyOffset + ti];
        }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE_3;
    const int ki = threadIdx.y - ii * KERNEL_SIZE_3;
    const int jj = threadIdx.x / KERNEL_SIZE_3;
    const int kj = threadIdx.x - jj * KERNEL_SIZE_3;
    const int i = si + ii*dilation, j = sj + jj*dilation;
    if (i < height && j < width){
        const int ni = get_window_start(i, height, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
        const int pi = get_pb_start(i, height, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
        const int pj = get_pb_start(j, width, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
        scalar_t updt = scalar_t(0);
        const int queryIdx = ii*TILE_3 + jj;
        const int keyIdx = int((ni+ki*dilation - sni)/dilation)*KTILE_3 + int((nj+kj*dilation - snj)/dilation);

        #pragma unroll
        for (int dimOffset=0; dimOffset < DIM_32; ++dimOffset)
            updt += tile[queryIdx][dimOffset] * kTile[keyIdx][dimOffset];

        const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE_3+kj;
        const int biasIndex = h * bias.stride(0) + (pi+ki) * bias.stride(1) + (pj+kj) * bias.stride(2);
        updt += bias.data()[biasIndex];
        attn.data()[index] = updt;
    }
    //}
}


/* TODO: FIX BANK CONFLICTS */
template <int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_bias_fp16_5x5_32( // QK
    const Tensor5D<scalar_t> query,                         // query
    const Tensor5D<scalar_t> key,                           // key  
    const Tensor3D<scalar_t> bias,                          // relative positional bias tensor
    Tensor5D<scalar_t> attn,                                // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use blockIdx since blockDim will be 1 and threadIdx will
    // always be 0.
    // const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / heads;
    const int h = z - b * heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILE_5*KERNEL_SIZE_5) + threadIdx.x;
    const int stride2 = DIMHALF_32 * width;
    const int batchHeadOffset = b * (stride2*height*heads) + h * (stride2*height);
    const int si = int(blockIdx.y / dilation) * (TILE_5 * dilation) + (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILE_5 * dilation) + (blockIdx.x % dilation);
    const int sni = get_window_start(si, height, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
    const int snj = get_window_start(sj, width, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
    __shared__ __half2 tile[TILE_5*TILE_5][DIM_32+3];
    __shared__ __half2 kTile[KTILE_5*KTILE_5][DIM_32+3];
    __half2* query2 = reinterpret_cast<__half2*>(query.data());
    __half2* key2 = reinterpret_cast<__half2*>(key.data());

    /* query tile */
    const int qtx = lti / DIMHALF_32;
    const int qty = lti - qtx * DIMHALF_32;
    if (qtx < TILE_5*TILE_5)
    {
        int qi = qtx / TILE_5;
        const int qj = (qtx - qi * TILE_5) * dilation + sj;
        qi = qi * dilation + si;
        if (qi < height && qj < width){
            tile[qtx][qty] = query2[batchHeadOffset + qi * stride2 + qj * DIMHALF_32 + qty];
        }
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KHALFITERS_32;
    if (ktx < KTILE_5*KTILE_5)
    {
        int bi = ktx / KTILE_5;
        const int bj = (ktx - bi * KTILE_5) * dilation + snj;
        bi = bi * dilation + sni;
        if (bi < height && bj < width){
            const int keyOffset = batchHeadOffset + bi * stride2 + bj * DIMHALF_32 + kty;
            #pragma unroll
            for (int ti=0; ti < KHALFITERS_32; ++ti)
                kTile[ktx][kty + ti] = key2[keyOffset + ti];
        }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE_5;
    const int ki = threadIdx.y - ii * KERNEL_SIZE_5;
    const int jj = threadIdx.x / KERNEL_SIZE_5;
    const int kj = threadIdx.x - jj * KERNEL_SIZE_5;
    const int i = si + ii*dilation, j = sj + jj*dilation;
    if (i < height && j < width){
        const int ni = get_window_start(i, height, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
        const int pi = get_pb_start(i, height, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
        const int pj = get_pb_start(j, width, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
        __half2 updt = __float2half2_rn(0.f);
        const int queryIdx = ii*TILE_5 + jj;
        const int keyIdx = int((ni+ki*dilation - sni)/dilation)*KTILE_5 + int((nj+kj*dilation - snj)/dilation);

        #pragma unroll
        for (int dimOffset=0; dimOffset < DIMHALF_32; ++dimOffset)
            updt = __hfma2(tile[queryIdx][dimOffset], kTile[keyIdx][dimOffset], updt);
        const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE_5+kj;
        const int biasIndex = h * bias.stride(0) + (pi+ki) * bias.stride(1) + (pj+kj) * bias.stride(2);
        attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y)) + bias.data()[biasIndex];
    }
    //}
}

/* TODO: CHECK BANK CONFLICTS */
template <int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_bias_5x5_32( // QK
    const Tensor5D<scalar_t> query,                    // query
    const Tensor5D<scalar_t> key,                      // key  
    const Tensor3D<scalar_t> bias,                     // relative positional bias tensor
    Tensor5D<scalar_t> attn,                           // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use blockIdx since blockDim will be 1 and threadIdx will
    // always be 0.
    // const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / heads;
    const int h = z - b * heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILE_5*KERNEL_SIZE_5) + threadIdx.x;
    const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
    const int si = int(blockIdx.y / dilation) * (TILE_5 * dilation) + (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILE_5 * dilation) + (blockIdx.x % dilation);
    const int sni = get_window_start(si, height, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
    const int snj = get_window_start(sj, width, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
    __shared__ scalar_t tile[TILE_5*TILE_5][DIM_32+3];
    __shared__ scalar_t kTile[KTILE_5*KTILE_5][DIM_32+3];

    /* query tile */
    const int qtx = lti / QSTRIDE_5;
    const int qty = (lti - qtx * QSTRIDE_5) * QITERS_5;
    if (qtx < TILE_5*TILE_5)
    {
        int qi = qtx / TILE_5;
        const int qj = (qtx - qi * TILE_5) * dilation + sj;
        qi = qi * dilation + si;
        if (qi < height && qj < width){
            #pragma unroll
            for (int ti=0; ti < QITERS_5; ++ti)
                tile[qtx][qty+ti] = query.data()[batchHeadOffset + qi * query.stride(2) + qj * query.stride(3) + qty+ti];
        }
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KITERS_32;
    if (ktx < KTILE_5*KTILE_5)
    {
        int bi = ktx / KTILE_5;
        const int bj = (ktx - bi * KTILE_5) * dilation + snj;
        bi = bi * dilation + sni;
        if (bi < height && bj < width){
            const int keyOffset = batchHeadOffset + bi * query.stride(2) + bj * query.stride(3) + kty;
            #pragma unroll
            for (int ti=0; ti < KITERS_32; ++ti)
                kTile[ktx][kty + ti] = key.data()[keyOffset + ti];
        }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE_5;
    const int ki = threadIdx.y - ii * KERNEL_SIZE_5;
    const int jj = threadIdx.x / KERNEL_SIZE_5;
    const int kj = threadIdx.x - jj * KERNEL_SIZE_5;
    const int i = si + ii*dilation, j = sj + jj*dilation;
    if (i < height && j < width){
        const int ni = get_window_start(i, height, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
        const int pi = get_pb_start(i, height, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
        const int pj = get_pb_start(j, width, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
        scalar_t updt = scalar_t(0);
        const int queryIdx = ii*TILE_5 + jj;
        const int keyIdx = int((ni+ki*dilation - sni)/dilation)*KTILE_5 + int((nj+kj*dilation - snj)/dilation);

        #pragma unroll
        for (int dimOffset=0; dimOffset < DIM_32; ++dimOffset)
            updt += tile[queryIdx][dimOffset] * kTile[keyIdx][dimOffset];

        const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE_5+kj;
        const int biasIndex = h * bias.stride(0) + (pi+ki) * bias.stride(1) + (pj+kj) * bias.stride(2);
        updt += bias.data()[biasIndex];
        attn.data()[index] = updt;
    }
    //}
}


template <int TILE, int KTILE, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_bias_fp16_7x7_9x9_32( // QK
    const Tensor5D<scalar_t> query,                             // query
    const Tensor5D<scalar_t> key,                               // key  
    const Tensor3D<scalar_t> bias,                              // relative positional bias tensor
    Tensor5D<scalar_t> attn,                                    // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use blockIdx since blockDim will be 1 and threadIdx will
    // always be 0.
    // const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / heads;
    const int h = z - b * heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILE*KERNEL_SIZE) + threadIdx.x;
    const int stride2 = DIMHALF_32 * width;
    const int batchHeadOffset = b * (stride2*height*heads) + h * (stride2*height);
    const int si = int(blockIdx.y / dilation) * (TILE * dilation) + (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILE * dilation) + (blockIdx.x % dilation);
    const int sni = get_window_start(si, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    const int snj = get_window_start(sj, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    __shared__ __half2 tile[TILE*TILE][DIM_32+3];
    __shared__ __half2 kTile[KTILE*KTILE][DIM_32+3];
    __half2* query2 = reinterpret_cast<__half2*>(query.data());
    __half2* key2 = reinterpret_cast<__half2*>(key.data());

    /* query tile */
    const int qtx = lti / DIM_32;
    const int qtyp = lti - qtx * DIM_32;
    const int qdi = qtyp / KHALFITERS_32;
    const int qdj = qtyp - qdi * KHALFITERS_32;
    const int qty = qdi*KITERS_32+qdj;
    if (qtx < TILE*TILE && qtyp < DIMHALF_32)
    {
        int qi = qtx / TILE;
        const int qj = (qtx - qi * TILE) * dilation + sj;
        qi = qi * dilation + si;
        if (qi < height && qj < width)
            tile[qtx][qty] = query2[batchHeadOffset + qi * stride2 + qj * DIMHALF_32 + qtyp];
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KHALFITERS_32;
    if (ktx < KTILE*KTILE)
    {
        int bi = ktx / KTILE;
        const int bj = (ktx - bi * KTILE) * dilation + snj;
        bi = bi * dilation + sni;
        if (bi < height && bj < width){
            const int keyOffset = batchHeadOffset + bi * stride2 + bj * DIMHALF_32 + kty;
            #pragma unroll
            for (int ti=0; ti < KHALFITERS_32; ++ti)
                kTile[ktx][kty*2 + ti] = key2[keyOffset + ti];
        }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE;
    const int ki = threadIdx.y - ii * KERNEL_SIZE;
    const int jj = threadIdx.x / KERNEL_SIZE;
    const int kj = threadIdx.x - jj * KERNEL_SIZE;
    const int i = si + ii*dilation, j = sj + jj*dilation;
    if (i < height && j < width){
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        __half2 updt = __float2half2_rn(0.f);
        const int queryIdx = ii*TILE + jj;
        const int keyIdx = int((ni+ki*dilation - sni)/dilation)*KTILE + int((nj+kj*dilation - snj)/dilation);

        #pragma unroll
        for (int di=0; di < KSTRIDE_32; ++di)
            #pragma unroll
            for (int dj=0; dj <KHALFITERS_32; ++dj)
                updt = __hfma2(tile[queryIdx][di*KITERS_32+dj], kTile[keyIdx][di*KITERS_32+dj], updt);
        const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE+kj;
        const int biasIndex = h * bias.stride(0) + (pi+ki) * bias.stride(1) + (pj+kj) * bias.stride(2);
        attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y)) + bias.data()[biasIndex];
    }
    //}
}

template <int TILE, int KTILE, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_bias_7x7_9x9_32( // QK
    const Tensor5D<scalar_t> query,                        // query
    const Tensor5D<scalar_t> key,                          // key  
    const Tensor3D<scalar_t> bias,                         // relative positional bias tensor
    Tensor5D<scalar_t> attn,                               // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use blockIdx since blockDim will be 1 and threadIdx will
    // always be 0.
    // const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / heads;
    const int h = z - b * heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILE*KERNEL_SIZE) + threadIdx.x;
    const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
    const int si = int(blockIdx.y / dilation) * (TILE * dilation) + (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILE * dilation) + (blockIdx.x % dilation);
    const int sni = get_window_start(si, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    const int snj = get_window_start(sj, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    __shared__ scalar_t tile[TILE*TILE][DIM_32+3];
    __shared__ scalar_t kTile[KTILE*KTILE][DIM_32+3];

    /* query tile */
    const int qtx = lti / DIM_32;
    const int qty = lti - qtx * DIM_32;
    if (qtx < TILE*TILE)
    {
        int qi = qtx / TILE;
        const int qj = (qtx - qi * TILE) * dilation + sj;
        qi = qi * dilation + si;
        if (qi < height && qj < width)
            tile[qtx][qty] = query.data()[batchHeadOffset + qi * query.stride(2) + qj * query.stride(3) + qty];
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KITERS_32;
    if (ktx < KTILE*KTILE)
    {
        int bi = ktx / KTILE;
        const int bj = (ktx - bi * KTILE) * dilation + snj;
        bi = bi * dilation + sni;
        if (bi < height && bj < width){
            const int keyOffset = batchHeadOffset + bi * query.stride(2) + bj * query.stride(3) + kty;
            #pragma unroll
            for (int ti=0; ti < KITERS_32; ++ti)
                kTile[ktx][kty + ti] = key.data()[keyOffset + ti];
        }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE;
    const int ki = threadIdx.y - ii * KERNEL_SIZE;
    const int jj = threadIdx.x / KERNEL_SIZE;
    const int kj = threadIdx.x - jj * KERNEL_SIZE;
    const int i = si + ii*dilation, j = sj + jj*dilation;
    if (i < height && j < width){
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        scalar_t updt = scalar_t(0);
        const int queryIdx = ii*TILE + jj;
        const int keyIdx = int((ni+ki*dilation - sni)/dilation)*KTILE + int((nj+kj*dilation - snj)/dilation);

        #pragma unroll
        for (int dimOffset=0; dimOffset < DIM_32; ++dimOffset)
            updt += tile[queryIdx][dimOffset] * kTile[keyIdx][dimOffset];

        const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE+kj;
        const int biasIndex = h * bias.stride(0) + (pi+ki) * bias.stride(1) + (pj+kj) * bias.stride(2);
        updt += bias.data()[biasIndex];
        attn.data()[index] = updt;
    }
    //}
}


template <int TILEX, int TILEY, int KTILEX, int KTILEY, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t, typename memscalar_t>
__global__ void pointwise_neighborhood_2d_bias_fp16_11x11_13x13_32( // QK
    const Tensor5D<scalar_t> query,                                 // query
    const Tensor5D<scalar_t> key,                                   // key  
    const Tensor3D<scalar_t> bias,                                  // relative positional bias tensor
    Tensor5D<scalar_t> attn,                                        // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use blockIdx since blockDim will be 1 and threadIdx will
    // always be 0.
    // const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / heads;
    const int h = z - b * heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILEY*KERNEL_SIZE) + threadIdx.x;
    const int stride2 = DIMHALF_32 * width;
    const int batchHeadOffset = b * (stride2*height*heads) + h * (stride2*height);
    const int si = int(blockIdx.y / dilation) * (TILEX * dilation) + (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILEY * dilation) + (blockIdx.x % dilation);
    const int sni = get_window_start(si, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    const int snj = get_window_start(sj, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    __shared__ __half2 tile[TILEX*TILEY][DIM_32+3];
    __shared__ __half2 kTile[KTILEX*KTILEY][DIM_32+3];
    __half2* query2 = reinterpret_cast<__half2*>(query.data());
    __half2* key2 = reinterpret_cast<__half2*>(key.data());

    /* query tile */
    const int qtx = lti / DIM_32;
    const int qtyp = lti - qtx * DIM_32;
    const int qdi = qtyp / KHALFITERS_32;
    const int qdj = qtyp - qdi * KHALFITERS_32;
    const int qty = qdi*KITERS_32+qdj;
    if (qtx < TILEX*TILEY && qtyp < DIMHALF_32)
    {
        int qi = qtx / TILEY;
        const int qj = (qtx - qi * TILEY) * dilation + sj;
        qi =  qi * dilation + si;
        if (qi < height && qj < width)
            tile[qtx][qty] = query2[batchHeadOffset + qi * stride2 + qj * DIMHALF_32 + qtyp];
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KHALFITERS_32;
    if (ktx < KTILEX*KTILEY)
    {
        int bi = ktx / KTILEY;
        const int bj = (ktx - bi * KTILEY) * dilation + snj;
        bi = bi * dilation + sni;
        if (bi < height && bj < width){
            const int keyOffset = batchHeadOffset + bi * stride2 + bj * DIMHALF_32 + kty;
            #pragma unroll
            for (int ti=0; ti < KHALFITERS_32; ++ti)
                kTile[ktx][kty*2 + ti] = key2[keyOffset + ti];
        }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE;
    const int ki = threadIdx.y - ii * KERNEL_SIZE;
    const int jj = threadIdx.x / KERNEL_SIZE;
    const int kj = threadIdx.x - jj * KERNEL_SIZE;
    const int i = si + ii*dilation, j = sj + jj*dilation;
    if (i < height && j < width){
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        __half2 updt = __float2half2_rn(0.f);
        const int queryIdx = ii*TILEY + jj;
        const int keyIdx = int((ni+ki*dilation - sni)/dilation)*KTILEY + int((nj+kj*dilation - snj)/dilation);

        #pragma unroll
        for (int di=0; di < KSTRIDE_32; ++di)
            #pragma unroll
            for (int dj=0; dj <KHALFITERS_32; ++dj)
                updt = __hfma2(tile[queryIdx][di*KITERS_32+dj], kTile[keyIdx][di*KITERS_32+dj], updt);
        const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE+kj;
        const int biasIndex = h * bias.stride(0) + (pi+ki) * bias.stride(1) + (pj+kj) * bias.stride(2);
        attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y)) + bias.data()[biasIndex];
    }
    //}
}

template <int TILEX, int TILEY, int KTILEX, int KTILEY, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t, typename memscalar_t>
__global__ void pointwise_neighborhood_2d_bias_11x11_13x13_32( // QK
    const Tensor5D<scalar_t> query,                            // query
    const Tensor5D<scalar_t> key,                              // key  
    const Tensor3D<scalar_t> bias,                             // relative positional bias tensor
    Tensor5D<scalar_t> attn,                                   // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use blockIdx since blockDim will be 1 and threadIdx will
    // always be 0.
    // const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / heads;
    const int h = z - b * heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILEY*KERNEL_SIZE) + threadIdx.x;
    const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
    const int si = int(blockIdx.y / dilation) * (TILEX * dilation) + (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILEY * dilation) + (blockIdx.x % dilation);
    const int sni = get_window_start(si, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    const int snj = get_window_start(sj, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    __shared__ memscalar_t tile[TILEX*TILEY][DIM_32+3];
    __shared__ memscalar_t kTile[KTILEX*KTILEY][DIM_32+3];

    /* query tile */
    const int qtx = lti / DIM_32;
    const int qty = lti - qtx * DIM_32;
    if (qtx < TILEX*TILEY)
    {
        int qi = qtx / TILEY;
        const int qj = (qtx - qi * TILEY) * dilation + sj;
        qi =  qi * dilation + si;
        if (qi < height && qj < width)
            tile[qtx][qty] = query.data()[batchHeadOffset + qi * query.stride(2) + qj * query.stride(3) + qty];
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KITERS_32;
    if (ktx < KTILEX*KTILEY)
    {
        int bi = ktx / KTILEY;
        const int bj = (ktx - bi * KTILEY) * dilation + snj;
        bi = bi * dilation + sni;
        if (bi < height && bj < width){
            const int keyOffset = batchHeadOffset + bi * query.stride(2) + bj * query.stride(3) + kty;
            #pragma unroll
            for (int ti=0; ti < KITERS_32; ++ti)
                kTile[ktx][kty + ti] = key.data()[keyOffset + ti];
        }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE;
    const int ki = threadIdx.y - ii * KERNEL_SIZE;
    const int jj = threadIdx.x / KERNEL_SIZE;
    const int kj = threadIdx.x - jj * KERNEL_SIZE;
    const int i = si + ii*dilation, j = sj + jj*dilation;
    if (i < height && j < width){
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        scalar_t updt = scalar_t(0);
        const int queryIdx = ii*TILEY + jj;
        const int keyIdx = int((ni+ki*dilation - sni)/dilation)*KTILEY + int((nj+kj*dilation - snj)/dilation);

        #pragma unroll
        for (int dimOffset=0; dimOffset < DIM_32; ++dimOffset)
            updt += tile[queryIdx][dimOffset] * kTile[keyIdx][dimOffset];

        const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE+kj;
        const int biasIndex = h * bias.stride(0) + (pi+ki) * bias.stride(1) + (pj+kj) * bias.stride(2);
        updt += bias.data()[biasIndex];
        attn.data()[index] = updt;
    }
    //}
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_fp16( // QK    / A-grad
    const Tensor5D<scalar_t> query,             // query / d_out
    const Tensor5D<scalar_t> key,               // key   / value
    Tensor5D<scalar_t> attn,                    // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in,
    const int dimhalf) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < height * width){
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (y < KERNEL_SIZE * KERNEL_SIZE){
                __half2* query2 = reinterpret_cast<__half2*>(query.data());
                __half2* key2 = reinterpret_cast<__half2*>(key.data());
                const int b = z / heads;
                const int h = z - b * heads;
                const int ki = y / KERNEL_SIZE;
                const int kj = y - ki * KERNEL_SIZE;
                const int i = x / width;
                const int j = x - i * width;
                const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                __half2 updt = __float2half2_rn(0.f);
                const int stride2 = dimhalf * width;
                const int batchHeadOffset = b * (stride2*height*heads) + h * (stride2*height);
                const int queryOffset = batchHeadOffset + i * stride2 + j * dimhalf;
                const int keyOffset = batchHeadOffset + (ki*dilation+ni) * stride2 + (kj*dilation+nj) * dimhalf;
                #pragma unroll
                for (int dimOffset=0; dimOffset < dimhalf; ++dimOffset)
                    updt = __hfma2(query2[queryOffset+dimOffset], key2[keyOffset+dimOffset], updt);
                const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + y * attn.stride(4);
                attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y));
            }
        }
    }
}


template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d( // QK    / A-grad
    const Tensor5D<scalar_t> query,        // query / d_out
    const Tensor5D<scalar_t> key,          // key   / value
    Tensor5D<scalar_t> attn,               // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in,
    const int dim) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < height * width){
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (y < KERNEL_SIZE * KERNEL_SIZE){
                const int b = z / heads;
                const int h = z - b * heads;
                const int ki = y / KERNEL_SIZE;
                const int kj = y - ki * KERNEL_SIZE;
                const int i = x / width;
                const int j = x - i * width;
                const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                scalar_t updt = scalar_t(0);
                const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
                const int queryOffset = batchHeadOffset + i * query.stride(2) + j * query.stride(3);
                const int keyOffset = batchHeadOffset + (ki*dilation+ni) * key.stride(2) + (kj*dilation+nj) * key.stride(3);
                #pragma unroll
                for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                    updt += query.data()[queryOffset+dimOffset] * key.data()[keyOffset+dimOffset];
                const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + y * attn.stride(4);
                attn.data()[index] = updt;
            }
        }
    }
}

/* TODO: FIX BANK CONFLICTS */
template <int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_fp16_3x3_32( // QK    / A-grad
    const Tensor5D<scalar_t> query,                    // query / d_out
    const Tensor5D<scalar_t> key,                      // key   / value
    Tensor5D<scalar_t> attn,                           // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use blockIdx since blockDim will be 1 and threadIdx will
    // always be 0.
    // const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / heads;
    const int h = z - b * heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILE_3*KERNEL_SIZE_3) + threadIdx.x;
    const int stride2 = DIMHALF_32 * width;
    const int batchHeadOffset = b * (stride2*height*heads) + h * (stride2*height);
    const int si = int(blockIdx.y / dilation) * (TILE_3 * dilation) + (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILE_3 * dilation) + (blockIdx.x % dilation);
    const int sni = get_window_start(si, height, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
    const int snj = get_window_start(sj, width, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
    __shared__ __half2 tile[TILE_3*TILE_3][DIM_32+3];
    __shared__ __half2 kTile[KTILE_3*KTILE_3][DIM_32+3];
    __half2* query2 = reinterpret_cast<__half2*>(query.data());
    __half2* key2 = reinterpret_cast<__half2*>(key.data());

    /* query tile */
    const int qtx = lti / QSTRIDE_3_HALF;
    const int qty = (lti - qtx * QSTRIDE_3_HALF) * QITERS_3_HALF;
    if (qtx < TILE_3*TILE_3)
    {
        int qi = qtx / TILE_3;
        const int qj = (qtx - qi * TILE_3) * dilation + sj;
        qi = qi * dilation + si;
        if (qi < height && qj < width){
            #pragma unroll
            for (int ti=0; ti < QITERS_3_HALF; ++ti)
                tile[qtx][qty+ti] = query2[batchHeadOffset + qi * stride2 + qj * DIMHALF_32 + qty+ti];
        }
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KHALFITERS_32;
    if (ktx < KTILE_3*KTILE_3)
    {
        int bi = ktx / KTILE_3;
        const int bj = (ktx - bi * KTILE_3) * dilation + snj;
        bi = bi * dilation + sni;
        if (bi < height && bj < width){
            const int keyOffset = batchHeadOffset + bi * stride2 + bj * DIMHALF_32 + kty;
            #pragma unroll
            for (int ti=0; ti < KHALFITERS_32; ++ti)
                kTile[ktx][kty + ti] = key2[keyOffset + ti];
        }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE_3;
    const int ki = threadIdx.y - ii * KERNEL_SIZE_3;
    const int jj = threadIdx.x / KERNEL_SIZE_3;
    const int kj = threadIdx.x - jj * KERNEL_SIZE_3;
    const int i = si + ii*dilation, j = sj + jj*dilation;
    if (i < height && j < width){
        const int ni = get_window_start(i, height, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
        __half2 updt = __float2half2_rn(0.f);
        const int queryIdx = ii*TILE_3 + jj;
        const int keyIdx = int((ni+ki*dilation - sni)/dilation)*KTILE_3 + int((nj+kj*dilation - snj)/dilation);

        #pragma unroll
        for (int dimOffset=0; dimOffset < DIMHALF_32; ++dimOffset)
            updt = __hfma2(tile[queryIdx][dimOffset], kTile[keyIdx][dimOffset], updt);
        const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE_3+kj;
        attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y));
    }
    //}
}

/* TODO: CHECK BANK CONFLICTS */
template <int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_3x3_32( // QK    / A-grad
    const Tensor5D<scalar_t> query,               // query / d_out
    const Tensor5D<scalar_t> key,                 // key   / value
    Tensor5D<scalar_t> attn,                      // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use blockIdx since blockDim will be 1 and threadIdx will
    // always be 0.
    // const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / heads;
    const int h = z - b * heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILE_3*KERNEL_SIZE_3) + threadIdx.x;
    const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
    const int si = int(blockIdx.y / dilation) * (TILE_3 * dilation) + (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILE_3 * dilation) + (blockIdx.x % dilation);
    const int sni = get_window_start(si, height, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
    const int snj = get_window_start(sj, width, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
    __shared__ scalar_t tile[TILE_3*TILE_3][DIM_32+3];
    __shared__ scalar_t kTile[KTILE_3*KTILE_3][DIM_32+3];

    /* query tile */
    const int qtx = lti / QSTRIDE_3;
    const int qty = (lti - qtx * QSTRIDE_3) * QITERS_3;
    if (qtx < TILE_3*TILE_3)
    {
        int qi = qtx / TILE_3;
        const int qj = (qtx - qi * TILE_3) * dilation + sj;
        qi = qi * dilation + si;
        if (qi < height && qj < width){
            #pragma unroll
            for (int ti=0; ti < QITERS_3; ++ti)
                tile[qtx][qty+ti] = query.data()[batchHeadOffset + qi * query.stride(2) + qj * query.stride(3) + qty+ti];
        }
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KITERS_32;
    if (ktx < KTILE_3*KTILE_3)
    {
        int bi = ktx / KTILE_3;
        const int bj = (ktx - bi * KTILE_3) * dilation + snj;
        bi = bi * dilation + sni;
        if (bi < height && bj < width){
            const int keyOffset = batchHeadOffset + bi * query.stride(2) + bj * query.stride(3) + kty;
            #pragma unroll
            for (int ti=0; ti < KITERS_32; ++ti)
                kTile[ktx][kty + ti] = key.data()[keyOffset + ti];
        }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE_3;
    const int ki = threadIdx.y - ii * KERNEL_SIZE_3;
    const int jj = threadIdx.x / KERNEL_SIZE_3;
    const int kj = threadIdx.x - jj * KERNEL_SIZE_3;
    const int i = si + ii*dilation, j = sj + jj*dilation;
    if (i < height && j < width){
        const int ni = get_window_start(i, height, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation);
        scalar_t updt = scalar_t(0);
        const int queryIdx = ii*TILE_3 + jj;
        const int keyIdx = int((ni+ki*dilation - sni)/dilation)*KTILE_3 + int((nj+kj*dilation - snj)/dilation);

        #pragma unroll
        for (int dimOffset=0; dimOffset < DIM_32; ++dimOffset)
            updt += tile[queryIdx][dimOffset] * kTile[keyIdx][dimOffset];

        const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE_3+kj;
        attn.data()[index] = updt;
    }
    //}
}

/* TODO: FIX BANK CONFLICTS */
template <int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_fp16_5x5_32( // QK    / A-grad
    const Tensor5D<scalar_t> query,                    // query / d_out
    const Tensor5D<scalar_t> key,                      // key   / value
    Tensor5D<scalar_t> attn,                           // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use blockIdx since blockDim will be 1 and threadIdx will
    // always be 0.
    // const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / heads;
    const int h = z - b * heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILE_5*KERNEL_SIZE_5) + threadIdx.x;
    const int stride2 = DIMHALF_32 * width;
    const int batchHeadOffset = b * (stride2*height*heads) + h * (stride2*height);
    const int si = int(blockIdx.y / dilation) * (TILE_5 * dilation) + (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILE_5 * dilation) + (blockIdx.x % dilation);
    const int sni = get_window_start(si, height, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
    const int snj = get_window_start(sj, width, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
    __shared__ __half2 tile[TILE_5*TILE_5][DIM_32+3];
    __shared__ __half2 kTile[KTILE_5*KTILE_5][DIM_32+3];
    __half2* query2 = reinterpret_cast<__half2*>(query.data());
    __half2* key2 = reinterpret_cast<__half2*>(key.data());

    /* query tile */
    const int qtx = lti / DIMHALF_32;
    const int qty = lti - qtx * DIMHALF_32;
    if (qtx < TILE_5*TILE_5)
    {
        int qi = qtx / TILE_5;
        const int qj = (qtx - qi * TILE_5) * dilation + sj;
        qi = qi * dilation + si;
        if (qi < height && qj < width){
            tile[qtx][qty] = query2[batchHeadOffset + qi * stride2 + qj * DIMHALF_32 + qty];
        }
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KHALFITERS_32;
    if (ktx < KTILE_5*KTILE_5)
    {
        int bi = ktx / KTILE_5;
        const int bj = (ktx - bi * KTILE_5) * dilation + snj;
        bi = bi * dilation + sni;
        if (bi < height && bj < width){
            const int keyOffset = batchHeadOffset + bi * stride2 + bj * DIMHALF_32 + kty;
            #pragma unroll
            for (int ti=0; ti < KHALFITERS_32; ++ti)
                kTile[ktx][kty + ti] = key2[keyOffset + ti];
        }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE_5;
    const int ki = threadIdx.y - ii * KERNEL_SIZE_5;
    const int jj = threadIdx.x / KERNEL_SIZE_5;
    const int kj = threadIdx.x - jj * KERNEL_SIZE_5;
    const int i = si + ii*dilation, j = sj + jj*dilation;
    if (i < height && j < width){
        const int ni = get_window_start(i, height, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
        __half2 updt = __float2half2_rn(0.f);
        const int queryIdx = ii*TILE_5 + jj;
        const int keyIdx = int((ni+ki*dilation - sni)/dilation)*KTILE_5 + int((nj+kj*dilation - snj)/dilation);

        #pragma unroll
        for (int dimOffset=0; dimOffset < DIMHALF_32; ++dimOffset)
            updt = __hfma2(tile[queryIdx][dimOffset], kTile[keyIdx][dimOffset], updt);
        const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE_5+kj;
        attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y));
    }
    //}
}

/* TODO: CHECK BANK CONFLICTS */
template <int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_5x5_32( // QK    / A-grad
    const Tensor5D<scalar_t> query,               // query / d_out
    const Tensor5D<scalar_t> key,                 // key   / value
    Tensor5D<scalar_t> attn,                      // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use blockIdx since blockDim will be 1 and threadIdx will
    // always be 0.
    // const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / heads;
    const int h = z - b * heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILE_5*KERNEL_SIZE_5) + threadIdx.x;
    const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
    const int si = int(blockIdx.y / dilation) * (TILE_5 * dilation) + (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILE_5 * dilation) + (blockIdx.x % dilation);
    const int sni = get_window_start(si, height, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
    const int snj = get_window_start(sj, width, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
    __shared__ scalar_t tile[TILE_5*TILE_5][DIM_32+3];
    __shared__ scalar_t kTile[KTILE_5*KTILE_5][DIM_32+3];

    /* query tile */
    const int qtx = lti / QSTRIDE_5;
    const int qty = (lti - qtx * QSTRIDE_5) * QITERS_5;
    if (qtx < TILE_5*TILE_5)
    {
        int qi = qtx / TILE_5;
        const int qj = (qtx - qi * TILE_5) * dilation + sj;
        qi = qi * dilation + si;
        if (qi < height && qj < width){
            #pragma unroll
            for (int ti=0; ti < QITERS_5; ++ti)
                tile[qtx][qty+ti] = query.data()[batchHeadOffset + qi * query.stride(2) + qj * query.stride(3) + qty+ti];
        }
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KITERS_32;
    if (ktx < KTILE_5*KTILE_5)
    {
        int bi = ktx / KTILE_5;
        const int bj = (ktx - bi * KTILE_5) * dilation + snj;
        bi = bi * dilation + sni;
        if (bi < height && bj < width){
            const int keyOffset = batchHeadOffset + bi * query.stride(2) + bj * query.stride(3) + kty;
            #pragma unroll
            for (int ti=0; ti < KITERS_32; ++ti)
                kTile[ktx][kty + ti] = key.data()[keyOffset + ti];
        }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE_5;
    const int ki = threadIdx.y - ii * KERNEL_SIZE_5;
    const int jj = threadIdx.x / KERNEL_SIZE_5;
    const int kj = threadIdx.x - jj * KERNEL_SIZE_5;
    const int i = si + ii*dilation, j = sj + jj*dilation;
    if (i < height && j < width){
        const int ni = get_window_start(i, height, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation);
        scalar_t updt = scalar_t(0);
        const int queryIdx = ii*TILE_5 + jj;
        const int keyIdx = int((ni+ki*dilation - sni)/dilation)*KTILE_5 + int((nj+kj*dilation - snj)/dilation);

        #pragma unroll
        for (int dimOffset=0; dimOffset < DIM_32; ++dimOffset)
            updt += tile[queryIdx][dimOffset] * kTile[keyIdx][dimOffset];

        const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE_5+kj;
        attn.data()[index] = updt;
    }
    //}
}

template <int TILE, int KTILE, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_fp16_7x7_9x9_32( // QK    / A-grad
    const Tensor5D<scalar_t> query,                        // query / d_out
    const Tensor5D<scalar_t> key,                          // key   / value
    Tensor5D<scalar_t> attn,                               // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use blockIdx since blockDim will be 1 and threadIdx will
    // always be 0.
    // const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / heads;
    const int h = z - b * heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILE*KERNEL_SIZE) + threadIdx.x;
    const int stride2 = DIMHALF_32 * width;
    const int batchHeadOffset = b * (stride2*height*heads) + h * (stride2*height);
    const int si = int(blockIdx.y / dilation) * (TILE * dilation) + (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILE * dilation) + (blockIdx.x % dilation);
    const int sni = get_window_start(si, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    const int snj = get_window_start(sj, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    __shared__ __half2 tile[TILE*TILE][DIM_32+3];
    __shared__ __half2 kTile[KTILE*KTILE][DIM_32+3];
    __half2* query2 = reinterpret_cast<__half2*>(query.data());
    __half2* key2 = reinterpret_cast<__half2*>(key.data());

    /* query tile */
    const int qtx = lti / DIM_32;
    const int qtyp = lti - qtx * DIM_32;
    const int qdi = qtyp / KHALFITERS_32;
    const int qdj = qtyp - qdi * KHALFITERS_32;
    const int qty = qdi*KITERS_32+qdj;
    if (qtx < TILE*TILE && qtyp < DIMHALF_32)
    {
        int qi = qtx / TILE;
        const int qj = (qtx - qi * TILE) * dilation + sj;
        qi = qi * dilation + si;
        if (qi < height && qj < width)
            tile[qtx][qty] = query2[batchHeadOffset + qi * stride2 + qj * DIMHALF_32 + qtyp];
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KHALFITERS_32;
    if (ktx < KTILE*KTILE)
    {
        int bi = ktx / KTILE;
        const int bj = (ktx - bi * KTILE) * dilation + snj;
        bi = bi * dilation + sni;
        if (bi < height && bj < width){
            const int keyOffset = batchHeadOffset + bi * stride2 + bj * DIMHALF_32 + kty;
            #pragma unroll
            for (int ti=0; ti < KHALFITERS_32; ++ti)
                kTile[ktx][kty*2 + ti] = key2[keyOffset + ti];
        }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE;
    const int ki = threadIdx.y - ii * KERNEL_SIZE;
    const int jj = threadIdx.x / KERNEL_SIZE;
    const int kj = threadIdx.x - jj * KERNEL_SIZE;
    const int i = si + ii*dilation, j = sj + jj*dilation;
    if (i < height && j < width){
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        __half2 updt = __float2half2_rn(0.f);
        const int queryIdx = ii*TILE + jj;
        const int keyIdx = int((ni+ki*dilation - sni)/dilation)*KTILE + int((nj+kj*dilation - snj)/dilation);

        #pragma unroll
        for (int di=0; di < KSTRIDE_32; ++di)
            #pragma unroll
            for (int dj=0; dj < KHALFITERS_32; ++dj)
                updt = __hfma2(tile[queryIdx][di*KITERS_32+dj], kTile[keyIdx][di*KITERS_32+dj], updt);
        const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE+kj;
        attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y));
    }
    //}
}

template <int TILE, int KTILE, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_7x7_9x9_32( // QK    / A-grad
    const Tensor5D<scalar_t> query,                   // query / d_out
    const Tensor5D<scalar_t> key,                     // key   / value
    Tensor5D<scalar_t> attn,                          // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use blockIdx since blockDim will be 1 and threadIdx will
    // always be 0.
    // const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / heads;
    const int h = z - b * heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILE*KERNEL_SIZE) + threadIdx.x;
    const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
    const int si = int(blockIdx.y / dilation) * (TILE * dilation) + (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILE * dilation) + (blockIdx.x % dilation);
    const int sni = get_window_start(si, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    const int snj = get_window_start(sj, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    __shared__ scalar_t tile[TILE*TILE][DIM_32+3];
    __shared__ scalar_t kTile[KTILE*KTILE][DIM_32+3];

    /* query tile */
    const int qtx = lti / DIM_32;
    const int qty = lti - qtx * DIM_32;
    if (qtx < TILE*TILE)
    {
        int qi = qtx / TILE;
        const int qj = (qtx - qi * TILE) * dilation + sj;
        qi = qi * dilation + si;
        if (qi < height && qj < width)
            tile[qtx][qty] = query.data()[batchHeadOffset + qi * query.stride(2) + qj * query.stride(3) + qty];
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KITERS_32;
    if (ktx < KTILE*KTILE)
    {
        int bi = ktx / KTILE;
        const int bj = (ktx - bi * KTILE) * dilation + snj;
        bi = bi * dilation + sni;
        if (bi < height && bj < width){
            const int keyOffset = batchHeadOffset + bi * query.stride(2) + bj * query.stride(3) + kty;
            #pragma unroll
            for (int ti=0; ti < KITERS_32; ++ti)
                kTile[ktx][kty + ti] = key.data()[keyOffset + ti];
        }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE;
    const int ki = threadIdx.y - ii * KERNEL_SIZE;
    const int jj = threadIdx.x / KERNEL_SIZE;
    const int kj = threadIdx.x - jj * KERNEL_SIZE;
    const int i = si + ii*dilation, j = sj + jj*dilation;
    if (i < height && j < width){
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        scalar_t updt = scalar_t(0);
        const int queryIdx = ii*TILE + jj;
        const int keyIdx = int((ni+ki*dilation - sni)/dilation)*KTILE + int((nj+kj*dilation - snj)/dilation);

        #pragma unroll
        for (int dimOffset=0; dimOffset < DIM_32; ++dimOffset)
            updt += tile[queryIdx][dimOffset] * kTile[keyIdx][dimOffset];

        const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE+kj;
        attn.data()[index] = updt;
    }
    //}
}

template <int TILEX, int TILEY, int KTILEX, int KTILEY, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t, typename memscalar_t>
__global__ void pointwise_neighborhood_2d_fp16_11x11_13x13_32( // QK    / A-grad
    const Tensor5D<scalar_t> query,                            // query / d_out
    const Tensor5D<scalar_t> key,                              // key   / value
    Tensor5D<scalar_t> attn,                                   // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use blockIdx since blockDim will be 1 and threadIdx will
    // always be 0.
    // const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / heads;
    const int h = z - b * heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILEY*KERNEL_SIZE) + threadIdx.x;
    const int stride2 = DIMHALF_32 * width;
    const int batchHeadOffset = b * (stride2*height*heads) + h * (stride2*height);
    const int si = int(blockIdx.y / dilation) * (TILEX * dilation) + (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILEY * dilation) + (blockIdx.x % dilation);
    const int sni = get_window_start(si, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    const int snj = get_window_start(sj, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    __shared__ __half2 tile[TILEX*TILEY][DIM_32+3];
    __shared__ __half2 kTile[KTILEX*KTILEY][DIM_32+3];
    __half2* query2 = reinterpret_cast<__half2*>(query.data());
    __half2* key2 = reinterpret_cast<__half2*>(key.data());

    /* query tile */
    const int qtx = lti / DIM_32;
    const int qtyp = lti - qtx * DIM_32;
    const int qdi = qtyp / KHALFITERS_32;
    const int qdj = qtyp - qdi * KHALFITERS_32;
    const int qty = qdi*KITERS_32+qdj;
    if (qtx < TILEX*TILEY && qtyp < DIMHALF_32)
    {
        int qi = qtx / TILEY;
        const int qj = (qtx - qi * TILEY) * dilation + sj;
        qi =  qi * dilation + si;
        if (qi < height && qj < width)
            tile[qtx][qty] = query2[batchHeadOffset + qi * stride2 + qj * DIMHALF_32 + qtyp];
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KHALFITERS_32;
    if (ktx < KTILEX*KTILEY)
    {
        int bi = ktx / KTILEY;
        const int bj = (ktx - bi * KTILEY) * dilation + snj;
        bi = bi * dilation + sni;
        if (bi < height && bj < width){
            const int keyOffset = batchHeadOffset + bi * stride2 + bj * DIMHALF_32 + kty;
            #pragma unroll
            for (int ti=0; ti < KHALFITERS_32; ++ti)
                kTile[ktx][kty*2 + ti] = key2[keyOffset + ti];
        }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE;
    const int ki = threadIdx.y - ii * KERNEL_SIZE;
    const int jj = threadIdx.x / KERNEL_SIZE;
    const int kj = threadIdx.x - jj * KERNEL_SIZE;
    const int i = si + ii*dilation, j = sj + jj*dilation;
    if (i < height && j < width){
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        __half2 updt = __float2half2_rn(0.f);
        const int queryIdx = ii*TILEY + jj;
        const int keyIdx = int((ni+ki*dilation - sni)/dilation)*KTILEY + int((nj+kj*dilation - snj)/dilation);

        #pragma unroll
        for (int di=0; di < KSTRIDE_32; ++di)
            #pragma unroll
            for (int dj=0; dj <KHALFITERS_32; ++dj)
                updt = __hfma2(tile[queryIdx][di*KITERS_32+dj], kTile[keyIdx][di*KITERS_32+dj], updt);
        const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE+kj;
        attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y));
    }
    //}
}

template <int TILEX, int TILEY, int KTILEX, int KTILEY, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t, typename memscalar_t>
__global__ void pointwise_neighborhood_2d_11x11_13x13_32( // QK    / A-grad
    const Tensor5D<scalar_t> query,                       // query / d_out
    const Tensor5D<scalar_t> key,                         // key   / value
    Tensor5D<scalar_t> attn,                              // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    // Because batch heads have stride 1 per threadblock, we can just use blockIdx since blockDim will be 1 and threadIdx will
    // always be 0.
    // const int z = blockIdx.z * blockDim.z + threadIdx.z;
    const int z = blockIdx.z;
    const int b = z / heads;
    const int h = z - b * heads;
    // Not needed again because it will always be true.
    // if (z < batch_size * heads)
    // {
    const int lti = threadIdx.y * (TILEY*KERNEL_SIZE) + threadIdx.x;
    const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
    const int si = int(blockIdx.y / dilation) * (TILEX * dilation) + (blockIdx.y % dilation);
    const int sj = int(blockIdx.x / dilation) * (TILEY * dilation) + (blockIdx.x % dilation);
    const int sni = get_window_start(si, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    const int snj = get_window_start(sj, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
    __shared__ memscalar_t tile[TILEX*TILEY][DIM_32+3];
    __shared__ memscalar_t kTile[KTILEX*KTILEY][DIM_32+3];

    /* query tile */
    const int qtx = lti / DIM_32;
    const int qty = lti - qtx * DIM_32;
    if (qtx < TILEX*TILEY)
    {
        int qi = qtx / TILEY;
        const int qj = (qtx - qi * TILEY) * dilation + sj;
        qi =  qi * dilation + si;
        if (qi < height && qj < width)
            tile[qtx][qty] = query.data()[batchHeadOffset + qi * query.stride(2) + qj * query.stride(3) + qty];
    }
    /* key tile */
    const int ktx = lti / KSTRIDE_32;
    const int kty = (lti - ktx * KSTRIDE_32) * KITERS_32;
    if (ktx < KTILEX*KTILEY)
    {
        int bi = ktx / KTILEY;
        const int bj = (ktx - bi * KTILEY) * dilation + snj;
        bi = bi * dilation + sni;
        if (bi < height && bj < width){
            const int keyOffset = batchHeadOffset + bi * query.stride(2) + bj * query.stride(3) + kty;
            #pragma unroll
            for (int ti=0; ti < KITERS_32; ++ti)
                kTile[ktx][kty + ti] = key.data()[keyOffset + ti];
        }
    }
    __syncthreads();
    const int ii = threadIdx.y / KERNEL_SIZE;
    const int ki = threadIdx.y - ii * KERNEL_SIZE;
    const int jj = threadIdx.x / KERNEL_SIZE;
    const int kj = threadIdx.x - jj * KERNEL_SIZE;
    const int i = si + ii*dilation, j = sj + jj*dilation;
    if (i < height && j < width){
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        scalar_t updt = scalar_t(0);
        const int queryIdx = ii*TILEY + jj;
        const int keyIdx = int((ni+ki*dilation - sni)/dilation)*KTILEY + int((nj+kj*dilation - snj)/dilation);

        #pragma unroll
        for (int dimOffset=0; dimOffset < DIM_32; ++dimOffset)
            updt += tile[queryIdx][dimOffset] * kTile[keyIdx][dimOffset];

        const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE+kj;
        attn.data()[index] = updt;
    }
    //}
}

} // namespace natten
