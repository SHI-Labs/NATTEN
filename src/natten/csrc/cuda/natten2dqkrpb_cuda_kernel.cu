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

namespace natten {

#define WARP_SIZE 32

#define KERNEL_SIZE_13 13
#define KERNEL_SIZE_11 11
#define KERNEL_SIZE_9 9
#define KERNEL_SIZE_7 7
#define KERNEL_SIZE_5 5
#define KERNEL_SIZE_3 3
#define NEIGHBORHOOD_SIZE_13 6
#define NEIGHBORHOOD_SIZE_11 5
#define NEIGHBORHOOD_SIZE_9 4
#define NEIGHBORHOOD_SIZE_7 3
#define NEIGHBORHOOD_SIZE_5 2
#define NEIGHBORHOOD_SIZE_3 1
// Always keep batchthreads 1, because we want each thread block to process one 1 sample 1 head
#define BATCHTHREADS_13 1
#define BATCHTHREADS_11 1
#define BATCHTHREADS_9 1
#define BATCHTHREADS_7 1
#define BATCHTHREADS_5 1
#define BATCHTHREADS_3 1
// Tile is the number of pixels across each axis that are processed within a single threadblock
// So far the best tile size for Kernel size 7 is 3x3.
#define TILE_9 3
#define TILE_7 3
#define TILE_5 4
#define TILE_3 7

#define TILE_11_X 2
#define TILE_11_Y 3
#define TILE_13_X 2
#define TILE_13_Y 3
// Each of the 3x3 pixels has 7x7 key neighbors in this case, therefore the tile size for keys will 7 + 3 - 1 = 9x9
#define KTILE_9 11
#define KTILE_7 9
#define KTILE_5 8
#define KTILE_3 9

#define KTILE_11_X 12
#define KTILE_11_Y 13
#define KTILE_13_X 14
#define KTILE_13_Y 15
// 7x7 kernel, and we want each threadblock to process the entire neighborhood for each QUERY in its tile,
// so we'll have 7x7 * 3x3 = 21x21
// Also keep in mind these 21 threads are across each axis, so it's 21x21 threads total
// 21x21 = 441 < 1024
// Ensure it's less than 1024, which is the max number of threads per threadblock
#define XYTHREADS_9 27
#define XYTHREADS_7 21
#define XYTHREADS_5 20
#define XYTHREADS_3 21

#define XTHREADS_11 33
#define YTHREADS_11 22
#define XTHREADS_13 39
#define YTHREADS_13 26

// DIM is fixed at 32 for now
#define DIM_32 32
#define DIMHALF_32 16 // FP16 stored in half2 => half the dims
// There's 32 * 3x3 QUERY cells to store, and 32 * 10x10 KEY cells
// The former is 288 < 441 threads, so each thread can copy over one QUERY cell exactly, and we'll have empty threads too
// But that's not the case for the latter, which is 3200 and it's not < 441
// But we can have each thread load more cells instead. 8 is optimal since it will maximize utility
// So copy 8 dims per KEY pixel in each thread
#define KITERS_32 8
#define KHALFITERS_32 4 // FP16 stored in half2 => half the dims
// and DIM = 32 / 8 = 4, hence 4 is the stride.
#define KSTRIDE_32 4
// For kernel size 5, we have to do 2 query dims per thread, because we have fewer threads in each threadblock than the total
// number of queries.
// For kernel size 3, we have to read 2 query dims per thread
#define QITERS_5 2
#define QSTRIDE_5 16
#define QITERS_3 4
#define QSTRIDE_3 8
#define QITERS_3_HALF 2
#define QSTRIDE_3_HALF 8

// This is just for the other kernels that are not using SMEM
#define CUDA_NUM_THREADS_Q 512
#define CUDA_NUM_THREADS_K 512
#define CUDA_NUM_THREADS_RPB 64
#define CUDA_NUM_THREADS_Q16 512
#define CUDA_NUM_THREADS_K16 256
#define CUDA_NUM_THREADS_RPB16 64


template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void natten2dqkrpb_cuda_forward_kernel_fp16(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> rpb,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> attn,
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
                const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1) + (pj+kj) * rpb.stride(2);
                attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y)) + rpb.data()[rpbIndex];
            }
        }
    }
}


template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void natten2dqkrpb_cuda_forward_kernel_fp32(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> rpb,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> attn,
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
                const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1) + (pj+kj) * rpb.stride(2);
                updt += rpb.data()[rpbIndex];
                attn.data()[index] = updt;
            }
        }
    }
}


/* TODO: FIX BANK CONFLICTS */
template <int DILATION, typename scalar_t>
__global__ void natten2dqkrpb_cuda_forward_kernel_fp16_3x3_32(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> rpb,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> attn,
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
        const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1) + (pj+kj) * rpb.stride(2);
        attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y)) + rpb.data()[rpbIndex];
    }
    //}
}

/* TODO: CHECK BANK CONFLICTS */
template <int DILATION, typename scalar_t>
__global__ void natten2dqkrpb_cuda_forward_kernel_fp32_3x3_32(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> rpb,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> attn,
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
        const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1) + (pj+kj) * rpb.stride(2);
        updt += rpb.data()[rpbIndex];
        attn.data()[index] = updt;
    }
    //}
}


/* TODO: FIX BANK CONFLICTS */
template <int DILATION, typename scalar_t>
__global__ void natten2dqkrpb_cuda_forward_kernel_fp16_5x5_32(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> rpb,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> attn,
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
        const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1) + (pj+kj) * rpb.stride(2);
        attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y)) + rpb.data()[rpbIndex];
    }
    //}
}

/* TODO: CHECK BANK CONFLICTS */
template <int DILATION, typename scalar_t>
__global__ void natten2dqkrpb_cuda_forward_kernel_fp32_5x5_32(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> rpb,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> attn,
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
        const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1) + (pj+kj) * rpb.stride(2);
        updt += rpb.data()[rpbIndex];
        attn.data()[index] = updt;
    }
    //}
}


template <int TILE, int KTILE, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void natten2dqkrpb_cuda_forward_kernel_fp16_7x7_9x9_32(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> rpb,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> attn,
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
        const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1) + (pj+kj) * rpb.stride(2);
        attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y)) + rpb.data()[rpbIndex];
    }
    //}
}

template <int TILE, int KTILE, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void natten2dqkrpb_cuda_forward_kernel_fp32_7x7_9x9_32(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> rpb,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> attn,
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
        const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1) + (pj+kj) * rpb.stride(2);
        updt += rpb.data()[rpbIndex];
        attn.data()[index] = updt;
    }
    //}
}


template <int TILEX, int TILEY, int KTILEX, int KTILEY, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t, typename memscalar_t>
__global__ void natten2dqkrpb_cuda_forward_kernel_fp16_11x11_13x13_32(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> rpb,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> attn,
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
        const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1) + (pj+kj) * rpb.stride(2);
        attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y)) + rpb.data()[rpbIndex];
    }
    //}
}

template <int TILEX, int TILEY, int KTILEX, int KTILEY, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t, typename memscalar_t>
__global__ void natten2dqkrpb_cuda_forward_kernel_fp32_11x11_13x13_32(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> query,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> key,
    const torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> rpb,
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> attn,
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
        const int rpbIndex = h * rpb.stride(0) + (pi+ki) * rpb.stride(1) + (pj+kj) * rpb.stride(2);
        updt += rpb.data()[rpbIndex];
        attn.data()[index] = updt;
    }
    //}
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void natten2dq_cuda_backward_kernel_fp32(
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_query,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> key,
    const int height,
    const int width,
    const int heads,
    const int dilation_in,
    const int dim,
    const int totalElements) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < totalElements){
        int indtmp1 = linearIndex/dim;
        const int d = linearIndex - indtmp1 * dim;
        int indtmp2 = indtmp1/width;
        const int j = indtmp1 - indtmp2 * width;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        scalar_t d_query_update = scalar_t(0);
        int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1) + i * d_attn.stride(2) + j * d_attn.stride(3);
        const int keyOffset = b * key.stride(0) + h * key.stride(1) + d;
        #pragma unroll
        for (int xi=ni; xi < ni + KERNEL_SIZE * dilation; xi+=dilation)
            #pragma unroll
            for (int xj=nj; xj < nj + KERNEL_SIZE * dilation; xj+=dilation){
                const int keyIndex = keyOffset + xi * key.stride(2) + xj * key.stride(3);
                d_query_update += d_attn.data()[attnOffset] * key.data()[keyIndex];
                ++attnOffset;
            }
        d_query.data()[linearIndex] = d_query_update;
    }
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void natten2dq_cuda_backward_kernel_fp16(
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_query,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> key,
    const int height,
    const int width,
    const int heads,
    const int dilation_in,
    const int dimhalf,
    const int totalElements) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < totalElements){
        __half2* d_query2 = reinterpret_cast<__half2*>(d_query.data());
        __half2* key2 = reinterpret_cast<__half2*>(key.data());
        int indtmp1 = linearIndex/dimhalf;
        const int d = linearIndex - indtmp1 * dimhalf;
        int indtmp2 = indtmp1/width;
        const int j = indtmp1 - indtmp2 * width;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        __half2 d_query_update = __float2half2_rn(0.f);
        int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1) + i * d_attn.stride(2) + j * d_attn.stride(3);
        const int stride2 = dimhalf * width;
        const int keyOffset = b * (stride2 * height * heads) + h * (stride2 * height) + d;
        #pragma unroll
        for (int xi=ni; xi < ni + KERNEL_SIZE * dilation; xi+=dilation)
            #pragma unroll
            for (int xj=nj; xj < nj + KERNEL_SIZE * dilation; xj+=dilation){
                const int keyIndex = keyOffset + xi * stride2 + xj * dimhalf;
                scalar_t a = d_attn.data()[attnOffset];
                d_query_update = __hfma2(__halves2half2(a, a), key2[keyIndex], d_query_update);
                ++attnOffset;
            }
        d_query2[linearIndex] = d_query_update;
    }
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void natten2drpb_cuda_backward_kernel_fp16(
    torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> d_rpb,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_attn,
    const int height,
    const int width,
    const int dilation_in,
    const int batch_size,
    const int d_rpb_numel,
    const int totalThreads) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < totalThreads){
        int indtmp1 = linearIndex/KERNEL_SIZE;
        const int kj = linearIndex - indtmp1 * KERNEL_SIZE;
        int indtmp2 = indtmp1/KERNEL_SIZE;
        const int ki = indtmp1 - indtmp2 * KERNEL_SIZE;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/width;
        const int j = indtmp1 - indtmp2 * width;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        const int h = indtmp2;
        const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        float d_rpb_update = scalar_t(0);
        int attnOffset = h * d_attn.stride(1) + i * d_attn.stride(2) + j * d_attn.stride(3) + (ki*KERNEL_SIZE+kj);
        #pragma unroll
        for (int b=0; b < batch_size; ++b){
            d_rpb_update += static_cast<float>(d_attn.data()[attnOffset]);
            attnOffset += d_attn.stride(0);
        }
        const int index = h * d_rpb.stride(0) + (pi+ki) * d_rpb.stride(1) + (pj+kj) * d_rpb.stride(2);
        at::native::fastAtomicAdd(d_rpb.data(), index, d_rpb_numel, static_cast<scalar_t>(d_rpb_update), true);
    }
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void natten2drpb_cuda_backward_kernel(
    torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> d_rpb,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_attn,
    const int height,
    const int width,
    const int dilation_in,
    const int batch_size,
    const int d_rpb_numel,
    const int totalThreads) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < totalThreads){
        int indtmp1 = linearIndex/KERNEL_SIZE;
        const int kj = linearIndex - indtmp1 * KERNEL_SIZE;
        int indtmp2 = indtmp1/KERNEL_SIZE;
        const int ki = indtmp1 - indtmp2 * KERNEL_SIZE;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/width;
        const int j = indtmp1 - indtmp2 * width;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        const int h = indtmp2;
        const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        scalar_t d_rpb_update = scalar_t(0);
        int attnOffset = h * d_attn.stride(1) + i * d_attn.stride(2) + j * d_attn.stride(3) + (ki*KERNEL_SIZE+kj);
        #pragma unroll
        for (int b=0; b < batch_size; ++b){
            d_rpb_update += d_attn.data()[attnOffset];
            attnOffset += d_attn.stride(0);
        }
        const int index = h * d_rpb.stride(0) + (pi+ki) * d_rpb.stride(1) + (pj+kj) * d_rpb.stride(2);
        at::native::fastAtomicAdd(d_rpb.data(), index, d_rpb_numel, static_cast<scalar_t>(d_rpb_update), true);
    }
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void natten2dk_cuda_backward_kernel_fp16(
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_key,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> query,
    const int height,
    const int width,
    const int heads,
    const int dilation_in,
    const int dimhalf,
    const int d_key_numel) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < d_key_numel){
        __half2* d_key2 = reinterpret_cast<__half2*>(d_key.data());
        __half2* query2 = reinterpret_cast<__half2*>(query.data());
        int indtmp1 = linearIndex/dimhalf;
        const int d = linearIndex - indtmp1 * dimhalf;
        int indtmp2 = indtmp1/width;
        const int j = indtmp1 - indtmp2 * width;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int ni = get_backward_window_start(i, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_backward_window_start(j, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int ei = get_backward_window_end(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int ej = get_backward_window_end(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1);
        const int stride2 = dimhalf * width;
        const int queryOffset = b * (stride2 * height * heads) + h * (stride2 * height) + d;
        __half2 d_key_update = __float2half2_rn(0.f);
        #pragma unroll
        for (int xi=ni; xi < ei; xi+=dilation){
            const int oni = get_window_start(xi, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            #pragma unroll
            for (int xj=nj; xj < ej; xj+=dilation){
                const int onj = get_window_start(xj, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int queryIndex = queryOffset + xi * stride2 + xj * dimhalf;
                const int attnIndex = attnOffset + xi * d_attn.stride(2) + xj * d_attn.stride(3) + int((i-oni)/dilation)*KERNEL_SIZE+int((j-onj)/dilation);
                scalar_t a = d_attn.data()[attnIndex];
                d_key_update = __hfma2(query2[queryIndex], __halves2half2(a, a), d_key_update);
            }
        }
        d_key2[linearIndex] = d_key_update;
    }
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void natten2dk_cuda_backward_kernel_fp32(
    torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_key,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_attn,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> query,
    const int height,
    const int width,
    const int heads,
    const int dilation_in,
    const int dim,
    const int d_key_numel) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < d_key_numel){
        int indtmp1 = linearIndex/dim;
        const int d = linearIndex - indtmp1 * dim;
        int indtmp2 = indtmp1/width;
        const int j = indtmp1 - indtmp2 * width;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int ni = get_backward_window_start(i, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_backward_window_start(j, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int ei = get_backward_window_end(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int ej = get_backward_window_end(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int attnOffset = b * d_attn.stride(0) + h * d_attn.stride(1);
        const int queryOffset = b * query.stride(0) + h * query.stride(1) + d;
        scalar_t d_key_update = scalar_t(0);
        #pragma unroll
        for (int xi=ni; xi < ei; xi+=dilation){
            const int oni = get_window_start(xi, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            #pragma unroll
            for (int xj=nj; xj < ej; xj+=dilation){
                const int onj = get_window_start(xj, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int queryIndex = queryOffset + xi * query.stride(2) + xj * query.stride(3);
                const int attnIndex = attnOffset + xi * d_attn.stride(2) + xj * d_attn.stride(3) + int((i-oni)/dilation)*KERNEL_SIZE+int((j-onj)/dilation);
                d_key_update += query.data()[queryIndex] * d_attn.data()[attnIndex];
            }
        }
        d_key.data()[linearIndex] = d_key_update;
    }
}

torch::Tensor natten2dqkrpb_cuda_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t dim = query.size(4);
    int64_t RPB_MAX = rpb.size(1);
    int kernel_size = (RPB_MAX + 1) / 2;
    int kernel_size_sq = pow(kernel_size, 2);
    int zsize = batch_size * heads;
    int xsize = height * width;
    CHECK_FEATMAP(height, width, kernel_size, dilation);
    CHECK_KERNELSIZE("natten2dqkrpb_cuda_forward", kernel_size);
    int KERNELTHREADS = min(CUDA_NUM_THREADS, kernel_size_sq);
    int PIXELTHREADS = min(int(CUDA_NUM_THREADS / KERNELTHREADS), xsize);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (PIXELTHREADS * KERNELTHREADS));

    auto attn = torch::zeros(
            {batch_size, heads, height, width, kernel_size_sq}, query.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
            (xsize + PIXELTHREADS - 1) / PIXELTHREADS,
            (kernel_size_sq + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(PIXELTHREADS, KERNELTHREADS, BATCHTHREADS);
    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "natten2dqkrpb_cuda_forward", ([&] {
        const auto query_a = query.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto rpb_a = rpb.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS(kernel_size, dilation, natten2dqkrpb_cuda_forward_kernel_fp32, 
                blocks, threads, 0, stream, 
                query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dilation, dim);
    }));
    return attn;
}

torch::Tensor natten2dqkrpb_cuda_forward_fp16(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t dimhalf = query.size(4) / 2;
    int64_t RPB_MAX = rpb.size(1);
    int kernel_size = (RPB_MAX + 1) / 2;
    int kernel_size_sq = pow(kernel_size, 2);
    int zsize = batch_size * heads;
    int xsize = height * width;
    CHECK_FEATMAP(height, width, kernel_size, dilation);
    CHECK_KERNELSIZE("natten2dqkrpb_cuda_forward_fp16", kernel_size);
    TORCH_CHECK(dimhalf*2 == query.size(4), "Dims per head must be an even number in FP16.");
    int KERNELTHREADS = min(CUDA_NUM_THREADS, kernel_size_sq);
    int PIXELTHREADS = min(int(CUDA_NUM_THREADS / KERNELTHREADS), xsize);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (PIXELTHREADS * KERNELTHREADS));

    auto attn = torch::zeros(
            {batch_size, heads, height, width, kernel_size_sq}, query.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
            (xsize + PIXELTHREADS - 1) / PIXELTHREADS,
            (kernel_size_sq + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(PIXELTHREADS, KERNELTHREADS, BATCHTHREADS);
    AT_DISPATCH_HALF_TYPES(at::kHalf, query.scalar_type(), "natten2dqkrpb_cuda_forward_fp16", ([&] {
        const auto query_a = query.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto rpb_a = rpb.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS(kernel_size, dilation, natten2dqkrpb_cuda_forward_kernel_fp16, 
                blocks, threads, 0, stream, 
                query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dilation, dimhalf);
    }));
    return attn;
}

torch::Tensor natten2dqkrpb_cuda_forward_tiled_32(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t dim = query.size(4);
    int64_t RPB_MAX = rpb.size(1);
    int kernel_size = (RPB_MAX + 1) / 2;
    CHECK_FEATMAP(height, width, kernel_size, dilation);
    TORCH_CHECK(dim == DIM_32, "natten2dqkrpb_cuda_forward_fp32_tiled_32", " only supports 32-dim attention heads.");
    TORCH_CHECK(kernel_size == KERNEL_SIZE_7 || kernel_size == KERNEL_SIZE_3 || kernel_size == KERNEL_SIZE_5 ||
            kernel_size == KERNEL_SIZE_9 || kernel_size == KERNEL_SIZE_11 || kernel_size == KERNEL_SIZE_13,
            "natten2dqkrpb_cuda_forward_fp32_tiled_32", " only supports kernel sizes 3, 5, 7, 9, 11, and 13.");
    int xsize = width * kernel_size;
    int ysize = height * kernel_size;
    int zsize = batch_size * heads;

    auto attn = torch::zeros({batch_size, heads, height, width, kernel_size*kernel_size}, query.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    int XTHREADS = -1;
    int YTHREADS = -1;
    int BATCHTHREADS = -1;
    if (kernel_size == KERNEL_SIZE_7)
    {
        XTHREADS = XYTHREADS_7;
        YTHREADS = XYTHREADS_7;
        BATCHTHREADS = BATCHTHREADS_7;
    }
    else if (kernel_size == KERNEL_SIZE_3)
    {
        XTHREADS = XYTHREADS_3;
        YTHREADS = XYTHREADS_3;
        BATCHTHREADS = BATCHTHREADS_3;
    }
    else if (kernel_size == KERNEL_SIZE_5)
    {
        XTHREADS = XYTHREADS_5;
        YTHREADS = XYTHREADS_5;
        BATCHTHREADS = BATCHTHREADS_5;
    }
    else if (kernel_size == KERNEL_SIZE_9)
    {
        XTHREADS = XYTHREADS_9;
        YTHREADS = XYTHREADS_9;
        BATCHTHREADS = BATCHTHREADS_9;
    }
    else if (kernel_size == KERNEL_SIZE_11)
    {
        XTHREADS = XTHREADS_11;
        YTHREADS = YTHREADS_11;
        BATCHTHREADS = BATCHTHREADS_11;
    }
    else if (kernel_size == KERNEL_SIZE_13)
    {
        XTHREADS = XTHREADS_13;
        YTHREADS = YTHREADS_13;
        BATCHTHREADS = BATCHTHREADS_13;
    }
    const dim3 blocks(
            (xsize + XTHREADS*dilation - 1) / XTHREADS,
            (ysize + YTHREADS*dilation - 1) / YTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(XTHREADS, YTHREADS, BATCHTHREADS);
    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "natten2dqkrpb_cuda_forward_fp32_tiled_32", ([&] {
        const auto query_a = query.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto rpb_a = rpb.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (kernel_size == KERNEL_SIZE_7)
            LAUNCH_DNA_KNS_TILED79(TILE_7, KTILE_7, KERNEL_SIZE_7, NEIGHBORHOOD_SIZE_7, dilation,
                    natten2dqkrpb_cuda_forward_kernel_fp32_7x7_9x9_32, blocks, threads, 0, stream, 
                    query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dilation);
        else if (kernel_size == KERNEL_SIZE_9)
            LAUNCH_DNA_KNS_TILED79(TILE_9, KTILE_9, KERNEL_SIZE_9, NEIGHBORHOOD_SIZE_9, dilation,
                    natten2dqkrpb_cuda_forward_kernel_fp32_7x7_9x9_32, blocks, threads, 0, stream, 
                    query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dilation);
        else if (kernel_size == KERNEL_SIZE_3)
            LAUNCH_DNA_DS(dilation, natten2dqkrpb_cuda_forward_kernel_fp32_3x3_32,
                    blocks, threads, 0, stream,
                    query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dilation);
        else if (kernel_size == KERNEL_SIZE_5)
            LAUNCH_DNA_DS(dilation, natten2dqkrpb_cuda_forward_kernel_fp32_5x5_32,
                    blocks, threads, 0, stream,
                    query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dilation);
        else if (kernel_size == KERNEL_SIZE_11)
            LAUNCH_DNA_KNS_TILED1113(TILE_11_X, TILE_11_Y, KTILE_11_X, KTILE_11_Y, 
                    KERNEL_SIZE_11, NEIGHBORHOOD_SIZE_11, dilation, scalar_t,
                    natten2dqkrpb_cuda_forward_kernel_fp32_11x11_13x13_32,
                    blocks, threads, 0, stream,
                    query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dilation);
        else if (kernel_size == KERNEL_SIZE_13)
            LAUNCH_DNA_KNS_TILED1113(TILE_13_X, TILE_13_Y, KTILE_13_X, KTILE_13_Y, 
                    KERNEL_SIZE_13, NEIGHBORHOOD_SIZE_13, dilation, float,
                    natten2dqkrpb_cuda_forward_kernel_fp32_11x11_13x13_32,
                    blocks, threads, 0, stream,
                    query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dilation);
    }));
    return attn;
}

torch::Tensor natten2dqkrpb_cuda_forward_fp16_tiled_32(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t dimhalf = query.size(4) / 2;
    int64_t RPB_MAX = rpb.size(1);
    int kernel_size = (RPB_MAX + 1) / 2;
    CHECK_FEATMAP(height, width, kernel_size, dilation);
    TORCH_CHECK(dimhalf*2 == query.size(4), "Dims per head must be an even number in FP16.");
    TORCH_CHECK(dimhalf*2 == DIM_32, "natten2dqkrpb_cuda_forward_fp16_tiled_32", " only supports 32-dim attention heads.");
    TORCH_CHECK(kernel_size == KERNEL_SIZE_7 || kernel_size == KERNEL_SIZE_3 || kernel_size == KERNEL_SIZE_5 ||
            kernel_size == KERNEL_SIZE_9 || kernel_size == KERNEL_SIZE_11 || kernel_size == KERNEL_SIZE_13,
            "natten2dqkrpb_cuda_forward_fp16_tiled_32", " only supports kernel sizes 3, 5, 7, 9, 11, and 13.");
    int xsize = width * kernel_size;
    int ysize = height * kernel_size;
    int zsize = batch_size * heads;

    auto attn = torch::zeros({batch_size, heads, height, width, kernel_size*kernel_size}, query.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    int XTHREADS = -1;
    int YTHREADS = -1;
    int BATCHTHREADS = -1;
    if (kernel_size == KERNEL_SIZE_7)
    {
        XTHREADS = XYTHREADS_7;
        YTHREADS = XYTHREADS_7;
        BATCHTHREADS = BATCHTHREADS_7;
    }
    else if (kernel_size == KERNEL_SIZE_3)
    {
        XTHREADS = XYTHREADS_3;
        YTHREADS = XYTHREADS_3;
        BATCHTHREADS = BATCHTHREADS_3;
    }
    else if (kernel_size == KERNEL_SIZE_5)
    {
        XTHREADS = XYTHREADS_5;
        YTHREADS = XYTHREADS_5;
        BATCHTHREADS = BATCHTHREADS_5;
    }
    else if (kernel_size == KERNEL_SIZE_9)
    {
        XTHREADS = XYTHREADS_9;
        YTHREADS = XYTHREADS_9;
        BATCHTHREADS = BATCHTHREADS_9;
    }
    else if (kernel_size == KERNEL_SIZE_11)
    {
        XTHREADS = XTHREADS_11;
        YTHREADS = YTHREADS_11;
        BATCHTHREADS = BATCHTHREADS_11;
    }
    else if (kernel_size == KERNEL_SIZE_13)
    {
        XTHREADS = XTHREADS_13;
        YTHREADS = YTHREADS_13;
        BATCHTHREADS = BATCHTHREADS_13;
    }
    const dim3 blocks(
            (xsize + XTHREADS*dilation - 1) / XTHREADS,
            (ysize + YTHREADS*dilation - 1) / YTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(XTHREADS, YTHREADS, BATCHTHREADS);
    AT_DISPATCH_HALF_TYPES(at::kHalf, query.scalar_type(), "natten2dqkrpb_cuda_forward_fp16_tiled_32", ([&] {
        const auto query_a = query.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto rpb_a = rpb.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (kernel_size == KERNEL_SIZE_7)
            LAUNCH_DNA_KNS_TILED79(TILE_7, KTILE_7, KERNEL_SIZE_7, NEIGHBORHOOD_SIZE_7, dilation,
                    natten2dqkrpb_cuda_forward_kernel_fp16_7x7_9x9_32, blocks, threads, 0, stream, 
                    query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dilation);
        else if (kernel_size == KERNEL_SIZE_9)
            LAUNCH_DNA_KNS_TILED79(TILE_9, KTILE_9, KERNEL_SIZE_9, NEIGHBORHOOD_SIZE_9, dilation,
                    natten2dqkrpb_cuda_forward_kernel_fp16_7x7_9x9_32, blocks, threads, 0, stream, 
                    query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dilation);
        else if (kernel_size == KERNEL_SIZE_3)
            LAUNCH_DNA_DS(dilation, natten2dqkrpb_cuda_forward_kernel_fp16_3x3_32,
                    blocks, threads, 0, stream,
                    query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dilation);
        else if (kernel_size == KERNEL_SIZE_5)
            LAUNCH_DNA_DS(dilation, natten2dqkrpb_cuda_forward_kernel_fp16_5x5_32,
                    blocks, threads, 0, stream,
                    query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dilation);
        else if (kernel_size == KERNEL_SIZE_11)
            LAUNCH_DNA_KNS_TILED1113(TILE_11_X, TILE_11_Y, KTILE_11_X, KTILE_11_Y, 
                    KERNEL_SIZE_11, NEIGHBORHOOD_SIZE_11, dilation, scalar_t,
                    natten2dqkrpb_cuda_forward_kernel_fp16_11x11_13x13_32,
                    blocks, threads, 0, stream,
                    query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dilation);
        else if (kernel_size == KERNEL_SIZE_13)
            LAUNCH_DNA_KNS_TILED1113(TILE_13_X, TILE_13_Y, KTILE_13_X, KTILE_13_Y, 
                    KERNEL_SIZE_13, NEIGHBORHOOD_SIZE_13, dilation, scalar_t,
                    natten2dqkrpb_cuda_forward_kernel_fp16_11x11_13x13_32,
                    blocks, threads, 0, stream,
                    query_a, key_a, rpb_a, attn_a, height, width, batch_size, heads, dilation);
    }));
    return attn;
}

std::vector<torch::Tensor> natten2dqkrpb_cuda_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t dim = query.size(4);
    int kernel_size_sq = d_attn.size(4);
    int kernel_size = sqrt(kernel_size_sq);
    CHECK_FEATMAP(height, width, kernel_size, dilation);
    CHECK_KERNELSIZE("natten2dqkrpb_cuda_backward", kernel_size);
    int64_t RPB_MAX = kernel_size * 2 - 1;
   
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    at::Tensor d_rpb;
    if (biasEnabled)
        d_rpb = torch::zeros({heads, RPB_MAX, RPB_MAX}, d_attn.options());

    int32_t n_rpb = heads * height * width * kernel_size_sq;
    int blocks_rpb = GET_BLOCKS(n_rpb, CUDA_NUM_THREADS_RPB);
    dim3 grid_rpb(blocks_rpb);
    dim3 blockr(CUDA_NUM_THREADS_RPB);
    int32_t n_query = d_query.numel();
    int blocks_query = GET_BLOCKS(n_query, CUDA_NUM_THREADS_Q);
    dim3 grid_query(blocks_query);
    dim3 blockq(CUDA_NUM_THREADS_Q);
    int32_t n_key = d_key.numel();
    int blocks_key = GET_BLOCKS(n_key, CUDA_NUM_THREADS_K);
    dim3 grid_key(blocks_key);
    dim3 blockk(CUDA_NUM_THREADS_K);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(d_query.scalar_type(), "natten2dqkrpb_backward_cuda", ([&] {
        const auto d_attn_a = d_attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto query_a = query.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        auto d_query_a = d_query.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        auto d_key_a = d_key.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (biasEnabled) {
            auto d_rpb_a = d_rpb.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
            LAUNCH_DNA_KNS(kernel_size, dilation, natten2drpb_cuda_backward_kernel, grid_rpb, blockr, 0, stream,
                    d_rpb_a, d_attn_a, height, width, dilation, batch_size, d_rpb.numel(), n_rpb);
        }
        LAUNCH_DNA_KNS(kernel_size, dilation, natten2dq_cuda_backward_kernel_fp32, grid_query, blockq, 0, stream,
                d_query_a, d_attn_a, key_a, height, width, heads, dilation, dim, n_query);
        LAUNCH_DNA_KNS(kernel_size, dilation, natten2dk_cuda_backward_kernel_fp32, grid_key, blockk, 0, stream,
                d_key_a, d_attn_a, query_a, height, width, heads, dilation, dim, n_key);
    }));
    return {d_query, d_key, d_rpb};
}

std::vector<torch::Tensor> natten2dqkrpb_cuda_backward_fp16(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t dimhalf = query.size(4) / 2;
    TORCH_CHECK(dimhalf*2 == query.size(4), "Dims per head must be an even number in FP16.");
    int64_t kernel_size_sq = d_attn.size(4);
    int kernel_size = sqrt(kernel_size_sq);
    CHECK_FEATMAP(height, width, kernel_size, dilation);
    CHECK_KERNELSIZE("natten2dqkrpb_cuda_backward_fp16", kernel_size);
    int64_t RPB_MAX = kernel_size * 2 - 1;
   
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    at::Tensor d_rpb;
    if (biasEnabled)
        d_rpb = torch::zeros({heads, RPB_MAX, RPB_MAX}, d_attn.options());

    int32_t n_rpb = heads * height * width * kernel_size_sq;
    int blocks_rpb = GET_BLOCKS(n_rpb, CUDA_NUM_THREADS_RPB16);
    dim3 grid_rpb(blocks_rpb);
    dim3 blockr(CUDA_NUM_THREADS_RPB16);
    int32_t nhalf_query = d_query.numel() / 2;
    int blocks_query = GET_BLOCKS(nhalf_query, CUDA_NUM_THREADS_Q16);
    dim3 grid_query(blocks_query);
    dim3 blockq(CUDA_NUM_THREADS_Q16);
    int32_t nhalf_key = d_key.numel() / 2;
    int blocks_key = GET_BLOCKS(nhalf_key, CUDA_NUM_THREADS_K16);
    dim3 grid_key(blocks_key);
    dim3 blockk(CUDA_NUM_THREADS_K16);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_HALF_TYPES(at::kHalf, d_query.scalar_type(), "natten2dqkrpb_backward_cuda_fp16", ([&] {
        const auto d_attn_a = d_attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto query_a = query.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        auto d_query_a = d_query.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        auto d_key_a = d_key.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (biasEnabled) {
            auto d_rpb_a = d_rpb.packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
            LAUNCH_DNA_KNS(kernel_size, dilation, natten2drpb_cuda_backward_kernel_fp16, grid_rpb, blockr, 0, stream,
                    d_rpb_a, d_attn_a, height, width, dilation, batch_size, d_rpb.numel(), n_rpb);
        }
        LAUNCH_DNA_KNS(kernel_size, dilation, natten2dq_cuda_backward_kernel_fp16, grid_query, blockq, 0, stream,
                d_query_a, d_attn_a, key_a, height, width, heads, dilation, dimhalf, nhalf_query);
        LAUNCH_DNA_KNS(kernel_size, dilation, natten2dk_cuda_backward_kernel_fp16, grid_key, blockk, 0, stream,
                d_key_a, d_attn_a, query_a, height, width, heads, dilation, dimhalf, nhalf_key);
    }));
    return {d_query, d_key, d_rpb};
}
} // namespace natten
