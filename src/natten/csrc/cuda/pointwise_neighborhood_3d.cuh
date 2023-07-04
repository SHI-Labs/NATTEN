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
    \brief Pointwise-Neighborhood kernel for 3D data.
           Computes attention weights between query points and their corresponding
           key neighborhood.
           Extra kernel with fused bias (relative positional bias.)
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

#include "cuda/natten_commons.cuh"

namespace natten {

template<class scalar_t>
using Tensor4D = typename torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits>;
template<class scalar_t>
using Tensor6D = typename torch::PackedTensorAccessor32<scalar_t,6,torch::DefaultPtrTraits>;

template <int KERNEL_SIZE, int KERNEL_SIZE_D, int NEIGHBORHOOD_SIZE, int NEIGHBORHOOD_SIZE_D, typename scalar_t>
__global__ void pointwise_neighborhood_3d_bias( // QK
    const Tensor6D<scalar_t> query,             // query
    const Tensor6D<scalar_t> key,               // key  
    const Tensor4D<scalar_t> bias,              // relative positional bias tensor
    Tensor6D<scalar_t> attn,                    // attn
    const int depth,
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation,
    const int dilation_d,
    const int dim) {
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < depth * height * width){
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (y < KERNEL_SIZE_D * KERNEL_SIZE * KERNEL_SIZE){
                int indtmp1 = y / KERNEL_SIZE;
                const int kk = indtmp1 / KERNEL_SIZE;
                const int kj = y - indtmp1 * KERNEL_SIZE;
                const int ki = indtmp1 - kk * KERNEL_SIZE;

                indtmp1 = x / width;
                const int k = indtmp1 / height;
                const int j = x - indtmp1 * width;
                const int i = indtmp1 - k * height;

                const int b = z / heads;
                const int h = z - b * heads;

                const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int nk = get_window_start(k, depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
                const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int pk = get_pb_start(k, depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);

                scalar_t updt = scalar_t(0);
                const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
                const int queryOffset = batchHeadOffset + k * query.stride(2) + i * query.stride(3) + j * query.stride(4);
                const int keyOffset = batchHeadOffset + (kk*dilation_d+nk) * key.stride(2) + (ki*dilation+ni) * key.stride(3) + (kj*dilation+nj) * key.stride(4);
                #pragma unroll
                for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                    updt += query.data()[queryOffset+dimOffset] * key.data()[keyOffset+dimOffset];
                const int index = b * attn.stride(0) + h * attn.stride(1) + k * attn.stride(2) + i * attn.stride(3) + j * attn.stride(4) + y * attn.stride(5);
                const int biasIndex = h * bias.stride(0) + (pk+kk) * bias.stride(1) + (pi+ki) * bias.stride(2) + (pj+kj) * bias.stride(3);
                updt += bias.data()[biasIndex];
                attn.data()[index] = updt;
            }
        }
    }
}

template <int KERNEL_SIZE, int KERNEL_SIZE_D, int NEIGHBORHOOD_SIZE, int NEIGHBORHOOD_SIZE_D, typename scalar_t>
__global__ void pointwise_neighborhood_3d_bias_fp16( // QK
    const Tensor6D<scalar_t> query,                  // query
    const Tensor6D<scalar_t> key,                    // key  
    const Tensor4D<scalar_t> bias,                   // relative positional bias tensor
    Tensor6D<scalar_t> attn,                         // attn
    const int depth,
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation,
    const int dilation_d,
    const int dimhalf) {
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < depth* height * width){
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (y < KERNEL_SIZE_D * KERNEL_SIZE * KERNEL_SIZE){
                __half2* query2 = reinterpret_cast<__half2*>(query.data());
                __half2* key2 = reinterpret_cast<__half2*>(key.data());
                int indtmp1 = y / KERNEL_SIZE;
                const int kk = indtmp1 / KERNEL_SIZE;
                const int kj = y - indtmp1 * KERNEL_SIZE;
                const int ki = indtmp1 - kk * KERNEL_SIZE;

                indtmp1 = x / width;
                const int k = indtmp1 / height;
                const int j = x - indtmp1 * width;
                const int i = indtmp1 - k * height;

                const int b = z / heads;
                const int h = z - b * heads;

                const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int nk = get_window_start(k, depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
                const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int pk = get_pb_start(k, depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);

                __half2 updt = __float2half2_rn(0.f);
                const int stride3 = dimhalf * width;
                const int stride2 = stride3 * height;
                const int batchHeadOffset = b * (stride2*depth*heads) + h * (stride2*depth);
                const int queryOffset = batchHeadOffset + k * stride2 + i * stride3 + j * dimhalf;
                const int keyOffset = batchHeadOffset + (kk*dilation_d+nk) * stride2 + (ki*dilation+ni) * stride3 + (kj*dilation+nj) * dimhalf;
                #pragma unroll
                for (int dimOffset=0; dimOffset < dimhalf; ++dimOffset)
                    updt = __hfma2(query2[queryOffset+dimOffset], key2[keyOffset+dimOffset], updt);
                const int index = b * attn.stride(0) + h * attn.stride(1) + k * attn.stride(2) + i * attn.stride(3) + j * attn.stride(4) + y * attn.stride(5);
                const int biasIndex = h * bias.stride(0) + (pk+kk) * bias.stride(1) + (pi+ki) * bias.stride(2) + (pj+kj) * bias.stride(3);
                attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y)) + bias.data()[biasIndex];
            }
        }
    }
}

template <int KERNEL_SIZE, int KERNEL_SIZE_D, int NEIGHBORHOOD_SIZE, int NEIGHBORHOOD_SIZE_D, typename scalar_t>
__global__ void pointwise_neighborhood_3d(      // QK
    const Tensor6D<scalar_t> query,             // query
    const Tensor6D<scalar_t> key,               // key  
    Tensor6D<scalar_t> attn,                    // attn
    const int depth,
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation,
    const int dilation_d,
    const int dim) {
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < depth * height * width){
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (y < KERNEL_SIZE_D * KERNEL_SIZE * KERNEL_SIZE){
                int indtmp1 = y / KERNEL_SIZE;
                const int kk = indtmp1 / KERNEL_SIZE;
                const int kj = y - indtmp1 * KERNEL_SIZE;
                const int ki = indtmp1 - kk * KERNEL_SIZE;

                indtmp1 = x / width;
                const int k = indtmp1 / height;
                const int j = x - indtmp1 * width;
                const int i = indtmp1 - k * height;

                const int b = z / heads;
                const int h = z - b * heads;

                const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int nk = get_window_start(k, depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);

                scalar_t updt = scalar_t(0);
                const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
                const int queryOffset = batchHeadOffset + k * query.stride(2) + i * query.stride(3) + j * query.stride(4);
                const int keyOffset = batchHeadOffset + (kk*dilation_d+nk) * key.stride(2) + (ki*dilation+ni) * key.stride(3) + (kj*dilation+nj) * key.stride(4);
                #pragma unroll
                for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                    updt += query.data()[queryOffset+dimOffset] * key.data()[keyOffset+dimOffset];
                const int index = b * attn.stride(0) + h * attn.stride(1) + k * attn.stride(2) + i * attn.stride(3) + j * attn.stride(4) + y * attn.stride(5);
                attn.data()[index] = updt;
            }
        }
    }
}

template <int KERNEL_SIZE, int KERNEL_SIZE_D, int NEIGHBORHOOD_SIZE, int NEIGHBORHOOD_SIZE_D, typename scalar_t>
__global__ void pointwise_neighborhood_3d_fp16(      // QK
    const Tensor6D<scalar_t> query,                  // query
    const Tensor6D<scalar_t> key,                    // key  
    Tensor6D<scalar_t> attn,                         // attn
    const int depth,
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation,
    const int dilation_d,
    const int dimhalf) {
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        if (x < depth* height * width){
            const int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (y < KERNEL_SIZE_D * KERNEL_SIZE * KERNEL_SIZE){
                __half2* query2 = reinterpret_cast<__half2*>(query.data());
                __half2* key2 = reinterpret_cast<__half2*>(key.data());
                int indtmp1 = y / KERNEL_SIZE;
                const int kk = indtmp1 / KERNEL_SIZE;
                const int kj = y - indtmp1 * KERNEL_SIZE;
                const int ki = indtmp1 - kk * KERNEL_SIZE;

                indtmp1 = x / width;
                const int k = indtmp1 / height;
                const int j = x - indtmp1 * width;
                const int i = indtmp1 - k * height;

                const int b = z / heads;
                const int h = z - b * heads;

                const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int nk = get_window_start(k, depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);

                __half2 updt = __float2half2_rn(0.f);
                const int stride3 = dimhalf * width;
                const int stride2 = stride3 * height;
                const int batchHeadOffset = b * (stride2*depth*heads) + h * (stride2*depth);
                const int queryOffset = batchHeadOffset + k * stride2 + i * stride3 + j * dimhalf;
                const int keyOffset = batchHeadOffset + (kk*dilation_d+nk) * stride2 + (ki*dilation+ni) * stride3 + (kj*dilation+nj) * dimhalf;
                #pragma unroll
                for (int dimOffset=0; dimOffset < dimhalf; ++dimOffset)
                    updt = __hfma2(query2[queryOffset+dimOffset], key2[keyOffset+dimOffset], updt);
                const int index = b * attn.stride(0) + h * attn.stride(1) + k * attn.stride(2) + i * attn.stride(3) + j * attn.stride(4) + y * attn.stride(5);
                attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y));
            }
        }
    }
}

} // namespace natten
