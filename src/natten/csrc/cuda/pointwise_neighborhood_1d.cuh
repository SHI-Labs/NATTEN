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
using Tensor2D = typename torch::PackedTensorAccessor32<scalar_t,2,torch::DefaultPtrTraits>;
template<class scalar_t>
using Tensor4D = typename torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits>;

template <int KS, int NS, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_1d( // QK    / A-grad
    const Tensor4D<scalar_t> query,        // query / d_out
    const Tensor4D<scalar_t> key,          // key   / value
    Tensor4D<scalar_t> attn,               // attn  / d_attn
    const int length,
    const int batch_size,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int dim) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < length){
            const int ki = blockIdx.y * blockDim.y + threadIdx.y;
            if (ki < KERNEL_SIZE){
                const int b = z / heads;
                const int h = z - b * heads;
                const int ni = get_window_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                scalar_t updt = scalar_t(0);
                const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
                const int queryOffset = batchHeadOffset + i * query.stride(2);
                const int keyOffset = batchHeadOffset + (ki*dilation+ni) * key.stride(2);
                #pragma unroll
                for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                    updt += query.data()[queryOffset+dimOffset] * key.data()[keyOffset+dimOffset];
                attn.data()[b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + ki] = updt;
            }
        }
    }
}

template <int KS, int NS, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_1d_fp16( // QK    / A-grad
    const Tensor4D<scalar_t> query,             // query / d_out
    const Tensor4D<scalar_t> key,               // key   / value
    Tensor4D<scalar_t> attn,                    // attn  / d_attn
    const int length,
    const int batch_size,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int dimhalf) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < length){
            const int ki = blockIdx.y * blockDim.y + threadIdx.y;
            if (ki < KERNEL_SIZE){
                __half2* query2 = reinterpret_cast<__half2*>(query.data());
                __half2* key2 = reinterpret_cast<__half2*>(key.data());
                const int b = z / heads;
                const int h = z - b * heads;
                const int ni = get_window_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                __half2 updt = __float2half2_rn(0.f);
                const int batchHeadOffset = b * (dimhalf*length*heads) + h * (dimhalf*length);
                const int queryOffset = batchHeadOffset + i * dimhalf;
                const int keyOffset = batchHeadOffset + (ki*dilation+ni) * dimhalf;
                #pragma unroll
                for (int dimOffset=0; dimOffset < dimhalf; ++dimOffset)
                    updt = __hfma2(query2[queryOffset+dimOffset], key2[keyOffset+dimOffset], updt);
                attn.data()[b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + ki] = static_cast<scalar_t>(__hadd(updt.x, updt.y));
            }
        }
    }
}

template <int KS, int NS, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_1d_bias( // QK   
    const Tensor4D<scalar_t> query,             // query
    const Tensor4D<scalar_t> key,               // key  
    const Tensor2D<scalar_t> bias,              // relative positional bias tensor
    Tensor4D<scalar_t> attn,                    // attn
    const int length,
    const int batch_size,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int dim) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < length){
            const int ki = blockIdx.y * blockDim.y + threadIdx.y;
            if (ki < KERNEL_SIZE){
                const int b = z / heads;
                const int h = z - b * heads;
                const int ni = get_window_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int pi = get_pb_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                scalar_t updt = scalar_t(0);
                const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
                const int queryOffset = batchHeadOffset + i * query.stride(2);
                const int keyOffset = batchHeadOffset + (ki*dilation+ni) * key.stride(2);
                #pragma unroll
                for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                    updt += query.data()[queryOffset+dimOffset] * key.data()[keyOffset+dimOffset];
                const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + ki;
                const int biasIndex = h * bias.stride(0) + (pi+ki) * bias.stride(1);
                updt += bias.data()[biasIndex];
                attn.data()[index] = updt;
            }
        }
    }
}

template <int KS, int NS, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_1d_bias_fp16( // QK   
    const Tensor4D<scalar_t> query,                  // query
    const Tensor4D<scalar_t> key,                    // key  
    const Tensor2D<scalar_t> bias,                   // relative positional bias tensor
    Tensor4D<scalar_t> attn,                         // attn
    const int length,
    const int batch_size,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int dimhalf) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (z < batch_size * heads){
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < length){
            const int ki = blockIdx.y * blockDim.y + threadIdx.y;
            if (ki < KERNEL_SIZE){
                __half2* query2 = reinterpret_cast<__half2*>(query.data());
                __half2* key2 = reinterpret_cast<__half2*>(key.data());
                const int b = z / heads;
                const int h = z - b * heads;
                const int ni = get_window_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int pi = get_pb_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                __half2 updt = __float2half2_rn(0.f);
                const int batchHeadOffset = b * (dimhalf*length*heads) + h * (dimhalf*length);
                const int queryOffset = batchHeadOffset + i * dimhalf;
                const int keyOffset = batchHeadOffset + (ki*dilation+ni) * dimhalf;
                #pragma unroll
                for (int dimOffset=0; dimOffset < dimhalf; ++dimOffset)
                    updt = __hfma2(query2[queryOffset+dimOffset], key2[keyOffset+dimOffset], updt);
                const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + ki;
                const int biasIndex = h * bias.stride(0) + (pi+ki) * bias.stride(1);
                attn.data()[index] = static_cast<scalar_t>(__hadd(updt.x, updt.y)) + bias.data()[biasIndex];
            }
        }
    }
}

} // namespace natten
