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
    \brief Relative positional bias backward pass kernel for 2D data.
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

namespace natten {

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void rel_pos_bias_gradient_2d(
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
__global__ void rel_pos_bias_gradient_2d_fp16(
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

} // namespace natten
