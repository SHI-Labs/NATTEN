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
    \brief Inverse-Neighborhood-Neighborhood kernel for 2D data.
           Applies inverse neighborhood attention weights to inverse neighborhood values.
           Used to compute key and value grads.
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

template<class scalar_t>
using Tensor5D = typename torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits>;

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void inverse_neighborhood_2d(           // K-grad / V-grad
    const Tensor5D<scalar_t> weights,              // d_attn / attn
    const Tensor5D<scalar_t> values,               // query  / d_out
    Tensor5D<scalar_t> output,                     // d_key  / d_value
    const int height,
    const int width,
    const int heads,
    const int dilation_in,
    const int dim,
    const int output_numel) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < output_numel){
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
        const int weightsOffset = b * weights.stride(0) + h * weights.stride(1);
        const int outOffset = b * values.stride(0) + h * values.stride(1) + d;
        scalar_t output_update = scalar_t(0);
        #pragma unroll
        for (int xi=ni; xi < ei; xi+=dilation){
            const int oni = get_window_start(xi, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            #pragma unroll
            for (int xj=nj; xj < ej; xj+=dilation){
                const int onj = get_window_start(xj, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int outIndex = outOffset + xi * values.stride(2) + xj * values.stride(3);
                const int weightsIndex = weightsOffset + xi * weights.stride(2) + xj * weights.stride(3) + int((i-oni)/dilation)*KERNEL_SIZE+int((j-onj)/dilation);
                output_update += values.data()[outIndex] * weights.data()[weightsIndex];
            }
        }
        output.data()[linearIndex] = output_update;
    }
}

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void inverse_neighborhood_2d_fp16(      // K-grad / V-grad
    const Tensor5D<scalar_t> weights,              // d_attn / attn
    const Tensor5D<scalar_t> values,               // query  / d_out
    Tensor5D<scalar_t> output,                     // d_key  / d_value
    const int height,
    const int width,
    const int heads,
    const int dilation_in,
    const int dimhalf,
    const int output_numel) {
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    const int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (linearIndex < output_numel){
        __half2* values2 = reinterpret_cast<__half2*>(values.data());
        __half2* output2 = reinterpret_cast<__half2*>(output.data());
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
        const int weightsOffset = b * weights.stride(0) + h * weights.stride(1);
        const int stride2 = dimhalf * width;
        const int outOffset = b * (stride2 * height * heads) + h * (stride2 * height) + d;
        __half2 output_update = __float2half2_rn(0.f);
        #pragma unroll
        for (int xi=ni; xi < ei; xi+=dilation){
            const int oni = get_window_start(xi, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            #pragma unroll
            for (int xj=nj; xj < ej; xj+=dilation){
                const int onj = get_window_start(xj, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int outIndex = outOffset + xi * stride2 + xj * dimhalf;
                const int weightsIndex = weightsOffset + xi * weights.stride(2) + xj * weights.stride(3) + int((i-oni)/dilation)*KERNEL_SIZE+int((j-onj)/dilation);
                scalar_t a = weights.data()[weightsIndex];
                output_update = __hfma2(values2[outIndex], __halves2half2(a, a), output_update);
            }
        }
        output2[linearIndex] = output_update;
    }
}

} // namespace natten
