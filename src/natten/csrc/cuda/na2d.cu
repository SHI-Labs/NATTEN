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
    \brief Neighborhood Attention 2D - CUDA interface
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
#include "pointwise_neighborhood_2d.cu"
#include "neighborhood_neighborhood_2d.cu"
#include "inverse_neighborhood_2d.cu"
#include "rel_pos_bias_2d.cu"

#define CUDA_NUM_THREADS_Q 512
#define CUDA_NUM_THREADS_K 512
#define CUDA_NUM_THREADS_RPB 64
#define CUDA_NUM_THREADS_Q16 512
#define CUDA_NUM_THREADS_K16 256
#define CUDA_NUM_THREADS_RPB16 64
#define CUDA_NUM_THREADS_F 512
#define CUDA_NUM_THREADS_FP16 512
#define CUDA_NUM_THREADS_V 512
#define CUDA_NUM_THREADS_V16 256

namespace natten {

torch::Tensor natten2dqkrpb_cuda_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const at::optional<at::Tensor> &rpb,
    const int kernel_size,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t dim = query.size(4);
    int zsize = batch_size * heads;
    int xsize = height * width;
    const int kernel_size_sq = pow(kernel_size, 2);
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
        auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (rpb.has_value()) {
            const auto rpb_a = rpb.value().packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
            LAUNCH_DNA_KNS(kernel_size, dilation, 
                    pointwise_neighborhood_2d_bias, 
                    blocks, threads, 0, stream, 
                    query_a, key_a, rpb_a, attn_a, 
                    height, width, batch_size, heads, dilation, dim);
        } else {
            LAUNCH_DNA_KNS(kernel_size, dilation, 
                    pointwise_neighborhood_2d, 
                    blocks, threads, 0, stream, 
                    query_a, key_a, attn_a, 
                    height, width, batch_size, heads, dilation, dim);
        }
    }));
    return attn;
}

torch::Tensor natten2dqkrpb_cuda_forward_fp16(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const at::optional<at::Tensor> &rpb,
    const int kernel_size,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t dimhalf = query.size(4) / 2;
    int zsize = batch_size * heads;
    int xsize = height * width;
    const int kernel_size_sq = pow(kernel_size, 2);
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
        auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (rpb.has_value()) {
            const auto rpb_a = rpb.value().packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
            LAUNCH_DNA_KNS(kernel_size, dilation, 
                    pointwise_neighborhood_2d_bias_fp16, 
                    blocks, threads, 0, stream, 
                    query_a, key_a, rpb_a, attn_a, 
                    height, width, batch_size, heads, dilation, dimhalf);
        } else {
            LAUNCH_DNA_KNS(kernel_size, dilation, 
                    pointwise_neighborhood_2d_fp16, 
                    blocks, threads, 0, stream, 
                    query_a, key_a, attn_a, 
                    height, width, batch_size, heads, dilation, dimhalf);
        }
    }));
    return attn;
}

torch::Tensor natten2dqkrpb_cuda_forward_tiled_32(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const at::optional<at::Tensor> &rpb,
    const int kernel_size,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t dim = query.size(4);
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
        auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (rpb.has_value()) {
            const auto rpb_a = rpb.value().packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
            if (kernel_size == KERNEL_SIZE_7)
                LAUNCH_DNA_KNS_TILED79(TILE_7, KTILE_7, KERNEL_SIZE_7, NEIGHBORHOOD_SIZE_7, dilation,
                        pointwise_neighborhood_2d_bias_7x7_9x9_32, 
                        blocks, threads, 0, stream, 
                        query_a, key_a, rpb_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_9)
                LAUNCH_DNA_KNS_TILED79(TILE_9, KTILE_9, KERNEL_SIZE_9, NEIGHBORHOOD_SIZE_9, dilation,
                        pointwise_neighborhood_2d_bias_7x7_9x9_32, 
                        blocks, threads, 0, stream, 
                        query_a, key_a, rpb_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_3)
                LAUNCH_DNA_DS(dilation, 
                        pointwise_neighborhood_2d_bias_3x3_32,
                        blocks, threads, 0, stream,
                        query_a, key_a, rpb_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_5)
                LAUNCH_DNA_DS(dilation, 
                        pointwise_neighborhood_2d_bias_5x5_32,
                        blocks, threads, 0, stream,
                        query_a, key_a, rpb_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_11)
                LAUNCH_DNA_KNS_TILED1113(TILE_11_X, TILE_11_Y, KTILE_11_X, KTILE_11_Y, 
                        KERNEL_SIZE_11, NEIGHBORHOOD_SIZE_11, dilation, scalar_t,
                        pointwise_neighborhood_2d_bias_11x11_13x13_32,
                        blocks, threads, 0, stream,
                        query_a, key_a, rpb_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_13)
                LAUNCH_DNA_KNS_TILED1113(TILE_13_X, TILE_13_Y, KTILE_13_X, KTILE_13_Y, 
                        KERNEL_SIZE_13, NEIGHBORHOOD_SIZE_13, dilation, float,
                        pointwise_neighborhood_2d_bias_11x11_13x13_32,
                        blocks, threads, 0, stream,
                        query_a, key_a, rpb_a, attn_a, 
                        height, width, batch_size, heads, dilation);
        } else {
            if (kernel_size == KERNEL_SIZE_7)
                LAUNCH_DNA_KNS_TILED79(TILE_7, KTILE_7, KERNEL_SIZE_7, NEIGHBORHOOD_SIZE_7, dilation,
                        pointwise_neighborhood_2d_7x7_9x9_32, 
                        blocks, threads, 0, stream, 
                        query_a, key_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_9)
                LAUNCH_DNA_KNS_TILED79(TILE_9, KTILE_9, KERNEL_SIZE_9, NEIGHBORHOOD_SIZE_9, dilation,
                        pointwise_neighborhood_2d_7x7_9x9_32, 
                        blocks, threads, 0, stream, 
                        query_a, key_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_3)
                LAUNCH_DNA_DS(dilation, 
                        pointwise_neighborhood_2d_3x3_32,
                        blocks, threads, 0, stream,
                        query_a, key_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_5)
                LAUNCH_DNA_DS(dilation, 
                        pointwise_neighborhood_2d_5x5_32,
                        blocks, threads, 0, stream,
                        query_a, key_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_11)
                LAUNCH_DNA_KNS_TILED1113(TILE_11_X, TILE_11_Y, KTILE_11_X, KTILE_11_Y, 
                        KERNEL_SIZE_11, NEIGHBORHOOD_SIZE_11, dilation, scalar_t,
                        pointwise_neighborhood_2d_11x11_13x13_32,
                        blocks, threads, 0, stream,
                        query_a, key_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_13)
                LAUNCH_DNA_KNS_TILED1113(TILE_13_X, TILE_13_Y, KTILE_13_X, KTILE_13_Y, 
                        KERNEL_SIZE_13, NEIGHBORHOOD_SIZE_13, dilation, float,
                        pointwise_neighborhood_2d_11x11_13x13_32,
                        blocks, threads, 0, stream,
                        query_a, key_a, attn_a, 
                        height, width, batch_size, heads, dilation);
        }
    }));
    return attn;
}

torch::Tensor natten2dqkrpb_cuda_forward_fp16_tiled_32(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const at::optional<at::Tensor> &rpb,
    const int kernel_size,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t dimhalf = query.size(4) / 2;
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
        auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (rpb.has_value()) {
            const auto rpb_a = rpb.value().packed_accessor32<scalar_t,3,torch::DefaultPtrTraits>();
            if (kernel_size == KERNEL_SIZE_7)
                LAUNCH_DNA_KNS_TILED79(TILE_7, KTILE_7, KERNEL_SIZE_7, NEIGHBORHOOD_SIZE_7, dilation,
                        pointwise_neighborhood_2d_bias_fp16_7x7_9x9_32, 
                        blocks, threads, 0, stream, 
                        query_a, key_a, rpb_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_9)
                LAUNCH_DNA_KNS_TILED79(TILE_9, KTILE_9, KERNEL_SIZE_9, NEIGHBORHOOD_SIZE_9, dilation,
                        pointwise_neighborhood_2d_bias_fp16_7x7_9x9_32, 
                        blocks, threads, 0, stream, 
                        query_a, key_a, rpb_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_3)
                LAUNCH_DNA_DS(dilation, 
                        pointwise_neighborhood_2d_bias_fp16_3x3_32,
                        blocks, threads, 0, stream,
                        query_a, key_a, rpb_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_5)
                LAUNCH_DNA_DS(dilation, 
                        pointwise_neighborhood_2d_bias_fp16_5x5_32,
                        blocks, threads, 0, stream,
                        query_a, key_a, rpb_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_11)
                LAUNCH_DNA_KNS_TILED1113(TILE_11_X, TILE_11_Y, KTILE_11_X, KTILE_11_Y, 
                        KERNEL_SIZE_11, NEIGHBORHOOD_SIZE_11, dilation, scalar_t,
                        pointwise_neighborhood_2d_bias_fp16_11x11_13x13_32,
                        blocks, threads, 0, stream,
                        query_a, key_a, rpb_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_13)
                LAUNCH_DNA_KNS_TILED1113(TILE_13_X, TILE_13_Y, KTILE_13_X, KTILE_13_Y, 
                        KERNEL_SIZE_13, NEIGHBORHOOD_SIZE_13, dilation, scalar_t,
                        pointwise_neighborhood_2d_bias_fp16_11x11_13x13_32,
                        blocks, threads, 0, stream,
                        query_a, key_a, rpb_a, attn_a, 
                        height, width, batch_size, heads, dilation);
        } else {
            if (kernel_size == KERNEL_SIZE_7)
                LAUNCH_DNA_KNS_TILED79(TILE_7, KTILE_7, KERNEL_SIZE_7, NEIGHBORHOOD_SIZE_7, dilation,
                        pointwise_neighborhood_2d_fp16_7x7_9x9_32, 
                        blocks, threads, 0, stream, 
                        query_a, key_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_9)
                LAUNCH_DNA_KNS_TILED79(TILE_9, KTILE_9, KERNEL_SIZE_9, NEIGHBORHOOD_SIZE_9, dilation,
                        pointwise_neighborhood_2d_fp16_7x7_9x9_32, 
                        blocks, threads, 0, stream, 
                        query_a, key_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_3)
                LAUNCH_DNA_DS(dilation, 
                        pointwise_neighborhood_2d_fp16_3x3_32,
                        blocks, threads, 0, stream,
                        query_a, key_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_5)
                LAUNCH_DNA_DS(dilation, 
                        pointwise_neighborhood_2d_fp16_5x5_32,
                        blocks, threads, 0, stream,
                        query_a, key_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_11)
                LAUNCH_DNA_KNS_TILED1113(TILE_11_X, TILE_11_Y, KTILE_11_X, KTILE_11_Y, 
                        KERNEL_SIZE_11, NEIGHBORHOOD_SIZE_11, dilation, scalar_t,
                        pointwise_neighborhood_2d_fp16_11x11_13x13_32,
                        blocks, threads, 0, stream,
                        query_a, key_a, attn_a, 
                        height, width, batch_size, heads, dilation);
            else if (kernel_size == KERNEL_SIZE_13)
                LAUNCH_DNA_KNS_TILED1113(TILE_13_X, TILE_13_Y, KTILE_13_X, KTILE_13_Y, 
                        KERNEL_SIZE_13, NEIGHBORHOOD_SIZE_13, dilation, scalar_t,
                        pointwise_neighborhood_2d_fp16_11x11_13x13_32,
                        blocks, threads, 0, stream,
                        query_a, key_a, attn_a, 
                        height, width, batch_size, heads, dilation);
        }
    }));
    return attn;
}

std::vector<torch::Tensor> natten2dqkrpb_cuda_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled,
    const int kernel_size,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t dim = query.size(4);
    CHECK_FEATMAP(height, width, kernel_size, dilation);
    CHECK_KERNELSIZE("natten2dqkrpb_cuda_backward", kernel_size);
    const int kernel_size_sq = pow(kernel_size, 2);
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
            LAUNCH_DNA_KNS(kernel_size, dilation, 
                    rel_pos_bias_gradient_2d, 
                    grid_rpb, blockr, 0, stream,
                    d_rpb_a, d_attn_a, 
                    height, width, dilation, batch_size, d_rpb.numel(), n_rpb);
        }
        LAUNCH_DNA_KNS(kernel_size, dilation, 
                neighborhood_neighborhood_2d, 
                grid_query, blockq, 0, stream,
                d_attn_a, key_a, d_query_a, 
                height, width, heads, dilation, dim, n_query);
        LAUNCH_DNA_KNS(kernel_size, dilation, 
                inverse_neighborhood_2d, 
                grid_key, blockk, 0, stream,
                d_attn_a, query_a, d_key_a, 
                height, width, heads, dilation, dim, n_key);
    }));
    return {d_query, d_key, d_rpb};
}

std::vector<torch::Tensor> natten2dqkrpb_cuda_backward_fp16(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled,
    const int kernel_size,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t height = query.size(2);
    int64_t width = query.size(3);
    int64_t dimhalf = query.size(4) / 2;
    TORCH_CHECK(dimhalf*2 == query.size(4), "Dims per head must be an even number in FP16.");
    CHECK_FEATMAP(height, width, kernel_size, dilation);
    CHECK_KERNELSIZE("natten2dqkrpb_cuda_backward_fp16", kernel_size);
    const int kernel_size_sq = pow(kernel_size, 2);
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
            LAUNCH_DNA_KNS(kernel_size, dilation, 
                    rel_pos_bias_gradient_2d_fp16, 
                    grid_rpb, blockr, 0, stream,
                    d_rpb_a, d_attn_a, 
                    height, width, dilation, batch_size, d_rpb.numel(), n_rpb);
        }
        LAUNCH_DNA_KNS(kernel_size, dilation, 
                neighborhood_neighborhood_2d_fp16, 
                grid_query, blockq, 0, stream,
                d_attn_a, key_a, d_query_a, 
                height, width, heads, dilation, dimhalf, nhalf_query);
        LAUNCH_DNA_KNS(kernel_size, dilation, 
                inverse_neighborhood_2d_fp16, 
                grid_key, blockk, 0, stream,
                d_attn_a, query_a, d_key_a, 
                height, width, heads, dilation, dimhalf, nhalf_key);
    }));
    return {d_query, d_key, d_rpb};
}

torch::Tensor natten2dav_cuda_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size,
    const int dilation) {
    int batch_size = value.size(0);
    int heads = value.size(1);
    int height = value.size(2);
    int width = value.size(3);
    int dim = value.size(4);
    const int kernel_size_sq = pow(kernel_size, 2);
    CHECK_FEATMAP(height, width, kernel_size, dilation);
    CHECK_KERNELSIZE("natten2dav_cuda_forward", kernel_size);

    auto out = torch::zeros_like(value);

    int32_t n = out.numel();
    int blocks = GET_BLOCKS(n, CUDA_NUM_THREADS_F);
    dim3 grid(blocks);
    dim3 block(CUDA_NUM_THREADS_F);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "natten2dav_forward_cuda", ([&] {
        const auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        auto out_a = out.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS(
                kernel_size, dilation, 
                neighborhood_neighborhood_2d, 
                grid, block, 0, stream, 
                attn_a, value_a, out_a, 
                height, width, heads, dilation, dim, n);
    }));
    return out;
}

torch::Tensor natten2dav_cuda_forward_fp16(
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size,
    const int dilation) {
    int batch_size = value.size(0);
    int heads = value.size(1);
    int height = value.size(2);
    int width = value.size(3);
    int dimhalf = value.size(4) / 2;
    TORCH_CHECK(dimhalf*2 == value.size(4), "Dims per head must be an even number in FP16.");
    const int kernel_size_sq = pow(kernel_size, 2);
    CHECK_FEATMAP(height, width, kernel_size, dilation);
    CHECK_KERNELSIZE("natten2dav_cuda_forward_fp16", kernel_size);

    auto out = torch::zeros_like(value);

    int32_t nhalf = out.numel() / 2;
    int blocks = GET_BLOCKS(nhalf, CUDA_NUM_THREADS_FP16);
    dim3 grid(blocks);
    dim3 block(CUDA_NUM_THREADS_FP16);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_HALF_TYPES(at::kHalf, value.scalar_type(), "natten2dav_forward_cuda_fp16", ([&] {
        const auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        auto out_a = out.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS(kernel_size, dilation, 
                neighborhood_neighborhood_2d_fp16, 
                grid, block, 0, stream, 
                attn_a, value_a, out_a, 
                height, width, heads, dilation, dimhalf, nhalf);
    }));
    return out;
}

std::vector<torch::Tensor> natten2dav_cuda_backward_tiled_32(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size,
    const int dilation) {
    int64_t batch_size = value.size(0);
    int64_t heads = value.size(1);
    int64_t height = value.size(2);
    int64_t width = value.size(3);
    int64_t dim = value.size(4);
    int xsize = width * kernel_size;
    int ysize = height * kernel_size;
    int zsize = batch_size * heads;
    const int kernel_size_sq = pow(kernel_size, 2);
    CHECK_FEATMAP(height, width, kernel_size, dilation);
    TORCH_CHECK(dim == DIM_32, "natten2dav_cuda_backward_tiled_32", " only supports 32-dim attention heads.");
    TORCH_CHECK(kernel_size == KERNEL_SIZE_7 || kernel_size == KERNEL_SIZE_3 ||  kernel_size == KERNEL_SIZE_5 ||
            kernel_size == KERNEL_SIZE_9 || kernel_size == KERNEL_SIZE_11 || kernel_size == KERNEL_SIZE_13,
            "natten2dav_cuda_backward_tiled_32", " only supports kernel sizes 3, 5, 7, 9, 11, and 13.");

    auto d_attn = torch::zeros_like(attn);
    auto d_value = torch::zeros_like(value);
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
    const dim3 attn_blocks(
            (xsize + XTHREADS*dilation - 1) / XTHREADS,
            (ysize + YTHREADS*dilation - 1) / YTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 attn_threads(XTHREADS, YTHREADS, BATCHTHREADS);

    int32_t n_value = d_value.numel();
    int blocks_value = GET_BLOCKS(n_value, CUDA_NUM_THREADS_V);
    dim3 grid_value(blocks_value);
    dim3 block(CUDA_NUM_THREADS_V);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(d_attn.scalar_type(), "natten2dav_cuda_backward_tiled_32", ([&] {
        auto d_attn_a = d_attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        auto d_value_a = d_value.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto d_out_a = d_out.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (kernel_size == KERNEL_SIZE_7)
        {
            LAUNCH_DNA_KNS_TILED79(TILE_7, KTILE_7, KERNEL_SIZE_7, NEIGHBORHOOD_SIZE_7, dilation,
                    pointwise_neighborhood_2d_7x7_9x9_32, 
                    attn_blocks, attn_threads, 0, stream,
                    d_out_a, value_a, d_attn_a, 
                    height, width, batch_size, heads, dilation);
            _IN_LAUNCH_DNA_KNS(KERNEL_SIZE_7, NEIGHBORHOOD_SIZE_7, dilation, inverse_neighborhood_2d, grid_value,
                    block, 0, stream, 
                    attn_a, d_out_a, d_value_a, 
                    height, width, heads, dilation, dim, n_value);
        }
        else if (kernel_size == KERNEL_SIZE_9)
        {
            LAUNCH_DNA_KNS_TILED79(TILE_9, KTILE_9, KERNEL_SIZE_9, NEIGHBORHOOD_SIZE_9, dilation,
                    pointwise_neighborhood_2d_7x7_9x9_32, 
                    attn_blocks, attn_threads, 0, stream, 
                    d_out_a, value_a, d_attn_a, 
                    height, width, batch_size, heads, dilation);
            _IN_LAUNCH_DNA_KNS(KERNEL_SIZE_9, NEIGHBORHOOD_SIZE_9, dilation, inverse_neighborhood_2d, grid_value,
                    block, 0, stream, 
                    attn_a, d_out_a, d_value_a, 
                    height, width, heads, dilation, dim, n_value);
        }
        else if (kernel_size == KERNEL_SIZE_3)
        {
            LAUNCH_DNA_DS(dilation, pointwise_neighborhood_2d_3x3_32, 
                    attn_blocks, attn_threads, 0, stream, 
                    d_out_a, value_a, d_attn_a, 
                    height, width, batch_size, heads, dilation);
            _IN_LAUNCH_DNA_KNS(KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation, inverse_neighborhood_2d, grid_value,
                    block, 0, stream, 
                    attn_a, d_out_a, d_value_a, 
                    height, width, heads, dilation, dim, n_value);
        }
        else if (kernel_size == KERNEL_SIZE_5)
        {
            LAUNCH_DNA_DS(dilation, pointwise_neighborhood_2d_5x5_32, 
                    attn_blocks, attn_threads, 0, stream, 
                    d_out_a, value_a, d_attn_a, 
                    height, width, batch_size, heads, dilation);
            _IN_LAUNCH_DNA_KNS(KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation, inverse_neighborhood_2d, grid_value,
                    block, 0, stream, 
                    attn_a, d_out_a, d_value_a, 
                    height, width, heads, dilation, dim, n_value);
        }
        else if (kernel_size == KERNEL_SIZE_11)
        {
            LAUNCH_DNA_KNS_TILED1113(TILE_11_X, TILE_11_Y, KTILE_11_X, KTILE_11_Y, 
                    KERNEL_SIZE_11, NEIGHBORHOOD_SIZE_11, dilation, scalar_t,
                    pointwise_neighborhood_2d_11x11_13x13_32, 
                    attn_blocks, attn_threads, 0, stream,
                    d_out_a, value_a, d_attn_a, 
                    height, width, batch_size, heads, dilation);
            _IN_LAUNCH_DNA_KNS(KERNEL_SIZE_11, NEIGHBORHOOD_SIZE_11, dilation, inverse_neighborhood_2d, grid_value,
                    block, 0, stream, 
                    attn_a, d_out_a, d_value_a, 
                    height, width, heads, dilation, dim, n_value);
        }
        else if (kernel_size == KERNEL_SIZE_13)
        {
            LAUNCH_DNA_KNS_TILED1113(TILE_13_X, TILE_13_Y, KTILE_13_X, KTILE_13_Y, 
                    KERNEL_SIZE_13, NEIGHBORHOOD_SIZE_13, dilation, float,
                    pointwise_neighborhood_2d_11x11_13x13_32, 
                    attn_blocks, attn_threads, 0, stream,
                    d_out_a, value_a, d_attn_a, 
                    height, width, batch_size, heads, dilation);
            _IN_LAUNCH_DNA_KNS(KERNEL_SIZE_13, NEIGHBORHOOD_SIZE_13, dilation, inverse_neighborhood_2d, grid_value,
                    block, 0, stream, 
                    attn_a, d_out_a, d_value_a, 
                    height, width, heads, dilation, dim, n_value);
        }
    }));
    return {d_attn, d_value};
}

std::vector<torch::Tensor> natten2dav_cuda_backward_fp16_tiled_32(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size,
    const int dilation) {
    int64_t batch_size = value.size(0);
    int64_t heads = value.size(1);
    int64_t height = value.size(2);
    int64_t width = value.size(3);
    int64_t dimhalf = value.size(4) / 2;
    TORCH_CHECK(dimhalf*2 == value.size(4), "Dims per head must be an even number in FP16.");
    int xsize = width * kernel_size;
    int ysize = height * kernel_size;
    int zsize = batch_size * heads;
    const int kernel_size_sq = pow(kernel_size, 2);
    CHECK_FEATMAP(height, width, kernel_size, dilation);
    TORCH_CHECK(dimhalf*2 == DIM_32, "natten2dav_cuda_backward_fp16_tiled_32", " only supports 32-dim attention heads.");
    TORCH_CHECK(kernel_size == KERNEL_SIZE_7 || kernel_size == KERNEL_SIZE_3 ||  kernel_size == KERNEL_SIZE_5 ||
            kernel_size == KERNEL_SIZE_9 || kernel_size == KERNEL_SIZE_11 || kernel_size == KERNEL_SIZE_13,
            "natten2dav_cuda_backward_fp16_tiled_32", " only supports kernel sizes 3, 5, 7, 9, 11, and 13.");

    auto d_attn = torch::zeros_like(attn);
    auto d_value = torch::zeros_like(value);
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
    const dim3 attn_blocks(
            (xsize + XTHREADS*dilation - 1) / XTHREADS,
            (ysize + YTHREADS*dilation - 1) / YTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 attn_threads(XTHREADS, YTHREADS, BATCHTHREADS);

    int32_t nhalf_value = d_value.numel() / 2;
    int blocks_value = GET_BLOCKS(nhalf_value, CUDA_NUM_THREADS_V16);
    dim3 grid_value(blocks_value);
    dim3 block(CUDA_NUM_THREADS_V16);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_HALF_TYPES(at::kHalf, d_attn.scalar_type(), "natten2dav_cuda_backward_fp16_tiled_32", ([&] {
        auto d_attn_a = d_attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        auto d_value_a = d_value.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto d_out_a = d_out.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        if (kernel_size == KERNEL_SIZE_7){
            LAUNCH_DNA_KNS_TILED79(TILE_7, KTILE_7, KERNEL_SIZE_7, NEIGHBORHOOD_SIZE_7, dilation,
                    pointwise_neighborhood_2d_fp16_7x7_9x9_32, 
                    attn_blocks, attn_threads, 0, stream, 
                    d_out_a, value_a, d_attn_a, 
                    height, width, batch_size, heads, dilation);
            _IN_LAUNCH_DNA_KNS(KERNEL_SIZE_7, NEIGHBORHOOD_SIZE_7, dilation, 
                    inverse_neighborhood_2d_fp16, 
                    grid_value, block, 0, stream, 
                    attn_a, d_out_a, d_value_a, 
                    height, width, heads, dilation, dimhalf, nhalf_value);
        }
        else if (kernel_size == KERNEL_SIZE_9){
            LAUNCH_DNA_KNS_TILED79(TILE_9, KTILE_9, KERNEL_SIZE_9, NEIGHBORHOOD_SIZE_9, dilation,
                    pointwise_neighborhood_2d_fp16_7x7_9x9_32, 
                    attn_blocks, attn_threads, 0, stream, 
                    d_out_a, value_a, d_attn_a, 
                    height, width, batch_size, heads, dilation);
            _IN_LAUNCH_DNA_KNS(KERNEL_SIZE_9, NEIGHBORHOOD_SIZE_9, dilation, 
                    inverse_neighborhood_2d_fp16, 
                    grid_value, block, 0, stream, 
                    attn_a, d_out_a, d_value_a, 
                    height, width, heads, dilation, dimhalf, nhalf_value);
        }
        else if (kernel_size == KERNEL_SIZE_5){
            LAUNCH_DNA_DS(dilation, pointwise_neighborhood_2d_fp16_5x5_32, 
                    attn_blocks, attn_threads, 0, stream, 
                    d_out_a, value_a, d_attn_a, 
                    height, width, batch_size, heads, dilation);
            _IN_LAUNCH_DNA_KNS(KERNEL_SIZE_5, NEIGHBORHOOD_SIZE_5, dilation, 
                    inverse_neighborhood_2d_fp16, 
                    grid_value, block, 0, stream, 
                    attn_a, d_out_a, d_value_a, 
                    height, width, heads, dilation, dimhalf, nhalf_value);
        }
        else if (kernel_size == KERNEL_SIZE_3){
            LAUNCH_DNA_DS(dilation, pointwise_neighborhood_2d_fp16_3x3_32,
                    attn_blocks, attn_threads, 0, stream,
                    d_out_a, value_a, d_attn_a, 
                    height, width, batch_size, heads, dilation);
            _IN_LAUNCH_DNA_KNS(KERNEL_SIZE_3, NEIGHBORHOOD_SIZE_3, dilation, 
                    inverse_neighborhood_2d_fp16, 
                    grid_value, block, 0, stream, 
                    attn_a, d_out_a, d_value_a, 
                    height, width, heads, dilation, dimhalf, nhalf_value);
        }
        else if (kernel_size == KERNEL_SIZE_11){
            LAUNCH_DNA_KNS_TILED1113(TILE_11_X, TILE_11_Y, KTILE_11_X, KTILE_11_Y, 
                    KERNEL_SIZE_11, NEIGHBORHOOD_SIZE_11, dilation, scalar_t,
                    pointwise_neighborhood_2d_fp16_11x11_13x13_32, 
                    attn_blocks, attn_threads, 0, stream,
                    d_out_a, value_a, d_attn_a, 
                    height, width, batch_size, heads, dilation);
            _IN_LAUNCH_DNA_KNS(KERNEL_SIZE_11, NEIGHBORHOOD_SIZE_11, dilation, 
                    inverse_neighborhood_2d_fp16, 
                    grid_value, block, 0, stream, 
                    attn_a, d_out_a, d_value_a, 
                    height, width, heads, dilation, dimhalf, nhalf_value);
        }
        else if (kernel_size == KERNEL_SIZE_13){
            LAUNCH_DNA_KNS_TILED1113(TILE_13_X, TILE_13_Y, KTILE_13_X, KTILE_13_Y, 
                    KERNEL_SIZE_13, NEIGHBORHOOD_SIZE_13, dilation, scalar_t,
                    pointwise_neighborhood_2d_fp16_11x11_13x13_32, 
                    attn_blocks, attn_threads, 0, stream,
                    d_out_a, value_a, d_attn_a, 
                    height, width, batch_size, heads, dilation);
            _IN_LAUNCH_DNA_KNS(KERNEL_SIZE_13, NEIGHBORHOOD_SIZE_13, dilation, 
                    inverse_neighborhood_2d_fp16, 
                    grid_value, block, 0, stream, 
                    attn_a, d_out_a, d_value_a, 
                    height, width, heads, dilation, dimhalf, nhalf_value);
        }
    }));
    return {d_attn, d_value};
}

std::vector<torch::Tensor> natten2dav_cuda_backward(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size,
    const int dilation) {
    int64_t batch_size = value.size(0);
    int64_t heads = value.size(1);
    int64_t height = value.size(2);
    int64_t width = value.size(3);
    int64_t dim = value.size(4);
    int zsize = batch_size * heads;
    int xsize = height * width;
    const int kernel_size_sq = pow(kernel_size, 2);
    CHECK_FEATMAP(height, width, kernel_size, dilation);
    CHECK_KERNELSIZE("natten2dav_cuda_backward", kernel_size);
    int KERNELTHREADS = min(CUDA_NUM_THREADS, kernel_size_sq);
    int PIXELTHREADS = min(int(CUDA_NUM_THREADS / KERNELTHREADS), xsize);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (PIXELTHREADS * KERNELTHREADS));

    auto d_attn = torch::zeros_like(attn);
    auto d_value = torch::zeros_like(value);

    const dim3 attn_blocks(
            (xsize + PIXELTHREADS - 1) / PIXELTHREADS,
            (kernel_size_sq + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 attn_threads(PIXELTHREADS, KERNELTHREADS, BATCHTHREADS);
    int32_t n_value = d_value.numel();
    int blocks_value = GET_BLOCKS(n_value);
    dim3 grid_value(blocks_value);
    dim3 block(CUDA_NUM_THREADS);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(d_attn.scalar_type(), "natten2dav_backward_cuda", ([&] {
        auto d_attn_a = d_attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        auto d_value_a = d_value.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto d_out_a = d_out.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS(kernel_size, dilation, 
                pointwise_neighborhood_2d, 
                attn_blocks, attn_threads, 0, stream, 
                d_out_a, value_a, d_attn_a, 
                height, width, batch_size, heads, dilation, dim);
        LAUNCH_DNA_KNS(kernel_size, dilation, 
                inverse_neighborhood_2d, grid_value, block, 0, stream, 
                attn_a, d_out_a, d_value_a, 
                height, width, heads, dilation, dim, n_value);
    }));
    return {d_attn, d_value};
}

std::vector<torch::Tensor> natten2dav_cuda_backward_fp16(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size,
    const int dilation) {
    int64_t batch_size = value.size(0);
    int64_t heads = value.size(1);
    int64_t height = value.size(2);
    int64_t width = value.size(3);
    int64_t dimhalf = value.size(4) / 2;
    TORCH_CHECK(dimhalf*2 == value.size(4), "Dims per head must be an even number in FP16.");
    int zsize = batch_size * heads;
    int xsize = height * width;
    const int kernel_size_sq = pow(kernel_size, 2);
    CHECK_FEATMAP(height, width, kernel_size, dilation);
    CHECK_KERNELSIZE("natten2dav_cuda_backward_fp16", kernel_size);
    int KERNELTHREADS = min(CUDA_NUM_THREADS, kernel_size_sq);
    int PIXELTHREADS = min(int(CUDA_NUM_THREADS / KERNELTHREADS), xsize);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (PIXELTHREADS * KERNELTHREADS));

    auto d_attn = torch::zeros_like(attn);
    auto d_value = torch::zeros_like(value);

    const dim3 attn_blocks(
            (xsize + PIXELTHREADS - 1) / PIXELTHREADS,
            (kernel_size_sq + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 attn_threads(PIXELTHREADS, KERNELTHREADS, BATCHTHREADS);
    int32_t nhalf_value = d_value.numel() / 2;
    int blocks_value = GET_BLOCKS(nhalf_value);
    dim3 grid_value(blocks_value);
    dim3 block(CUDA_NUM_THREADS);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_HALF_TYPES(at::kHalf, d_attn.scalar_type(), "natten2dav_backward_cuda_fp16", ([&] {
        auto d_attn_a = d_attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        auto d_value_a = d_value.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto d_out_a = d_out.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        const auto attn_a = attn.packed_accessor32<scalar_t,5,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS(kernel_size, dilation, 
                pointwise_neighborhood_2d_fp16, 
                attn_blocks, attn_threads, 0, stream, 
                d_out_a, value_a, d_attn_a,  
                height, width, batch_size, heads, dilation, dimhalf);
        LAUNCH_DNA_KNS(kernel_size, dilation, 
                inverse_neighborhood_2d_fp16, 
                grid_value, block, 0, stream, 
                d_out_a, d_value_a, attn_a, 
                height, width, heads, dilation, dimhalf, nhalf_value);
    }));
    return {d_attn, d_value};
}

} // namespace natten
