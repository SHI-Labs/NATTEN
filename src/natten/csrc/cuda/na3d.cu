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
    \brief Neighborhood Attention 3D - CUDA interface
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
#include "pointwise_neighborhood_3d.cu"
#include "neighborhood_neighborhood_3d.cu"
#include "inverse_neighborhood_3d.cu"
#include "rel_pos_bias_3d.cu"

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

torch::Tensor natten3dqkrpb_cuda_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const at::optional<at::Tensor> &rpb,
    const int kernel_size_d,
    const int kernel_size,
    const int dilation_d,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t depth = query.size(2);
    int64_t height = query.size(3);
    int64_t width = query.size(4);
    int64_t dim = query.size(5);
    int ysize = kernel_size_d * kernel_size * kernel_size;
    int zsize = batch_size * heads;
    int xsize = depth * height * width;
    CHECK_3DFEATMAP(depth, height, width, kernel_size, kernel_size_d, dilation, dilation_d);
    CHECK_KERNELSIZE("natten3dqkrpb_cuda_forward", kernel_size);
    CHECK_KERNELSIZE("natten3dqkrpb_cuda_forward", kernel_size_d);
    int KERNELTHREADS = min(CUDA_NUM_THREADS, ysize);
    int PIXELTHREADS = min(int(CUDA_NUM_THREADS / KERNELTHREADS), xsize);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (PIXELTHREADS * KERNELTHREADS));

    auto attn = torch::zeros(
            {batch_size, heads, depth, height, width, ysize}, query.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
            (xsize + PIXELTHREADS - 1) / PIXELTHREADS,
            (ysize + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(PIXELTHREADS, KERNELTHREADS, BATCHTHREADS);
    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "natten3dqkrpb_cuda_forward", ([&] {
        const auto query_a = query.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        if (rpb.has_value()) {
            const auto rpb_a = rpb.value().packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
            LAUNCH_NA_KDNDS(kernel_size, kernel_size_d, 
                    pointwise_neighborhood_3d_bias, 
                    blocks, threads, 0, stream, 
                    query_a, key_a, rpb_a, attn_a, 
                    depth, height, width, batch_size, heads, dilation, dilation_d, dim);
        } else {
            LAUNCH_NA_KDNDS(kernel_size, kernel_size_d, 
                    pointwise_neighborhood_3d, 
                    blocks, threads, 0, stream, 
                    query_a, key_a, attn_a, 
                    depth, height, width, batch_size, heads, dilation, dilation_d, dim);
        }
    }));
    return attn;
}

torch::Tensor natten3dqkrpb_cuda_forward_fp16(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const at::optional<at::Tensor> &rpb,
    const int kernel_size_d,
    const int kernel_size,
    const int dilation_d,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t depth = query.size(2);
    int64_t height = query.size(3);
    int64_t width = query.size(4);
    int64_t dimhalf = query.size(5) / 2;
    int ysize = kernel_size_d * kernel_size * kernel_size;
    int zsize = batch_size * heads;
    int xsize = depth * height * width;
    CHECK_3DFEATMAP(depth, height, width, kernel_size, kernel_size_d, dilation, dilation_d);
    CHECK_KERNELSIZE("natten3dqkrpb_cuda_forward_fp16", kernel_size);
    CHECK_KERNELSIZE("natten3dqkrpb_cuda_forward_fp16", kernel_size_d);
    TORCH_CHECK(dimhalf*2 == query.size(5), "Dims per head must be an even number in FP16.");
    int KERNELTHREADS = min(CUDA_NUM_THREADS, ysize);
    int PIXELTHREADS = min(int(CUDA_NUM_THREADS / KERNELTHREADS), xsize);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (PIXELTHREADS * KERNELTHREADS));

    auto attn = torch::zeros(
            {batch_size, heads, depth, height, width, ysize}, query.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
            (xsize + PIXELTHREADS - 1) / PIXELTHREADS,
            (ysize + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(PIXELTHREADS, KERNELTHREADS, BATCHTHREADS);
    AT_DISPATCH_HALF_TYPES(at::kHalf, query.scalar_type(), "natten3dqkrpb_cuda_forward_fp16", ([&] {
        const auto query_a = query.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        if (rpb.has_value()) {
            const auto rpb_a = rpb.value().packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
            LAUNCH_NA_KDNDS(kernel_size, kernel_size_d, 
                    pointwise_neighborhood_3d_bias_fp16, 
                    blocks, threads, 0, stream, 
                    query_a, key_a, rpb_a, attn_a, 
                    depth, height, width, batch_size, heads, dilation, dilation_d, dimhalf);
        } else {
            LAUNCH_NA_KDNDS(kernel_size, kernel_size_d, 
                    pointwise_neighborhood_3d_fp16, 
                    blocks, threads, 0, stream, 
                    query_a, key_a, attn_a, 
                    depth, height, width, batch_size, heads, dilation, dilation_d, dimhalf);
        }
    }));
    return attn;
}

std::vector<torch::Tensor> natten3dqkrpb_cuda_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled,
    const int kernel_size_d,
    const int kernel_size,
    const int dilation_d,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t depth = query.size(2);
    int64_t height = query.size(3);
    int64_t width = query.size(4);
    int64_t dim = query.size(5);
    CHECK_3DFEATMAP(depth, height, width, kernel_size, kernel_size_d, dilation, dilation_d);
    CHECK_KERNELSIZE("natten3dqkrpb_cuda_backward", kernel_size);
    CHECK_KERNELSIZE("natten3dqkrpb_cuda_backward", kernel_size_d);
    int64_t RPB_MAX_D = kernel_size_d * 2 - 1;
    int64_t RPB_MAX = kernel_size * 2 - 1;
   
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    at::Tensor d_rpb;
    if (biasEnabled)
        d_rpb = torch::zeros({heads, RPB_MAX_D, RPB_MAX, RPB_MAX}, d_attn.options());

    int32_t n_rpb = heads * depth * height * width * d_attn.size(5);
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
    AT_DISPATCH_FLOATING_TYPES(d_query.scalar_type(), "natten3dqkrpb_backward_cuda", ([&] {
        const auto d_attn_a = d_attn.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto query_a = query.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        auto d_query_a = d_query.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        auto d_key_a = d_key.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        if (biasEnabled) {
            auto d_rpb_a = d_rpb.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
            LAUNCH_NA_KDNDS(kernel_size, kernel_size_d, 
                    rel_pos_bias_gradient_3d, 
                    grid_rpb, blockr, 0, stream, 
                    d_rpb_a, d_attn_a, depth, height, width, dilation, dilation_d, batch_size, d_rpb.numel(), n_rpb);
        }
        LAUNCH_NA_KDNDS(kernel_size, kernel_size_d,
                neighborhood_neighborhood_3d, 
                grid_query, blockq, 0, stream,
                d_attn_a, key_a, d_query_a, 
                depth, height, width, heads, dilation, dilation_d, dim, n_query);
        LAUNCH_NA_KDNDS(kernel_size, kernel_size_d,
                inverse_neighborhood_3d, 
                grid_key, blockk, 0, stream,
                d_attn_a, query_a, d_key_a, 
                depth, height, width, heads, dilation, dilation_d, dim, n_key);
    }));
    return {d_query, d_key, d_rpb};
}

std::vector<torch::Tensor> natten3dqkrpb_cuda_backward_fp16(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled,
    const int kernel_size_d,
    const int kernel_size,
    const int dilation_d,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t depth = query.size(2);
    int64_t height = query.size(3);
    int64_t width = query.size(4);
    int64_t dimhalf = query.size(5) / 2;
    CHECK_3DFEATMAP(depth, height, width, kernel_size, kernel_size_d, dilation, dilation_d);
    CHECK_KERNELSIZE("natten3dqkrpb_cuda_backward_fp16", kernel_size);
    CHECK_KERNELSIZE("natten3dqkrpb_cuda_backward_fp16", kernel_size_d);
    TORCH_CHECK(dimhalf*2 == query.size(5), "Dims per head must be an even number in FP16.");
    int64_t RPB_MAX_D = kernel_size_d * 2 - 1;
    int64_t RPB_MAX = kernel_size * 2 - 1;
   
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    at::Tensor d_rpb;
    if (biasEnabled)
        d_rpb = torch::zeros({heads, RPB_MAX_D, RPB_MAX, RPB_MAX}, d_attn.options());

    int32_t n_rpb = heads * depth * height * width * d_attn.size(5);
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
    AT_DISPATCH_HALF_TYPES(at::kHalf, d_query.scalar_type(), "natten3dqkrpb_backward_cuda_fp16", ([&] {
        const auto d_attn_a = d_attn.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto query_a = query.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        auto d_query_a = d_query.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        auto d_key_a = d_key.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        if (biasEnabled) {
            auto d_rpb_a = d_rpb.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
            LAUNCH_NA_KDNDS(kernel_size, kernel_size_d,
                    rel_pos_bias_gradient_3d_fp16, 
                    grid_rpb, blockr, 0, stream,
                    d_rpb_a, d_attn_a, 
                    depth, height, width, dilation, dilation_d, batch_size, d_rpb.numel(), n_rpb);
        }
        LAUNCH_NA_KDNDS(kernel_size, kernel_size_d, 
                neighborhood_neighborhood_3d_fp16, 
                grid_query, blockq, 0, stream,
                d_attn_a, key_a, d_query_a, 
                depth, height, width, heads, dilation, dilation_d, dimhalf, nhalf_query);
        LAUNCH_NA_KDNDS(kernel_size, kernel_size_d,
                inverse_neighborhood_3d_fp16, 
                grid_key, blockk, 0, stream,
                d_attn_a, query_a, d_key_a, 
                depth, height, width, heads, dilation, dilation_d, dimhalf, nhalf_key);
    }));
    return {d_query, d_key, d_rpb};
}

torch::Tensor natten3dav_cuda_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size_d,
    const int kernel_size,
    const int dilation_d,
    const int dilation) {
    int64_t batch_size = value.size(0);
    int64_t heads = value.size(1);
    int64_t depth = value.size(2);
    int64_t height = value.size(3);
    int64_t width = value.size(4);
    int64_t dim = value.size(5);
    CHECK_3DFEATMAP(depth, height, width, kernel_size, kernel_size_d, dilation, dilation_d);
    CHECK_KERNELSIZE("natten3dav_cuda_forward", kernel_size);
    CHECK_KERNELSIZE("natten3dav_cuda_forward", kernel_size_d);

    auto out = torch::zeros_like(value);

    int32_t n = out.numel();
    int blocks = GET_BLOCKS(n, CUDA_NUM_THREADS_F);
    dim3 grid(blocks);
    dim3 block(CUDA_NUM_THREADS_F);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "natten3dav_forward_cuda", ([&] {
        const auto attn_a = attn.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        auto out_a = out.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        LAUNCH_NA_KDNDS(
                kernel_size, kernel_size_d,
                neighborhood_neighborhood_3d, 
                grid, block, 0, stream, 
                attn_a, value_a, out_a, 
                depth, height, width, heads, dilation, dilation_d, dim, n);
    }));
    return out;
}

torch::Tensor natten3dav_cuda_forward_fp16(
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size_d,
    const int kernel_size,
    const int dilation_d,
    const int dilation) {
    int64_t batch_size = value.size(0);
    int64_t heads = value.size(1);
    int64_t depth = value.size(2);
    int64_t height = value.size(3);
    int64_t width = value.size(4);
    int64_t dimhalf = value.size(5) / 2;
    CHECK_3DFEATMAP(depth, height, width, kernel_size, kernel_size_d, dilation, dilation_d);
    CHECK_KERNELSIZE("natten3dav_cuda_forward_fp16", kernel_size);
    CHECK_KERNELSIZE("natten3dav_cuda_forward_fp16", kernel_size_d);
    TORCH_CHECK(dimhalf*2 == value.size(5), "Dims per head must be an even number in FP16.");

    auto out = torch::zeros_like(value);

    int32_t nhalf = out.numel() / 2;
    int blocks = GET_BLOCKS(nhalf, CUDA_NUM_THREADS_FP16);
    dim3 grid(blocks);
    dim3 block(CUDA_NUM_THREADS_FP16);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_HALF_TYPES(at::kHalf, value.scalar_type(), "natten3dav_forward_cuda_fp16", ([&] {
        const auto attn_a = attn.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        auto out_a = out.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        LAUNCH_NA_KDNDS(
                kernel_size, kernel_size_d,
                neighborhood_neighborhood_3d_fp16, 
                grid, block, 0, stream, 
                attn_a, value_a, out_a, 
                depth, height, width, heads, dilation, dilation_d, dimhalf, nhalf);
    }));
    return out;
}

std::vector<torch::Tensor> natten3dav_cuda_backward(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size_d,
    const int kernel_size,
    const int dilation_d,
    const int dilation) {
    int64_t batch_size = value.size(0);
    int64_t heads = value.size(1);
    int64_t depth = value.size(2);
    int64_t height = value.size(3);
    int64_t width = value.size(4);
    int64_t dim = value.size(5);
    int attn_size = attn.size(5); 
    int zsize = batch_size * heads;
    int xsize = depth * height * width;
    CHECK_3DFEATMAP(depth, height, width, kernel_size, kernel_size_d, dilation, dilation_d);
    CHECK_KERNELSIZE("natten3dav_cuda_backward", kernel_size);
    CHECK_KERNELSIZE("natten3dav_cuda_backward", kernel_size_d);
    int KERNELTHREADS = min(CUDA_NUM_THREADS, attn_size);
    int PIXELTHREADS = min(int(CUDA_NUM_THREADS / KERNELTHREADS), xsize);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (PIXELTHREADS * KERNELTHREADS));

    auto d_attn = torch::zeros_like(attn);
    auto d_value = torch::zeros_like(value);

    const dim3 attn_blocks(
            (xsize + PIXELTHREADS - 1) / PIXELTHREADS,
            (attn_size + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 attn_threads(PIXELTHREADS, KERNELTHREADS, BATCHTHREADS);
    int32_t n_value = d_value.numel();
    int blocks_value = GET_BLOCKS(n_value);
    dim3 grid_value(blocks_value);
    dim3 block(CUDA_NUM_THREADS);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(d_attn.scalar_type(), "natten3dav_backward_cuda", ([&] {
        auto d_attn_a = d_attn.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        auto d_value_a = d_value.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto d_out_a = d_out.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto attn_a = attn.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        LAUNCH_NA_KDNDS(kernel_size, kernel_size_d,
                pointwise_neighborhood_3d, 
                attn_blocks, attn_threads, 0, stream, 
                d_out_a, value_a, d_attn_a, 
                depth, height, width, batch_size, heads, dilation, dilation_d, dim);
        LAUNCH_NA_KDNDS(kernel_size, kernel_size_d,
                inverse_neighborhood_3d, grid_value, block, 0, stream, 
                attn_a, d_out_a, d_value_a, 
                depth, height, width, heads, dilation, dilation_d, dim, n_value);
    }));
    return {d_attn, d_value};
}

std::vector<torch::Tensor> natten3dav_cuda_backward_fp16(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size_d,
    const int kernel_size,
    const int dilation_d,
    const int dilation) {
    int64_t batch_size = value.size(0);
    int64_t heads = value.size(1);
    int64_t depth = value.size(2);
    int64_t height = value.size(3);
    int64_t width = value.size(4);
    int64_t dimhalf = value.size(5) / 2;
    int attn_size = attn.size(5); 
    int zsize = batch_size * heads;
    int xsize = depth * height * width;
    CHECK_3DFEATMAP(depth, height, width, kernel_size, kernel_size_d, dilation, dilation_d);
    CHECK_KERNELSIZE("natten3dav_cuda_backward_fp16", kernel_size);
    CHECK_KERNELSIZE("natten3dav_cuda_backward_fp16", kernel_size_d);
    TORCH_CHECK(dimhalf*2 == value.size(5), "Dims per head must be an even number in FP16.");
    int KERNELTHREADS = min(CUDA_NUM_THREADS, attn_size);
    int PIXELTHREADS = min(int(CUDA_NUM_THREADS / KERNELTHREADS), xsize);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (PIXELTHREADS * KERNELTHREADS));

    auto d_attn = torch::zeros_like(attn);
    auto d_value = torch::zeros_like(value);

    const dim3 attn_blocks(
            (xsize + PIXELTHREADS - 1) / PIXELTHREADS,
            (attn_size + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 attn_threads(PIXELTHREADS, KERNELTHREADS, BATCHTHREADS);
    int32_t nhalf_value = d_value.numel() / 2;
    int blocks_value = GET_BLOCKS(nhalf_value);
    dim3 grid_value(blocks_value);
    dim3 block(CUDA_NUM_THREADS);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_HALF_TYPES(at::kHalf, d_attn.scalar_type(), "natten3dav_backward_cuda_fp16", ([&] {
        auto d_attn_a = d_attn.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        auto d_value_a = d_value.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto d_out_a = d_out.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto attn_a = attn.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        LAUNCH_NA_KDNDS(kernel_size, kernel_size_d, 
                pointwise_neighborhood_3d_fp16, 
                attn_blocks, attn_threads, 0, stream, 
                d_out_a, value_a, d_attn_a,  
                depth, height, width, batch_size, heads, dilation, dilation_d, dimhalf);
        LAUNCH_NA_KDNDS(kernel_size, kernel_size_d, 
                inverse_neighborhood_3d_fp16, 
                grid_value, block, 0, stream, 
                d_out_a, d_value_a, attn_a, 
                depth, height, width, heads, dilation, dilation_d, dimhalf, nhalf_value);
    }));
    return {d_attn, d_value};
}

} // namespace natten
