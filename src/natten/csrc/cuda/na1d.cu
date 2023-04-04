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
    \brief Neighborhood Attention 1D - CUDA interface
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
#include "pointwise_neighborhood_1d.cu"
#include "neighborhood_neighborhood_1d.cu"
#include "inverse_neighborhood_1d.cu"
#include "rel_pos_bias_1d.cu"

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

torch::Tensor natten1dqkrpb_cuda_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const at::optional<at::Tensor> &rpb,
    const int kernel_size,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t length = query.size(2);
    int64_t dim = query.size(3);
    int zsize = batch_size * heads;
    CHECK_SEQUENCE(length, kernel_size, dilation);
    int KERNELTHREADS = min(CUDA_NUM_THREADS, kernel_size);
    int TOKENTHREADS = min(int64_t(CUDA_NUM_THREADS / KERNELTHREADS), length);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * KERNELTHREADS));

    auto attn = torch::zeros(
            {batch_size, heads, length, kernel_size}, query.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
            (length + TOKENTHREADS - 1) / TOKENTHREADS,
            (kernel_size + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(TOKENTHREADS, KERNELTHREADS, BATCHTHREADS);
    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "natten1dqkrpb_cuda_forward", ([&] {
        const auto query_a = query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        if (rpb.has_value()) {
            const auto rpb_a = rpb.value().packed_accessor32<scalar_t,2,torch::DefaultPtrTraits>();
            LAUNCH_DNA_KNS_1D(kernel_size, dilation, pointwise_neighborhood_1d_bias, 
                    blocks, threads, 0, stream, 
                    query_a, key_a, rpb_a, attn_a, length, batch_size, heads, kernel_size, dilation, dim);
        } else {
            LAUNCH_DNA_KNS_1D(kernel_size, dilation, pointwise_neighborhood_1d, 
                    blocks, threads, 0, stream, 
                    query_a, key_a, attn_a, length, batch_size, heads, kernel_size, dilation, dim);
        }
    }));
    return attn;
}

torch::Tensor natten1dqkrpb_cuda_forward_fp16(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const at::optional<at::Tensor> &rpb,
    const int kernel_size,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t length = query.size(2);
    int64_t dimhalf = query.size(3) / 2;
    int zsize = batch_size * heads;
    CHECK_SEQUENCE(length, kernel_size, dilation);
    TORCH_CHECK(dimhalf*2 == query.size(3), "Dims per head must be an even number in FP16.");
    int KERNELTHREADS = min(CUDA_NUM_THREADS, kernel_size);
    int TOKENTHREADS = min(int64_t(CUDA_NUM_THREADS / KERNELTHREADS), length);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * KERNELTHREADS));

    auto attn = torch::zeros(
            {batch_size, heads, length, kernel_size}, query.options());

    const auto stream = c10::cuda::getCurrentCUDAStream();
    const dim3 blocks(
            (length + TOKENTHREADS - 1) / TOKENTHREADS,
            (kernel_size + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(TOKENTHREADS, KERNELTHREADS, BATCHTHREADS);
    AT_DISPATCH_HALF_TYPES(at::kHalf, query.scalar_type(), "natten1dqkrpb_cuda_forward_fp16", ([&] {
        const auto query_a = query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        if (rpb.has_value()) {
            const auto rpb_a = rpb.value().packed_accessor32<scalar_t,2,torch::DefaultPtrTraits>();
            LAUNCH_DNA_KNS_1D(kernel_size, dilation, pointwise_neighborhood_1d_bias_fp16, 
                    blocks, threads, 0, stream, 
                    query_a, key_a, rpb_a, attn_a, length, batch_size, heads, kernel_size, dilation, dimhalf);
        } else {
            LAUNCH_DNA_KNS_1D(kernel_size, dilation, pointwise_neighborhood_1d_fp16, 
                    blocks, threads, 0, stream, 
                    query_a, key_a, attn_a, length, batch_size, heads, kernel_size, dilation, dimhalf);
        }
    }));
    return attn;
}

std::vector<torch::Tensor> natten1dqkrpb_cuda_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled,
    const int kernel_size,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t length = query.size(2);
    int64_t dim = query.size(3);
    CHECK_SEQUENCE(length, kernel_size, dilation);
    int64_t RPB_MAX = kernel_size * 2 - 1;
   
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    at::Tensor d_rpb;
    if (biasEnabled)
        d_rpb = torch::zeros({heads, RPB_MAX}, d_attn.options());
    int32_t n_rpb = heads * length * kernel_size;
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
    AT_DISPATCH_FLOATING_TYPES(d_query.scalar_type(), "natten1dqkrpb_backward_cuda", ([&] {
        const auto d_attn_a = d_attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto query_a = query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto d_query_a = d_query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto d_key_a = d_key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        if (biasEnabled) {
            auto d_rpb_a = d_rpb.packed_accessor32<scalar_t,2,torch::DefaultPtrTraits>();
            LAUNCH_DNA_KNS_1D(kernel_size, dilation, rel_pos_bias_gradient_1d, grid_rpb, blockr, 0, stream, 
                    d_rpb_a, d_attn_a, length, kernel_size, dilation, batch_size, d_rpb.numel(), n_rpb);
        }
        LAUNCH_DNA_KNS_1D(kernel_size, dilation, neighborhood_neighborhood_1d, grid_query, blockq, 0, stream, 
                d_attn_a, key_a, d_query_a, length, heads, kernel_size, dilation, dim, n_query);
        LAUNCH_DNA_KNS_1D(kernel_size, dilation, inverse_neighborhood_1d, grid_key, blockk, 0, stream, 
                d_attn_a, query_a, d_key_a, length, heads, kernel_size, dilation, dim, n_key);
    }));
    return {d_query, d_key, d_rpb};
}

std::vector<torch::Tensor> natten1dqkrpb_cuda_backward_fp16(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled,
    const int kernel_size,
    const int dilation) {
    int64_t batch_size = query.size(0);
    int64_t heads = query.size(1);
    int64_t length = query.size(2);
    int64_t dimhalf = query.size(3) / 2;
    TORCH_CHECK(dimhalf*2 == query.size(3), "Dims per head must be an even number in FP16.");
    CHECK_SEQUENCE(length, kernel_size, dilation);
    int64_t RPB_MAX = kernel_size * 2 - 1;
   
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    at::Tensor d_rpb;
    if (biasEnabled)
        d_rpb = torch::zeros({heads, RPB_MAX}, d_attn.options());

    int32_t n_rpb = heads * length * kernel_size;
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
    AT_DISPATCH_HALF_TYPES(at::kHalf, d_query.scalar_type(), "natten1dqkrpb_backward_cuda_fp16", ([&] {
        const auto d_attn_a = d_attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto query_a = query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto d_query_a = d_query.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto d_key_a = d_key.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        if (biasEnabled) {
            auto d_rpb_a = d_rpb.packed_accessor32<scalar_t,2,torch::DefaultPtrTraits>();
            LAUNCH_DNA_KNS_1D(kernel_size, dilation, rel_pos_bias_gradient_1d_fp16, grid_rpb, blockr, 0, stream, 
                    d_rpb_a, d_attn_a, length, kernel_size, dilation, batch_size, d_rpb.numel(), n_rpb);
        }
        LAUNCH_DNA_KNS_1D(kernel_size, dilation, neighborhood_neighborhood_1d_fp16, grid_query, blockq, 0, stream, 
                d_attn_a, key_a, d_query_a, length, heads, kernel_size, dilation, dimhalf, nhalf_query);
        LAUNCH_DNA_KNS_1D(kernel_size, dilation, inverse_neighborhood_1d_fp16, grid_key, blockk, 0, stream, 
                d_attn_a, query_a, d_key_a, length, heads, kernel_size, dilation, dimhalf, nhalf_key);
    }));
    return {d_query, d_key, d_rpb};
}

torch::Tensor natten1dav_cuda_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size,
    const int dilation) {
    int batch_size = value.size(0);
    int heads = value.size(1);
    int length = value.size(2);
    int dim = value.size(3);
    CHECK_SEQUENCE(length, kernel_size, dilation);

    auto out = torch::zeros_like(value);

    int32_t n = out.numel();
    int blocks = GET_BLOCKS(n, CUDA_NUM_THREADS_F);
    dim3 grid(blocks);
    dim3 block(CUDA_NUM_THREADS_F);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "natten1dav_forward_cuda", ([&] {
        const auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto out_a = out.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS_1D(kernel_size, dilation, neighborhood_neighborhood_1d, grid, block, 0, stream, 
                attn_a, value_a, out_a, length, heads, kernel_size, dilation, dim, n);
    }));
    return out;
}

torch::Tensor natten1dav_cuda_forward_fp16(
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size,
    const int dilation) {
    int batch_size = value.size(0);
    int heads = value.size(1);
    int length = value.size(2);
    int dimhalf = value.size(3) / 2;
    TORCH_CHECK(dimhalf*2 == value.size(3), "Dims per head must be an even number in FP16.");
    CHECK_SEQUENCE(length, kernel_size, dilation);

    auto out = torch::zeros_like(value);

    int32_t nhalf = out.numel() / 2;
    int blocks = GET_BLOCKS(nhalf, CUDA_NUM_THREADS_FP16);
    dim3 grid(blocks);
    dim3 block(CUDA_NUM_THREADS_FP16);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_HALF_TYPES(at::kHalf, value.scalar_type(), "natten1dav_forward_cuda_fp16", ([&] {
        const auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto out_a = out.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS_1D(kernel_size, dilation, neighborhood_neighborhood_1d_fp16, grid, block, 0, stream, 
                attn_a, value_a, out_a, length, heads, kernel_size, dilation, dimhalf, nhalf);
    }));
    return out;
}

std::vector<torch::Tensor> natten1dav_cuda_backward(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size,
    const int dilation) {
    int64_t batch_size = value.size(0);
    int64_t heads = value.size(1);
    int64_t length = value.size(2);
    int64_t dim = value.size(3);
    int zsize = batch_size * heads;
    CHECK_SEQUENCE(length, kernel_size, dilation);
    int KERNELTHREADS = min(CUDA_NUM_THREADS, kernel_size);
    int TOKENTHREADS = min(int64_t(CUDA_NUM_THREADS / KERNELTHREADS), length);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * KERNELTHREADS));

    auto d_attn = torch::zeros_like(attn);
    auto d_value = torch::zeros_like(value);

    const dim3 attn_blocks(
            (length + TOKENTHREADS - 1) / TOKENTHREADS,
            (kernel_size + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 attn_threads(TOKENTHREADS, KERNELTHREADS, BATCHTHREADS);
    int32_t n_value = d_value.numel();
    int blocks_value = GET_BLOCKS(n_value);
    dim3 grid_value(blocks_value);
    dim3 block(CUDA_NUM_THREADS);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(d_attn.scalar_type(), "natten1dav_backward_cuda", ([&] {
        auto d_attn_a = d_attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto d_value_a = d_value.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto d_out_a = d_out.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS_1D(kernel_size, dilation, pointwise_neighborhood_1d, attn_blocks, attn_threads, 0, stream, 
                d_out_a, value_a, d_attn_a, length, batch_size, heads, kernel_size, dilation, dim);
        LAUNCH_DNA_KNS_1D(kernel_size, dilation, inverse_neighborhood_1d, grid_value, block, 0, stream, 
                attn_a, d_out_a, d_value_a, length, heads, kernel_size, dilation, dim, n_value);
    }));
    return {d_attn, d_value};
}

std::vector<torch::Tensor> natten1dav_cuda_backward_fp16(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size,
    const int dilation) {
    int64_t batch_size = value.size(0);
    int64_t heads = value.size(1);
    int64_t length = value.size(2);
    int64_t dimhalf = value.size(3) / 2;
    TORCH_CHECK(dimhalf*2 == value.size(3), "Dims per head must be an even number in FP16.");
    int zsize = batch_size * heads;
    CHECK_SEQUENCE(length, kernel_size, dilation);
    int KERNELTHREADS = min(CUDA_NUM_THREADS, kernel_size);
    int TOKENTHREADS = min(int64_t(CUDA_NUM_THREADS / KERNELTHREADS), length);
    int BATCHTHREADS = max(1, CUDA_NUM_THREADS / (TOKENTHREADS * KERNELTHREADS));

    auto d_attn = torch::zeros_like(attn);
    auto d_value = torch::zeros_like(value);

    const dim3 attn_blocks(
            (length + TOKENTHREADS - 1) / TOKENTHREADS,
            (kernel_size + KERNELTHREADS - 1) / KERNELTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 attn_threads(TOKENTHREADS, KERNELTHREADS, BATCHTHREADS);
    int32_t nhalf_value = d_value.numel() / 2;
    int blocks_value = GET_BLOCKS(nhalf_value);
    dim3 grid_value(blocks_value);
    dim3 block(CUDA_NUM_THREADS);
    const auto stream = c10::cuda::getCurrentCUDAStream();
    AT_DISPATCH_HALF_TYPES(at::kHalf, d_attn.scalar_type(), "natten1dav_backward_cuda_fp16", ([&] {
        auto d_attn_a = d_attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        auto d_value_a = d_value.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto d_out_a = d_out.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto value_a = value.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        const auto attn_a = attn.packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
        LAUNCH_DNA_KNS_1D(kernel_size, dilation, pointwise_neighborhood_1d_fp16, attn_blocks, attn_threads, 0, stream, 
                d_out_a, value_a, d_attn_a, length, batch_size, heads, kernel_size, dilation, dimhalf);
        LAUNCH_DNA_KNS_1D(kernel_size, dilation, inverse_neighborhood_1d_fp16, grid_value, block, 0, stream, 
                attn_a, d_out_a, d_value_a, length, heads, kernel_size, dilation, dimhalf, nhalf_value);
    }));
    return {d_attn, d_value};
}

} // namespace natten
