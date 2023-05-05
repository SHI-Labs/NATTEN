
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

#ifdef NATTEN_DEBUG_MODE

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

void natten_pn3d_tiled_dim32(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const at::optional<at::Tensor> &rpb,
    torch::Tensor &attn,
    int batch_size,
    int heads,
    int depth,
    int height,
    int width,
    int kernel_size,
    int kernel_size_d) {
    int xsize = width * kernel_size;
    int ysize = height * kernel_size;
    int zsize = batch_size * heads * depth * kernel_size_d;

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
            (xsize + XTHREADS - 1) / XTHREADS,
            (ysize + YTHREADS - 1) / YTHREADS,
            (zsize + BATCHTHREADS - 1) / BATCHTHREADS);
    const dim3 threads(XTHREADS, YTHREADS, BATCHTHREADS);
    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "natten3dqkrpb_cuda_forward_fp32_tiled_32", ([&] {
        const auto query_a = query.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        const auto key_a = key.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        auto attn_a = attn.packed_accessor32<scalar_t,6,torch::DefaultPtrTraits>();
        if (rpb.has_value()) {
            const auto rpb_a = rpb.value().packed_accessor32<scalar_t,4,torch::DefaultPtrTraits>();
            if (kernel_size == KERNEL_SIZE_7)
                pointwise_neighborhood_3d_bias_7x7_9x9_32<
                    TILE_7, KTILE_7, 
                    KERNEL_SIZE_7, NEIGHBORHOOD_SIZE_7, 
                    XYTHREADS_7, 
                    scalar_t><<<blocks, threads, 0, stream>>>(
                            query_a, key_a, rpb_a, attn_a, 
                            depth, height, width, batch_size, heads, kernel_size_d);
            else if (kernel_size == KERNEL_SIZE_9)
                pointwise_neighborhood_3d_bias_7x7_9x9_32<
                    TILE_9, KTILE_9, 
                    KERNEL_SIZE_9, NEIGHBORHOOD_SIZE_9, 
                    XYTHREADS_9, 
                    scalar_t><<<blocks, threads, 0, stream>>>(
                            query_a, key_a, rpb_a, attn_a, 
                            depth, height, width, batch_size, heads, kernel_size_d);
            else if (kernel_size == KERNEL_SIZE_3)
                pointwise_neighborhood_3d_bias_3x3_32<scalar_t><<<blocks, threads, 0, stream>>>(
                            query_a, key_a, rpb_a, attn_a, 
                            depth, height, width, batch_size, heads, kernel_size_d);
            else if (kernel_size == KERNEL_SIZE_5)
                pointwise_neighborhood_3d_bias_5x5_32<scalar_t><<<blocks, threads, 0, stream>>>(
                            query_a, key_a, rpb_a, attn_a, 
                            depth, height, width, batch_size, heads, kernel_size_d);
            else if (kernel_size == KERNEL_SIZE_11)
                pointwise_neighborhood_3d_bias_11x11_13x13_32<
                    TILE_11_X, TILE_11_Y, KTILE_11_X, KTILE_11_Y,
                    KERNEL_SIZE_11, NEIGHBORHOOD_SIZE_11, 
                    XTHREADS_11, YTHREADS_11,
                    scalar_t><<<blocks, threads, 0, stream>>>(
                            query_a, key_a, rpb_a, attn_a, 
                            depth, height, width, batch_size, heads, kernel_size_d);
            else if (kernel_size == KERNEL_SIZE_13)
                pointwise_neighborhood_3d_bias_11x11_13x13_32<
                    TILE_13_X, TILE_13_Y, KTILE_13_X, KTILE_13_Y,
                    KERNEL_SIZE_13, NEIGHBORHOOD_SIZE_13, 
                    XTHREADS_13, YTHREADS_13,
                    scalar_t><<<blocks, threads, 0, stream>>>(
                            query_a, key_a, rpb_a, attn_a, 
                            depth, height, width, batch_size, heads, kernel_size_d);
        } else {
            if (kernel_size == KERNEL_SIZE_7)
                pointwise_neighborhood_3d_7x7_9x9_32<
                    TILE_7, KTILE_7, 
                    KERNEL_SIZE_7, NEIGHBORHOOD_SIZE_7, 
                    XYTHREADS_7, 
                    scalar_t><<<blocks, threads, 0, stream>>>(
                            query_a, key_a, attn_a, 
                            depth, height, width, batch_size, heads, kernel_size_d);
            else if (kernel_size == KERNEL_SIZE_9)
                pointwise_neighborhood_3d_7x7_9x9_32<
                    TILE_9, KTILE_9, 
                    KERNEL_SIZE_9, NEIGHBORHOOD_SIZE_9, 
                    XYTHREADS_9, 
                    scalar_t><<<blocks, threads, 0, stream>>>(
                            query_a, key_a, attn_a, 
                            depth, height, width, batch_size, heads, kernel_size_d);
            else if (kernel_size == KERNEL_SIZE_3)
                pointwise_neighborhood_3d_3x3_32<scalar_t><<<blocks, threads, 0, stream>>>(
                            query_a, key_a, attn_a, 
                            depth, height, width, batch_size, heads, kernel_size_d);
            else if (kernel_size == KERNEL_SIZE_5)
                pointwise_neighborhood_3d_5x5_32<scalar_t><<<blocks, threads, 0, stream>>>(
                            query_a, key_a, attn_a, 
                            depth, height, width, batch_size, heads, kernel_size_d);
            else if (kernel_size == KERNEL_SIZE_11)
                pointwise_neighborhood_3d_11x11_13x13_32<
                    TILE_11_X, TILE_11_Y, KTILE_11_X, KTILE_11_Y,
                    KERNEL_SIZE_11, NEIGHBORHOOD_SIZE_11, 
                    XTHREADS_11, YTHREADS_11,
                    scalar_t><<<blocks, threads, 0, stream>>>(
                            query_a, key_a, attn_a, 
                            depth, height, width, batch_size, heads, kernel_size_d);
            else if (kernel_size == KERNEL_SIZE_13)
                pointwise_neighborhood_3d_11x11_13x13_32<
                    TILE_13_X, TILE_13_Y, KTILE_13_X, KTILE_13_Y,
                    KERNEL_SIZE_13, NEIGHBORHOOD_SIZE_13, 
                    XTHREADS_13, YTHREADS_13,
                    scalar_t><<<blocks, threads, 0, stream>>>(
                            query_a, key_a, attn_a, 
                            depth, height, width, batch_size, heads, kernel_size_d);
        }
    }));
}

} // namespace natten
#endif
