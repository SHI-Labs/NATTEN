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
    \brief Pointwise-Neighborhood CPU kernel for 3D data.
           Computes attention weights between query points and their corresponding
           key neighborhood.
           Extra kernel with fused bias (relative positional bias.)
*/

#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include "cpu/natten_cpu_commons.h"

namespace natten {

template<class scalar_t>
using Tensor4D = typename at::TensorAccessor<scalar_t, 4>;
template<class scalar_t>
using Tensor6D = typename at::TensorAccessor<scalar_t, 6>;

#define GRAIN_SIZE 0

template <int KS, int DKS, int NS, int DNS, typename scalar_t>
void pointwise_neighborhood_3d(     // QK    / A-grad
    const Tensor6D<scalar_t> query, // query / d_out
    const Tensor6D<scalar_t> key,   // key   / value
    Tensor6D<scalar_t> attn,        // attn  / d_attn
    const int depth, 
    const int height, 
    const int width,
    const int heads,
    const int kernel_size_in,
    const int kernel_size_d_in,
    const int dilation,
    const int dilation_d,
    const int dim,
    const int batch_size) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int KERNEL_SIZE_D = (DKS>1) ? DKS : kernel_size_d_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int NEIGHBORHOOD_SIZE_D = (DNS>0) ? DNS : KERNEL_SIZE_D / 2;
    at::parallel_for(0, batch_size*heads*depth*height*width, GRAIN_SIZE, [&](int start, int end) {
    for (int x = start; x < end; x++) {
        int indtmp1 = x/width;
        const int j = x - indtmp1 * width;
        int indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/depth;
        const int k = indtmp1 - indtmp2 * depth;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nk = get_window_start(k, depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
        for (int kk = 0; kk < KERNEL_SIZE_D; kk++) {
        for (int ki = 0; ki < KERNEL_SIZE; ki++) {
        for (int kj = 0; kj < KERNEL_SIZE; kj++) {
            scalar_t updt = scalar_t(0);
            const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
            const int queryOffset = batchHeadOffset + k * query.stride(2) + i * query.stride(3) + j * query.stride(4);
            const int keyOffset = batchHeadOffset + (kk*dilation_d+nk) * key.stride(2) + (ki*dilation+ni) * key.stride(3) + (kj*dilation+nj) * key.stride(4);
            for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                updt += query.data()[queryOffset+dimOffset] * key.data()[keyOffset+dimOffset];
            const int index = b * attn.stride(0) + h * attn.stride(1) + k * attn.stride(2) + i * attn.stride(3) + j * attn.stride(4) + kk*(KERNEL_SIZE*KERNEL_SIZE)+ki*KERNEL_SIZE+kj;
            attn.data()[index] = updt;
        }}}
    }});
}

template <int KS, int DKS, int NS, int DNS, typename scalar_t>
void pointwise_neighborhood_3d_bias( // QK    / A-grad
    const Tensor6D<scalar_t> query,  // query / d_out
    const Tensor6D<scalar_t> key,    // key   / value
    const Tensor4D<scalar_t> bias,   // relative positional bias tensor
    Tensor6D<scalar_t> attn,         // attn  / d_attn
    const int depth, 
    const int height, 
    const int width,
    const int heads,
    const int kernel_size_in,
    const int kernel_size_d_in,
    const int dilation,
    const int dilation_d,
    const int dim,
    const int batch_size) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int KERNEL_SIZE_D = (DKS>1) ? DKS : kernel_size_d_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int NEIGHBORHOOD_SIZE_D = (DNS>0) ? DNS : KERNEL_SIZE_D / 2;
    at::parallel_for(0, batch_size*heads*depth*height*width, GRAIN_SIZE, [&](int start, int end) {
    for (int x = start; x < end; x++) {
        int indtmp1 = x/width;
        const int j = x - indtmp1 * width;
        int indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/depth;
        const int k = indtmp1 - indtmp2 * depth;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nk = get_window_start(k, depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
        const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pk = get_pb_start(k, depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
        for (int kk = 0; kk < KERNEL_SIZE_D; kk++) {
        for (int ki = 0; ki < KERNEL_SIZE; ki++) {
        for (int kj = 0; kj < KERNEL_SIZE; kj++) {
            scalar_t updt = scalar_t(0);
            const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
            const int queryOffset = batchHeadOffset + k * query.stride(2) + i * query.stride(3) + j * query.stride(4);
            const int keyOffset = batchHeadOffset + (kk*dilation_d+nk) * key.stride(2) + (ki*dilation+ni) * key.stride(3) + (kj*dilation+nj) * key.stride(4);
            for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                updt += query.data()[queryOffset+dimOffset] * key.data()[keyOffset+dimOffset];
            const int index = b * attn.stride(0) + h * attn.stride(1) + k * attn.stride(2) + i * attn.stride(3) + j * attn.stride(4) + kk*(KERNEL_SIZE*KERNEL_SIZE)+ki*KERNEL_SIZE+kj;
            const int biasIndex = h * bias.stride(0) + (pk+kk) * bias.stride(1) + (pi+ki) * bias.stride(2) + (pj+kj) * bias.stride(3);
            updt += bias.data()[biasIndex];
            attn.data()[index] = updt;
        }}}
    }});
}

} // namespace natten
