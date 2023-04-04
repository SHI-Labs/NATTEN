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
    \brief Pointwise-Neighborhood CPU kernel for 2D data.
           Computes attention weights between query points and their corresponding
           key neighborhood.
           Extra kernel with fused bias (relative positional bias.)
*/

#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#if defined(AVX_INT)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif

#include "natten_cpu_commons.h"

namespace natten {

template<class scalar_t>
using Tensor3D = typename at::TensorAccessor<scalar_t, 3>;
template<class scalar_t>
using Tensor5D = typename at::TensorAccessor<scalar_t, 5>;

#define GRAIN_SIZE 0

template <int KS, int NS, int DILATION, typename scalar_t>
void pointwise_neighborhood_2d(     // QK    / A-grad
    const Tensor5D<scalar_t> query, // query / d_out
    const Tensor5D<scalar_t> key,   // key   / value
    Tensor5D<scalar_t> attn,        // attn  / d_attn
    const int height, 
    const int width,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int dim,
    const int batch_size) {
#if defined(AVX_INT)
    using Vec = at::vec::Vectorized<scalar_t>;
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    at::parallel_for(0, batch_size*heads*height*width, GRAIN_SIZE, [&](int start, int end) {
    for (int x = start; x < end; x++) {
        int indtmp1 = x/width;
        const int j = x - indtmp1 * width;
        int indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
        const int queryOffset = batchHeadOffset + i * query.stride(2) + j * query.stride(3);
        int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3);
        scalar_t* _qaddr = query.data() + queryOffset;
        for (int ki = 0; ki < KERNEL_SIZE; ki++) {
        for (int kj = 0; kj < KERNEL_SIZE; kj++) {
            Vec updt = Vec(scalar_t(0));
            const int keyOffset = batchHeadOffset + (ki*dilation+ni) * key.stride(2) + (kj*dilation+nj) * key.stride(3);
            scalar_t* _kaddr = key.data() + keyOffset;
            int64_t d1 = 0;
            for (; d1 < dim - (dim % Vec::size()); d1+=Vec::size())
                updt = at::vec::fmadd(Vec::loadu(_qaddr + d1), Vec::loadu(_kaddr + d1), updt);
            scalar_t sum_val = at::vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; }, updt, Vec::size());
            for (; d1 < dim; ++d1)
                sum_val += _qaddr[d1] * _kaddr[d1];
            attn.data()[index] = sum_val;
            index++;
        }}
    }});
#else
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    at::parallel_for(0, batch_size*heads*height*width, GRAIN_SIZE, [&](int start, int end) {
    for (int x = start; x < end; x++) {
        int indtmp1 = x/width;
        const int j = x - indtmp1 * width;
        int indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        for (int ki = 0; ki < KERNEL_SIZE; ki++) {
        for (int kj = 0; kj < KERNEL_SIZE; kj++) {
            scalar_t updt = scalar_t(0);
            const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
            const int queryOffset = batchHeadOffset + i * query.stride(2) + j * query.stride(3);
            const int keyOffset = batchHeadOffset + (ki*dilation+ni) * key.stride(2) + (kj*dilation+nj) * key.stride(3);
            for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                updt += query.data()[queryOffset+dimOffset] * key.data()[keyOffset+dimOffset];
            const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE+kj;
            attn.data()[index] = updt;
        }}
    }});
#endif
}


template <int KS, int NS, int DILATION, typename scalar_t>
void pointwise_neighborhood_2d_bias( // QK    / A-grad
    const Tensor5D<scalar_t> query,  // query / d_out
    const Tensor5D<scalar_t> key,    // key   / value
    const Tensor3D<scalar_t> bias,   // relative positional bias tensor
    Tensor5D<scalar_t> attn,         // attn  / d_attn
    const int height, 
    const int width,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int dim,
    const int batch_size) {
#if defined(AVX_INT)
    using Vec = at::vec::Vectorized<scalar_t>;
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    at::parallel_for(0, batch_size*heads*height*width, GRAIN_SIZE, [&](int start, int end) {
    for (int x = start; x < end; x++) {
        int indtmp1 = x/width;
        const int j = x - indtmp1 * width;
        int indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
        const int queryOffset = batchHeadOffset + i * query.stride(2) + j * query.stride(3);
        int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3);
        scalar_t* _qaddr = query.data() + queryOffset;
        for (int ki = 0; ki < KERNEL_SIZE; ki++) {
        for (int kj = 0; kj < KERNEL_SIZE; kj++) {
            Vec updt = Vec(scalar_t(0));
            const int keyOffset = batchHeadOffset + (ki*dilation+ni) * key.stride(2) + (kj*dilation+nj) * key.stride(3);
            scalar_t* _kaddr = key.data() + keyOffset;
            int64_t d1 = 0;
            for (; d1 < dim - (dim % Vec::size()); d1+=Vec::size())
                updt = at::vec::fmadd(Vec::loadu(_qaddr + d1), Vec::loadu(_kaddr + d1), updt);
            scalar_t sum_val = at::vec::vec_reduce_all([](Vec& x, Vec& y) { return x + y; }, updt, Vec::size());
            for (; d1 < dim; ++d1)
                sum_val += _qaddr[d1] * _kaddr[d1];
            const int biasIndex = h * bias.stride(0) + (pi+ki) * bias.stride(1) + (pj+kj) * bias.stride(2);
            attn.data()[index] = bias.data()[biasIndex] + sum_val;
            index++;
        }}
    }});
#else
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    at::parallel_for(0, batch_size*heads*height*width, GRAIN_SIZE, [&](int start, int end) {
    for (int x = start; x < end; x++) {
        int indtmp1 = x/width;
        const int j = x - indtmp1 * width;
        int indtmp2 = indtmp1/height;
        const int i = indtmp1 - indtmp2 * height;
        indtmp1 = indtmp2;
        indtmp2 = indtmp1/heads;
        const int h = indtmp1 - indtmp2 * heads;
        const int b = indtmp2;
        const int ni = get_window_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int nj = get_window_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        for (int ki = 0; ki < KERNEL_SIZE; ki++) {
        for (int kj = 0; kj < KERNEL_SIZE; kj++) {
            scalar_t updt = scalar_t(0);
            const int batchHeadOffset = b * query.stride(0) + h * query.stride(1);
            const int queryOffset = batchHeadOffset + i * query.stride(2) + j * query.stride(3);
            const int keyOffset = batchHeadOffset + (ki*dilation+ni) * key.stride(2) + (kj*dilation+nj) * key.stride(3);
            for (int dimOffset=0; dimOffset < dim; ++dimOffset)
                updt += query.data()[queryOffset+dimOffset] * key.data()[keyOffset+dimOffset];
            const int index = b * attn.stride(0) + h * attn.stride(1) + i * attn.stride(2) + j * attn.stride(3) + ki*KERNEL_SIZE+kj;
            const int biasIndex = h * bias.stride(0) + (pi+ki) * bias.stride(1) + (pj+kj) * bias.stride(2);
            updt += bias.data()[biasIndex];
            attn.data()[index] = updt;
        }}
    }});
#endif
}

} // namespace natten
