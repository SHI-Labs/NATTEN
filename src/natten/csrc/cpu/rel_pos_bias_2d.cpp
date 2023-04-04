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
    \brief Relative positional bias backward pass CPU kernel for 2D data.
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

#define GRAIN_SIZE 0

// TODO: AVX

template <int KS, int NS, int DILATION, typename scalar_t>
void rel_pos_bias_gradient_2d(
    at::TensorAccessor<scalar_t, 3> d_bias,
    const at::TensorAccessor<scalar_t, 5> d_attn,
    const int height, 
    const int width,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int batch_size) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    at::parallel_for(0, heads, GRAIN_SIZE, [&](int start, int end) {
    for (int h = start; h < end; h++) {
        for (int i = 0; i < height; i++) {
        const int pi = get_pb_start(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
        for (int j = 0; j < width; j++) {
            const int pj = get_pb_start(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
            for (int ki = 0; ki < KERNEL_SIZE; ki++) {
            for (int kj = 0; kj < KERNEL_SIZE; kj++) {
                scalar_t d_bias_update = scalar_t(0);
                int attnOffset = h * d_attn.stride(1) + i * d_attn.stride(2) + j * d_attn.stride(3) + ki*KERNEL_SIZE+kj;
                for (int b=0; b < batch_size; ++b){
                    d_bias_update += d_attn.data()[attnOffset];
                    attnOffset += d_attn.stride(0);
                }
                const int index = h * d_bias.stride(0) + (pi+ki) * d_bias.stride(1) + (pj+kj) * d_bias.stride(2);
                d_bias.data()[index] += d_bias_update;
            }}
        }}
    }});
}

} // namespace natten
