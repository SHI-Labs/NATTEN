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
    \brief Neighborhood-Neighborhood CPU kernel for 1D data.
           Applies neighborhood attention weights to neighborhood values.
*/

#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#if defined(AVX_INT)
#include <ATen/cpu/vec/functional.h>
#include <ATen/cpu/vec/vec.h>
#endif

#include "cpu/natten_cpu_commons.h"

namespace natten {

template<class scalar_t>
using Tensor4D = typename at::TensorAccessor<scalar_t, 4>;

#define GRAIN_SIZE 0

// TODO: AVX

template <int KS, int NS, int DILATION, typename scalar_t>
void neighborhood_neighborhood_1d(           // AV     / Q-grad
    const Tensor4D<scalar_t> weights,        // attn   / d_attn
    const Tensor4D<scalar_t> values,         // value  / key
    Tensor4D<scalar_t> output,               // output / d_query
    const int length,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int dim,
    const int batch_size) {
    const int KERNEL_SIZE = (KS>1) ? KS : kernel_size_in;
    const int NEIGHBORHOOD_SIZE = (NS>0) ? NS : KERNEL_SIZE / 2;
    const int dilation = (DILATION>0) ? DILATION : dilation_in;
    for (int b = 0; b < batch_size; b++) {
        at::parallel_for(0, heads, GRAIN_SIZE, [&](int start, int end) {
        for (int h = start; h < end; h++) {
            for (int i = 0; i < length; i++) {
                const int ni = get_window_start(i, length, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                for (int d = 0; d < dim; d++) {
                    scalar_t output_update = scalar_t(0);
                    int attnOffset = b * weights.stride(0) + h * weights.stride(1) + i * weights.stride(2);
                    const int valuesOffset = b * values.stride(0) + h * values.stride(1) + d;
                    for (int xi=ni; xi < ni + KERNEL_SIZE * dilation; xi+=dilation){
                        const int valuesIndex = valuesOffset + xi * values.stride(2);
                        output_update += weights.data()[attnOffset] * values.data()[valuesIndex];
                        ++attnOffset;
                    }
                    const int linearIndex = b * output.stride(0) + h * output.stride(1) +i * output.stride(2) + d;
                    output.data()[linearIndex] = output_update;
                }
            }
        }});
    }
}

} // namespace natten
