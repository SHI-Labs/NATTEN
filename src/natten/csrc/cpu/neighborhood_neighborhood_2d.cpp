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
    \brief Neighborhood-Neighborhood CPU kernel for 2D data.
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

#include "natten_cpu_commons.h"

namespace natten {

template<class scalar_t>
using Tensor5D = typename at::TensorAccessor<scalar_t, 5>;

#define GRAIN_SIZE 0

template <int KS, int NS, int DILATION, typename scalar_t>
void neighborhood_neighborhood_2d(           // AV     / Q-grad
    const Tensor5D<scalar_t> weights,        // attn   / d_attn
    const Tensor5D<scalar_t> values,         // value  / key
    Tensor5D<scalar_t> output,               // output / d_query
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
        int weightsOffset = b * weights.stride(0) + h * weights.stride(1) + i * weights.stride(2) + j * weights.stride(3);
        const int valuesOffset = b * values.stride(0) + h * values.stride(1);
        const int outputIndex = b*output.stride(0) + h*output.stride(1) + i*output.stride(2) + j*output.stride(3);
        scalar_t* _oaddr = output.data() + outputIndex;
        for (int xi=ni; xi < ni + KERNEL_SIZE * dilation; xi+=dilation){
        for (int xj=nj; xj < nj + KERNEL_SIZE * dilation; xj+=dilation){
            const int valuesIndex = valuesOffset + xi * values.stride(2)+ xj * values.stride(3);
            scalar_t* _vaddr = values.data() + valuesIndex;
            Vec a = Vec(weights.data()[weightsOffset]);
            at::vec::map2([a](Vec& x, Vec& y) { return fmadd(a, x, y); }, _oaddr, _vaddr, _oaddr, dim);
            ++weightsOffset;
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
        for (int d = 0; d < dim; d++) {
            scalar_t updt = scalar_t(0);
            int weightsOffset = b * weights.stride(0) + h * weights.stride(1) + i * weights.stride(2) + j * weights.stride(3);
            const int valuesOffset = b * values.stride(0) + h * values.stride(1) + d;
            for (int xi=ni; xi < ni + KERNEL_SIZE * dilation; xi+=dilation){
            for (int xj=nj; xj < nj + KERNEL_SIZE * dilation; xj+=dilation){
                const int valuesIndex = valuesOffset + xi * values.stride(2)+ xj * values.stride(3);
                updt += weights.data()[weightsOffset] * values.data()[valuesIndex];
                ++weightsOffset;
            }}
            const int linearIndex = b*output.stride(0) + h*output.stride(1) + i*output.stride(2) + j*output.stride(3) + d*output.stride(4);
            output.data()[linearIndex] = updt;
        }
    }});
#endif
}

} // namespace natten
