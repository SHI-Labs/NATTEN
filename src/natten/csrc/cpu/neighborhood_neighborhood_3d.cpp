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
    \brief Neighborhood-Neighborhood CPU kernel for 3D data.
           Applies neighborhood attention weights to neighborhood values.
*/

#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include "natten_cpu_commons.h"

namespace natten {

template<class scalar_t>
using Tensor6D = typename at::TensorAccessor<scalar_t, 6>;

#define GRAIN_SIZE 0

template <int KS, int DKS, int NS, int DNS, typename scalar_t>
void neighborhood_neighborhood_3d(           // AV     / Q-grad
    const Tensor6D<scalar_t> weights,        // attn   / d_attn
    const Tensor6D<scalar_t> values,         // value  / key
    Tensor6D<scalar_t> output,               // output / d_query
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
        for (int d = 0; d < dim; d++) {
            scalar_t updt = scalar_t(0);
            int weightsOffset = b * weights.stride(0) + h * weights.stride(1) + k * weights.stride(2) + i * weights.stride(3) + j * weights.stride(4);
            const int valuesOffset = b * values.stride(0) + h * values.stride(1) + d;
            for (int xk=nk; xk < nk + KERNEL_SIZE_D * dilation_d; xk+=dilation_d){
            for (int xi=ni; xi < ni + KERNEL_SIZE * dilation; xi+=dilation){
            for (int xj=nj; xj < nj + KERNEL_SIZE * dilation; xj+=dilation){
                const int valuesIndex = valuesOffset + xk * values.stride(2) + xi * values.stride(3) + xj * values.stride(4);
                updt += weights.data()[weightsOffset] * values.data()[valuesIndex];
                ++weightsOffset;
            }}}
            const int linearIndex = b*output.stride(0) + h*output.stride(1) + k*output.stride(2) + i*output.stride(3) + j*output.stride(4) + d*output.stride(5);
            output.data()[linearIndex] = updt;
        }
    }});
}

} // namespace natten
