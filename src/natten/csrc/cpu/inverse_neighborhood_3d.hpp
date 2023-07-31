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
    \brief Inverse-Neighborhood-Neighborhood CPU kernel for 3D data.
           Applies inverse neighborhood attention weights to inverse neighborhood values.
           Used to compute key and value grads.
*/

#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include "cpu/natten_cpu_commons.h"

namespace natten {

template<class scalar_t>
using Tensor6D = typename at::TensorAccessor<scalar_t, 6>;

#define GRAIN_SIZE 0

template <int KS, int DKS, int NS, int DNS, typename scalar_t>
void inverse_neighborhood_3d(          // K-grad / V-grad
    const Tensor6D<scalar_t> weights,  // d_attn / attn
    const Tensor6D<scalar_t> values,   // query  / d_out
    Tensor6D<scalar_t> output,         // d_key  / d_value
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
    for (int b = 0; b < batch_size; b++) {
        at::parallel_for(0, heads, GRAIN_SIZE, [&](int start, int end) {
        for (int h = start; h < end; h++) {
            for (int k = 0; k < depth; k++) {
            const int nk = get_backward_window_start(k, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
            const int ek = get_backward_window_end(k, depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
                for (int i = 0; i < height; i++) {
                const int ni = get_backward_window_start(i, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                const int ei = get_backward_window_end(i, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                for (int j = 0; j < width; j++) {
                    const int nj = get_backward_window_start(j, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                    const int ej = get_backward_window_end(j, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                    for (int d = 0; d < dim; d++) {
                        const int weightsOffset = b * weights.stride(0) + h * weights.stride(1);
                        const int outOffset = b * values.stride(0) + h * values.stride(1) + d;
                        scalar_t output_update = scalar_t(0);
                        for (int xk=nk; xk < ek; xk+=dilation_d){
                        const int onk = get_window_start(xk, depth, KERNEL_SIZE_D, NEIGHBORHOOD_SIZE_D, dilation_d);
                        for (int xi=ni; xi < ei; xi+=dilation){
                        const int oni = get_window_start(xi, height, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                        for (int xj=nj; xj < ej; xj+=dilation){
                            const int onj = get_window_start(xj, width, KERNEL_SIZE, NEIGHBORHOOD_SIZE, dilation);
                            const int outIndex = outOffset + xk * values.stride(2) + xi * values.stride(3) + xj * values.stride(4);
                            const int weightsIndex = weightsOffset + xk * weights.stride(2) + xi * weights.stride(3) + xj * weights.stride(4) + int((k-onk)/dilation_d)*(KERNEL_SIZE*KERNEL_SIZE) + int((i-oni)/dilation)*KERNEL_SIZE + int((j-onj)/dilation);
                            output_update += values.data()[outIndex] * weights.data()[weightsIndex];
                        }}}
                        const int linearIndex = b*output.stride(0) + h*output.stride(1) + k*output.stride(2) + i*output.stride(3) + j*output.stride(4) + d;
                        output.data()[linearIndex] = output_update;
                    }
                }}}
        }});
    }
}

} // namespace natten
