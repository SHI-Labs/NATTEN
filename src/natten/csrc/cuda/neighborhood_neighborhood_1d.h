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
    \brief Neighborhood-Neighborhood kernel for 1D data.
           Applies neighborhood attention weights to neighborhood values.
*/

#include <cuda.h>
#include <torch/extension.h>

namespace natten {

template<class scalar_t>
using Tensor4D = typename torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits>;

template <int KS, int NS, int DILATION, typename scalar_t>
__global__ void neighborhood_neighborhood_1d(           // AV     / Q-grad
    const Tensor4D<scalar_t> weights,                   // attn   / d_attn
    const Tensor4D<scalar_t> values,                    // value  / key
    Tensor4D<scalar_t> output,                          // output / d_query
    const int length,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int dim,
    const int totalElements);

template <int KS, int NS, int DILATION, typename scalar_t>
__global__ void neighborhood_neighborhood_1d_fp16(           // AV     / Q-grad
    const Tensor4D<scalar_t> weights,                        // attn   / d_attn
    const Tensor4D<scalar_t> values,                         // value  / key
    Tensor4D<scalar_t> output,                               // output / d_query
    const int length,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int dimhalf,
    const int totalElements);

} // namespace natten
