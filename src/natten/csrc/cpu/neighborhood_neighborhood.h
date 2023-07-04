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
template<class scalar_t>
using Tensor5D = typename at::TensorAccessor<scalar_t, 5>;
template<class scalar_t>
using Tensor6D = typename at::TensorAccessor<scalar_t, 6>;


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
    const int batch_size);


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
    const int batch_size);


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
    const int batch_size);

} // namespace natten
