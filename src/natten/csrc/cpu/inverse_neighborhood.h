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
    \brief Inverse-Neighborhood-Neighborhood CPU kernel for 1D data.
           Applies inverse neighborhood attention weights to inverse neighborhood values.
           Used to compute key and value grads.
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
void inverse_neighborhood_1d(          // K-grad / V-grad
    const Tensor4D<scalar_t> weights,  // d_attn / attn
    const Tensor4D<scalar_t> values,   // query  / d_out
    Tensor4D<scalar_t> output,         // d_key  / d_value
    const int length,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int dim,
    const int batch_size);


template <int KS, int NS, int DILATION, typename scalar_t>
void inverse_neighborhood_2d(          // K-grad / V-grad
    const Tensor5D<scalar_t> weights,  // d_attn / attn
    const Tensor5D<scalar_t> values,   // query  / d_out
    Tensor5D<scalar_t> output,         // d_key  / d_value
    const int height, 
    const int width,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int dim,
    const int batch_size);


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
    const int batch_size);

} // namespace natten
