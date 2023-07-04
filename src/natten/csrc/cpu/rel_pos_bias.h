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
    \brief Relative positional bias backward pass CPU kernel for 3D data.
*/

#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include "cpu/natten_cpu_commons.h"

namespace natten {

template <int KS, int NS, int DILATION, typename scalar_t>
void rel_pos_bias_gradient_1d(
    at::TensorAccessor<scalar_t, 2> d_bias,
    const at::TensorAccessor<scalar_t, 4> d_attn,
    const int length,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int batch_size);

template <int KS, int NS, int DILATION, typename scalar_t>
void rel_pos_bias_gradient_2d(
    at::TensorAccessor<scalar_t, 3> d_bias,
    const at::TensorAccessor<scalar_t, 5> d_attn,
    const int height, 
    const int width,
    const int heads,
    const int kernel_size_in,
    const int dilation_in,
    const int batch_size);

template <int KS, int DKS, int NS, int DNS, typename scalar_t>
void rel_pos_bias_gradient_3d(
    at::TensorAccessor<scalar_t, 4> d_bias,
    const at::TensorAccessor<scalar_t, 6> d_attn,
    const int depth, 
    const int height, 
    const int width,
    const int heads,
    const int kernel_size_in,
    const int kernel_size_d_in,
    const int dilation,
    const int dilation_d,
    const int batch_size);

} // namespace natten
