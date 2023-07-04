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
    \brief Relative positional bias backward pass kernel for 2D data.
*/

#include <cuda.h>
#include <torch/extension.h>

namespace natten {

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void rel_pos_bias_gradient_2d(
    torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> d_rpb,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_attn,
    const int height,
    const int width,
    const int dilation_in,
    const int batch_size,
    const int d_rpb_numel,
    const int totalThreads);

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void rel_pos_bias_gradient_2d_fp16(
    torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits> d_rpb,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits> d_attn,
    const int height,
    const int width,
    const int dilation_in,
    const int batch_size,
    const int d_rpb_numel,
    const int totalThreads);

} // namespace natten
