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
    \brief Neighborhood-Neighborhood kernel for 3D data.
           Applies neighborhood attention weights to neighborhood values.
*/

#include <cuda.h>
#include <torch/extension.h>

namespace natten {

template<class scalar_t>
using Tensor6D = typename torch::PackedTensorAccessor32<scalar_t,6,torch::DefaultPtrTraits>;

template <int KERNEL_SIZE, int KERNEL_SIZE_D, int NEIGHBORHOOD_SIZE, int NEIGHBORHOOD_SIZE_D, typename scalar_t>
__global__ void neighborhood_neighborhood_3d(           // AV     / Q-grad
    const Tensor6D<scalar_t> weights,                   // attn   / d_attn
    const Tensor6D<scalar_t> values,                    // value  / key
    Tensor6D<scalar_t> output,                          // output / d_query
    const int depth,
    const int height,
    const int width,
    const int heads,
    const int dilation,
    const int dilation_d,
    const int dim,
    const int totalElements);

template <int KERNEL_SIZE, int KERNEL_SIZE_D, int NEIGHBORHOOD_SIZE, int NEIGHBORHOOD_SIZE_D, typename scalar_t>
__global__ void neighborhood_neighborhood_3d_fp16(      // AV     / Q-grad
    const Tensor6D<scalar_t> weights,                   // attn   / d_attn
    const Tensor6D<scalar_t> values,                    // value  / key
    Tensor6D<scalar_t> output,                          // output / d_query
    const int depth,
    const int height,
    const int width,
    const int heads,
    const int dilation,
    const int dilation_d,
    const int dimhalf,
    const int totalElements);

} // namespace natten
