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
    \brief Pointwise-Neighborhood kernel for 3D data.
           Computes attention weights between query points and their corresponding
           key neighborhood.
           Extra kernel with fused bias (relative positional bias.)
*/

#include <cuda.h>
#include <torch/extension.h>

namespace natten {

template<class scalar_t>
using Tensor4D = typename torch::PackedTensorAccessor32<scalar_t,4,torch::DefaultPtrTraits>;
template<class scalar_t>
using Tensor6D = typename torch::PackedTensorAccessor32<scalar_t,6,torch::DefaultPtrTraits>;

template <int KERNEL_SIZE, int KERNEL_SIZE_D, int NEIGHBORHOOD_SIZE, int NEIGHBORHOOD_SIZE_D, typename scalar_t>
__global__ void pointwise_neighborhood_3d_bias( // QK
    const Tensor6D<scalar_t> query,             // query
    const Tensor6D<scalar_t> key,               // key  
    const Tensor4D<scalar_t> bias,              // relative positional bias tensor
    Tensor6D<scalar_t> attn,                    // attn
    const int depth,
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation,
    const int dilation_d,
    const int dim);

template <int KERNEL_SIZE, int KERNEL_SIZE_D, int NEIGHBORHOOD_SIZE, int NEIGHBORHOOD_SIZE_D, typename scalar_t>
__global__ void pointwise_neighborhood_3d_bias_fp16( // QK
    const Tensor6D<scalar_t> query,                  // query
    const Tensor6D<scalar_t> key,                    // key  
    const Tensor4D<scalar_t> bias,                   // relative positional bias tensor
    Tensor6D<scalar_t> attn,                         // attn
    const int depth,
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation,
    const int dilation_d,
    const int dimhalf);

template <int KERNEL_SIZE, int KERNEL_SIZE_D, int NEIGHBORHOOD_SIZE, int NEIGHBORHOOD_SIZE_D, typename scalar_t>
__global__ void pointwise_neighborhood_3d(      // QK
    const Tensor6D<scalar_t> query,             // query
    const Tensor6D<scalar_t> key,               // key  
    Tensor6D<scalar_t> attn,                    // attn
    const int depth,
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation,
    const int dilation_d,
    const int dim);

template <int KERNEL_SIZE, int KERNEL_SIZE_D, int NEIGHBORHOOD_SIZE, int NEIGHBORHOOD_SIZE_D, typename scalar_t>
__global__ void pointwise_neighborhood_3d_fp16(      // QK
    const Tensor6D<scalar_t> query,                  // query
    const Tensor6D<scalar_t> key,                    // key  
    Tensor6D<scalar_t> attn,                         // attn
    const int depth,
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation,
    const int dilation_d,
    const int dimhalf);

} // namespace natten
