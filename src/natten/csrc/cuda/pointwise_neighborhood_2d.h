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
    \brief Pointwise-Neighborhood kernel for 2D data.
           Computes attention weights between query points and their corresponding
           key neighborhood.
           Extra kernel with fused bias (relative positional bias.)
           + Tiled kernels for NA with window size 3, 5, 7, 9, 11, and 13 (only 32 dim per head 
           supported at the moment.)
*/

#include <cuda.h>
#include <torch/extension.h>

namespace natten {

template<class scalar_t>
using Tensor3D = typename torch::PackedTensorAccessor32<scalar_t,3,torch::DefaultPtrTraits>;
template<class scalar_t>
using Tensor5D = typename torch::PackedTensorAccessor32<scalar_t,5,torch::DefaultPtrTraits>;

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_bias_fp16( // QK
    const Tensor5D<scalar_t> query,                  // query
    const Tensor5D<scalar_t> key,                    // key  
    const Tensor3D<scalar_t> bias,                   // relative positional bias tensor
    Tensor5D<scalar_t> attn,                         // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in,
    const int dimhalf);

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_bias( // QK
    const Tensor5D<scalar_t> query,             // query
    const Tensor5D<scalar_t> key,               // key  
    const Tensor3D<scalar_t> bias,              // relative positional bias tensor
    Tensor5D<scalar_t> attn,                    // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in,
    const int dim);

template <int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_bias_fp16_3x3_32( // QK
    const Tensor5D<scalar_t> query,                         // query
    const Tensor5D<scalar_t> key,                           // key  
    const Tensor3D<scalar_t> bias,                          // relative positional bias tensor
    Tensor5D<scalar_t> attn,                                // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in);

template <int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_bias_3x3_32( // QK
    const Tensor5D<scalar_t> query,                    // query
    const Tensor5D<scalar_t> key,                      // key  
    const Tensor3D<scalar_t> bias,                     // relative positional bias tensor
    Tensor5D<scalar_t> attn,                           // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in);

template <int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_bias_fp16_5x5_32( // QK
    const Tensor5D<scalar_t> query,                         // query
    const Tensor5D<scalar_t> key,                           // key  
    const Tensor3D<scalar_t> bias,                          // relative positional bias tensor
    Tensor5D<scalar_t> attn,                                // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in);

template <int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_bias_5x5_32( // QK
    const Tensor5D<scalar_t> query,                    // query
    const Tensor5D<scalar_t> key,                      // key  
    const Tensor3D<scalar_t> bias,                     // relative positional bias tensor
    Tensor5D<scalar_t> attn,                           // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in);

template <int TILE, int KTILE, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_bias_fp16_7x7_9x9_32( // QK
    const Tensor5D<scalar_t> query,                             // query
    const Tensor5D<scalar_t> key,                               // key  
    const Tensor3D<scalar_t> bias,                              // relative positional bias tensor
    Tensor5D<scalar_t> attn,                                    // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in);

template <int TILE, int KTILE, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_bias_7x7_9x9_32( // QK
    const Tensor5D<scalar_t> query,                        // query
    const Tensor5D<scalar_t> key,                          // key  
    const Tensor3D<scalar_t> bias,                         // relative positional bias tensor
    Tensor5D<scalar_t> attn,                               // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in);

template <int TILEX, int TILEY, int KTILEX, int KTILEY, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t, typename memscalar_t>
__global__ void pointwise_neighborhood_2d_bias_fp16_11x11_13x13_32( // QK
    const Tensor5D<scalar_t> query,                                 // query
    const Tensor5D<scalar_t> key,                                   // key  
    const Tensor3D<scalar_t> bias,                                  // relative positional bias tensor
    Tensor5D<scalar_t> attn,                                        // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in);

template <int TILEX, int TILEY, int KTILEX, int KTILEY, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t, typename memscalar_t>
__global__ void pointwise_neighborhood_2d_bias_11x11_13x13_32( // QK
    const Tensor5D<scalar_t> query,                            // query
    const Tensor5D<scalar_t> key,                              // key  
    const Tensor3D<scalar_t> bias,                             // relative positional bias tensor
    Tensor5D<scalar_t> attn,                                   // attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in);

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_fp16( // QK    / A-grad
    const Tensor5D<scalar_t> query,             // query / d_out
    const Tensor5D<scalar_t> key,               // key   / value
    Tensor5D<scalar_t> attn,                    // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in,
    const int dimhalf);

template <int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d( // QK    / A-grad
    const Tensor5D<scalar_t> query,        // query / d_out
    const Tensor5D<scalar_t> key,          // key   / value
    Tensor5D<scalar_t> attn,               // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in,
    const int dim);

template <int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_fp16_3x3_32( // QK    / A-grad
    const Tensor5D<scalar_t> query,                    // query / d_out
    const Tensor5D<scalar_t> key,                      // key   / value
    Tensor5D<scalar_t> attn,                           // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in);

template <int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_3x3_32( // QK    / A-grad
    const Tensor5D<scalar_t> query,               // query / d_out
    const Tensor5D<scalar_t> key,                 // key   / value
    Tensor5D<scalar_t> attn,                      // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in);

template <int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_fp16_5x5_32( // QK    / A-grad
    const Tensor5D<scalar_t> query,                    // query / d_out
    const Tensor5D<scalar_t> key,                      // key   / value
    Tensor5D<scalar_t> attn,                           // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in);

template <int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_5x5_32( // QK    / A-grad
    const Tensor5D<scalar_t> query,               // query / d_out
    const Tensor5D<scalar_t> key,                 // key   / value
    Tensor5D<scalar_t> attn,                      // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in);

template <int TILE, int KTILE, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_fp16_7x7_9x9_32( // QK    / A-grad
    const Tensor5D<scalar_t> query,                        // query / d_out
    const Tensor5D<scalar_t> key,                          // key   / value
    Tensor5D<scalar_t> attn,                               // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in);

template <int TILE, int KTILE, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t>
__global__ void pointwise_neighborhood_2d_7x7_9x9_32( // QK    / A-grad
    const Tensor5D<scalar_t> query,                   // query / d_out
    const Tensor5D<scalar_t> key,                     // key   / value
    Tensor5D<scalar_t> attn,                          // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in);

template <int TILEX, int TILEY, int KTILEX, int KTILEY, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t, typename memscalar_t>
__global__ void pointwise_neighborhood_2d_fp16_11x11_13x13_32( // QK    / A-grad
    const Tensor5D<scalar_t> query,                            // query / d_out
    const Tensor5D<scalar_t> key,                              // key   / value
    Tensor5D<scalar_t> attn,                                   // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in);

template <int TILEX, int TILEY, int KTILEX, int KTILEY, int KERNEL_SIZE, int NEIGHBORHOOD_SIZE, int DILATION, typename scalar_t, typename memscalar_t>
__global__ void pointwise_neighborhood_2d_11x11_13x13_32( // QK    / A-grad
    const Tensor5D<scalar_t> query,                       // query / d_out
    const Tensor5D<scalar_t> key,                         // key   / value
    Tensor5D<scalar_t> attn,                              // attn  / d_attn
    const int height,
    const int width,
    const int batch_size,
    const int heads,
    const int dilation_in);

} // namespace natten
