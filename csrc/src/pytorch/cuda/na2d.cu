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
    \brief Neighborhood Attention 2D Torch interface
*/

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "natten/pytorch/cuda/helpers.cuh"
#include "natten/dtypes.cuh"
#include "natten/cuda/na2d.cuh"

namespace natten {
namespace pytorch {
namespace cuda {

void na2d_qk_forward(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::optional<at::Tensor> &bias,
    at::Tensor &attn,
    const int batch_size,
    const int heads,
    const int height,
    const int width,
    const int dim,
    const int kernel_size,
    const int dilation) {
    DISPATCH_DTYPE(query.scalar_type(), natten::cuda::na2d_qk_forward,
            static_cast<void *>(query.data_ptr()),
            static_cast<void *>(key.data_ptr()),
            bias.has_value() ? static_cast<void *>(bias.value().data_ptr()) : nullptr,
            static_cast<void *>(attn.data_ptr()),
            batch_size, heads, height, width, dim,
            kernel_size, dilation);
}

void na2d_qk_backward(
    const at::Tensor &d_attn,
    const at::Tensor &query,
    const at::Tensor &key,
    at::Tensor &d_query,
    at::Tensor &d_key,
    at::Tensor &d_bias,
    const int batch_size,
    const int heads,
    const int height,
    const int width,
    const int dim,
    const int kernel_size,
    const int dilation) {
    DISPATCH_DTYPE(d_attn.scalar_type(), natten::cuda::na2d_qk_backward,
            static_cast<void *>(query.data_ptr()),
            static_cast<void *>(key.data_ptr()),
            static_cast<void *>(d_attn.data_ptr()),
            static_cast<void *>(d_query.data_ptr()),
            static_cast<void *>(d_key.data_ptr()),
            d_bias.has_storage() ? static_cast<void *>(d_bias.data_ptr()) : nullptr,
            batch_size, heads, height, width, dim,
            kernel_size, dilation);
}

void na2d_av_forward(
    const at::Tensor &attn,
    const at::Tensor &value,
    at::Tensor &output,
    const int batch_size,
    const int heads,
    const int height,
    const int width,
    const int dim,
    const int kernel_size,
    const int dilation) {
    DISPATCH_DTYPE(attn.scalar_type(), natten::cuda::na2d_av_forward,
            static_cast<void *>(attn.data_ptr()),
            static_cast<void *>(value.data_ptr()),
            static_cast<void *>(output.data_ptr()),
            batch_size, heads, height, width, dim,
            kernel_size, dilation);
}

void na2d_av_backward(
    const at::Tensor &d_out,
    const at::Tensor &attn,
    const at::Tensor &value,
    at::Tensor &d_attn,
    at::Tensor &d_value,
    const int batch_size,
    const int heads,
    const int height,
    const int width,
    const int dim,
    const int kernel_size,
    const int dilation) {
    DISPATCH_DTYPE(d_out.scalar_type(), natten::cuda::na2d_av_backward,
            static_cast<void *>(attn.data_ptr()),
            static_cast<void *>(value.data_ptr()),
            static_cast<void *>(d_out.data_ptr()),
            static_cast<void *>(d_attn.data_ptr()),
            static_cast<void *>(d_value.data_ptr()),
            batch_size, heads, height, width, dim,
            kernel_size, dilation);
}

} // namespace cuda
} // namespace pytorch
} // namespace natten
