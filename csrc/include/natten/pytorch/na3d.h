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
    \brief Neighborhood Attention 3D Torch interface
*/

#pragma once
#include <ATen/ATen.h>
#include <vector>

namespace natten {
namespace pytorch {

at::Tensor na3d_qk_forward(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::optional<at::Tensor> &bias,
    const int kernel_size,
    const int dilation,
    const int depth_kernel_size,
    const int depth_dilation);

std::vector<at::Tensor> na3d_qk_backward(
    const at::Tensor &d_attn,
    const at::Tensor &query,
    const at::Tensor &key,
    const bool has_bias,
    const int kernel_size,
    const int dilation,
    const int depth_kernel_size,
    const int depth_dilation);

at::Tensor na3d_av_forward(
    const at::Tensor &attn,
    const at::Tensor &value,
    const int kernel_size,
    const int dilation,
    const int depth_kernel_size,
    const int depth_dilation);

std::vector<at::Tensor> na3d_av_backward(
    const at::Tensor &d_out,
    const at::Tensor &attn,
    const at::Tensor &value,
    const int kernel_size,
    const int dilation,
    const int depth_kernel_size,
    const int depth_dilation);

} // namespace pytorch
} // namespace natten

