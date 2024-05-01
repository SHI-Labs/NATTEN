/***************************************************************************************************
 * Copyright (c) 2022-2024 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
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
    \brief Neighborhood Attention 1D Torch interface
*/

#pragma once
#include <ATen/ATen.h>

#include <natten/natten.h>

namespace natten {
namespace pytorch {
namespace cuda {

void na1d_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& out,
    const at::optional<at::Tensor>& rpb,
    const at::optional<at::Tensor>& logsumexp,
    int32_t batch_size,
    int32_t length,
    int32_t heads,
    int32_t dim,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t>& query_tile_size,
    const std::tuple<int32_t>& key_tile_size);

void na1d_backward(
    const at::Tensor& grad_out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& logsumexp,
    const at::Tensor& out,
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    int32_t batch_size,
    int32_t length,
    int32_t heads,
    int32_t dim,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t>& query_tile_size,
    const std::tuple<int32_t>& key_tile_size,
    const std::tuple<int32_t>& num_splits_key,
    bool compute_delta_with_torch);

void na1d_qk_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::optional<at::Tensor>& bias,
    at::Tensor& attn,
    int32_t batch_size,
    int32_t heads,
    int32_t length,
    int32_t dim,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal);

void na1d_qk_backward(
    const at::Tensor& d_attn,
    const at::Tensor& query,
    const at::Tensor& key,
    at::Tensor& d_query,
    at::Tensor& d_key,
    at::optional<at::Tensor>& d_bias,
    int32_t batch_size,
    int32_t heads,
    int32_t length,
    int32_t dim,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal);

void na1d_av_forward(
    const at::Tensor& attn,
    const at::Tensor& value,
    at::Tensor& output,
    int32_t batch_size,
    int32_t heads,
    int32_t length,
    int32_t dim,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal);

void na1d_av_backward(
    const at::Tensor& d_out,
    const at::Tensor& attn,
    const at::Tensor& value,
    at::Tensor& d_attn,
    at::Tensor& d_value,
    int32_t batch_size,
    int32_t heads,
    int32_t length,
    int32_t dim,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal);

} // namespace cuda
} // namespace pytorch
} // namespace natten
