/***************************************************************************************************
 * Copyright (c) 2022-2025 Ali Hassani.
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
    \brief Hopper FNA interface
*/

#pragma once
#include <ATen/ATen.h>

#include <natten/natten.h>

namespace natten {

// Forward

void hopper_na1d_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::optional<at::Tensor>& logsumexp,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& stride,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t>& q_shape_,
    const std::tuple<int32_t>& kv_shape_,
    const std::tuple<int32_t>& qkv_shape_,
    const std::tuple<int32_t>& query_tile_size,
    const std::tuple<int32_t>& key_tile_size,
    int kernel_type);

void hopper_na2d_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::optional<at::Tensor>& logsumexp,
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t>& dilation,
    const std::tuple<bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t>& q_shape_,
    const std::tuple<int32_t, int32_t>& kv_shape_,
    const std::tuple<int32_t, int32_t>& qkv_shape_,
    const std::tuple<int32_t, int32_t>& query_tile_size,
    const std::tuple<int32_t, int32_t>& key_tile_size,
    int kernel_type);

void hopper_na3d_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::optional<at::Tensor>& logsumexp,
    const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t, int32_t>& dilation,
    const std::tuple<bool, bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t, int32_t>& q_shape_,
    const std::tuple<int32_t, int32_t, int32_t>& kv_shape_,
    const std::tuple<int32_t, int32_t, int32_t>& qkv_shape_,
    const std::tuple<int32_t, int32_t, int32_t>& query_tile_size,
    const std::tuple<int32_t, int32_t, int32_t>& key_tile_size,
    int kernel_type);

// Backward

void hopper_na1d_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& stride,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t>& q_shape_,
    const std::tuple<int32_t>& kv_shape_,
    const std::tuple<int32_t>& qkv_shape_,
    const std::tuple<int32_t>& query_tile_shape,
    const std::tuple<int32_t>& key_tile_shape);

void hopper_na2d_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t>& dilation,
    const std::tuple<bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t>& q_shape_,
    const std::tuple<int32_t, int32_t>& kv_shape_,
    const std::tuple<int32_t, int32_t>& qkv_shape_,
    const std::tuple<int32_t, int32_t>& query_tile_shape,
    const std::tuple<int32_t, int32_t>& key_tile_shape);

void hopper_na3d_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t, int32_t>& dilation,
    const std::tuple<bool, bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t, int32_t>& q_shape_,
    const std::tuple<int32_t, int32_t, int32_t>& kv_shape_,
    const std::tuple<int32_t, int32_t, int32_t>& qkv_shape_,
    const std::tuple<int32_t, int32_t, int32_t>& query_tile_shape,
    const std::tuple<int32_t, int32_t, int32_t>& key_tile_shape);

} // namespace natten
