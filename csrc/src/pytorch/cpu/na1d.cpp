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

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <natten/cpu/na1d.h>
#include <natten/dtypes.h>
#include <natten/natten.h>
#include <natten/pytorch/cpu/helpers.h>

namespace natten {
namespace pytorch {
namespace cpu {

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
    const std::tuple<int32_t>& key_tile_size) {
  // TODO: implement CPU reference
  TORCH_CHECK(false, "Fused kernels are not available on CPU yet.");
}

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
    bool compute_delta_with_torch) {
  // TODO: implement CPU reference
  TORCH_CHECK(false, "Fused kernels are not available on CPU yet.");
}

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
    const std::tuple<bool>& is_causal) {
  DISPATCH_DTYPE(
      query.scalar_type(),
      natten::cpu::na1d_qk_forward,
      static_cast<void*>(query.data_ptr()),
      static_cast<void*>(key.data_ptr()),
      bias.has_value() ? static_cast<void*>(bias.value().data_ptr()) : nullptr,
      static_cast<void*>(attn.data_ptr()),
      batch_size,
      heads,
      length,
      dim,
      attn.stride(0),
      attn.stride(1),
      attn.stride(2),
      kernel_size,
      dilation,
      is_causal);
}

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
    const std::tuple<bool>& is_causal) {
  DISPATCH_DTYPE(
      d_attn.scalar_type(),
      natten::cpu::na1d_qk_backward,
      static_cast<void*>(query.data_ptr()),
      static_cast<void*>(key.data_ptr()),
      static_cast<void*>(d_attn.data_ptr()),
      static_cast<void*>(d_query.data_ptr()),
      static_cast<void*>(d_key.data_ptr()),
      d_bias.has_value() ? static_cast<void*>(d_bias.value().data_ptr())
                         : nullptr,
      batch_size,
      heads,
      length,
      dim,
      d_attn.stride(0),
      d_attn.stride(1),
      d_attn.stride(2),
      kernel_size,
      dilation,
      is_causal);
}

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
    const std::tuple<bool>& is_causal) {
  DISPATCH_DTYPE(
      attn.scalar_type(),
      natten::cpu::na1d_av_forward,
      static_cast<void*>(attn.data_ptr()),
      static_cast<void*>(value.data_ptr()),
      static_cast<void*>(output.data_ptr()),
      batch_size,
      heads,
      length,
      dim,
      attn.stride(0),
      attn.stride(1),
      attn.stride(2),
      kernel_size,
      dilation,
      is_causal);
}

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
    const std::tuple<bool>& is_causal) {
  DISPATCH_DTYPE(
      d_out.scalar_type(),
      natten::cpu::na1d_av_backward,
      static_cast<void*>(attn.data_ptr()),
      static_cast<void*>(value.data_ptr()),
      static_cast<void*>(d_out.data_ptr()),
      static_cast<void*>(d_attn.data_ptr()),
      static_cast<void*>(d_value.data_ptr()),
      batch_size,
      heads,
      length,
      dim,
      attn.stride(0),
      attn.stride(1),
      attn.stride(2),
      kernel_size,
      dilation,
      is_causal);
}

} // namespace cpu
} // namespace pytorch
} // namespace natten
