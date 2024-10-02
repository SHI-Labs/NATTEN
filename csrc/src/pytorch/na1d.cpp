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

#include <natten/natten.h>
#include <natten/pytorch/cpu/na1d.h>
#ifdef NATTEN_WITH_CUDA
#include <natten/pytorch/cuda/na1d.cuh>
#endif
#include <natten/pytorch/helpers.h>

namespace natten {
namespace pytorch {

void na1d_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::optional<at::Tensor>& rpb,
    const at::optional<at::Tensor>& logsumexp,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t>& query_tile_size,
    const std::tuple<int32_t>& key_tile_size) {
  AssertDimsAre128BitAligned(query, value);
  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(value);
  CheckArgs(kernel_size, dilation);
  CheckIfPropertiesMatch(query, key, value);
  CheckIfTensorShapesMatch<1>(query, key);
  CheckIfTensorShapesMatch<1>(query, value);
  CheckIfTensorShapesMatch<1>(out, value);
  int32_t batch_size = query.size(0);
  int32_t length = query.size(1);
  int32_t heads = query.size(2);
  int32_t dim = query.size(3);
  CheckArgsAgainstDim({length}, kernel_size, dilation);
  if (rpb.has_value()) {
    CheckBias<1>(query, rpb.value(), heads, kernel_size);
  }
  if (logsumexp.has_value()) {
    CheckLogSumExp<1>(out, logsumexp.value());
  }
  DISPATCH_DEVICE(
      query.device(),
      na1d_forward,
      query,
      key,
      value,
      out,
      rpb,
      logsumexp,
      batch_size,
      length,
      heads,
      dim,
      kernel_size,
      dilation,
      is_causal,
      attn_scale,
      query_tile_size,
      key_tile_size);
}

void na1d_backward(
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
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t>& query_tile_size,
    const std::tuple<int32_t>& key_tile_size,
    const std::tuple<int32_t>& num_splits_key,
    bool compute_delta_with_torch) {
  AssertDimsAre128BitAligned(query, value);
  // TODO: please please simplify these checks!!!
  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(value);
  CHECK_CONTIGUOUS(grad_query);
  CHECK_CONTIGUOUS(grad_key);
  CHECK_CONTIGUOUS(grad_value);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(logsumexp);
  CheckArgs(kernel_size, dilation);
  CheckIfPropertiesMatch(query, key, value);
  CheckIfPropertiesMatch(grad_value, grad_out, out);
  CheckIfPropertiesMatch(grad_query, grad_key, grad_value);
  CheckIfPropertiesMatch(grad_query, query, value);
  CheckIfTensorShapesMatch<1>(query, key);
  CheckIfTensorShapesMatch<1>(query, value);
  CheckIfTensorShapesMatch<1>(out, value);
  CheckIfTensorShapesMatch<1>(grad_query, grad_key);
  CheckIfTensorShapesMatch<1>(grad_query, grad_value);
  CheckIfTensorShapesMatch<1>(grad_out, grad_value);
  CheckIfTensorShapesMatch<1>(grad_out, out);
  CheckLogSumExp<1>(out, logsumexp);
  int32_t batch_size = query.size(0);
  int32_t length = query.size(1);
  int32_t heads = query.size(2);
  int32_t dim = query.size(3);
  CheckArgsAgainstDim({length}, kernel_size, dilation);
  DISPATCH_DEVICE(
      query.device(),
      na1d_backward,
      grad_out,
      query,
      key,
      value,
      logsumexp,
      out,
      grad_query,
      grad_key,
      grad_value,
      batch_size,
      length,
      heads,
      dim,
      kernel_size,
      dilation,
      is_causal,
      attn_scale,
      query_tile_size,
      key_tile_size,
      num_splits_key,
      compute_delta_with_torch);
}

void na1d_qk_forward(
    at::Tensor& attn,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::optional<at::Tensor>& bias,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal) {
  TORCH_CHECK(
      !any_true(is_causal) || !bias.has_value(),
      "Neighborhood attention with causal masking does not support positional biases yet.");
  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CheckArgs(kernel_size, dilation);
  CheckIfPropertiesMatch(query, key, attn);
  CheckIfTensorShapesMatch<1>(query, key);
  CheckAttnShape<1>(query, attn, kernel_size);
  int32_t batch_size = query.size(0);
  int32_t heads = query.size(1);
  int32_t length = query.size(2);
  int32_t dim = query.size(3);
  CheckArgsAgainstDim({length}, kernel_size, dilation);
  if (bias.has_value()) {
    CheckBias<1>(query, bias.value(), heads, kernel_size);
  }
  DISPATCH_DEVICE(
      query.device(),
      na1d_qk_forward,
      query,
      key,
      bias,
      attn,
      batch_size,
      heads,
      length,
      dim,
      kernel_size,
      dilation,
      is_causal);
}

void na1d_qk_backward(
    at::Tensor& d_query,
    at::Tensor& d_key,
    at::optional<at::Tensor>& d_bias,
    const at::Tensor& d_attn,
    const at::Tensor& query,
    const at::Tensor& key,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal) {
  TORCH_CHECK(
      !any_true(is_causal) || !d_bias.has_value(),
      "Neighborhood attention with causal masking does not support positional biases yet.");
  CHECK_CONTIGUOUS(d_query);
  CHECK_CONTIGUOUS(d_key);
  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CheckArgs(kernel_size, dilation);
  CheckIfPropertiesMatch(query, key);
  CheckIfPropertiesMatch(d_query, d_key, d_attn);
  CheckIfTensorShapesMatch<1>(query, key);
  CheckIfTensorShapesMatch<1>(d_query, d_key);
  CheckIfTensorShapesMatch<1>(query, d_key);
  CheckAttnShape<1>(query, d_attn, kernel_size);
  int32_t batch_size = query.size(0);
  int32_t heads = query.size(1);
  int32_t length = query.size(2);
  int32_t dim = query.size(3);
  CheckArgsAgainstDim({length}, kernel_size, dilation);
  if (d_bias.has_value()) {
    CheckBias<1>(query, d_bias.value(), heads, kernel_size);
  }
  DISPATCH_DEVICE(
      d_attn.device(),
      na1d_qk_backward,
      d_attn,
      query,
      key,
      d_query,
      d_key,
      d_bias,
      batch_size,
      heads,
      length,
      dim,
      kernel_size,
      dilation,
      is_causal);
}

void na1d_av_forward(
    at::Tensor& out,
    const at::Tensor& attn,
    const at::Tensor& value,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal) {
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(value);
  CheckArgs(kernel_size, dilation);
  CheckIfPropertiesMatch(out, value, attn);
  CheckIfTensorShapesMatch<1>(out, value);
  CheckAttnShape<1>(value, attn, kernel_size);
  int32_t batch_size = value.size(0);
  int32_t heads = value.size(1);
  int32_t length = value.size(2);
  int32_t dim = value.size(3);
  CheckArgsAgainstDim({length}, kernel_size, dilation);
  DISPATCH_DEVICE(
      attn.device(),
      na1d_av_forward,
      attn,
      value,
      out,
      batch_size,
      heads,
      length,
      dim,
      kernel_size,
      dilation,
      is_causal);
}

void na1d_av_backward(
    at::Tensor& d_attn,
    at::Tensor& d_value,
    const at::Tensor& d_out,
    const at::Tensor& attn,
    const at::Tensor& value,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal) {
  CHECK_CONTIGUOUS(d_out);
  CHECK_CONTIGUOUS(d_value);
  CHECK_CONTIGUOUS(value);
  CheckArgs(kernel_size, dilation);
  CheckIfPropertiesMatch(attn, value);
  CheckIfPropertiesMatch(d_attn, d_value, d_out);
  CheckIfTensorShapesMatch<1>(value, d_value);
  CheckIfTensorShapesMatch<1>(attn, d_attn);
  CheckIfTensorShapesMatch<1>(value, d_out);
  CheckAttnShape<1>(value, attn, kernel_size);
  int32_t batch_size = value.size(0);
  int32_t heads = value.size(1);
  int32_t length = value.size(2);
  int32_t dim = value.size(3);
  CheckArgsAgainstDim({length}, kernel_size, dilation);
  DISPATCH_DEVICE(
      attn.device(),
      na1d_av_backward,
      d_out,
      attn,
      value,
      d_attn,
      d_value,
      batch_size,
      heads,
      length,
      dim,
      kernel_size,
      dilation,
      is_causal);
}

} // namespace pytorch
} // namespace natten
