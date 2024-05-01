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
    \brief Neighborhood Attention 2D Torch interface
*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <natten/natten.h>
#include <natten/cuda/na2d.cuh>
#include <natten/dtypes.cuh>
#include <natten/pytorch/cuda/compute_delta.cuh>
#include <natten/pytorch/cuda/helpers.cuh>

namespace natten {
namespace pytorch {
namespace cuda {

void na2d_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& out,
    const at::optional<at::Tensor>& rpb,
    const at::optional<at::Tensor>& logsumexp,
    int32_t batch_size,
    int32_t height,
    int32_t width,
    int32_t heads,
    int32_t dim,
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& dilation,
    const std::tuple<bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t>& query_tile_size,
    const std::tuple<int32_t, int32_t>& key_tile_size) {
  at::Tensor workspace;
  // TODO: figure out a better solution than this that doesn't
  // involve calling the FNA dispatcher for the sole purpose of
  // getting workspace size, and doesn't force us to use torch API
  // in the backend. This works just okay, but I hate the idea of
  // passing down a lambda function like this.
  auto alloc_bytes = [&workspace, &query](
                         void** ptr, int64_t bytes, bool zfill) {
    workspace = at::empty({bytes}, query.options().dtype(at::ScalarType::Byte));
    if (zfill) {
      workspace.zero_();
    }
    *ptr = static_cast<void*>(workspace.data_ptr());
  };
  DISPATCH_DTYPE(
      query.device().index(),
      at::cuda::getCurrentCUDAStream(query.device().index()),
      query.scalar_type(),
      natten::cuda::na2d_forward,
      alloc_bytes,
      static_cast<void*>(query.data_ptr()),
      static_cast<void*>(key.data_ptr()),
      static_cast<void*>(value.data_ptr()),
      static_cast<void*>(out.data_ptr()),
      rpb.has_value() ? static_cast<void*>(rpb.value().data_ptr()) : nullptr,
      logsumexp.has_value() ? static_cast<void*>(logsumexp.value().data_ptr())
                            : nullptr,
      batch_size,
      height,
      width,
      heads,
      dim,
      kernel_size,
      dilation,
      is_causal,
      attn_scale,
      query_tile_size,
      key_tile_size);
}

void na2d_backward(
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
    int32_t height,
    int32_t width,
    int32_t heads,
    int32_t dim,
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& dilation,
    const std::tuple<bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t>& query_tile_size,
    const std::tuple<int32_t, int32_t>& key_tile_size,
    const std::tuple<int32_t, int32_t>& num_splits_key,
    bool compute_delta_with_torch) {
  at::Tensor workspace;
  // TODO: figure out a better solution than this that doesn't
  // involve calling the FNA dispatcher for the sole purpose of
  // getting workspace size, and doesn't force us to use torch API
  // in the backend. This works just okay, but I hate the idea of
  // passing down a lambda function like this.
  auto alloc_bytes = [&workspace, &query](
                         void** ptr, int64_t bytes, bool zfill) {
    workspace = at::empty({bytes}, query.options().dtype(at::ScalarType::Byte));
    if (zfill) {
      workspace.zero_();
    }
    *ptr = static_cast<void*>(workspace.data_ptr());
  };
  at::Tensor delta;
  if (compute_delta_with_torch) {
    delta =
        (grad_out.to(at::kFloat) * out.to(at::kFloat)).flatten(1, 2).sum(-1);
  } else {
    delta = torch::empty(
        {batch_size, height * width, heads}, query.options().dtype(at::kFloat));
    compute_delta(out, grad_out, delta, (int32_t)delta.numel(), dim);
  }
  TORCH_CHECK(delta.size(0) == batch_size);
  TORCH_CHECK(delta.size(1) == height * width);
  TORCH_CHECK(delta.size(2) == heads);
  if (at::globalContext().deterministicAlgorithms()) {
    TORCH_CHECK(
        natten::flatten(num_splits_key) <= 1,
        "FNA-backward was called with KV parallelism, "
        "which makes it algorithm non-deterministic, "
        "but PyTorch's deterministic mode is enabled. "
        "NATTEN Python API should have avoided this; which means "
        "you're probably calling the C function directly.");
  }
  DISPATCH_DTYPE(
      query.device().index(),
      at::cuda::getCurrentCUDAStream(query.device().index()),
      query.scalar_type(),
      natten::cuda::na2d_backward,
      alloc_bytes,
      static_cast<void*>(grad_out.data_ptr()),
      static_cast<void*>(query.data_ptr()),
      static_cast<void*>(key.data_ptr()),
      static_cast<void*>(value.data_ptr()),
      static_cast<void*>(logsumexp.data_ptr()),
      static_cast<void*>(delta.data_ptr()),
      static_cast<void*>(out.data_ptr()),
      static_cast<void*>(grad_query.data_ptr()),
      static_cast<void*>(grad_key.data_ptr()),
      static_cast<void*>(grad_value.data_ptr()),
      batch_size,
      height,
      width,
      heads,
      dim,
      kernel_size,
      dilation,
      is_causal,
      attn_scale,
      query_tile_size,
      key_tile_size,
      num_splits_key);
}

void na2d_qk_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::optional<at::Tensor>& bias,
    at::Tensor& attn,
    int32_t batch_size,
    int32_t heads,
    int32_t height,
    int32_t width,
    int32_t dim,
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& dilation,
    const std::tuple<bool, bool>& is_causal) {
  DISPATCH_DTYPE(
      query.device().index(),
      at::cuda::getCurrentCUDAStream(query.device().index()),
      query.scalar_type(),
      natten::cuda::na2d_qk_forward,
      static_cast<void*>(query.data_ptr()),
      static_cast<void*>(key.data_ptr()),
      bias.has_value() ? static_cast<void*>(bias.value().data_ptr()) : nullptr,
      static_cast<void*>(attn.data_ptr()),
      batch_size,
      heads,
      height,
      width,
      dim,
      attn.stride(0),
      attn.stride(1),
      attn.stride(2),
      attn.stride(3),
      kernel_size,
      dilation,
      is_causal);
}

void na2d_qk_backward(
    const at::Tensor& d_attn,
    const at::Tensor& query,
    const at::Tensor& key,
    at::Tensor& d_query,
    at::Tensor& d_key,
    at::optional<at::Tensor>& d_bias,
    int32_t batch_size,
    int32_t heads,
    int32_t height,
    int32_t width,
    int32_t dim,
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& dilation,
    const std::tuple<bool, bool>& is_causal) {
  // dRPB is always computed in FP32; there is no FP16/BF16 kernel for it.
  auto should_cast_bias = false;
  void* d_bias_ptr = nullptr;
  at::optional<at::Tensor> d_bias_fp32;
  if (d_bias.has_value()) {
    auto d_bias_tensor = d_bias.value();
    if (d_bias_tensor.scalar_type() == torch::kFloat16 ||
        d_bias_tensor.scalar_type() == torch::kBFloat16) {
      should_cast_bias = true;
      d_bias_fp32 = at::zeros(
          d_bias_tensor.sizes(), d_bias_tensor.options().dtype(torch::kFloat));
      d_bias_ptr = static_cast<void*>(d_bias_fp32.value().data_ptr());
    } else {
      d_bias_ptr = static_cast<void*>(d_bias_tensor.data_ptr());
    }
    if (at::globalContext().deterministicAlgorithms()) {
      TORCH_CHECK(
          false,
          "Training NATTEN with relative positional "
          "is non-deterministic, "
          "but PyTorch's deterministic mode is enabled. ");
    }
  }
  DISPATCH_DTYPE(
      d_attn.device().index(),
      at::cuda::getCurrentCUDAStream(query.device().index()),
      d_attn.scalar_type(),
      natten::cuda::na2d_qk_backward,
      static_cast<void*>(query.data_ptr()),
      static_cast<void*>(key.data_ptr()),
      static_cast<void*>(d_attn.data_ptr()),
      static_cast<void*>(d_query.data_ptr()),
      static_cast<void*>(d_key.data_ptr()),
      d_bias_ptr,
      batch_size,
      heads,
      height,
      width,
      dim,
      d_attn.stride(0),
      d_attn.stride(1),
      d_attn.stride(2),
      d_attn.stride(3),
      kernel_size,
      dilation,
      is_causal);
  if (should_cast_bias) {
    TORCH_CHECK(
        d_bias.has_value() && d_bias_fp32.has_value(),
        "Something went wrong when casting biases. Please open an issue.");
    auto d_bias_tensor = d_bias.value();
    auto d_bias_fp32_tensor = d_bias_fp32.value();
    auto d_bias_half = d_bias_fp32_tensor.toType(d_bias_tensor.scalar_type());
    d_bias_tensor.copy_(d_bias_half);
  }
}

void na2d_av_forward(
    const at::Tensor& attn,
    const at::Tensor& value,
    at::Tensor& output,
    int32_t batch_size,
    int32_t heads,
    int32_t height,
    int32_t width,
    int32_t dim,
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& dilation,
    const std::tuple<bool, bool>& is_causal) {
  DISPATCH_DTYPE(
      attn.device().index(),
      at::cuda::getCurrentCUDAStream(attn.device().index()),
      attn.scalar_type(),
      natten::cuda::na2d_av_forward,
      static_cast<void*>(attn.data_ptr()),
      static_cast<void*>(value.data_ptr()),
      static_cast<void*>(output.data_ptr()),
      batch_size,
      heads,
      height,
      width,
      dim,
      attn.stride(0),
      attn.stride(1),
      attn.stride(2),
      attn.stride(3),
      kernel_size,
      dilation,
      is_causal);
}

void na2d_av_backward(
    const at::Tensor& d_out,
    const at::Tensor& attn,
    const at::Tensor& value,
    at::Tensor& d_attn,
    at::Tensor& d_value,
    int32_t batch_size,
    int32_t heads,
    int32_t height,
    int32_t width,
    int32_t dim,
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& dilation,
    const std::tuple<bool, bool>& is_causal) {
  DISPATCH_DTYPE(
      d_out.device().index(),
      at::cuda::getCurrentCUDAStream(attn.device().index()),
      d_out.scalar_type(),
      natten::cuda::na2d_av_backward,
      static_cast<void*>(attn.data_ptr()),
      static_cast<void*>(value.data_ptr()),
      static_cast<void*>(d_out.data_ptr()),
      static_cast<void*>(d_attn.data_ptr()),
      static_cast<void*>(d_value.data_ptr()),
      batch_size,
      heads,
      height,
      width,
      dim,
      attn.stride(0),
      attn.stride(1),
      attn.stride(2),
      attn.stride(3),
      kernel_size,
      dilation,
      is_causal);
}

} // namespace cuda
} // namespace pytorch
} // namespace natten
