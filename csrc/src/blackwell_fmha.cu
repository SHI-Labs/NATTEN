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
    \brief Blackwell FMHA interface
*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <natten/helpers.h>
#include <natten/natten.h>

#if defined(NATTEN_WITH_CUTLASS) && defined(NATTEN_WITH_BLACKWELL_FNA)
#include <natten_autogen/cuda/blackwell_fmha/interface.h>
#include <natten_autogen/cuda/blackwell_fmha_bwd/interface.h>
#include <natten/cuda/fmha_blackwell/fmha_backward.cuh>
#include <natten/cuda/fmha_blackwell/fmha_forward.cuh>
#endif

namespace natten {

void blackwell_fmha_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::optional<at::Tensor>& logsumexp,
    bool is_causal,
    float attn_scale,
    int query_tile_size,
    int key_tile_size,
    bool run_persistent,
    // varlen
    const at::optional<at::Tensor>& cumulative_seqlen_Q,
    const at::optional<at::Tensor>& cumulative_seqlen_KV,
    // only used if cumulative_seqlen_Q and cumulative_seqlen_KV are specified
    int max_seqlen_Q,
    int max_seqlen_KV) {
#if defined(NATTEN_WITH_CUTLASS) && defined(NATTEN_WITH_BLACKWELL_FNA)
  AssertDimsAre128BitAligned(query, value);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(value);
  CHECK_CONTIGUOUS(out);

  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(out);

  at::cuda::OptionalCUDAGuard device_guard(query.device());

  CheckIfPropertiesMatch(query, key, value);
  CheckIfTensorShapesMatch<1>(key, value); // head_dim == head_dim_value
  CheckIfTensorShapesMatch<1>(query, out); // head_dim == head_dim_value

  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");
  TORCH_CHECK(key.dim() == 4, "Tensors must be 4-D.");
  TORCH_CHECK(value.dim() == 4, "Tensors must be 4-D.");

  TORCH_CHECK(
      query.size(0) == key.size(0),
      "Blackwell FMHA forward: Query and key must match in batch size, got ",
      "query.shape[0]=",
      query.size(0),
      ", key.shape[0]=",
      key.size(0));

  TORCH_CHECK(
      query.size(3) == key.size(3),
      "Blackwell FMHA forward: Query and key must match in head dim, got ",
      "query.shape[3]=",
      query.size(3),
      ", key.shape[3]=",
      key.size(3));

  // GQA/MQA is supported
  TORCH_CHECK(
      query.size(2) >= key.size(2),
      "Blackwell FMHA forward: Query heads must be greater than or equal to key/value heads, got ",
      "query.shape[2]=",
      query.size(2),
      ", key.shape[2]=",
      key.size(2));

  TORCH_CHECK(
      query.size(2) % key.size(2) == 0,
      "Blackwell FMHA forward: Query heads must evenly divide key/value heads, got ",
      "query.shape[2]=",
      query.size(2),
      ", key.shape[2]=",
      key.size(2));

  int batch_size = query.size(0);
  int seqlen_q = query.size(1);
  int seqlen_kv = key.size(1);
  int heads_q = query.size(2);
  int heads_kv = key.size(2);
  int dim = query.size(3);

  if (logsumexp.has_value()) {
    CheckLogSumExp<1>(out, logsumexp.value());
    CHECK_CUDA(logsumexp.value());
  }

  TORCH_CHECK(
      dim == 32 || dim == 64 || dim == 128,
      "Blackwell FMHA forward only supports head dims 32, 64, and 128 for now.");

  cudaDeviceProp* device_props =
      at::cuda::getDeviceProperties(query.device().index());
  const int cc = device_props->major * 10 + device_props->minor;

  TORCH_CHECK(
      cc == 100 || cc == 103,
      "Blackwell FMHA forward can only run on the Blackwell (datacenter-class) architecture (SM100, SM103).");

  TORCH_CHECK(
      query.scalar_type() == key.scalar_type() &&
          query.scalar_type() == value.scalar_type() &&
          query.scalar_type() == out.scalar_type(),
      "Blackwell FMHA forward: Query, key, value, and output must match in dtype.");

  TORCH_CHECK(
      query.scalar_type() == torch::kFloat16 ||
          query.scalar_type() == torch::kBFloat16 ||
          query.scalar_type() == c10::ScalarType::Float8_e4m3fn ||
          query.scalar_type() == c10::ScalarType::Float8_e5m2,
      "Blackwell FMHA forward only supports FP16, BF16, FP8_E4M3, and FP8_E5M2.");

  // varlen
  bool is_varlen =
      cumulative_seqlen_Q.has_value() || cumulative_seqlen_KV.has_value();

  void* ptr_cumulative_seqlen_Q = nullptr;
  void* ptr_cumulative_seqlen_KV = nullptr;
  if (is_varlen) {
    TORCH_CHECK(
        cumulative_seqlen_Q.has_value() && cumulative_seqlen_KV.has_value(),
        "Blackwell FMHA: Both cumulative_seqlen_Q and cumulative_seqlen_KV must be specified when using varlen.");

    TORCH_CHECK(
        batch_size == 1,
        "Blackwell FMHA: Tensor batch size must be 1 (packed sequence layout), got ",
        batch_size);

    auto& cumulative_seqlen_Q_tensor = cumulative_seqlen_Q.value();
    auto& cumulative_seqlen_KV_tensor = cumulative_seqlen_KV.value();

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.dim() == 1,
        "Blackwell FMHA: cumulative_seqlen_Q is expected to be a 1-D tensor.");
    TORCH_CHECK(
        cumulative_seqlen_KV_tensor.dim() == 1,
        "Blackwell FMHA: cumulative_seqlen_KV is expected to be a 1-D tensor.");

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.size(0) ==
            cumulative_seqlen_KV_tensor.size(0),
        "Blackwell FMHA: cumulative_seqlen_Q and cumulative_seqlen_KV must be the same size.");

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.size(0) > 1,
        "Blackwell FMHA: cumulative_seqlen_Q and cumulative_seqlen_KV size must be greater than 1.");

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.scalar_type() == torch::kInt,
        "Blackwell FMHA: cumulative_seqlen_Q is expected to be an int32 tensor, got ",
        cumulative_seqlen_Q_tensor.scalar_type());
    TORCH_CHECK(
        cumulative_seqlen_KV_tensor.scalar_type() == torch::kInt,
        "Blackwell FMHA: cumulative_seqlen_KV is expected to be an int32 tensor, got ",
        cumulative_seqlen_KV_tensor.scalar_type());

    batch_size = cumulative_seqlen_Q_tensor.size(0) - 1;
    ptr_cumulative_seqlen_Q =
        static_cast<void*>(cumulative_seqlen_Q_tensor.data_ptr());
    ptr_cumulative_seqlen_KV =
        static_cast<void*>(cumulative_seqlen_KV_tensor.data_ptr());
  }
  //

  int device_id = query.device().index();
  auto cuda_stream = at::cuda::getCurrentCUDAStream(device_id);

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  DISPATCH_BLACKWELL_FMHA_FORWARD(
      query.scalar_type(),
      dim,
      query_tile_size,
      key_tile_size,
      run_persistent,
      static_cast<void*>(query.data_ptr()),
      static_cast<void*>(key.data_ptr()),
      static_cast<void*>(value.data_ptr()),
      static_cast<void*>(out.data_ptr()),
      logsumexp.has_value() ? static_cast<void*>(logsumexp.value().data_ptr())
                            : nullptr,
      batch_size,
      seqlen_q,
      seqlen_kv,
      heads_q,
      heads_kv,
      dim,
      is_causal,
      attn_scale,
      // varlen parameters
      is_varlen,
      max_seqlen_Q,
      max_seqlen_KV,
      ptr_cumulative_seqlen_Q,
      ptr_cumulative_seqlen_KV,
      // init/launch params
      device_id,
      cuda_stream,
      query.options());

#else
  TORCH_CHECK(
      false,
      "Blackwell FMHA forward: libnatten was not compiled with CUTLASS_ARCH_MMA_SM100_SUPPORTED.");
#endif
#else
  TORCH_CHECK(
      false,
      "Blackwell FMHA forward: libnatten was not compiled for Blackwell (SM100/SM103).");
#endif
}

void blackwell_fmha_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    bool is_causal,
    float attn_scale,
    int query_tile_size,
    int key_tile_size,
    // varlen
    const at::optional<at::Tensor>& cumulative_seqlen_Q,
    const at::optional<at::Tensor>& cumulative_seqlen_KV,
    // only used if cumulative_seqlen_Q and cumulative_seqlen_KV are specified
    int max_seqlen_Q,
    int max_seqlen_KV) {
#if defined(NATTEN_WITH_CUTLASS) && defined(NATTEN_WITH_BLACKWELL_FNA)
  AssertDimsAre128BitAligned(query, value);

  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(out);
  CHECK_CUDA(grad_query);
  CHECK_CUDA(grad_key);
  CHECK_CUDA(grad_value);
  CHECK_CUDA(grad_out);
  CHECK_CUDA(logsumexp);

  at::cuda::OptionalCUDAGuard device_guard(query.device());

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(value);
  CHECK_CONTIGUOUS(grad_query);
  CHECK_CONTIGUOUS(grad_key);
  CHECK_CONTIGUOUS(grad_value);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(logsumexp);

  CheckIfPropertiesMatch(query, key, value);
  CheckIfPropertiesMatch(grad_value, grad_out, out);
  CheckIfPropertiesMatch(grad_query, grad_key, grad_value);
  CheckIfPropertiesMatch(grad_query, query, value);

  CheckIfTensorShapesMatch<1>(query, out); // head_dim == head_dim_value
  CheckIfTensorShapesMatch<1>(key, value); // head_dim == head_dim_value

  CheckIfHeadDimsMatch(out, value);
  CheckIfTensorShapesMatch<1>(grad_query, query);
  CheckIfTensorShapesMatch<1>(grad_key, key);
  CheckIfTensorShapesMatch<1>(grad_value, value);
  CheckIfTensorShapesMatch<1>(grad_out, out);

  CheckLogSumExp<1>(out, logsumexp);

  TORCH_CHECK(
      query.size(0) == key.size(0),
      "Blackwell FMHA forward: Query and key must match in batch size, got ",
      "query.shape[0]=",
      query.size(0),
      ", key.shape[0]=",
      key.size(0));

  TORCH_CHECK(
      query.size(3) == key.size(3),
      "Blackwell FMHA forward: Query and key must match in head dim, got ",
      "query.shape[3]=",
      query.size(3),
      ", key.shape[3]=",
      key.size(3));

  // GQA/MQA is supported
  TORCH_CHECK(
      query.size(2) >= key.size(2),
      "Blackwell FMHA forward: Query heads must be greater than or equal to key/value heads, got ",
      "query.shape[2]=",
      query.size(2),
      ", key.shape[2]=",
      key.size(2));

  TORCH_CHECK(
      query.size(2) % key.size(2) == 0,
      "Blackwell FMHA forward: Query heads must evenly divide key/value heads, got ",
      "query.shape[2]=",
      query.size(2),
      ", key.shape[2]=",
      key.size(2));

  int batch_size = query.size(0);
  int seqlen_q = query.size(1);
  int seqlen_kv = key.size(1);
  int heads_q = query.size(2);
  int heads_kv = key.size(2);
  int dim = query.size(3);

  TORCH_CHECK(
      dim == 32 || dim == 64 || dim == 128,
      "Blackwell FMHA backward only supports head dims 32, 64, and 128 for now.");

  TORCH_CHECK(
      query.scalar_type() == torch::kFloat16 ||
          query.scalar_type() == torch::kBFloat16,
      "Blackwell FMHA backward only supports FP16 and BF16.");

  // varlen
  bool is_varlen =
      cumulative_seqlen_Q.has_value() || cumulative_seqlen_KV.has_value();

  void* ptr_cumulative_seqlen_Q = nullptr;
  void* ptr_cumulative_seqlen_KV = nullptr;
  if (is_varlen) {
    TORCH_CHECK(
        cumulative_seqlen_Q.has_value() && cumulative_seqlen_KV.has_value(),
        "Blackwell FMHA: Both cumulative_seqlen_Q and cumulative_seqlen_KV must be specified when using varlen.");

    TORCH_CHECK(
        batch_size == 1,
        "Blackwell FMHA: Tensor batch size must be 1 (packed sequence layout), got ",
        batch_size);

    auto& cumulative_seqlen_Q_tensor = cumulative_seqlen_Q.value();
    auto& cumulative_seqlen_KV_tensor = cumulative_seqlen_KV.value();

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.dim() == 1,
        "Blackwell FMHA: cumulative_seqlen_Q is expected to be a 1-D tensor.");
    TORCH_CHECK(
        cumulative_seqlen_KV_tensor.dim() == 1,
        "Blackwell FMHA: cumulative_seqlen_KV is expected to be a 1-D tensor.");

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.size(0) ==
            cumulative_seqlen_KV_tensor.size(0),
        "Blackwell FMHA: cumulative_seqlen_Q and cumulative_seqlen_KV must be the same size.");

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.size(0) > 1,
        "Blackwell FMHA: cumulative_seqlen_Q and cumulative_seqlen_KV size must be greater than 1.");

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.scalar_type() == torch::kInt,
        "Blackwell FMHA: cumulative_seqlen_Q is expected to be an int32 tensor, got ",
        cumulative_seqlen_Q_tensor.scalar_type());
    TORCH_CHECK(
        cumulative_seqlen_KV_tensor.scalar_type() == torch::kInt,
        "Blackwell FMHA: cumulative_seqlen_KV is expected to be an int32 tensor, got ",
        cumulative_seqlen_KV_tensor.scalar_type());

    batch_size = cumulative_seqlen_Q_tensor.size(0) - 1;
    ptr_cumulative_seqlen_Q =
        static_cast<void*>(cumulative_seqlen_Q_tensor.data_ptr());
    ptr_cumulative_seqlen_KV =
        static_cast<void*>(cumulative_seqlen_KV_tensor.data_ptr());
  }
  //

  TORCH_CHECK(
      not at::globalContext().deterministicAlgorithms(),
      "Blackwell FMHA backward is non-deterministic, "
      "but PyTorch's deterministic mode is enabled. "
      "NATTEN Python API should have avoided this; which means "
      "you're probably calling the C function directly.");

  int device_id = query.device().index();
  auto cuda_stream = at::cuda::getCurrentCUDAStream(device_id);
  cudaDeviceProp* device_props = at::cuda::getDeviceProperties(device_id);
  const int cc = device_props->major * 10 + device_props->minor;

  TORCH_CHECK(
      cc == 100 || cc == 103,
      "Blackwell FMHA backward can only run on the Blackwell (datacenter-class) architecture (SM100, SM103).");

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  DISPATCH_BLACKWELL_FMHA_BACKWARD(
      query.scalar_type(),
      dim,
      query_tile_size,
      key_tile_size,
      static_cast<void*>(query.data_ptr()),
      static_cast<void*>(key.data_ptr()),
      static_cast<void*>(value.data_ptr()),
      static_cast<void*>(out.data_ptr()),
      static_cast<void*>(logsumexp.data_ptr()),
      static_cast<void*>(grad_query.data_ptr()),
      static_cast<void*>(grad_key.data_ptr()),
      static_cast<void*>(grad_value.data_ptr()),
      static_cast<void*>(grad_out.data_ptr()),
      batch_size,
      seqlen_q,
      seqlen_kv,
      heads_q,
      heads_kv,
      dim,
      is_causal,
      attn_scale,
      // varlen parameters
      is_varlen,
      max_seqlen_Q,
      max_seqlen_KV,
      ptr_cumulative_seqlen_Q,
      ptr_cumulative_seqlen_KV,
      // init/launch params
      device_id,
      cuda_stream,
      query.options());

#else
  TORCH_CHECK(
      false,
      "Blackwell FMHA backward: libnatten was not compiled with CUTLASS_ARCH_MMA_SM100_SUPPORTED.");
#endif
#else
  TORCH_CHECK(
      false,
      "Blackwell FMHA backward: libnatten was not compiled for Blackwell (SM100/SM103).");
#endif
}

} // namespace natten
