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
    \brief FMHA interface
*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <natten/compute_delta.h>
#include <natten/helpers.h>
#include <natten/natten.h>

#ifdef NATTEN_WITH_CUTLASS
#include <natten_autogen/cuda/fmha/interface.h>
#include <natten/cuda/fmha/fmha_backward.cuh>
#include <natten/cuda/fmha/fmha_forward.cuh>
#endif

namespace natten {

void fmha_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::optional<at::Tensor>& logsumexp,
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
#ifdef NATTEN_WITH_CUTLASS
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
  CheckIfTensorShapesMatchExceptHeadDim<1>(key, value);
  CheckIfTensorShapesMatchExceptHeadDim<1>(query, out);
  CheckIfBatchHeadsHeadDimMatch(query, key);
  CheckIfHeadDimsMatch(out, value);

  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");
  TORCH_CHECK(key.dim() == 4, "Tensors must be 4-D.");
  TORCH_CHECK(value.dim() == 4, "Tensors must be 4-D.");

  int batch_size = query.size(0);
  int seqlen_q = query.size(1);
  int seqlen_kv = key.size(1);
  int heads = query.size(2);
  int dim = query.size(3);
  int dim_value = value.size(3);

  if (logsumexp.has_value()) {
    CheckLogSumExp<1>(out, logsumexp.value());
    CHECK_CUDA(logsumexp.value());
  }

  // varlen
  bool is_varlen =
      cumulative_seqlen_Q.has_value() || cumulative_seqlen_KV.has_value();

  void* ptr_cumulative_seqlen_Q = nullptr;
  void* ptr_cumulative_seqlen_KV = nullptr;
  if (is_varlen) {
    TORCH_CHECK(
        cumulative_seqlen_Q.has_value() && cumulative_seqlen_KV.has_value(),
        "CUTLASS FMHA: Both cumulative_seqlen_Q and cumulative_seqlen_KV must be specified when using varlen.");

    TORCH_CHECK(
        batch_size == 1,
        "CUTLASS FMHA: Tensor batch size must be 1 (packed sequence layout), got ",
        batch_size);

    auto& cumulative_seqlen_Q_tensor = cumulative_seqlen_Q.value();
    auto& cumulative_seqlen_KV_tensor = cumulative_seqlen_KV.value();

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.dim() == 1,
        "CUTLASS FMHA: cumulative_seqlen_Q is expected to be a 1-D tensor.");
    TORCH_CHECK(
        cumulative_seqlen_KV_tensor.dim() == 1,
        "CUTLASS FMHA: cumulative_seqlen_KV is expected to be a 1-D tensor.");

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.size(0) ==
            cumulative_seqlen_KV_tensor.size(0),
        "CUTLASS FMHA: cumulative_seqlen_Q and cumulative_seqlen_KV must be the same size.");

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.size(0) > 1,
        "CUTLASS FMHA: cumulative_seqlen_Q and cumulative_seqlen_KV size must be greater than 1.");

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.scalar_type() == torch::kInt,
        "CUTLASS FMHA: cumulative_seqlen_Q is expected to be an int32 tensor, got ",
        cumulative_seqlen_Q_tensor.scalar_type());
    TORCH_CHECK(
        cumulative_seqlen_KV_tensor.scalar_type() == torch::kInt,
        "CUTLASS FMHA: cumulative_seqlen_KV is expected to be an int32 tensor, got ",
        cumulative_seqlen_KV_tensor.scalar_type());

    batch_size = cumulative_seqlen_Q_tensor.size(0) - 1;
    ptr_cumulative_seqlen_Q =
        static_cast<void*>(cumulative_seqlen_Q_tensor.data_ptr());
    ptr_cumulative_seqlen_KV =
        static_cast<void*>(cumulative_seqlen_KV_tensor.data_ptr());
  }
  //

  at::Tensor workspace;
  auto alloc_bytes = [&workspace, &query](
                         void** ptr, int64_t bytes, bool zfill) {
    workspace = at::empty({bytes}, query.options().dtype(at::ScalarType::Byte));
    if (zfill) {
      workspace.zero_();
    }
    *ptr = static_cast<void*>(workspace.data_ptr());
  };

  cudaDeviceProp* device_props =
      at::cuda::getDeviceProperties(query.device().index());
  const int cc = device_props->major * 10 + device_props->minor;
  const size_t max_smem = device_props->sharedMemPerBlockOptin;

  if (cc >= 80 || (cc >= 50 && query.scalar_type() != torch::kBFloat16)) {
    natten::cuda::fmha::fmha_forward_generic(
        query.scalar_type(),
        cc,
        max_smem,
        at::cuda::getCurrentCUDAStream(query.device().index()),
        alloc_bytes,
        static_cast<void*>(query.data_ptr()),
        static_cast<void*>(key.data_ptr()),
        static_cast<void*>(value.data_ptr()),
        static_cast<void*>(out.data_ptr()),
        batch_size,
        seqlen_q,
        seqlen_kv,
        heads,
        dim,
        dim_value,
        is_causal,
        attn_scale,
        logsumexp.has_value() ? static_cast<void*>(logsumexp.value().data_ptr())
                              : nullptr,
        // varlen parameters
        is_varlen,
        max_seqlen_Q,
        max_seqlen_KV,
        ptr_cumulative_seqlen_Q,
        ptr_cumulative_seqlen_KV,
        // init/launch params
        query_tile_size,
        key_tile_size);
  } else {
    NATTEN_FAILURE(
        "FMHA kernels are only available on devices with "
        "compute capability >= 50 for FP16/FP32 inputs, and devices with "
        "compute capability >= 80 for FP32, BF16, and FP16 inputs.");
  }
#else
  TORCH_CHECK(false, "libnatten not compiled with CUTLASS.");
#endif
}

void fmha_backward(
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
    int num_splits_key,
    bool compute_delta_with_torch,
    // varlen
    const at::optional<at::Tensor>& cumulative_seqlen_Q,
    const at::optional<at::Tensor>& cumulative_seqlen_KV,
    // only used if cumulative_seqlen_Q and cumulative_seqlen_KV are specified
    int max_seqlen_Q,
    int max_seqlen_KV) {
#ifdef NATTEN_WITH_CUTLASS
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

  CheckIfTensorShapesMatchExceptHeadDim<1>(query, out);
  CheckIfTensorShapesMatchExceptHeadDim<1>(key, value);
  CheckIfBatchHeadsHeadDimMatch(query, key);
  CheckIfHeadDimsMatch(out, value);
  CheckIfTensorShapesMatch<1>(grad_query, query);
  CheckIfTensorShapesMatch<1>(grad_key, key);
  CheckIfTensorShapesMatch<1>(grad_value, value);
  CheckIfTensorShapesMatch<1>(grad_out, out);

  CheckLogSumExp<1>(out, logsumexp);

  int batch_size = query.size(0);
  int seqlen_q = query.size(1);
  int seqlen_kv = key.size(1);
  int heads = query.size(2);
  int dim = query.size(3);
  int dim_value = value.size(3);

  at::Tensor workspace;
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
    delta = (grad_out.to(at::kFloat) * out.to(at::kFloat)).sum(-1);
  } else {
    delta = torch::empty(
        {batch_size, seqlen_q, heads}, query.options().dtype(at::kFloat));
    compute_delta_(out, grad_out, delta, (int32_t)delta.numel(), dim_value);
  }
  TORCH_CHECK(delta.size(0) == batch_size);
  TORCH_CHECK(delta.size(1) == seqlen_q);
  TORCH_CHECK(delta.size(2) == heads);
  if (at::globalContext().deterministicAlgorithms()) {
    TORCH_CHECK(
        num_splits_key <= 1,
        "CUTLASS FMHA backward was called with KV parallelism, "
        "which makes it algorithm non-deterministic, "
        "but PyTorch's deterministic mode is enabled. "
        "NATTEN Python API should have avoided this; which means "
        "you're probably calling the C function directly.");
  }

  // varlen
  bool is_varlen =
      cumulative_seqlen_Q.has_value() || cumulative_seqlen_KV.has_value();

  void* ptr_cumulative_seqlen_Q = nullptr;
  void* ptr_cumulative_seqlen_KV = nullptr;
  if (is_varlen) {
    TORCH_CHECK(
        cumulative_seqlen_Q.has_value() && cumulative_seqlen_KV.has_value(),
        "CUTLASS FMHA: Both cumulative_seqlen_Q and cumulative_seqlen_KV must be specified when using varlen.");

    TORCH_CHECK(
        batch_size == 1,
        "CUTLASS FMHA: Tensor batch size must be 1 (packed sequence layout), got ",
        batch_size);

    auto& cumulative_seqlen_Q_tensor = cumulative_seqlen_Q.value();
    auto& cumulative_seqlen_KV_tensor = cumulative_seqlen_KV.value();

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.dim() == 1,
        "CUTLASS FMHA: cumulative_seqlen_Q is expected to be a 1-D tensor.");
    TORCH_CHECK(
        cumulative_seqlen_KV_tensor.dim() == 1,
        "CUTLASS FMHA: cumulative_seqlen_KV is expected to be a 1-D tensor.");

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.size(0) ==
            cumulative_seqlen_KV_tensor.size(0),
        "CUTLASS FMHA: cumulative_seqlen_Q and cumulative_seqlen_KV must be the same size.");

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.size(0) > 1,
        "CUTLASS FMHA: cumulative_seqlen_Q and cumulative_seqlen_KV size must be greater than 1.");

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.scalar_type() == torch::kInt,
        "CUTLASS FMHA: cumulative_seqlen_Q is expected to be an int32 tensor, got ",
        cumulative_seqlen_Q_tensor.scalar_type());
    TORCH_CHECK(
        cumulative_seqlen_KV_tensor.scalar_type() == torch::kInt,
        "CUTLASS FMHA: cumulative_seqlen_KV is expected to be an int32 tensor, got ",
        cumulative_seqlen_KV_tensor.scalar_type());

    batch_size = cumulative_seqlen_Q_tensor.size(0) - 1;
    ptr_cumulative_seqlen_Q =
        static_cast<void*>(cumulative_seqlen_Q_tensor.data_ptr());
    ptr_cumulative_seqlen_KV =
        static_cast<void*>(cumulative_seqlen_KV_tensor.data_ptr());
  }
  //

  auto seqlen_kv_ = is_varlen ? max_seqlen_KV : seqlen_kv;
  auto num_kv_tiles = (seqlen_kv_ + key_tile_size - 1) / key_tile_size;
  TORCH_CHECK(
      num_splits_key <= num_kv_tiles,
      "CUTLASS FMHA backward: backward_kv_splits must be less than or equal to ",
      "KV sequence length divided by tile size, got backward_kv_splits=",
      num_splits_key,
      ", ceil_div(seqlen_kv, key_tile_size)=",
      num_kv_tiles,
      ".");

  cudaDeviceProp* device_props =
      at::cuda::getDeviceProperties(query.device().index());
  const int cc = device_props->major * 10 + device_props->minor;
  const size_t max_smem = device_props->sharedMemPerBlockOptin;

  if (cc >= 80 || (cc >= 50 && query.scalar_type() != torch::kBFloat16)) {
    natten::cuda::fmha::fmha_backward_generic(
        query.scalar_type(),
        cc,
        max_smem,
        at::cuda::getCurrentCUDAStream(query.device().index()),
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
        seqlen_q,
        seqlen_kv,
        heads,
        dim,
        dim_value,
        is_causal,
        attn_scale,
        // varlen parameters
        is_varlen,
        max_seqlen_Q,
        max_seqlen_KV,
        ptr_cumulative_seqlen_Q,
        ptr_cumulative_seqlen_KV,
        // init/launch params
        query_tile_size,
        key_tile_size,
        num_splits_key);
  } else {
    NATTEN_FAILURE(
        "FMHA are only available on devices with "
        "compute capability >= 50 for FP16/FP32 inputs, and devices with "
        "compute capability >= 80 for FP32, BF16, and FP16 inputs.");
  }
#else
  TORCH_CHECK(false, "libnatten not compiled with CUTLASS.");
#endif
}

} // namespace natten
