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
    \brief FNA backward interface
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
#include <natten_autogen/cuda/fna/interface.h>
#include <natten/cuda/fna/fna_backward.cuh>
#endif

namespace natten {

template <typename StdTuple>
auto tuple_product(StdTuple a) {
  static_assert(
      std::tuple_size_v<StdTuple> > 0 && std::tuple_size_v<StdTuple> < 4);

  if constexpr (std::tuple_size_v<StdTuple> == 1) {
    return std::get<0>(a);
  } else if constexpr (std::tuple_size_v<StdTuple> == 2) {
    return std::get<0>(a) * std::get<1>(a);
  } else {
    return std::get<0>(a) * std::get<1>(a) * std::get<2>(a);
  }
}

template <class StdNADim, class StdCausal>
void fna_generic_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    const StdNADim& kernel_size,
    const StdNADim& stride,
    const StdNADim& dilation,
    const StdCausal& is_causal,
    float attn_scale,
    const StdNADim& qkv_shape,
    const StdNADim& query_tile_size,
    const StdNADim& key_tile_size,
    const StdNADim& num_splits_key,
    bool compute_delta_with_torch) {
  static_assert(
      std::tuple_size_v<StdNADim> > 0 && std::tuple_size_v<StdNADim> < 4);
  static constexpr int kNADim = std::tuple_size_v<StdNADim>;
  static_assert(std::tuple_size_v<StdCausal> == kNADim);

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

  CheckArgs(kernel_size, stride, dilation);
  CheckIfPropertiesMatch(query, key, value);
  CheckIfPropertiesMatch(grad_value, grad_out, out);
  CheckIfPropertiesMatch(grad_query, grad_key, grad_value);
  CheckIfPropertiesMatch(grad_query, query, value);

  CheckIfTensorShapesMatch<kNADim>(query, key);
  CheckIfTensorShapesMatch<kNADim>(query, value);
  CheckIfTensorShapesMatch<kNADim>(out, value);
  CheckIfTensorShapesMatch<kNADim>(grad_query, grad_key);
  CheckIfTensorShapesMatch<kNADim>(grad_query, grad_value);
  CheckIfTensorShapesMatch<kNADim>(grad_out, grad_value);
  CheckIfTensorShapesMatch<kNADim>(grad_out, out);

  CheckLogSumExp<kNADim>(out, logsumexp);

  int batch_size = query.size(0);
  int heads = query.size(kNADim + 1);
  int dim = query.size(kNADim + 2);
  auto seqlen = tuple_product(qkv_shape);
  CheckArgsAgainstDim(qkv_shape, kernel_size, dilation);

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
    delta = (grad_out.to(at::kFloat) * out.to(at::kFloat))
                .flatten(1, kNADim)
                .sum(-1);
  } else {
    delta = torch::empty(
        {batch_size, seqlen, heads}, query.options().dtype(at::kFloat));
    compute_delta_(out, grad_out, delta, (int32_t)delta.numel(), dim);
  }
  TORCH_CHECK(delta.size(0) == batch_size);
  TORCH_CHECK(delta.size(1) == seqlen);
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

  cudaDeviceProp* device_props =
      at::cuda::getDeviceProperties(query.device().index());
  const int cc = device_props->major * 10 + device_props->minor;
  const size_t max_smem = device_props->sharedMemPerBlockOptin;

  if (cc >= 80 || (cc >= 50 && query.scalar_type() != torch::kBFloat16)) {
    natten::cuda::fna::fna_backward_generic(
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
        qkv_shape,
        heads,
        dim,
        dim, // dim_value
        kernel_size,
        stride,
        dilation,
        is_causal,
        attn_scale,
        query_tile_size,
        key_tile_size,
        num_splits_key);
  } else {
    NATTEN_FAILURE(
        "Fused kernels are only available on devices with "
        "compute capability >= 50 for FP16/FP32 inputs, and devices with "
        "compute capability >= 80 for FP32, BF16, and FP16 inputs.");
  }
#else
  TORCH_CHECK(false, "libnatten not compiled with CUTLASS.");
#endif
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
    const std::tuple<int32_t>& stride,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t>& query_tile_size,
    const std::tuple<int32_t>& key_tile_size,
    const std::tuple<int32_t>& num_splits_key,
    bool compute_delta_with_torch) {
  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");

  fna_generic_backward(
      grad_query,
      grad_key,
      grad_value,
      query,
      key,
      value,
      out,
      grad_out,
      logsumexp,
      kernel_size,
      stride,
      dilation,
      is_causal,
      attn_scale,
      {query.size(1)},
      query_tile_size,
      key_tile_size,
      num_splits_key,
      compute_delta_with_torch);
}

void na2d_backward(
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
    const std::tuple<int32_t, int32_t>& query_tile_size,
    const std::tuple<int32_t, int32_t>& key_tile_size,
    const std::tuple<int32_t, int32_t>& num_splits_key,
    bool compute_delta_with_torch) {
  TORCH_CHECK(query.dim() == 5, "Tensors must be 5-D.");

  fna_generic_backward(
      grad_query,
      grad_key,
      grad_value,
      query,
      key,
      value,
      out,
      grad_out,
      logsumexp,
      kernel_size,
      stride,
      dilation,
      is_causal,
      attn_scale,
      {query.size(1), query.size(2)},
      query_tile_size,
      key_tile_size,
      num_splits_key,
      compute_delta_with_torch);
}

void na3d_backward(
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
    const std::tuple<int32_t, int32_t, int32_t>& query_tile_size,
    const std::tuple<int32_t, int32_t, int32_t>& key_tile_size,
    const std::tuple<int32_t, int32_t, int32_t>& num_splits_key,
    bool compute_delta_with_torch) {
  TORCH_CHECK(query.dim() == 6, "Tensors must be 6-D.");

  fna_generic_backward(
      grad_query,
      grad_key,
      grad_value,
      query,
      key,
      value,
      out,
      grad_out,
      logsumexp,
      kernel_size,
      stride,
      dilation,
      is_causal,
      attn_scale,
      {query.size(1), query.size(2), query.size(3)},
      query_tile_size,
      key_tile_size,
      num_splits_key,
      compute_delta_with_torch);
}

} // namespace natten
