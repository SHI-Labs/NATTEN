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
    \brief Blackwell FNA backward Torch interface
*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <natten/helpers.h>
#include <natten/natten.h>

#ifdef NATTEN_WITH_CUTLASS
#include <natten_autogen/cuda/blackwell_fna_bwd/interface.h>
#include <natten/cuda/fna_blackwell/fna_backward.cuh>

template <typename StdTuple>
auto std_tuple_to_cute_tuple(StdTuple a) {
  static_assert(
      std::tuple_size_v<StdTuple> > 0 && std::tuple_size_v<StdTuple> < 4);

  if constexpr (std::tuple_size_v<StdTuple> == 1) {
    return cute::make_tuple(std::get<0>(a));
  } else if constexpr (std::tuple_size_v<StdTuple> == 2) {
    return cute::make_tuple(std::get<0>(a), std::get<1>(a));
  } else {
    return cute::make_tuple(std::get<0>(a), std::get<1>(a), std::get<2>(a));
  }
}
#endif

namespace natten {

template <class StdNADim, class StdCausal>
void blackwell_fna_generic_backward(
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
    const StdNADim& stride_,
    const StdNADim& dilation_,
    const StdCausal& is_causal_,
    float attn_scale,
    const StdNADim& q_shape_, // after token permute and padding
    const StdNADim&
        kv_shape_, // after token permute and padding, including extra KV
    const StdNADim&
        qkv_shape_, // before token permute and padding, not including extra KV
    const StdNADim& query_tile_shape_,
    const StdNADim& key_tile_shape_) {
  static_assert(
      std::tuple_size_v<StdNADim> > 0 && std::tuple_size_v<StdNADim> < 4);
  static constexpr int kNADim = std::tuple_size_v<StdNADim>;
  static_assert(std::tuple_size_v<StdCausal> == kNADim);

#ifdef NATTEN_WITH_CUTLASS
#ifdef NATTEN_WITH_BLACKWELL_FNA
  AssertDimsAre128BitAligned(query, value);

  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CHECK_CONTIGUOUS(value);
  CHECK_CONTIGUOUS(grad_query);
  CHECK_CONTIGUOUS(grad_key);
  CHECK_CONTIGUOUS(grad_value);
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(grad_out);
  CHECK_CONTIGUOUS(logsumexp);

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

  CheckArgs(kernel_size, stride_, dilation_);
  CheckIfPropertiesMatch(query, key, value);
  CheckIfPropertiesMatch(grad_value, grad_out, out);
  CheckIfPropertiesMatch(grad_query, grad_key, grad_value);
  CheckIfPropertiesMatch(grad_query, query, value);

  // NOTE (alih): q and kv might have slightly different shapes because we're
  // padding to multiples of the tile shape. We're also supporting extra KV
  // tokens, so we can't have 5D/6D tensors anymore. Seqlen mode must be
  // flattened.
  CheckIfTensorShapesMatch<1>(query, out);
  CheckIfTensorShapesMatch<1>(key, value);
  CheckIfTensorShapesMatch<1>(query, grad_query);
  CheckIfTensorShapesMatch<1>(key, grad_key);
  CheckIfTensorShapesMatch<1>(value, grad_value);
  CheckIfTensorShapesMatch<1>(out, grad_out);

  int batch_size = query.size(0);
  int seqlen_q = query.size(1);
  int seqlen_kv = key.size(1);
  int heads = query.size(2);
  int dim = query.size(3);

  CheckArgsAgainstDim(qkv_shape_, kernel_size, dilation_);

  CheckLogSumExp<1>(out, logsumexp);

  auto qkv_shape = std_tuple_to_cute_tuple(qkv_shape_);
  auto q_shape = std_tuple_to_cute_tuple(q_shape_);
  auto kv_shape = std_tuple_to_cute_tuple(kv_shape_);

  auto query_tile_shape = std_tuple_to_cute_tuple(query_tile_shape_);
  auto key_tile_shape = std_tuple_to_cute_tuple(key_tile_shape_);

  auto window_size = std_tuple_to_cute_tuple(kernel_size);
  auto stride = std_tuple_to_cute_tuple(stride_);
  auto dilation = std_tuple_to_cute_tuple(dilation_);
  auto is_causal = std_tuple_to_cute_tuple(is_causal_);

  TORCH_CHECK(
      size(q_shape) == seqlen_q,
      "Q's sequence length (q.shape[1]) must match the size of Q shape.");
  TORCH_CHECK(
      size(kv_shape) == seqlen_kv,
      "KV's sequence length ({k,v}.shape[1]) must match the size of KV shape.");

  TORCH_CHECK(
      cute::evenly_divides(q_shape, query_tile_shape) &&
          cute::evenly_divides(kv_shape, key_tile_shape),
      "Tile shapes must evenly divide input. Please pad your inputs.");

  TORCH_CHECK(
      query.scalar_type() == torch::kFloat16 ||
          query.scalar_type() == torch::kBFloat16,
      "Only FP16/BF16 is supported for now.");

  TORCH_CHECK(
      dim == 32 || dim == 64 || dim == 128,
      "FNA Blackwell only supports head dims 32, 64, and 128.");

  TORCH_CHECK(
      not at::globalContext().deterministicAlgorithms(),
      "Blackwell FNA backward pass is non-deterministic, "
      "but PyTorch's deterministic mode is enabled. "
      "NATTEN Python API should have avoided this; which means "
      "you're probably calling the C function directly.");

  int device_id = query.device().index();
  auto cuda_stream = at::cuda::getCurrentCUDAStream(device_id);
  cudaDeviceProp* device_props = at::cuda::getDeviceProperties(device_id);
  const int cc = device_props->major * 10 + device_props->minor;

  TORCH_CHECK(
      cc == 100,
      "This operation can only run on the Blackwell (datacenter-class) architecture (SM100).");

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  DISPATCH_BLACKWELL_FNA_BACKWARD(
      kNADim,
      query.scalar_type(),
      dim,
      is_causal,
      query_tile_shape,
      key_tile_shape,
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
      heads,
      dim,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale,
      cuda_stream,
      query.options());

#else
  TORCH_CHECK(
      false,
      "libnatten was not compiled with CUTLASS_ARCH_MMA_SM100_SUPPORTED.");
#endif
#else
  TORCH_CHECK(false, "libnatten was not compiled for Blackwell (SM100).");
#endif
#else
  TORCH_CHECK(false, "libnatten not compiled with CUTLASS.");
#endif
}

void blackwell_na1d_backward(
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
    const std::tuple<int32_t>& q_shape,
    const std::tuple<int32_t>& kv_shape,
    const std::tuple<int32_t>& qkv_shape,
    const std::tuple<int32_t>& query_tile_shape,
    const std::tuple<int32_t>& key_tile_shape) {
  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");

  blackwell_fna_generic_backward(
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
      q_shape,
      kv_shape,
      qkv_shape,
      query_tile_shape,
      key_tile_shape);
}

void blackwell_na2d_backward(
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
    const std::tuple<int32_t, int32_t>& q_shape,
    const std::tuple<int32_t, int32_t>& kv_shape,
    const std::tuple<int32_t, int32_t>& qkv_shape,
    const std::tuple<int32_t, int32_t>& query_tile_shape,
    const std::tuple<int32_t, int32_t>& key_tile_shape) {
  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");

  blackwell_fna_generic_backward(
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
      q_shape,
      kv_shape,
      qkv_shape,
      query_tile_shape,
      key_tile_shape);
}

void blackwell_na3d_backward(
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
    const std::tuple<int32_t, int32_t, int32_t>& q_shape,
    const std::tuple<int32_t, int32_t, int32_t>& kv_shape,
    const std::tuple<int32_t, int32_t, int32_t>& qkv_shape,
    const std::tuple<int32_t, int32_t, int32_t>& query_tile_shape,
    const std::tuple<int32_t, int32_t, int32_t>& key_tile_shape) {
  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");

  blackwell_fna_generic_backward(
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
      q_shape,
      kv_shape,
      qkv_shape,
      query_tile_shape,
      key_tile_shape);
}

} // namespace natten
