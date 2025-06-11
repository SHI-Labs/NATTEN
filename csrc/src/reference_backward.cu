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
    \brief Reference backward
*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <natten/helpers.h>
#include <natten/natten.h>

#ifdef NATTEN_WITH_CUTLASS
#include <natten/cuda/reference/fna_reference_backward.hpp>
#include <natten_autogen/cuda/reference/interface.h>

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
void reference_na_generic_backward(
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
    int num_extra_kv) {
  static_assert(
      std::tuple_size_v<StdNADim> > 0 && std::tuple_size_v<StdNADim> < 4);
  static constexpr int kNADim = std::tuple_size_v<StdNADim>;
  static_assert(std::tuple_size_v<StdCausal> == kNADim);

#ifdef NATTEN_WITH_CUTLASS
  at::cuda::OptionalCUDAGuard device_guard(query.device());

  // TODO: please please simplify these checks!!!
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(out);
  CHECK_CUDA(grad_query);
  CHECK_CUDA(grad_key);
  CHECK_CUDA(grad_value);
  CHECK_CUDA(grad_out);
  CHECK_CUDA(logsumexp);

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

  // Everything's flattened to 1D, because we concat additional kvs
  CheckIfTensorShapesMatchExceptHeadDim<1>(query, out);
  CheckIfTensorShapesMatchExceptHeadDim<1>(key, value);
  CheckIfBatchHeadsHeadDimMatch(query, key);
  CheckIfTensorShapesMatch<1>(grad_query, query);
  CheckIfTensorShapesMatch<1>(grad_key, key);
  CheckIfTensorShapesMatch<1>(grad_value, value);
  CheckIfTensorShapesMatch<1>(grad_out, out);
  CheckLogSumExp<1>(out, logsumexp);

  int batch_size = query.size(0);
  int heads = query.size(2);
  int dim = query.size(3);

  int seqlen_q = query.size(1);
  int seqlen_kv = key.size(1);
  int dim_value = value.size(3);

  CheckArgsAgainstDim(qkv_shape, kernel_size, dilation);

  auto qkv_shape_ = std_tuple_to_cute_tuple(qkv_shape);
  auto window_size = std_tuple_to_cute_tuple(kernel_size);
  auto stride_ = std_tuple_to_cute_tuple(stride);
  auto dilation_ = std_tuple_to_cute_tuple(dilation);
  auto is_causal_ = std_tuple_to_cute_tuple(is_causal);

  int seqlen = size(qkv_shape_);

  TORCH_CHECK(
      query.scalar_type() == torch::kFloat16 ||
          query.scalar_type() == torch::kBFloat16 ||
          query.scalar_type() == torch::kFloat32,
      "Only FP32, FP16, and BF16 operands are supported for now.");

  TORCH_CHECK(
      size(qkv_shape_) == seqlen_q,
      "Q's sequence length (q.shape[1]) must match the size of QKV shape.");
  TORCH_CHECK(
      size(qkv_shape_) + num_extra_kv == seqlen_kv,
      "KV's sequence length ({k,v}.shape[1]) must match the size of QKV shape + num_extra_kv.");

  int device_id = query.device().index();
  auto cuda_stream = at::cuda::getCurrentCUDAStream(device_id);

  DISPATCH_REFERENCE_FNA_BACKWARD(
      kNADim,
      query.scalar_type(),
      is_causal_,
      static_cast<void*>(query.data_ptr()),
      static_cast<void*>(key.data_ptr()),
      static_cast<void*>(value.data_ptr()),
      static_cast<void*>(out.data_ptr()),
      static_cast<void*>(grad_out.data_ptr()),
      static_cast<void*>(grad_query.data_ptr()),
      static_cast<void*>(grad_key.data_ptr()),
      static_cast<void*>(grad_value.data_ptr()),
      static_cast<void*>(logsumexp.data_ptr()),
      batch_size,
      seqlen,
      heads,
      dim,
      dim_value,
      num_extra_kv,
      qkv_shape_,
      window_size,
      stride_,
      dilation_,
      attn_scale,
      cuda_stream);
#else
  TORCH_CHECK(false, "libnatten not compiled with CUTLASS.");
#endif
}

void reference_na1d_backward(
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
    const std::tuple<int32_t>& qkv_shape,
    int num_extra_kv) {
  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");

  reference_na_generic_backward(
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
      qkv_shape,
      num_extra_kv);
}

void reference_na2d_backward(
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
    const std::tuple<int32_t, int32_t>& qkv_shape,
    int num_extra_kv) {
  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");

  reference_na_generic_backward(
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
      qkv_shape,
      num_extra_kv);
}

void reference_na3d_backward(
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
    const std::tuple<int32_t, int32_t, int32_t>& qkv_shape,
    int num_extra_kv) {
  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");

  reference_na_generic_backward(
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
      qkv_shape,
      num_extra_kv);
}

} // namespace natten
