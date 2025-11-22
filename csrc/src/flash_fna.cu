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
    \brief Flash Attention FMHA interface
*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <natten/helpers.h>
#include <natten/natten.h>

#if defined(NATTEN_WITH_CUTLASS)
#include <natten_autogen/cuda/flash_fna/interface.h>        // Dispatcher
// #include <natten_autogen/cuda/flash_fna_bwd/interface.h>    // Dispatcher
#include <natten/cuda/flash_fna/flash_kernel/flash.h>       // Flash_fna_fwd_params
#include <natten/cuda/flash_fna/flash_kernel/param_utils.h> // Param conversion utils

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

// In this file, define the two entry functions for flash fwd and bwd, and add a dispatcher call in
// the end. The dispatcher will then route to the correct instantiation with the correct
// template arguments.

namespace natten {

template<class StdNADim, class StdCausal>
void flash_fna_generic_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::optional<at::Tensor>& logsumexp,
    const StdNADim& kernel_size,
    const StdNADim& stride_,
    const StdNADim& dilation_,
    const StdCausal& is_causal_,
    float attn_scale,
    // int query_tile_size,
    // int key_tile_size,
    const StdNADim& q_shape_, // after token permute and padding
    const StdNADim& kv_shape_, // after token permute and padding
    const StdNADim& qkv_shape_, // before token permute and padding
    const StdNADim& query_tile_shape_,
    const StdNADim& key_tile_shape_) {

  // NOTE (aditya): Copied over relevant sections from Hopper FNA.

  static_assert(
      std::tuple_size_v<StdNADim> > 0 && std::tuple_size_v<StdNADim> < 4);
  static constexpr int kNADim = std::tuple_size_v<StdNADim>;
  static_assert(std::tuple_size_v<StdCausal> == kNADim);

#ifdef NATTEN_WITH_CUTLASS
  // Here:
  //  1. Do all host-side checks.
  //  2. Initialize flash params object.
  //  3. Call the dispatcher.
  // Note that workspace init must be in autogen'd dispatcher, but here we don't need any.
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

  // 1. Host side checks
  CheckIfBatchHeadsHeadDimMatch(query, key);
  CheckArgs(kernel_size, stride_, dilation_);
  CheckIfPropertiesMatch(query, key, value);

  // NOTE (alih): q and kv might have slightly different shapes because we're
  // padding to multiples of the tile shape. Seqlen mode must be flattened.
  CheckIfTensorShapesMatch<1>(query, out);
  CheckIfTensorShapesMatch<1>(key, value);

  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");
  TORCH_CHECK(key.dim() == 4, "Tensors must be 4-D.");
  TORCH_CHECK(value.dim() == 4, "Tensors must be 4-D.");

  int batch_size = query.size(0);
  int seqlen_q = query.size(1);
  int seqlen_kv = key.size(1);
  int heads = query.size(2);
  int dim = query.size(3);

  CheckArgsAgainstDim(qkv_shape_, kernel_size, dilation_);

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

  // Check inputs are 16 bit
  TORCH_CHECK(
      query.scalar_type() == key.scalar_type() &&
      query.scalar_type() == value.scalar_type() &&
      query.scalar_type() == out.scalar_type(),
      "Query, key, value, and output must match in dtype.");

  TORCH_CHECK(
      query.scalar_type() == torch::kFloat16 ||
          query.scalar_type() == torch::kBFloat16,
      "Only FP16/BF16 is supported for now.");

  TORCH_CHECK(logsumexp.has_value(),
      "Logsumexp should not be a nullptr, it should be allocated by Python frontend.");

  // NOTE (aditya): LSE with Flash has BHQ format
  // CheckLogSumExp<1>(out, logsumexp.value());
  CHECK_CUDA(logsumexp.value());

  // 2. Init flash params
  cudaDeviceProp* device_props =
      at::cuda::getDeviceProperties(query.device().index());
  const int cc = device_props->major * 10 + device_props->minor;
  const int num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  auto flash_fna_fwd_params = natten::cuda::flash_fna::set_flash_fna_fwd_params(
      cc,
      num_sm,
      out,
      query,
      key,
      value,
      logsumexp,
      attn_scale,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation
    );

  // 3. Call dispatch
  if (cc >= 80) {
    int device_id = query.device().index();
    auto cuda_stream = at::cuda::getCurrentCUDAStream(device_id);

    DISPATCH_FLASH_FNA_FORWARD(
      query.scalar_type(), // dtype
      query.size(3),       // dim
      cc,
      query_tile_size,
      key_tile_size,
      flash_fna_fwd_params,
      cuda_stream);
  } else {
    NATTEN_FAILURE(
        "Flash FMHA kernels are only available on devices with "
        "compute capability >= 80 for BF16 and FP16 inputs.");
  }

#else
  TORCH_CHECK(false, "libnatten not compiled with CUTLASS.");
#endif
}


void flash_na1d_forward(
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
    const std::tuple<int32_t>& q_shape,
    const std::tuple<int32_t>& kv_shape,
    const std::tuple<int32_t>& qkv_shape,
    const std::tuple<int32_t>& query_tile_shape,
    const std::tuple<int32_t>& key_tile_shape) {
  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");

  flash_fna_generic_forward(
      out,
      query,
      key,
      value,
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
      key_tile_shape,
      kernel_type);
}

void flash_na2d_forward(
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
    const std::tuple<int32_t, int32_t>& q_shape,
    const std::tuple<int32_t, int32_t>& kv_shape,
    const std::tuple<int32_t, int32_t>& qkv_shape,
    const std::tuple<int32_t, int32_t>& query_tile_shape,
    const std::tuple<int32_t, int32_t>& key_tile_shape,
    int kernel_type) {
  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");

  flash_fna_generic_forward(
      out,
      query,
      key,
      value,
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

void flash_na3d_forward(
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
    const std::tuple<int32_t, int32_t, int32_t>& q_shape,
    const std::tuple<int32_t, int32_t, int32_t>& kv_shape,
    const std::tuple<int32_t, int32_t, int32_t>& qkv_shape,
    const std::tuple<int32_t, int32_t, int32_t>& query_tile_shape,
    const std::tuple<int32_t, int32_t, int32_t>& key_tile_shape,
    int kernel_type) {
  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");

  flash_fna_generic_forward(
      out,
      query,
      key,
      value,
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
      key_tile_shape,
      kernel_type);
}


void flash_fna_backward(
    at::Tensor& grad_query,
    at::Tensor& grad_key,
    at::Tensor& grad_value,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::Tensor& grad_out,
    const at::Tensor& logsumexp,
    float attn_scale,
    int query_tile_size,
    int key_tile_size,
    bool deterministic) {

  TORCH_CHECK(false, "Flash FNA backward not implemented yet.");

#ifdef NATTEN_WITH_CUTLASS
  // Here:
  //  1. Do all host-side checks.
  //  2. Initialize flash params object.
  //  3. Allocate workspace.
  //  4. Call the dispatcher.
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

  CheckIfTensorShapesMatch<1>(query, out);
  CheckIfTensorShapesMatch<1>(key, value);
  CheckIfBatchHeadsHeadDimMatch(query, key);
  CheckIfHeadDimsMatch(out, value);
  CheckIfTensorShapesMatch<1>(grad_query, query);
  CheckIfTensorShapesMatch<1>(grad_key, key);
  CheckIfTensorShapesMatch<1>(grad_value, value);
  CheckIfTensorShapesMatch<1>(grad_out, out);

  int batch_size = query.size(0);
  int seqlen_q = query.size(1);
  int seqlen_kv = key.size(1);
  int heads = query.size(2);
  int dim = query.size(3);

  TORCH_CHECK(
      query.scalar_type() == torch::kFloat16 ||
          query.scalar_type() == torch::kBFloat16,
      "Only FP16/BF16 is supported for now.");

  TORCH_CHECK(
      not (deterministic ^ at::globalContext().deterministicAlgorithms()),
      "The provided deterministic argument does not "
      "match with PyTorch's global setting. "
      "NATTEN Python API should have avoided this; which means "
      "you're probably calling the C function directly.");

  // NOTE (aditya): Caution!! Here workspace_size is a struct, and NOT a native int/int64.
  // Use workspace_size.total_bytes to fetch the number of workspace bytes needed.
  auto workspace_size = natten::cuda::flash::get_flash_bwd_workspace_size(
      batch_size,
      seqlen_q,
      seqlen_kv,
      heads,
      dim,
      query_tile_size,
      key_tile_size,
      deterministic);

  int64_t bytes = workspace_size.total_bytes;

  auto workspace = at::empty({bytes}, query.options().dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  
  auto workspace_alloc = natten::cuda::flash::allocate_flash_bwd_workspace(workspace_ptr,
      workspace_size);

  cudaDeviceProp* device_props =
      at::cuda::getDeviceProperties(query.device().index());
  const int cc = device_props->major * 10 + device_props->minor;
  const int num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  auto flash_bwd_params = natten::cuda::flash::set_flash_bwd_params(
      cc,
      num_sm,
      grad_query,
      grad_key,
      grad_value,
      grad_out,
      query,
      key,
      value,
      out,
      logsumexp,
      workspace_alloc.softmax_lse_log2_ptr,
      workspace_alloc.dsoftmax_sum_ptr,
      workspace_alloc.dQ_semaphore_ptr,
      workspace_alloc.dK_semaphore_ptr,
      workspace_alloc.dV_semaphore_ptr,
      workspace_alloc.dQ_accum_ptr,
      query_tile_size,
      key_tile_size,
      attn_scale,
      deterministic
      );

  if (cc >= 80) {

    int device_id = query.device().index();
    auto cuda_stream = at::cuda::getCurrentCUDAStream(device_id);
    cudaDeviceProp* device_props = at::cuda::getDeviceProperties(device_id);
    const int cc = device_props->major * 10 + device_props->minor;

    DISPATCH_FLASH_FMHA_BACKWARD(
        query.scalar_type(),
        dim,
        cc,
        deterministic,
        query_tile_size,
        key_tile_size,
        flash_bwd_params,
        cuda_stream);

  } else {
    NATTEN_FAILURE(
        "Flash FMHA kernels are only available on devices with "
        "compute capability >= 80 for BF16 and FP16 inputs.");
  }

#else
  TORCH_CHECK(false, "libnatten not compiled with CUTLASS.");
#endif
}

} // namespace natten
