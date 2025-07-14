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
    \brief Blackwell FNA Torch interface
*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <natten/helpers.h>
#include <natten/natten.h>

#ifdef NATTEN_WITH_CUTLASS
#include <natten_autogen/cuda/blackwell_fna/interface.h>
#include <natten/cuda/fna_blackwell/fna_forward.cuh>

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
void blackwell_fna_generic_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::optional<at::Tensor>& logsumexp,
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
    int num_extra_kv,
    const StdNADim& query_tile_shape_,
    const StdNADim& key_tile_shape_,
    bool run_persistent) {
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
  CHECK_CONTIGUOUS(out);

  CHECK_CUDA(query);
  CHECK_CUDA(key);
  CHECK_CUDA(value);
  CHECK_CUDA(out);

  at::cuda::OptionalCUDAGuard device_guard(query.device());

  CheckArgs(kernel_size, stride_, dilation_);
  CheckIfPropertiesMatch(query, key, value);

  // NOTE (alih): q and kv might have slightly different shapes because we're
  // padding to multiples of the tile shape. We're also supporting extra KV
  // tokens, so we can't have 5D/6D tensors anymore. Seqlen mode must be
  // flattened.
  CheckIfTensorShapesMatch<1>(query, out);
  CheckIfTensorShapesMatch<1>(key, value);

  int batch_size = query.size(0);
  int seqlen_q = query.size(1);
  int seqlen_kv = key.size(1);
  int heads = query.size(2);
  int dim = query.size(3);

  if (logsumexp.has_value()) {
    CheckLogSumExp<1>(out, logsumexp.value());
    CHECK_CUDA(logsumexp.value());
  }

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
      "Q's sequence length (q.shape[1]) must match the size of QKV shape.");
  TORCH_CHECK(
      size(kv_shape) + num_extra_kv == seqlen_kv,
      "KV's sequence length ({k,v}.shape[1]) must match the size of QKV shape + num_extra_kv.");

  TORCH_CHECK(
      dim == 32 || dim == 64 || dim == 128,
      "FNA Blackwell only supports head dims 32, 64, and 128.");

  cudaDeviceProp* device_props =
      at::cuda::getDeviceProperties(query.device().index());
  const int cc = device_props->major * 10 + device_props->minor;

  TORCH_CHECK(
      cc == 100,
      "This operation can only run on the Blackwell (datacenter-class) architecture (SM100).");

  TORCH_CHECK(
      query.scalar_type() == key.scalar_type() &&
          query.scalar_type() == value.scalar_type() &&
          query.scalar_type() == out.scalar_type(),
      "Query, key, value, and output must match in dtype.");

  TORCH_CHECK(
      query.scalar_type() == torch::kFloat16 ||
          query.scalar_type() == torch::kBFloat16,
      "Only FP16/BF16 is supported for now.");

  int device_id = query.device().index();
  auto cuda_stream = at::cuda::getCurrentCUDAStream(device_id);

  TORCH_CHECK(
      cute::evenly_divides(q_shape, query_tile_shape) &&
          cute::evenly_divides(kv_shape, key_tile_shape),
      "Tile shapes must evenly divide input. Please pad your inputs.");

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

  DISPATCH_BLACKWELL_FNA_FORWARD(
      kNADim,
      query.scalar_type(),
      dim,
      is_causal,
      query_tile_shape,
      key_tile_shape,
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
      heads,
      dim,
      num_extra_kv,
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

void blackwell_na1d_forward(
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
    int num_extra_kv,
    const std::tuple<int32_t>& query_tile_shape,
    const std::tuple<int32_t>& key_tile_shape,
    bool run_persistent) {
  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");

  blackwell_fna_generic_forward(
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
      num_extra_kv,
      query_tile_shape,
      key_tile_shape,
      run_persistent);
}

void blackwell_na2d_forward(
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
    int num_extra_kv,
    const std::tuple<int32_t, int32_t>& query_tile_shape,
    const std::tuple<int32_t, int32_t>& key_tile_shape,
    bool run_persistent) {
  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");

  blackwell_fna_generic_forward(
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
      num_extra_kv,
      query_tile_shape,
      key_tile_shape,
      run_persistent);
}

void blackwell_na3d_forward(
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
    int num_extra_kv,
    const std::tuple<int32_t, int32_t, int32_t>& query_tile_shape,
    const std::tuple<int32_t, int32_t, int32_t>& key_tile_shape,
    bool run_persistent) {
  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");

  blackwell_fna_generic_forward(
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
      num_extra_kv,
      query_tile_shape,
      key_tile_shape,
      run_persistent);
}

} // namespace natten
