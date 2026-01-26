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

#if defined(NATTEN_WITH_CUTLASS) && defined(NATTEN_WITH_BLACKWELL_FNA)
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
    const StdNADim& key_tile_shape_,
    // varlen
    const at::optional<at::Tensor>& cumulative_seqlen_Q,
    const at::optional<at::Tensor>& cumulative_seqlen_KV,
    const at::optional<at::Tensor>& token_layouts,
    const at::optional<at::Tensor>& batch_map,
    int max_seqlen_Q,
    int max_seqlen_KV,
    // var-param
    const at::optional<at::Tensor>& kernel_sizes,
    const at::optional<at::Tensor>& strides,
    const at::optional<at::Tensor>& dilations) {
  static_assert(
      std::tuple_size_v<StdNADim> > 0 && std::tuple_size_v<StdNADim> < 4);
  static constexpr int kNADim = std::tuple_size_v<StdNADim>;
  static_assert(std::tuple_size_v<StdCausal> == kNADim);

#if defined(NATTEN_WITH_CUTLASS) && defined(NATTEN_WITH_BLACKWELL_FNA)
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

  // varlen
  bool is_varlen = cumulative_seqlen_Q.has_value() ||
      cumulative_seqlen_KV.has_value() || token_layouts.has_value();
  bool is_varparam =
      kernel_sizes.has_value() || strides.has_value() || dilations.has_value();

  at::cuda::OptionalCUDAGuard device_guard(query.device());

  if (not is_varparam) {
    CheckArgs(kernel_size, stride_, dilation_);
  }
  CheckIfPropertiesMatch(query, key, value);
  CheckIfPropertiesMatch(grad_value, grad_out, out);
  CheckIfPropertiesMatch(grad_query, grad_key, grad_value);
  CheckIfPropertiesMatch(grad_query, query, value);

  // NOTE (alih): q and kv might have slightly different shapes because we're
  // padding to multiples of the tile shape. We're also supporting extra KV
  // tokens, so we can't have 5D/6D tensors anymore. Seqlen mode must be
  // flattened.
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
      "Blackwell FNA backward only supports head dims 32, 64, and 128 for now.");

  if (not is_varlen) {
    CheckArgsAgainstDim(qkv_shape_, kernel_size, dilation_);
  }

  auto qkv_shape = std_tuple_to_cute_tuple(qkv_shape_);
  auto q_shape = std_tuple_to_cute_tuple(q_shape_);
  auto kv_shape = std_tuple_to_cute_tuple(kv_shape_);

  auto query_tile_shape = std_tuple_to_cute_tuple(query_tile_shape_);
  auto key_tile_shape = std_tuple_to_cute_tuple(key_tile_shape_);

  auto window_size = std_tuple_to_cute_tuple(kernel_size);
  auto stride = std_tuple_to_cute_tuple(stride_);
  auto dilation = std_tuple_to_cute_tuple(dilation_);
  auto is_causal = std_tuple_to_cute_tuple(is_causal_);

  if (not is_varlen) {
    TORCH_CHECK(
        size(q_shape) == seqlen_q,
        "Blackwell FNA backward: Q sequence length (q.shape[1]) must match the size of Q shape.");
    TORCH_CHECK(
        size(kv_shape) == seqlen_kv,
        "Blackwell FNA backward: KV sequence length ({k,v}.shape[1]) must match the size of KV shape.");

    TORCH_CHECK(
        cute::evenly_divides(q_shape, query_tile_shape) &&
            cute::evenly_divides(kv_shape, key_tile_shape),
        "Blackwell FNA backward: Tile shapes must evenly divide input. Please pad your inputs.");
  }

  int device_id = query.device().index();
  auto cuda_stream = at::cuda::getCurrentCUDAStream(device_id);
  cudaDeviceProp* device_props = at::cuda::getDeviceProperties(device_id);
  const int cc = device_props->major * 10 + device_props->minor;

  TORCH_CHECK(
      cc == 100 || cc == 103,
      "Blackwell FMHA backward can only run on the Blackwell (datacenter-class) architecture (SM100, SM103).");

  TORCH_CHECK(
      query.scalar_type() == torch::kFloat16 ||
          query.scalar_type() == torch::kBFloat16,
      "Blackwell FNA backward only supports FP16 and BF16.");

  void* ptr_cumulative_seqlen_Q = nullptr;
  void* ptr_cumulative_seqlen_KV = nullptr;
  void* ptr_token_layouts = nullptr;
  void* ptr_batch_map = nullptr;

  void* ptr_window_sizes = nullptr;
  void* ptr_strides = nullptr;
  void* ptr_dilations = nullptr;

  if (is_varlen) {
    TORCH_CHECK(
        cumulative_seqlen_Q.has_value() && cumulative_seqlen_KV.has_value() &&
            token_layouts.has_value(),
        "Blackwell FNA: cumulative_seqlen_Q, cumulative_seqlen_KV, and token_layouts must all be specified when using varlen.");

    TORCH_CHECK(
        batch_size == 1,
        "Blackwell FNA: Tensor batch size must be 1 (packed sequence layout), got ",
        batch_size);

    auto& cumulative_seqlen_Q_tensor = cumulative_seqlen_Q.value();
    auto& cumulative_seqlen_KV_tensor = cumulative_seqlen_KV.value();
    auto& token_layouts_tensor = token_layouts.value();

    CHECK_CONTIGUOUS(cumulative_seqlen_Q_tensor);
    CHECK_CONTIGUOUS(cumulative_seqlen_KV_tensor);
    CHECK_CONTIGUOUS(token_layouts_tensor);

    CHECK_CUDA(cumulative_seqlen_Q_tensor);
    CHECK_CUDA(cumulative_seqlen_KV_tensor);
    CHECK_CUDA(token_layouts_tensor);

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.dim() == 1,
        "Blackwell FMHA: cumulative_seqlen_Q is expected to be a 1-D tensor.");
    TORCH_CHECK(
        cumulative_seqlen_KV_tensor.dim() == 1,
        "Blackwell FMHA: cumulative_seqlen_KV is expected to be a 1-D tensor.");

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.size(0) ==
            cumulative_seqlen_KV_tensor.size(0),
        "Blackwell FNA: cumulative_seqlen_Q and cumulative_seqlen_KV must be the same size.");

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.size(0) > 1,
        "Blackwell FNA: cumulative_seqlen_Q and cumulative_seqlen_KV size must be greater than 1.");

    TORCH_CHECK(
        cumulative_seqlen_Q_tensor.scalar_type() == torch::kInt,
        "Blackwell FNA: cumulative_seqlen_Q is expected to be an int32 tensor, got ",
        cumulative_seqlen_Q_tensor.scalar_type());
    TORCH_CHECK(
        cumulative_seqlen_KV_tensor.scalar_type() == torch::kInt,
        "Blackwell FNA: cumulative_seqlen_KV is expected to be an int32 tensor, got ",
        cumulative_seqlen_KV_tensor.scalar_type());

    batch_size = cumulative_seqlen_Q_tensor.size(0) - 1;
    auto batch_size_original = token_layouts_tensor.size(0);

    TORCH_CHECK(
        token_layouts_tensor.dim() == 2,
        "Blackwell FNA: token_layouts is expected to be a 2-D tensor.");

    // TORCH_CHECK(
    //     token_layouts_tensor.size(0) == batch_size_original,
    //     "Blackwell FNA: token_layouts.shape[0] must be
    //     cumulative_seqlen_{Q,KV}.shape[0] - 1.");

    TORCH_CHECK(
        token_layouts_tensor.size(1) == kNADim,
        "Blackwell FNA",
        kNADim,
        "-D: token_layouts.shape[1] must be ",
        kNADim,
        ", got ",
        token_layouts_tensor.size(1),
        ".");

    TORCH_CHECK(
        token_layouts_tensor.scalar_type() == torch::kInt,
        "Blackwell FNA: token_layouts is expected to be an int32 tensor, got ",
        token_layouts_tensor.scalar_type());

    ptr_cumulative_seqlen_Q =
        static_cast<void*>(cumulative_seqlen_Q_tensor.data_ptr());
    ptr_cumulative_seqlen_KV =
        static_cast<void*>(cumulative_seqlen_KV_tensor.data_ptr());
    ptr_token_layouts = static_cast<void*>(token_layouts_tensor.data_ptr());

    if (is_varparam) {
      if (kernel_sizes.has_value()) {
        CHECK_CONTIGUOUS(kernel_sizes.value());
        CHECK_CUDA(kernel_sizes.value());
        TORCH_CHECK(
            kernel_sizes.value().dim() == 2,
            "Blackwell FNA: kernel_sizes is expected to be a 2-D tensor.");

        TORCH_CHECK(
            kernel_sizes.value().size(0) == batch_size_original,
            "Blackwell FNA: kernel_sizes.shape[0] must match token_layouts.shape[0].");

        TORCH_CHECK(
            kernel_sizes.value().size(1) == kNADim,
            "Blackwell FNA",
            kNADim,
            "-D: kernel_sizes.shape[1] must be ",
            kNADim,
            ", got ",
            kernel_sizes.value().size(1),
            ".");

        TORCH_CHECK(
            kernel_sizes.value().scalar_type() == torch::kInt,
            "Blackwell FNA: kernel_sizes is expected to be an int32 tensor, got ",
            kernel_sizes.value().scalar_type());

        ptr_window_sizes = static_cast<void*>(kernel_sizes.value().data_ptr());
      }

      if (strides.has_value()) {
        CHECK_CONTIGUOUS(strides.value());
        CHECK_CUDA(strides.value());
        TORCH_CHECK(
            strides.value().dim() == 2,
            "Blackwell FNA: strides is expected to be a 2-D tensor.");

        TORCH_CHECK(
            strides.value().size(0) == batch_size_original,
            "Blackwell FNA: strides.shape[0] must be token_layouts.shape[0].");

        TORCH_CHECK(
            strides.value().size(1) == kNADim,
            "Blackwell FNA",
            kNADim,
            "-D: strides.shape[1] must be ",
            kNADim,
            ", got ",
            strides.value().size(1),
            ".");

        TORCH_CHECK(
            strides.value().scalar_type() == torch::kInt,
            "Blackwell FNA: strides is expected to be an int32 tensor, got ",
            strides.value().scalar_type());

        ptr_strides = static_cast<void*>(strides.value().data_ptr());
      }

      if (dilations.has_value()) {
        CHECK_CONTIGUOUS(dilations.value());
        CHECK_CUDA(dilations.value());
        TORCH_CHECK(
            dilations.value().dim() == 2,
            "Blackwell FNA: dilations is expected to be a 2-D tensor.");

        TORCH_CHECK(
            dilations.value().size(0) == batch_size_original,
            "Blackwell FNA: dilations.shape[0] must be token_layouts.shape[0].");

        TORCH_CHECK(
            dilations.value().size(1) == kNADim,
            "Blackwell FNA",
            kNADim,
            "-D: dilations.shape[1] must be ",
            kNADim,
            ", got ",
            dilations.value().size(1),
            ".");

        TORCH_CHECK(
            dilations.value().scalar_type() == torch::kInt,
            "Blackwell FNA: dilations is expected to be an int32 tensor, got ",
            dilations.value().scalar_type());

        ptr_dilations = static_cast<void*>(dilations.value().data_ptr());

        // variable dilations requires batch_map
        TORCH_CHECK(
            batch_map.has_value(),
            "Blackwell FNA: batch_map must all be specified when using variable dilations.");

        auto& batch_map_tensor = batch_map.value();
        CHECK_CONTIGUOUS(batch_map_tensor);
        CHECK_CUDA(batch_map_tensor);

        TORCH_CHECK(
            batch_map_tensor.dim() == 2,
            "Blackwell FNA: batch_map is expected to be a 2-D tensor.");

        TORCH_CHECK(
            batch_map_tensor.size(0) == batch_size,
            "Blackwell FNA: batch_map.shape[0] must be cumulative_seqlen_{Q,KV}.shape[0] - 1.");

        TORCH_CHECK(
            batch_map_tensor.size(1) == 2,
            "Blackwell FNA: batch_map.shape[1] must be ",
            2,
            ", got ",
            batch_map_tensor.size(1),
            ".");

        TORCH_CHECK(
            batch_map_tensor.scalar_type() == torch::kInt,
            "Blackwell FNA: batch_map is expected to be an int32 tensor, got ",
            batch_map_tensor.scalar_type());

        ptr_batch_map = static_cast<void*>(batch_map_tensor.data_ptr());
      }
    }

  } else {
    TORCH_CHECK(
        not is_varparam,
        "Blackwell FNA: variable-parameter FNA is only supported with variable-length FNA.");
  }

  TORCH_CHECK(
      not at::globalContext().deterministicAlgorithms(),
      "Blackwell FNA backward is non-deterministic, "
      "but PyTorch's deterministic mode is enabled. "
      "NATTEN Python API should have avoided this; which means "
      "you're probably calling the C function directly.");

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
      heads_q,
      heads_kv,
      dim,
      attn_scale,
      // fna parameters
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      // varlen parameters
      is_varlen,
      max_seqlen_Q,
      max_seqlen_KV,
      ptr_cumulative_seqlen_Q,
      ptr_cumulative_seqlen_KV,
      ptr_token_layouts,
      ptr_batch_map,
      // var-param parameters
      ptr_window_sizes,
      ptr_strides,
      ptr_dilations,
      // init/launch params
      device_id,
      cuda_stream,
      query.options());

#else
  TORCH_CHECK(
      false,
      "Blackwell FNA backward: libnatten was not compiled with CUTLASS_ARCH_MMA_SM100_SUPPORTED.");
#endif
#else
  TORCH_CHECK(
      false,
      "Blackwell FNA backward: libnatten was not compiled for Blackwell (SM100/SM103).");
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
    const std::tuple<int32_t>& key_tile_shape,
    // varlen
    const at::optional<at::Tensor>& cumulative_seqlen_Q,
    const at::optional<at::Tensor>& cumulative_seqlen_KV,
    const at::optional<at::Tensor>& token_layouts,
    const at::optional<at::Tensor>& batch_map,
    int max_seqlen_Q,
    int max_seqlen_KV,
    // var-param
    const at::optional<at::Tensor>& kernel_sizes,
    const at::optional<at::Tensor>& strides,
    const at::optional<at::Tensor>& dilations) {
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
      key_tile_shape,
      cumulative_seqlen_Q,
      cumulative_seqlen_KV,
      token_layouts,
      batch_map,
      max_seqlen_Q,
      max_seqlen_KV,
      kernel_sizes,
      strides,
      dilations);
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
    const std::tuple<int32_t, int32_t>& key_tile_shape,
    // varlen
    const at::optional<at::Tensor>& cumulative_seqlen_Q,
    const at::optional<at::Tensor>& cumulative_seqlen_KV,
    const at::optional<at::Tensor>& token_layouts,
    const at::optional<at::Tensor>& batch_map,
    int max_seqlen_Q,
    int max_seqlen_KV,
    // var-param
    const at::optional<at::Tensor>& kernel_sizes,
    const at::optional<at::Tensor>& strides,
    const at::optional<at::Tensor>& dilations) {
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
      key_tile_shape,
      cumulative_seqlen_Q,
      cumulative_seqlen_KV,
      token_layouts,
      batch_map,
      max_seqlen_Q,
      max_seqlen_KV,
      kernel_sizes,
      strides,
      dilations);
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
    const std::tuple<int32_t, int32_t, int32_t>& key_tile_shape,
    // varlen
    const at::optional<at::Tensor>& cumulative_seqlen_Q,
    const at::optional<at::Tensor>& cumulative_seqlen_KV,
    const at::optional<at::Tensor>& token_layouts,
    const at::optional<at::Tensor>& batch_map,
    int max_seqlen_Q,
    int max_seqlen_KV,
    // var-param
    const at::optional<at::Tensor>& kernel_sizes,
    const at::optional<at::Tensor>& strides,
    const at::optional<at::Tensor>& dilations) {
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
      key_tile_shape,
      cumulative_seqlen_Q,
      cumulative_seqlen_KV,
      token_layouts,
      batch_map,
      max_seqlen_Q,
      max_seqlen_KV,
      kernel_sizes,
      strides,
      dilations);
}

} // namespace natten
