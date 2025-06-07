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
    \brief Hopper FMHA interface
*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <natten/helpers.h>
#include <natten/natten.h>

#ifdef NATTEN_WITH_CUTLASS
#include <natten_autogen/cuda/hopper_fmha/interface.h>
#include <natten/cuda/fmha_hopper/fmha_forward.cuh>
#endif

namespace natten {

#ifdef NATTEN_WITH_CUTLASS
#ifdef NATTEN_WITH_HOPPER_FNA
namespace {

auto kernel_type_int_to_enum_type(int kernel_type) {
  switch (kernel_type) {
    case 0:
      return natten::cuda::fmha_hopper::HopperFmhaKernelType::NonPersistent;
    case 1:
      return natten::cuda::fmha_hopper::HopperFmhaKernelType::WSCooperative;
    case 2:
      return natten::cuda::fmha_hopper::HopperFmhaKernelType::WSPingpong;
  }
  std::cerr
      << "Invalid value for argument kernel_type; expected either 0, 1, or 2, got "
      << kernel_type << ".\n";
  return natten::cuda::fmha_hopper::HopperFmhaKernelType::Invalid;
}

} // namespace
#endif
#endif

void hopper_fmha_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::optional<at::Tensor>& logsumexp,
    float attn_scale,
    int query_tile_size,
    int key_tile_size,
    int kernel_type) {
#ifdef NATTEN_WITH_CUTLASS
#ifdef NATTEN_WITH_HOPPER_FNA
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
  CheckIfTensorShapesMatch<1>(key, value);
  CheckIfTensorShapesMatch<1>(query, out);

  TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");
  TORCH_CHECK(key.dim() == 4, "Tensors must be 4-D.");
  TORCH_CHECK(value.dim() == 4, "Tensors must be 4-D.");

  int batch_size = query.size(0);
  int seqlen_q = query.size(1);
  int seqlen_kv = key.size(1);
  int heads = query.size(2);
  int dim = query.size(3);

  if (logsumexp.has_value()) {
    CheckLogSumExp<1>(out, logsumexp.value());
    CHECK_CUDA(logsumexp.value());
  }

  TORCH_CHECK(
      dim == 32 || dim == 64 || dim == 128 || dim == 256,
      "Hopper FMHA only supports head dims 32, 64, 128, and 256 for now.");

  cudaDeviceProp* device_props =
      at::cuda::getDeviceProperties(query.device().index());
  const int cc = device_props->major * 10 + device_props->minor;
  TORCH_CHECK(
      cc == 90,
      "This operation can only run on the Hopper architecture (SM90).");

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

  auto kernel_type_ = kernel_type_int_to_enum_type(kernel_type);
  TORCH_CHECK(
      kernel_type_ != natten::cuda::fmha_hopper::HopperFmhaKernelType::Invalid,
      "Got invalid kernel_type argument.");

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

  DISPATCH_HOPPER_FMHA_FORWARD(
      query.scalar_type(),
      dim,
      query_tile_size,
      key_tile_size,
      kernel_type_,
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
      device_id,
      attn_scale,
      cuda_stream,
      query.options());

#else
  TORCH_CHECK(
      false,
      "libnatten was not compiled with CUTLASS_ARCH_MMA_SM90_SUPPORTED.");
#endif
#else
  TORCH_CHECK(false, "libnatten was not compiled for Hopper (SM90).");
#endif
#else
  TORCH_CHECK(false, "libnatten not compiled with CUTLASS.");
#endif
}

} // namespace natten
