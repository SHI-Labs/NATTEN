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

#pragma once
#include <ATen/ATen.h>

#define CHECK_CONTIGUOUS(x)                                  \
  TORCH_CHECK(!x.is_sparse(), #x " must be a dense tensor"); \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous");

#ifdef NATTEN_WITH_CUDA
#define DISPATCH_DEVICE(c10_type, kernel_name, ...)                          \
  [&] {                                                                      \
    if (c10_type.is_cpu()) {                                                 \
      natten::pytorch::cpu::kernel_name(__VA_ARGS__);                        \
    } else if (c10_type.is_cuda()) {                                         \
      natten::pytorch::cuda::kernel_name(__VA_ARGS__);                       \
    } else {                                                                 \
      std::cerr << "NATTEN does not support " << c10_type << " devices yet." \
                << std::endl;                                                \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  }()
#else
#define DISPATCH_DEVICE(c10_type, kernel_name, ...)                          \
  [&] {                                                                      \
    if (c10_type.is_cpu()) {                                                 \
      natten::pytorch::cpu::kernel_name(__VA_ARGS__);                        \
    } else if (c10_type.is_cuda()) {                                         \
      std::cerr << "NATTEN was not built with " << c10_type << " support."   \
                << std::endl;                                                \
    } else {                                                                 \
      std::cerr << "NATTEN does not support " << c10_type << " devices yet." \
                << std::endl;                                                \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  }()
#endif

namespace natten {
namespace pytorch {

inline void CheckArgs(int32_t kernel_size, int32_t dilation) {
  TORCH_CHECK(
      kernel_size > 1 && kernel_size % 2 == 1,
      "Kernel size must be an odd number greater than 1, got ",
      kernel_size,
      ".");
  TORCH_CHECK(
      dilation >= 1,
      "Dilation must be a nonnegative integer, got ",
      dilation,
      ".");
}

inline void CheckArgs(
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& dilation) {
  CheckArgs(std::get<0>(kernel_size), std::get<0>(dilation));
}

inline void CheckArgs(
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& dilation) {
  CheckArgs(std::get<0>(kernel_size), std::get<0>(dilation));
  CheckArgs(std::get<1>(kernel_size), std::get<1>(dilation));
}

inline void CheckArgs(
    const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  CheckArgs(std::get<0>(kernel_size), std::get<0>(dilation));
  CheckArgs(std::get<1>(kernel_size), std::get<1>(dilation));
  CheckArgs(std::get<2>(kernel_size), std::get<2>(dilation));
}

inline void CheckArgsAgainstDim(
    int32_t dim,
    int32_t kernel_size,
    int32_t dilation) {
  TORCH_CHECK(
      kernel_size * dilation <= dim,
      "Input axes must be greater than or equal to the product of kernel size and dilation. "
      "Got kernel size ",
      kernel_size,
      ", dilation ",
      dilation,
      ", but dimension size was ",
      dim,
      ".");
}

inline void CheckArgsAgainstDim(
    const std::tuple<int32_t>& dim,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& dilation) {
  CheckArgsAgainstDim(
      std::get<0>(dim), std::get<0>(kernel_size), std::get<0>(dilation));
}

inline void CheckArgsAgainstDim(
    const std::tuple<int32_t, int32_t>& dim,
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& dilation) {
  CheckArgsAgainstDim(
      std::get<0>(dim), std::get<0>(kernel_size), std::get<0>(dilation));
  CheckArgsAgainstDim(
      std::get<1>(dim), std::get<1>(kernel_size), std::get<1>(dilation));
}

inline void CheckArgsAgainstDim(
    const std::tuple<int32_t, int32_t, int32_t>& dim,
    const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  CheckArgsAgainstDim(
      std::get<0>(dim), std::get<0>(kernel_size), std::get<0>(dilation));
  CheckArgsAgainstDim(
      std::get<1>(dim), std::get<1>(kernel_size), std::get<1>(dilation));
  CheckArgsAgainstDim(
      std::get<2>(dim), std::get<2>(kernel_size), std::get<2>(dilation));
}

inline void CheckIfPropertiesMatch(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(
      a.device().is_cuda() == b.device().is_cuda(),
      "Expected all tensors to be on the same device.");
  TORCH_CHECK(
      a.scalar_type() == b.scalar_type(), "Input tensors must match in dtype!");
}

inline void CheckIfPropertiesMatch(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& c) {
  TORCH_CHECK(
      a.device().is_cuda() == b.device().is_cuda() &&
          b.device().is_cuda() == c.device().is_cuda(),
      "Expected all tensors to be on the same device.");
  TORCH_CHECK(
      a.scalar_type() == b.scalar_type() && b.scalar_type() == c.scalar_type(),
      "Input tensors must match in dtype!");
}

template <size_t NaDim>
void CheckIfTensorShapesMatch(const at::Tensor& a, const at::Tensor& b) {
  static_assert(NaDim >= 1 && NaDim < 4);
  static constexpr size_t Rank = NaDim + 3;
  TORCH_CHECK(
      a.dim() == b.dim() && a.dim() == Rank, "Expected ", Rank, "-D tensors.");
  for (size_t i = 0; i < Rank; ++i) {
    TORCH_CHECK(
        a.size(i) == b.size(i),
        "Tensor shape mismatch at dimension ",
        i,
        ": ",
        a.size(i),
        " != ",
        b.size(i));
  }
}

template <size_t NaDim, typename KernelType>
void CheckAttnShape(
    const at::Tensor& input,
    const at::Tensor& attn,
    KernelType kernel_size) {
  static_assert(NaDim >= 1 && NaDim < 4);
  TORCH_CHECK(attn.dim() == NaDim + 3, "Expected ", NaDim + 3, "-D tensors.");
  for (size_t i = 0; i < NaDim + 2; ++i) {
    TORCH_CHECK(
        input.size(i) == attn.size(i),
        "Tensor shape mismatch at dimension ",
        i,
        ": ",
        input.size(i),
        " != ",
        input.size(i));
  }
  auto expected_kernel_size = natten::flatten(kernel_size);
  TORCH_CHECK(
      attn.size(NaDim + 2) == expected_kernel_size,
      "Expected attention dim was ",
      expected_kernel_size,
      ", got ",
      attn.size(NaDim + 2));
}

template <size_t NaDim, typename KernelType>
void CheckBias(
    const at::Tensor& input,
    const at::Tensor& bias,
    int32_t num_heads,
    KernelType kernel_size) {
  static_assert(NaDim >= 1 && NaDim < 4);
  TORCH_CHECK(
      input.scalar_type() == bias.scalar_type(),
      "Inputs and bias must match in dtype.");
  TORCH_CHECK(
      bias.device().is_cuda() == input.device().is_cuda(),
      "Expected positional bias to be on the same device as the inputs.");
  CHECK_CONTIGUOUS(bias);
  TORCH_CHECK(bias.size(0) == num_heads, "Expected bias.shape[0] == heads.");
  for (size_t i = 0; i < NaDim; ++i) {
    auto expected_bias_dim = natten::get_from_tuple(kernel_size, i) * 2 - 1;
    TORCH_CHECK(
        bias.size(i + 1) == expected_bias_dim,
        "Invalid bias shape at dim ",
        i + 1,
        "; "
        "expected ",
        expected_bias_dim,
        ", got ",
        bias.size(i + 1),
        ".");
  }
}

template <size_t NaDim>
void CheckLogSumExp(const at::Tensor& output, const at::Tensor& logsumexp) {
  // Output: [batch, *, heads, dim]
  // Logsumexp: [batch, *, heads]
  static_assert(NaDim >= 1 && NaDim < 4);
  TORCH_CHECK(
      logsumexp.scalar_type() == torch::kFloat,
      "`logsumexp` must be stored in float32 data type, got ",
      logsumexp.scalar_type());
  TORCH_CHECK(
      logsumexp.device().is_cuda() == output.device().is_cuda(),
      "Expected logsumexp to be on the same device as the operands.");
  CHECK_CONTIGUOUS(logsumexp);
  TORCH_CHECK(
      output.dim() == NaDim + 3,
      NaDim,
      "-D NA expects operands to be ",
      NaDim + 3,
      " rank tensors, got ",
      output.dim());
  TORCH_CHECK(
      logsumexp.dim() == output.dim() - 1,
      NaDim,
      "-D NA expects logsumexp to be a ",
      NaDim + 2,
      " rank tensor, got ",
      logsumexp.dim());
  for (size_t i = 0; i < NaDim + 2; ++i) {
    TORCH_CHECK(
        logsumexp.size(i) == output.size(i),
        "Invalid logsumexp shape at dim ",
        i,
        "; "
        "expected ",
        output.size(i),
        ", got ",
        logsumexp.size(i),
        ".");
  }
}

inline void AssertDimsAre128BitAligned(
    const at::Tensor& query,
    const at::Tensor& value) {
  auto head_dim = query.size(-1);
  auto head_dim_value = value.size(-1);
  TORCH_CHECK(
      query.scalar_type() == value.scalar_type(),
      "QKV must match in data type, got query.dtype=",
      query.scalar_type(),
      ", but value.dtype=",
      value.scalar_type(),
      ".");
  TORCH_CHECK(
      query.scalar_type() == torch::kFloat ||
          query.scalar_type() == torch::kFloat16 ||
          query.scalar_type() == torch::kBFloat16,
      "This NATTEN operation only supports FP32, FP16, and BF16 data types, got ",
      query.scalar_type(),
      ".");

  if (query.scalar_type() == torch::kFloat) {
    TORCH_CHECK(
        head_dim % 4 == 0,
        "Query dimension must be a multiple of 4 for FP32 operands, got ",
        head_dim,
        ".");
    TORCH_CHECK(
        head_dim_value % 4 == 0,
        "Value dimension must be a multiple of 4 for FP32 operands, got ",
        head_dim_value,
        ".");
  } else if (
      query.scalar_type() == torch::kFloat16 ||
      query.scalar_type() == torch::kBFloat16) {
    TORCH_CHECK(
        head_dim % 8 == 0,
        "Query dimension must be a multiple of 8 for FP16/BF16 operands, got ",
        head_dim,
        ".");
    TORCH_CHECK(
        head_dim_value % 8 == 0,
        "Value dimension must be a multiple of 8 for FP16/BF16 operands, got ",
        head_dim_value,
        ".");
  }
}

} // namespace pytorch
} // namespace natten
