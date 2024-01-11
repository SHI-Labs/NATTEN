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

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

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

inline void CheckArgs(int kernel_size, int dilation) {
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

inline void CheckArgsAgainstDim(int dim, int kernel_size, int dilation) {
  TORCH_CHECK(
      kernel_size * dilation <= dim,
      "Input axes must be less than or equal to the product of kernel size and dilation. "
      "Got kernel size ",
      kernel_size,
      ", dilation ",
      dilation,
      ", but dimension size was ",
      dim,
      ".");
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

template <size_t NaDim>
void CheckAttnShape(
    const at::Tensor& input,
    const at::Tensor& attn,
    int kernel_size) {
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
  auto expected_kernel_size = std::pow(kernel_size, NaDim);
  TORCH_CHECK(
      attn.size(NaDim + 2) == expected_kernel_size,
      "Expected attention dim was ",
      expected_kernel_size,
      ", got ",
      attn.size(NaDim + 2));
}

template <size_t NaDim>
void CheckBias(
    const at::Tensor& input,
    const at::Tensor& bias,
    int kernel_size) {
  static_assert(NaDim >= 1 && NaDim < 4);
  TORCH_CHECK(
      input.scalar_type() == bias.scalar_type(),
      "Inputs and bias must match in dtype.");
  TORCH_CHECK(
      bias.device().is_cuda() == input.device().is_cuda(),
      "Expected positional bias to be on the same device as the inputs.");
  CHECK_CONTIGUOUS(bias);
  TORCH_CHECK(
      bias.size(0) == input.size(1),
      "Expected bias.shape[0] == input.shape[1] == heads.");
  for (size_t i = 0; i < NaDim; ++i) {
    auto expected_bias_dim = kernel_size * 2 - 1;
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

// TODO: I resent this; please do it the right way.
template <size_t NaDim>
void CheckAttnShape(
    const at::Tensor& input,
    const at::Tensor& attn,
    int kernel_size,
    int kernel_size_d) {
  static_assert(NaDim == 3);
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
  auto expected_kernel_size = kernel_size * kernel_size * kernel_size_d;
  TORCH_CHECK(
      attn.size(NaDim + 2) == expected_kernel_size,
      "Expected attention dim was ",
      expected_kernel_size,
      ", got ",
      attn.size(NaDim + 2));
}

template <size_t NaDim>
void CheckBias(
    const at::Tensor& input,
    const at::Tensor& bias,
    int kernel_size,
    int kernel_size_d) {
  static_assert(NaDim == 3);
  TORCH_CHECK(
      input.scalar_type() == bias.scalar_type(),
      "Inputs and bias must match in dtype.");
  TORCH_CHECK(
      bias.device().is_cuda() == input.device().is_cuda(),
      "Expected positional bias to be on the same device as the inputs.");
  CHECK_CONTIGUOUS(bias);
  TORCH_CHECK(
      bias.size(0) == input.size(1),
      "Expected bias.shape[0] == input.shape[1] == heads.");

  auto expected_bias_dim_0 = kernel_size_d * 2 - 1;
  TORCH_CHECK(
      bias.size(1) == expected_bias_dim_0,
      "Invalid bias shape at dim 1; expected ",
      expected_bias_dim_0,
      ", got ",
      bias.size(1),
      ".");
  for (size_t i = 1; i < NaDim; ++i) {
    auto expected_bias_dim = kernel_size * 2 - 1;
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

} // namespace pytorch
} // namespace natten
