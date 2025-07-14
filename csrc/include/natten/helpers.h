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

#pragma once
#include <ATen/ATen.h>

#define CHECK_CONTIGUOUS(x)                                     \
  TORCH_CHECK(not x.is_sparse(), #x " must be a dense tensor"); \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous");

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor.");

namespace natten {

inline void AssertOddKernelSize(int32_t kernel_size) {
  TORCH_CHECK(
      kernel_size % 2 == 1,
      "This operation only supports odd-sized kernel sizes, got ",
      kernel_size,
      ".");
}

inline void AssertOddKernelSize(const std::tuple<int32_t>& kernel_size) {
  AssertOddKernelSize(std::get<0>(kernel_size));
}

inline void AssertOddKernelSize(
    const std::tuple<int32_t, int32_t>& kernel_size) {
  AssertOddKernelSize(std::get<0>(kernel_size));
  AssertOddKernelSize(std::get<1>(kernel_size));
}

inline void AssertOddKernelSize(
    const std::tuple<int32_t, int32_t, int32_t>& kernel_size) {
  AssertOddKernelSize(std::get<0>(kernel_size));
  AssertOddKernelSize(std::get<1>(kernel_size));
  AssertOddKernelSize(std::get<2>(kernel_size));
}

inline void FnaRPBChecks(int32_t kernel_size, int32_t stride, bool has_rpb) {
  TORCH_CHECK(
      !has_rpb || (kernel_size % 2 == 1 && stride == 1),
      "This operation only supports odd-sized kernel sizes and stride=1 with RPB enabled, got ",
      kernel_size,
      ".");
}

inline void FnaRPBChecks(
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& stride,
    bool has_rpb) {
  FnaRPBChecks(std::get<0>(kernel_size), std::get<0>(stride), has_rpb);
}

inline void FnaRPBChecks(
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& stride,
    bool has_rpb) {
  FnaRPBChecks(std::get<0>(kernel_size), std::get<0>(stride), has_rpb);
  FnaRPBChecks(std::get<1>(kernel_size), std::get<1>(stride), has_rpb);
}

inline void FnaRPBChecks(
    const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t, int32_t>& stride,
    bool has_rpb) {
  FnaRPBChecks(std::get<0>(kernel_size), std::get<0>(stride), has_rpb);
  FnaRPBChecks(std::get<1>(kernel_size), std::get<1>(stride), has_rpb);
  FnaRPBChecks(std::get<2>(kernel_size), std::get<2>(stride), has_rpb);
}

inline void CheckArgs(int32_t kernel_size, int32_t dilation) {
  TORCH_CHECK(
      kernel_size > 1,
      "Kernel size must be greater than 1, got ",
      kernel_size,
      ".");
  TORCH_CHECK(
      dilation >= 1,
      "Dilation must be a positive integer, got ",
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

inline void CheckArgs(int32_t kernel_size, int32_t stride, int32_t dilation) {
  CheckArgs(kernel_size, dilation);
  TORCH_CHECK(
      stride >= 1, "Stride must be a positive integer, got ", stride, ".");
  TORCH_CHECK(
      stride <= kernel_size,
      "Stride must be smaller than or equal to kernel size, got ",
      "kernel_size=",
      kernel_size,
      ", stride=",
      stride,
      ".");
}

inline void CheckArgs(
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& stride,
    const std::tuple<int32_t>& dilation) {
  CheckArgs(
      std::get<0>(kernel_size), std::get<0>(stride), std::get<0>(dilation));
}

inline void CheckArgs(
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t>& dilation) {
  CheckArgs(
      std::get<0>(kernel_size), std::get<0>(stride), std::get<0>(dilation));
  CheckArgs(
      std::get<1>(kernel_size), std::get<1>(stride), std::get<1>(dilation));
}

inline void CheckArgs(
    const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  CheckArgs(
      std::get<0>(kernel_size), std::get<0>(stride), std::get<0>(dilation));
  CheckArgs(
      std::get<1>(kernel_size), std::get<1>(stride), std::get<1>(dilation));
  CheckArgs(
      std::get<2>(kernel_size), std::get<2>(stride), std::get<2>(dilation));
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

template <size_t NaDim>
void CheckIfTensorShapesMatchExceptHeadDim(
    const at::Tensor& a,
    const at::Tensor& b) {
  static_assert(NaDim >= 1 && NaDim < 4);
  static constexpr size_t Rank = NaDim + 3;
  TORCH_CHECK(
      a.dim() == b.dim() && a.dim() == Rank, "Expected ", Rank, "-D tensors.");
  for (size_t i = 0; i < Rank - 1; ++i) {
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

inline void CheckIfBatchHeadsMatch(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.dim() == b.dim(), "Expected tensors to match in rank.");
  auto tensor_rank = a.dim();
  TORCH_CHECK(
      a.size(0) == b.size(0),
      "Tensors don't match in batch size; ",
      a.size(0),
      " != ",
      b.size(0));
  TORCH_CHECK(
      a.size(tensor_rank - 2) == b.size(tensor_rank - 2),
      "Tensors don't match in number of heads; ",
      a.size(tensor_rank - 2),
      " != ",
      b.size(tensor_rank - 2));
}

inline void CheckIfHeadDimsMatch(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.dim() == b.dim(), "Expected tensors to match in rank.");
  auto tensor_rank = a.dim();
  TORCH_CHECK(
      a.size(tensor_rank - 1) == b.size(tensor_rank - 1),
      "Tensors don't match in head dim; ",
      a.size(tensor_rank - 1),
      " != ",
      b.size(tensor_rank - 1));
}

inline void CheckIfBatchHeadsHeadDimMatch(
    const at::Tensor& a,
    const at::Tensor& b) {
  TORCH_CHECK(a.dim() == b.dim(), "Expected tensors to match in rank.");
  auto tensor_rank = a.dim();
  CheckIfBatchHeadsMatch(a, b);
  CheckIfHeadDimsMatch(a, b);
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

inline void CheckLogSumExpHeadsFirst(
    const at::Tensor& output,
    const at::Tensor& logsumexp) {
  // Output: [batch, seqlen, heads, dim]
  // Logsumexp: [batch, heads, seqlen]
  TORCH_CHECK(
      logsumexp.scalar_type() == torch::kFloat,
      "`logsumexp` must be stored in float32 data type, got ",
      logsumexp.scalar_type());
  TORCH_CHECK(
      logsumexp.device().is_cuda() == output.device().is_cuda(),
      "Expected logsumexp to be on the same device as the operands.");
  CHECK_CONTIGUOUS(logsumexp);
  TORCH_CHECK(
      output.dim() == 4, "Expected 4D tensor, got ", output.dim(), "D.");
  TORCH_CHECK(
      logsumexp.dim() == 3,
      "Expected 3D logsumexp tensor, got ",
      logsumexp.dim(),
      "D/");

  TORCH_CHECK(
      logsumexp.size(0) == output.size(0),
      "Logsumexp and input tensor don't match in the batch dimension. ",
      "Input tensor has batch=",
      output.size(0),
      ", logsumexp has batch=",
      logsumexp.size(0),
      ".");

  TORCH_CHECK(
      logsumexp.size(1) == output.size(2),
      "Logsumexp and input tensor don't match in the head dimension. ",
      "Input tensor has heads=",
      output.size(2),
      ", logsumexp has heads=",
      logsumexp.size(1),
      ".");

  // NOTE: seqlen check is >= instead of ==, since LSE can be padded
  TORCH_CHECK(
      logsumexp.size(2) >= output.size(1),
      "Logsumexp and input tensor don't match in the seqlen dimension. ",
      "Input tensor has seqlen=",
      output.size(1),
      ", logsumexp has seqlen=",
      logsumexp.size(2),
      ".");
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

} // namespace natten
