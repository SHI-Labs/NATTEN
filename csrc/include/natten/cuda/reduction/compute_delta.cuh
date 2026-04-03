/***************************************************************************************************
 * Copyright (c) 2022 - 2026 Ali Hassani.
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

#include <natten/natten.h>
#ifdef NATTEN_WITH_CUTLASS
#include <natten/cuda/reduction/fmha_kernel_bwd_sum_OdO.hpp>
#include <natten/cuda/utils/generic_cutlass_device.hpp>
#include <natten/cuda/utils/cutlass.cuh>
#endif

namespace natten {
namespace cuda {

template <typename ElementIn, typename ElementOut>
void compute_delta(
    cudaStream_t stream,
    const ElementIn* ptr_O,
    const ElementIn* ptr_dO,
    ElementOut* ptr_sum_OdO,
    int batch,
    int heads,
    int seqlen_Q,
    int dim) {
#ifdef NATTEN_WITH_CUTLASS
  using namespace cute;
  // B H Q D
  using ProblemShape = cute::tuple<int, int, int, int>;
  using OperationSumOdO = cutlass::device::DeviceKernel<
      cutlass::fmha::kernel::
          FmhaKernelBwdSumOdO<ProblemShape, ElementIn, ElementOut>>;

  OperationSumOdO op_sum_OdO;

  ProblemShape problem_shape = cute::make_tuple(batch, heads, seqlen_Q, dim);
  auto stride_O = make_stride(
      static_cast<int64_t>(dim * heads) * static_cast<int64_t>(seqlen_Q),
      dim,
      dim * heads,
      _1{});
  auto stride_dO = stride_O;
  auto stride_sum_OdO = make_stride(
      static_cast<int64_t>(heads) * static_cast<int64_t>(seqlen_Q),
      _1{},
      heads);

  auto args = typename OperationSumOdO::Arguments{
      problem_shape,
      ptr_O,
      stride_O,
      ptr_dO,
      stride_dO,
      ptr_sum_OdO,
      stride_sum_OdO};

  auto status = OperationSumOdO::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    NATTEN_FAILURE(
        "`compute_delta` kernel is not supported for this use case.");
  }

  status = op_sum_OdO.initialize(args, nullptr, stream);
  if (status != cutlass::Status::kSuccess) {
    NATTEN_FAILURE("`compute_delta` kernel failed to initialize.");
  }

  auto result = op_sum_OdO.run(stream);
  if (result != cutlass::Status::kSuccess) {
    NATTEN_FAILURE("`compute_delta` kernel launch failed.");
  }
#else
  NATTEN_FAILURE(
      "`compute_delta` is only available when NATTEN is built with CUTLASS.");
#endif
}

} // namespace cuda
} // namespace natten
