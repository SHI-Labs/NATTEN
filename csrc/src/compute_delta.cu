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
/*! \file
    \brief compute_delta interface
    mostly used to test the kernel.
*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <natten/compute_delta.h>
#include <natten/helpers.h>
#include <natten/natten.h>

#ifdef NATTEN_WITH_CUTLASS
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <natten/cuda/reduction/compute_delta.cuh>
#endif

namespace natten {

void compute_delta(
    const at::Tensor& out,
    const at::Tensor& d_out,
    at::Tensor& delta) {
#ifdef NATTEN_WITH_CUTLASS
  at::cuda::OptionalCUDAGuard device_guard(out.device());

  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(d_out);
  CHECK_CONTIGUOUS(delta);
  CHECK_CUDA(out);
  CHECK_CUDA(d_out);
  CHECK_CUDA(delta);

  TORCH_CHECK(out.dim() == 4, "out must be 4-D.");
  TORCH_CHECK(d_out.dim() == 4, "d_out must be 4-D.");
  TORCH_CHECK(delta.dim() == 3, "delta must be 3-D.");

  CheckIfTensorShapesMatch<1>(out, d_out);
  CheckLogSumExp<1>(out, delta);

  int batch = out.size(0);
  int seqlen_Q = out.size(1);
  int heads = out.size(2);
  int dim = out.size(3);

  NATTEN_CHECK(
      out.scalar_type() == torch::kFloat ||
          out.scalar_type() == torch::kFloat16 ||
          out.scalar_type() == torch::kBFloat16,
      "`compute_delta` only supports FP32, FP16 and BF16 operands.");
  NATTEN_CHECK(
      d_out.scalar_type() == out.scalar_type(),
      "`compute_delta` input operands must match in dtype.");
  NATTEN_CHECK(
      delta.scalar_type() == torch::kFloat, "Delta is always in fp32.");

  if (out.scalar_type() == torch::kFloat) {
    natten::cuda::compute_delta<float, float>(
        at::cuda::getCurrentCUDAStream(out.device().index()),
        static_cast<const float*>(out.data_ptr()),
        static_cast<const float*>(d_out.data_ptr()),
        static_cast<float*>(delta.data_ptr()),
        batch,
        heads,
        seqlen_Q,
        dim);
  } else if (out.scalar_type() == torch::kFloat16) {
    natten::cuda::compute_delta(
        at::cuda::getCurrentCUDAStream(out.device().index()),
        static_cast<const cutlass::half_t*>(out.data_ptr()),
        static_cast<const cutlass::half_t*>(d_out.data_ptr()),
        static_cast<float*>(delta.data_ptr()),
        batch,
        heads,
        seqlen_Q,
        dim);
  } else if (out.scalar_type() == torch::kBFloat16) {
    natten::cuda::compute_delta(
        at::cuda::getCurrentCUDAStream(out.device().index()),
        static_cast<const cutlass::bfloat16_t*>(out.data_ptr()),
        static_cast<const cutlass::bfloat16_t*>(d_out.data_ptr()),
        static_cast<float*>(delta.data_ptr()),
        batch,
        heads,
        seqlen_Q,
        dim);
  } else {
    NATTEN_FAILURE("`compute_delta` is unavailable for this element type.");
  }

#else
  NATTEN_FAILURE(
      "`compute_delta` is only available when NATTEN is built with CUTLASS.");
#endif
}

} // namespace natten
