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
/*! \file
    \brief compute_delta interface
    mostly used to test the kernel.
*/

#include <ATen/ATen.h>
#include <torch/extension.h>

#include <natten/natten.h>
#include <natten/pytorch/cpu/compute_delta.h>
#ifdef NATTEN_WITH_CUDA
#include <natten/pytorch/cuda/compute_delta.cuh>
#endif
#include <natten/pytorch/helpers.h>

namespace natten {
namespace pytorch {

void compute_delta(
    const at::Tensor& out,
    const at::Tensor& d_out,
    at::Tensor& delta) {
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(d_out);
  CHECK_CONTIGUOUS(delta);
  int32_t num_rows = 1;
  NATTEN_CHECK(
      out.numel() == d_out.numel() && out.dim() == d_out.dim(),
      "`compute_delta` expects operands `out` and `d_out` to be contiguous, and "
      "of the same shape.");
  NATTEN_CHECK(
      out.dim() >= 2,
      "`compute_delta` expects operands `out` and `d_out` to be at least rank 2.");
  for (size_t i = 0; i < out.dim() - 1; ++i) {
    NATTEN_CHECK(
        out.size(i) == d_out.size(i) && delta.size(i) == out.size(i),
        "`compute_delta` expects all operands to match in shape up to the last dimension.");
    num_rows *= out.size(i);
  }

  int32_t dim = out.size(-1);
  DISPATCH_DEVICE(
      out.device(), compute_delta, out, d_out, delta, num_rows, dim);
}

} // namespace pytorch
} // namespace natten
