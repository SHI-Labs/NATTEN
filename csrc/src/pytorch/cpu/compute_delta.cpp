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

#include <natten/cpu/na1d.h>
#include <natten/dtypes.h>
#include <natten/natten.h>
#include <natten/pytorch/cpu/helpers.h>

namespace natten {
namespace pytorch {
namespace cpu {

void compute_delta(
    const at::Tensor& out,
    const at::Tensor& d_out,
    at::Tensor& delta,
    int32_t num_rows,
    int32_t dim) {
  // TODO: implement CPU reference
  NATTEN_FAILURE("`compute_delta` is not available on CPU.");
}

} // namespace cpu
} // namespace pytorch
} // namespace natten
