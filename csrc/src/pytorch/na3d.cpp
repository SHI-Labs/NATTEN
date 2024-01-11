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
    \brief Neighborhood Attention 3D Torch interface
*/

#include "natten/pytorch/cpu/na3d.h"
#include <ATen/ATen.h>
#include <torch/extension.h>
#ifdef NATTEN_WITH_CUDA
#include "natten/pytorch/cuda/na3d.cuh"
#endif

#include "natten/pytorch/helpers.h"

namespace natten {
namespace pytorch {

void na3d_qk_forward(
    at::Tensor& attn,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::optional<at::Tensor>& bias,
    const int kernel_size,
    const int dilation,
    const int kernel_size_d,
    const int dilation_d) {
  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CheckArgs(kernel_size, dilation);
  CheckArgs(kernel_size_d, dilation_d);
  CheckIfPropertiesMatch(query, key, attn);
  CheckIfTensorShapesMatch<3>(query, key);
  CheckAttnShape<3>(query, attn, kernel_size, kernel_size_d);
  if (bias.has_value()) {
    CheckBias<3>(query, bias.value(), kernel_size, kernel_size_d);
  }
  int batch_size = query.size(0);
  int heads = query.size(1);
  int depth = query.size(2);
  int height = query.size(3);
  int width = query.size(4);
  int dim = query.size(5);
  CheckArgsAgainstDim(depth, kernel_size_d, dilation_d);
  CheckArgsAgainstDim(height, kernel_size, dilation);
  CheckArgsAgainstDim(width, kernel_size, dilation);
  DISPATCH_DEVICE(
      query.device(),
      na3d_qk_forward,
      query,
      key,
      bias,
      attn,
      batch_size,
      heads,
      depth,
      height,
      width,
      dim,
      kernel_size,
      dilation,
      kernel_size_d,
      dilation_d);
}

void na3d_qk_backward(
    at::Tensor& d_query,
    at::Tensor& d_key,
    at::optional<at::Tensor>& d_bias,
    const at::Tensor& d_attn,
    const at::Tensor& query,
    const at::Tensor& key,
    const int kernel_size,
    const int dilation,
    const int kernel_size_d,
    const int dilation_d) {
  CHECK_CONTIGUOUS(d_query);
  CHECK_CONTIGUOUS(d_key);
  CHECK_CONTIGUOUS(query);
  CHECK_CONTIGUOUS(key);
  CheckArgs(kernel_size, dilation);
  CheckArgs(kernel_size_d, dilation_d);
  CheckIfPropertiesMatch(query, key);
  CheckIfPropertiesMatch(d_query, d_key, d_attn);
  CheckIfTensorShapesMatch<3>(query, key);
  CheckIfTensorShapesMatch<3>(d_query, d_key);
  CheckIfTensorShapesMatch<3>(query, d_key);
  CheckAttnShape<3>(query, d_attn, kernel_size, kernel_size_d);
  if (d_bias.has_value()) {
    CheckBias<3>(query, d_bias.value(), kernel_size, kernel_size_d);
  }
  int batch_size = query.size(0);
  int heads = query.size(1);
  int depth = query.size(2);
  int height = query.size(3);
  int width = query.size(4);
  int dim = query.size(5);
  CheckArgsAgainstDim(depth, kernel_size_d, dilation_d);
  CheckArgsAgainstDim(height, kernel_size, dilation);
  CheckArgsAgainstDim(width, kernel_size, dilation);
  DISPATCH_DEVICE(
      d_attn.device(),
      na3d_qk_backward,
      d_attn,
      query,
      key,
      d_query,
      d_key,
      d_bias,
      batch_size,
      heads,
      depth,
      height,
      width,
      dim,
      kernel_size,
      dilation,
      kernel_size_d,
      dilation_d);
}

void na3d_av_forward(
    at::Tensor& out,
    const at::Tensor& attn,
    const at::Tensor& value,
    const int kernel_size,
    const int dilation,
    const int kernel_size_d,
    const int dilation_d) {
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(value);
  CheckArgs(kernel_size, dilation);
  CheckArgs(kernel_size_d, dilation_d);
  CheckIfPropertiesMatch(out, value, attn);
  CheckIfTensorShapesMatch<3>(out, value);
  CheckAttnShape<3>(value, attn, kernel_size, kernel_size_d);
  int batch_size = value.size(0);
  int heads = value.size(1);
  int depth = value.size(2);
  int height = value.size(3);
  int width = value.size(4);
  int dim = value.size(5);
  CheckArgsAgainstDim(depth, kernel_size_d, dilation_d);
  CheckArgsAgainstDim(height, kernel_size, dilation);
  CheckArgsAgainstDim(width, kernel_size, dilation);
  DISPATCH_DEVICE(
      attn.device(),
      na3d_av_forward,
      attn,
      value,
      out,
      batch_size,
      heads,
      depth,
      height,
      width,
      dim,
      kernel_size,
      dilation,
      kernel_size_d,
      dilation_d);
}

void na3d_av_backward(
    at::Tensor& d_attn,
    at::Tensor& d_value,
    const at::Tensor& d_out,
    const at::Tensor& attn,
    const at::Tensor& value,
    const int kernel_size,
    const int dilation,
    const int kernel_size_d,
    const int dilation_d) {
  CHECK_CONTIGUOUS(d_out);
  CHECK_CONTIGUOUS(d_value);
  CHECK_CONTIGUOUS(value);
  CheckArgs(kernel_size, dilation);
  CheckArgs(kernel_size_d, dilation_d);
  CheckIfPropertiesMatch(attn, value);
  CheckIfPropertiesMatch(d_attn, d_value, d_out);
  CheckIfTensorShapesMatch<3>(value, d_value);
  CheckIfTensorShapesMatch<3>(attn, d_attn);
  CheckIfTensorShapesMatch<3>(value, d_out);
  CheckAttnShape<3>(value, attn, kernel_size, kernel_size_d);
  int batch_size = d_out.size(0);
  int heads = d_out.size(1);
  int depth = d_out.size(2);
  int height = d_out.size(3);
  int width = d_out.size(4);
  int dim = d_out.size(5);
  CheckArgsAgainstDim(depth, kernel_size_d, dilation_d);
  CheckArgsAgainstDim(height, kernel_size, dilation);
  CheckArgsAgainstDim(width, kernel_size, dilation);
  DISPATCH_DEVICE(
      attn.device(),
      na3d_av_backward,
      d_out,
      attn,
      value,
      d_attn,
      d_value,
      batch_size,
      heads,
      depth,
      height,
      width,
      dim,
      kernel_size,
      dilation,
      kernel_size_d,
      dilation_d);
}

} // namespace pytorch
} // namespace natten
