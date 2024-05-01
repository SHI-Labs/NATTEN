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
    \brief Neighborhood Attention 1D Torch interface
*/

#pragma once

#include <natten/natten.h>
#ifdef NATTEN_WITH_CUTLASS
#include <natten/cuda/reduction/device/compute_delta.h>
#include <natten/cuda/utils/cutlass.h>
#endif

namespace natten {
namespace cuda {

using Coord = cutlass::Coord<2>;

template <typename InputType, typename OutputType, int VectorLength>
void compute_delta_get_workspace_size(
    int64_t& workspace_size,
    int32_t num_rows,
    int32_t dim) {
  auto cta_count = std::min(num_rows, 65535);

#ifdef NATTEN_WITH_CUTLASS
  auto extent = Coord({(int)num_rows, (int)dim});

  using Kernel = natten::cuda::reduction::device::ComputeDelta<
      OutputType,
      InputType,
      VectorLength,
      /* ElementCompute = */ float,
      /* Threads = */ 32,
      /* BatchSize = */ 1>;

  auto kernel = Kernel(extent, cta_count);
  workspace_size = kernel.workspace_size();
#else
  NATTEN_FAILURE(
      "`compute_delta` is only available when NATTEN is built with CUTLASS.");
#endif
}

template <typename InputType, typename OutputType, int VectorLength>
void compute_delta(
    cudaStream_t stream,
    void* out_ptr,
    void* d_out_ptr,
    void* delta_ptr,
    void* workspace_ptr,
    int64_t workspace_size,
    int32_t num_rows,
    int32_t dim) {
#ifdef NATTEN_WITH_CUTLASS
  auto cta_count = std::min(num_rows, 65535);

  auto extent = Coord({(int)num_rows, (int)dim});

  using Kernel = natten::cuda::reduction::device::ComputeDelta<
      OutputType,
      InputType,
      VectorLength,
      /* ElementCompute = */ float,
      /* Threads = */ 32,
      /* BatchSize = */ 1>;

  auto kernel = Kernel(extent, cta_count);
  NATTEN_CHECK(
      workspace_size == kernel.workspace_size(),
      "NATTEN failure: workspace allocated for `compute_delta` was wrong.");
  int64_t stride[] = {(int64_t)dim};
  int64_t stride_out[] = {(int64_t)(1)};
  NATTEN_CUTLASS_CHECK(kernel(
      /* out */ reinterpret_cast<OutputType*>(delta_ptr),
      /* stride_out */ stride_out,
      reinterpret_cast<InputType*>(out_ptr),
      /* stride_a */ stride,
      reinterpret_cast<InputType*>(d_out_ptr),
      /* stride_b */ stride,
      workspace_ptr,
      stream));
#else
  NATTEN_FAILURE(
      "`compute_delta` is only available when NATTEN is built with CUTLASS.");
#endif
}

} // namespace cuda
} // namespace natten
