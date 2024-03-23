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

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/functional.h>

#include <natten/cuda/utils/cuda.h>
#include <natten/cuda/utils/cutlass.h>
#include <natten/natten.h>
#include <iostream>
#include <natten/cuda/fna/clear_workspace.cuh>
#include <natten/cuda/fna/clear_workspace_kernel.cuh>

namespace natten {
namespace cuda {
namespace fna {
namespace utils {

void clear_workspace(
    cudaStream_t stream,
    float* workspace_ptr,
    int64_t stride_batch,
    int64_t stride_row,
    int64_t stride_col,
    int32_t num_batches,
    int32_t num_rows,
    int32_t num_cols) {
  constexpr int kAccessSize = 4;
  constexpr int max_threads_per_cta = 512;

  int cta_shape_batch = cutlass::fast_min(max_threads_per_cta, num_batches);
  int cta_shape_row =
      cutlass::fast_min(max_threads_per_cta / cta_shape_batch, num_rows);

  NATTEN_CHECK(
      stride_col == 1,
      "FNA `clear_workspace` failed! workspace_q must be contiguous in the last dim.");
  NATTEN_CHECK(
      num_cols % kAccessSize == 0,
      "FNA `clear_workspace` failed! Expected the number of columns to be a multiple of 4.");

  auto blocks_batch = cutlass::ceil_div(num_batches, cta_shape_batch);
  auto blocks_row = cutlass::ceil_div(num_rows, cta_shape_row);

  NATTEN_CHECK(
      blocks_row * blocks_batch < 65535,
      "`clear_workspace` kernel launch failed.");

  dim3 blocks(blocks_row, blocks_batch);
  dim3 threads(cta_shape_row, cta_shape_batch);

  zero_out_workspace_counters<float, kAccessSize>
      <<<blocks, threads, 0, stream>>>(
          workspace_ptr,
          stride_batch,
          stride_row,
          // stride_col,
          num_batches,
          num_rows,
          num_cols);

  NATTEN_CUDA_CHECK(cudaGetLastError());
}

} // namespace utils
} // namespace fna
} // namespace cuda
} // namespace natten
