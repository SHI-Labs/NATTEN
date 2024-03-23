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
    \brief Just a simple naive kernel for clearing out workspace for backward.

    FMHA backward may require an accumulator for dQ, dK, and dV depending on
    the GEMM config and element type, similar to FMHA forward.

    However, FMHA backward will always require an accumulator for dQ if
    num_splits_key > 1.
    The reason is that multiple CTAs may race to update the same tile in dQ,
    hence a race condition that is handled with a mutex lock in the kernel.
    The mutex lock and counter are part of dQ accum region of workspace, and
    consequently they must be initialized as zeros since they have values added
    to them and not initialized any other way.

    Whether we use a cudaMemset or a kernel to zero out workspace, it's going
    to be somewhat inefficient. In some large-batch large-parallelism cases, it
    can be as big as approximately 5% of the latency of the backward kernel
   itself.

    Given that the mutex lock and counter are the only things we need to zero
   out, it makes sense to just have a naive kernel do that, and because the
   workload is significantly lower (the ratio of dQ blocks to the whole
   workspace), it'll probably be a lot faster.

    In some examples, I see roughly a 100x improvement (i.e. cudaMemset and
   torch's zero fill were around 500 us, this kernel was only 5 us.)

    The kernel itself is very simple. Threads in CTAs don't interact, and each
   thread is responsible for writing out 128 bits of zeros corresponding to one
   GradQTempStorage (the first 128 bits of GradQTempStorage is reserved for
   lock, counter, and 64 bits of padding.)
*/

#pragma once

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/functional.h>

#include <natten/cuda/utils/cuda.h>
#include <natten/cuda/utils/cutlass.h>
#include <natten/natten.h>

namespace natten {
namespace cuda {
namespace fna {
namespace utils {

template <typename T, int kElements>
__global__ void zero_out_workspace_counters(
    T* workspace_ptr,
    int64_t stride_batch,
    int64_t stride_row,
    // NOTE: stride_col assumed to be 1
    int32_t num_batches,
    int32_t num_rows,
    int32_t num_cols) {
  // NOTE: not aligned array; workspace_ptr post gk/gv offset
  // isn't guaranteed to be aligned.
  using Fragment = cutlass::Array<float, kElements>;

  int32_t batch_idx = (int32_t)(blockIdx.y * blockDim.y + threadIdx.y);
  int32_t row_idx = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (batch_idx < num_batches && row_idx < num_rows) {
    auto ptr = reinterpret_cast<Fragment*>(
        workspace_ptr + batch_idx * stride_batch + row_idx * stride_row);
    auto zeros = Fragment();
    zeros.clear();
#pragma unroll
    for (int32_t col_idx = 0; col_idx < num_cols; col_idx += kElements, ptr++) {
      *ptr = zeros;
    }
  }
}

} // namespace utils
} // namespace fna
} // namespace cuda
} // namespace natten
