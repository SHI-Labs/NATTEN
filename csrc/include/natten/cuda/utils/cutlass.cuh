/***************************************************************************************************
 * Copyright (c) 2022-2025 Ali Hassani.
 **************************************************************************************************/
/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

/// SM90 Device Kernel wrapper
template <typename Operator>
CUTLASS_GLOBAL
#ifdef __CUDACC__
// Enclosing this in __CUDACC__ suppresses MSVC warnings.
__launch_bounds__(
    Operator::MaxThreadsPerBlock,
    Operator::MinBlocksPerMultiprocessor)
#endif // __CUDACC__
    void device_kernel_sm90(CUTLASS_GRID_CONSTANT
                            typename Operator::Params const params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 900
  // Dynamic shared memory base pointer
  extern __shared__ char smem[];
  Operator op;
  op(params, smem);
  cutlass::arch::synclog_print();
#else
  printf(
      "FATAL: This kernel was built for SM90, but attempted to launch from SM%d\n",
      int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif
}

/// SM100 Device Kernel wrapper
template <typename Operator>
CUTLASS_GLOBAL
#ifdef __CUDACC__
// Enclosing this in __CUDACC__ suppresses MSVC warnings.
__launch_bounds__(
    Operator::MaxThreadsPerBlock,
    Operator::MinBlocksPerMultiprocessor)
#endif // __CUDACC__
    void device_kernel_sm100(CUTLASS_GRID_CONSTANT
                             typename Operator::Params const params) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 1000
  // Dynamic shared memory base pointer
  extern __shared__ char smem[];
  Operator op;
  op(params, smem);
  cutlass::arch::synclog_print();
#else
  printf(
      "FATAL: This kernel was built for SM100, but attempted to launch from SM%d\n",
      int(__CUDA_ARCH__ + 0) / 10);
#endif
#endif
}

} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////

#define NATTEN_CUTLASS_CHECK(status)                                    \
  [&] {                                                                 \
    cutlass::Status error = status;                                     \
    if (error != cutlass::Status::kSuccess) {                           \
      std::cerr << "NATTEN failure: cutlass error: "                    \
                << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                           \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }();
