/***************************************************************************************************
 * Copyright (c) 2022-2024 Ali Hassani.
 * NATTEN's CUTLASS kernels started off from CUTLASS 2.X's implicit GEMM kernels
 *for convolution.
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

/*! \file
    \brief Architecture-specific operators on memory added for SM80

      Based on and started off of the original cutlass version.

      Context: neighborhood attention weights (and by extension the gradient for
   weights) will not meet the minimum alignment requirement for asynchronous
   reads.

        This is due to NA kernel sizes being odd numbers, and the contiguous
   dimension in attention weights is exactly equal to kernel size (or kernel
   size squared if 2D.)

        Because of this, we need to fall back to synchronous reads in FP16/BF16
   kernels, since attention weights are only going to be read one-by-one, and
   the minimum alignment is 32 bits, but attention weights will be 16-bit.
*/

#pragma once

#include <cutlass/arch/cache_operation.h>
#include <cutlass/arch/memory.h>
#include <cutlass/arch/memory_sm75.h>
#include <cutlass/complex.h>
#include <cutlass/cutlass.h>

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
#define CUDA_CP_ASYNC_ACTIVATED 1
#else
#define CUDA_CP_ASYNC_ACTIVATED 0
#endif

namespace natten {
namespace cuda {
namespace gemm {
namespace arch {

/// Initiates an asynchronous copy from global memory to shared memory. Rather
/// than predicate the entire transfer, zeros are written to SMEM if the guard
/// predicate is false.
///
/// LDGSTS
///
template <
    /// Size of the access in bytes
    int SizeInBytes,
    /// Cache operation
    cutlass::arch::CacheOperation::Kind cache_op =
        cutlass::arch::CacheOperation::Always>
struct cp_async_zfill;

/// Partial specialization
template <>
struct cp_async_zfill<2, cutlass::arch::CacheOperation::Always> {
  /// Copy with zero fill
  CUTLASS_DEVICE
  cp_async_zfill(void* smem_ptr, void const* global_ptr, bool pred_guard) {
    using AccessType = cutlass::Array<uint8_t, 2>;

    if (pred_guard) {
      *static_cast<AccessType*>(smem_ptr) =
          *static_cast<AccessType const*>(global_ptr);
    } else {
      AccessType zeros;
      zeros.clear();
      *static_cast<AccessType*>(smem_ptr) = zeros;
    }
  }
};

/// Partial specialization
template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async_zfill<SizeInBytes, cutlass::arch::CacheOperation::Always> {
  /// Copy with zero fill
  CUTLASS_DEVICE
  cp_async_zfill(void* smem_ptr, void const* global_ptr, bool pred_guard) {
#if CUDA_CP_ASYNC_ACTIVATED

    // Make sure the size is supported.
    static_assert(
        (SizeInBytes == 4 || SizeInBytes == 8 || SizeInBytes == 16),
        "Size is not supported");

    unsigned smem_int_ptr = cutlass::arch::cutlass_get_smem_pointer(smem_ptr);
    int src_in_bytes = (pred_guard ? SizeInBytes : 0);

    asm volatile(
#if CUTLASS_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
            smem_int_ptr),
#else
        "cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
#endif
        "l"(global_ptr),
        "n"(SizeInBytes),
        "r"(src_in_bytes));

#else
    using AccessType = cutlass::Array<uint8_t, SizeInBytes>;

    if (pred_guard) {
      *static_cast<AccessType*>(smem_ptr) =
          *static_cast<AccessType const*>(global_ptr);
    } else {
      AccessType zeros;
      zeros.clear();
      *static_cast<AccessType*>(smem_ptr) = zeros;
    }
#endif
  }
};

/// Partial specialization
template <
    /// Size of the access in bytes
    int SizeInBytes>
struct cp_async_zfill<SizeInBytes, cutlass::arch::CacheOperation::Global> {
  /// Copy with zero fill
  CUTLASS_DEVICE
  cp_async_zfill(
      void* smem_ptr,
      void const* global_ptr,
      bool pred_guard = true) {
#if CUDA_CP_ASYNC_ACTIVATED

    static_assert(
        SizeInBytes == 16,
        "cp.async only supports CacheOperation::Global when access size is 16B.");

    unsigned smem_int_ptr = cutlass::arch::cutlass_get_smem_pointer(smem_ptr);
    int src_in_bytes = (pred_guard ? SizeInBytes : 0);

    asm volatile(
#if CUTLASS_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(
            smem_int_ptr),
#else
        "cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
#endif
        "l"(global_ptr),
        "n"(SizeInBytes),
        "r"(src_in_bytes));

#else
    using AccessType = cutlass::Array<uint8_t, SizeInBytes>;

    if (pred_guard) {
      *static_cast<AccessType*>(smem_ptr) =
          *static_cast<AccessType const*>(global_ptr);
    } else {
      AccessType zeros;
      zeros.clear();
      *static_cast<AccessType*>(smem_ptr) = zeros;
    }
#endif
  }
};

} // namespace arch
} // namespace gemm
} // namespace cuda
} // namespace natten

/////////////////////////////////////////////////////////////////////////////////////////////////
