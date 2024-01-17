/***************************************************************************************************
 * Copyright (c) 2022-2024 Ali Hassani.
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

#include "natten/cuda/gemm/kernel/implicit_gemm_na1d.cuh"
#include "natten/cuda/gemm/kernel/implicit_gemm_na2d.cuh"
#include "natten/cuda/gemm/neighborhood_attention.cuh"
#include "natten/cuda/gemm/threadblock/implicit_gemm_multistage.cuh"
#include "natten/cuda/gemm/threadblock/na_tile_iterator.cuh"
#include "natten/cuda/gemm/threadblock/threadblock_swizzle.cuh"

// Pipelined supports 16-bit alignment, so we don't have to modify it
#include "cutlass/conv/threadblock/implicit_gemm_pipelined.h"

namespace natten {
namespace cuda {
namespace gemm {
namespace kernel {

namespace {

template <typename ArchTag, typename EpilogueBase>
struct GetFragmentsPerIter {
  static_assert(
      ArchTag::kMinComputeCapability == 80 ||
      ArchTag::kMinComputeCapability == 75);
  static constexpr int value = EpilogueBase::kFragmentsPerIteration;
};

template <typename EpilogueBase>
struct GetFragmentsPerIter<cutlass::arch::Sm70, EpilogueBase> {
  static constexpr int value = 1;
};

} // namespace

} // namespace kernel
} // namespace gemm
} // namespace cuda
} // namespace natten
