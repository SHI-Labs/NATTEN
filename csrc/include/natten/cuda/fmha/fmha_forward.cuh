/***************************************************************************************************
 * Copyright (c) 2022-2025 Ali Hassani.
 *
 * Fused Neighborhood Attention kernels are heavily based on the
 * memory-efficient attention kernels from the xFormers project by Meta
 * Platforms, Inc.
 *
 * Copyright (c) Facebook, Inc. and its affiliates
 *
 * BSD 3-Clause License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC
 * Laboratories America and IDIAP Research Institute nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <natten/cuda/utils/cuda.h>
#include <natten/natten.h>
#include <natten/cuda/utils/cutlass.cuh>

#include <natten_autogen/cuda/fmha/interface.h>

namespace natten {
namespace cuda {
namespace fmha {

template <typename MemoryAllocator // In lieu of a caching allocator, and while
                                   // we only bind with torch
          >
void fmha_forward_generic(
    at::ScalarType dtype,
    const int cc,
    const size_t max_smem,
    cudaStream_t stream,
    MemoryAllocator alloc_bytes,
    void* query_ptr,
    void* key_ptr,
    void* value_ptr,
    void* out_ptr,
    int32_t batch_size,
    int32_t seqlen_q,
    int32_t seqlen_kv,
    int32_t heads,
    int32_t dim,
    int32_t dim_value,
    float attn_scale,
    void* logsumexp_ptr,
    int query_tile_size,
    int key_tile_size) {
  bool kernel_launched = false;
  auto launchKernel = [&](auto _k, auto kernel_fn) {
    using Kernel = decltype(_k);
    using scalar_t = typename Kernel::scalar_t;
    (void)_k;

    if (kernel_launched) {
      return;
    }

    if (dim_value > Kernel::kMaxK || dim > Kernel::kMaxK) {
      return;
    }

    if (query_tile_size != Kernel::kQueriesPerBlock ||
        key_tile_size != Kernel::kKeysPerBlock) {
      return;
    }

    // Alignment
    if ((dim % Kernel::kAlignmentQ) || (dim % Kernel::kAlignmentK) ||
        (dim_value % Kernel::kAlignmentV)) {
      return;
    }
    // Uses too much smem
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    if (smem_bytes > max_smem) {
      return;
    }

    kernel_launched = true;

    typename Kernel::Params p;
    p.query_ptr = (scalar_t*)query_ptr;
    p.key_ptr = (scalar_t*)key_ptr;
    p.value_ptr = (scalar_t*)value_ptr;
    p.logsumexp_ptr = logsumexp_ptr != nullptr
        ? (typename Kernel::lse_scalar_t*)logsumexp_ptr
        : nullptr;

    // void* accum_ptr = nullptr;
    if (Kernel::kNeedsOutputAccumulatorBuffer) {
      using AccumType = typename Kernel::output_accum_t;
      int64_t workspace_size_bytes =
          batch_size * seqlen_q * heads * dim_value * sizeof(AccumType);
      void* accum_ptr = nullptr;
      alloc_bytes(&accum_ptr, workspace_size_bytes, /* zero_fill = */ false);
      p.output_accum_ptr = (AccumType*)accum_ptr;
    } else {
      p.output_accum_ptr = nullptr;
    }
    p.output_ptr = (typename Kernel::output_t*)out_ptr;

    p.num_heads = heads;
    p.head_dim = dim;
    p.head_dim_value = dim_value;
    p.num_queries = seqlen_q;
    p.num_keys = seqlen_kv;
    p.num_batches = batch_size;

    p.scale = attn_scale;

    p.q_strideM = heads * dim;
    p.k_strideM = heads * dim;
    p.v_strideM = heads * dim_value;
    p.o_strideM = heads * dim_value;

    p.q_strideH = dim;
    p.k_strideH = dim;
    p.v_strideH = dim_value;

    p.q_strideB = seqlen_q * heads * dim;
    p.k_strideB = seqlen_kv * heads * dim;
    p.v_strideB = seqlen_kv * heads * dim_value;

    if (smem_bytes > 0xc000) {
      auto err = cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      NATTEN_CHECK(
          err != cudaErrorInvalidValue,
          "This GPU does not have enough shared-memory.");
      NATTEN_CUDA_CHECK(err);
    }
    auto blocks = p.getBlocksGrid();
    Kernel::check_supported(p);
    kernel_fn<<<blocks, p.getThreadsGrid(), smem_bytes, stream>>>(p);
  };

  DISPATCH_FMHA_FORWARD(cc, dtype, launchKernel);
  NATTEN_CHECK(
      kernel_launched, "Could not find a compatible FMHA forward kernel.");
  NATTEN_CUDA_CHECK(cudaGetLastError());
}

} // namespace fmha
} // namespace cuda
} // namespace natten
