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
void fmha_backward_generic(
    at::ScalarType dtype,
    const int cc,
    const size_t max_smem,
    cudaStream_t stream,
    MemoryAllocator alloc_bytes,
    void* grad_out_ptr,
    void* query_ptr,
    void* key_ptr,
    void* value_ptr,
    void* logsumexp_ptr,
    void* delta_ptr,
    void* out_ptr,
    //  Outputs:
    void* grad_query_ptr,
    void* grad_key_ptr,
    void* grad_value_ptr,
    // Params
    int32_t batch_size,
    int32_t seqlen_q,
    int32_t seqlen_kv,
    int32_t heads,
    int32_t dim,
    int32_t dim_value,
    float attn_scale,
    int query_tile_size,
    int key_tile_size,
    int num_splits_key) {
  bool kernel_launched = false;
  const auto maxK = std::max(dim, dim_value);

  auto launchKernel = [&](auto _k, auto kernel_fn) {
    using Kernel = decltype(_k);
    using scalar_t = typename Kernel::scalar_t;
    (void)_k;

    if (kernel_launched) {
      return;
    }
    // Check if this kernel is compatible
    if (Kernel::kMaxK < maxK) {
      return;
    }

    if (query_tile_size != Kernel::kBlockSizeI ||
        key_tile_size != Kernel::kBlockSizeJ) {
      return;
    }

    if ((dim % Kernel::kMinimumAlignment) ||
        (dim % Kernel::kMinimumAlignment) ||
        (dim_value % Kernel::kMinimumAlignment)) {
      return;
    }
    // Uses too much shmem
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    if (smem_bytes > max_smem) {
      return;
    }

    kernel_launched = true;

    typename Kernel::Params p;
    p.query_ptr = (scalar_t*)query_ptr;
    p.key_ptr = (scalar_t*)key_ptr;
    p.value_ptr = (scalar_t*)value_ptr;
    p.logsumexp_ptr = (typename Kernel::lse_scalar_t*)logsumexp_ptr;
    p.output_ptr = (scalar_t*)out_ptr;
    p.grad_output_ptr = (scalar_t*)grad_out_ptr;
    p.grad_query_ptr = (scalar_t*)grad_query_ptr;
    p.grad_key_ptr = (scalar_t*)grad_key_ptr;
    p.grad_value_ptr = (scalar_t*)grad_value_ptr;
    p.delta_ptr = (float*)delta_ptr;

    p.scale = attn_scale;

    p.num_heads = heads;
    p.head_dim = dim;
    p.head_dim_value = dim_value;
    p.num_queries = seqlen_q;
    p.num_keys = seqlen_kv;
    p.num_batches = batch_size;

    p.num_splits_key = num_splits_key;

    p.o_strideH = dim_value;
    p.q_strideH = dim;
    p.k_strideH = dim;
    p.v_strideH = dim_value;
    p.o_strideB = seqlen_q * heads * dim_value;
    p.q_strideB = seqlen_q * heads * dim;
    p.k_strideB = seqlen_kv * heads * dim;
    p.v_strideB = seqlen_kv * heads * dim_value;
    p.lse_strideB = seqlen_q * heads;
    p.delta_strideB = seqlen_q * heads;

    p.gO_strideB = seqlen_q * heads * dim_value;
    p.gQ_strideB = seqlen_q * heads * dim;
    p.gK_strideB = seqlen_kv * heads * dim;
    p.gV_strideB = seqlen_kv * heads * dim_value;
    p.gO_strideH = dim_value;
    p.gQ_strideH = dim;
    p.gK_strideH = dim;
    p.gV_strideH = dim_value;
    p.q_strideM = heads * dim;
    p.k_strideM = heads * dim;
    p.v_strideM = heads * dim_value;
    p.gO_strideM = heads * dim_value;

    int64_t size_bytes = p.workspace_size();
    if (size_bytes) {
      void* workspace_ptr = nullptr;
      alloc_bytes(
          &workspace_ptr, size_bytes, true /*p.should_zero_workspace()*/);
      p.workspace = (float*)workspace_ptr;
    }

    Kernel::check_supported(p);

    if (smem_bytes > 0xc000) {
      // https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications-technical-specifications-per-compute-capability
      auto err = cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      NATTEN_CHECK(
          err != cudaErrorInvalidValue,
          "This GPU does not have enough shared-memory.");
      NATTEN_CUDA_CHECK(err);
    }

    // second syntax resulted in the error below on windows
    // error C3495: 'kernel_fn': a simple capture must be a variable
    // with automatic storage duration declared
    // in the reaching scope of the lambda
#ifdef NATTEN_WINDOWS
    cudaFuncAttributes attr;
    NATTEN_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_fn));
    NATTEN_CHECK(
        attr.binaryVersion >= Kernel::ArchTag::kMinComputeCapability,
        "Something went wrong in the build process");
#else
    auto checkBinaryArchMatches = [&]() {
      cudaFuncAttributes attr;
      NATTEN_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel_fn));
      return attr.binaryVersion >= Kernel::ArchTag::kMinComputeCapability;
    };
    NATTEN_CHECK(
        checkBinaryArchMatches(), "Something went wrong in the build process");
#endif

    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);
  };

  DISPATCH_FMHA_BACKWARD(cc, dtype, launchKernel);
  NATTEN_CHECK(
      kernel_launched, "Could not find a compatible FMHA backward kernel.");
  NATTEN_CUDA_CHECK(cudaGetLastError());
}

} // namespace fmha
} // namespace cuda
} // namespace natten
