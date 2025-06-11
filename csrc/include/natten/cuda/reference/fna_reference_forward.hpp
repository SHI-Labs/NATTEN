/***************************************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_types.h>

#include <natten/cuda/reference/mask.hpp>
#include <natten/cuda/reference/utils.hpp>
#include <natten/cuda/utils/cuda.h>
#include <natten/natten.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace natten {
namespace cuda {
namespace reference {

template <
    int KVTileSize,
    int DimPerThread,
    class ProblemShape,
    class TensorQ,
    class TensorK,
    class TensorV,
    class TensorO,
    class TensorLSE,
    class NADim,
    class Causal,
    class QKVLayout>
void __global__ fna_reference_kernel(
    ProblemShape problem_shape,
    TensorQ mQ,
    TensorK mK,
    TensorV mV,
    TensorO mO,
    TensorLSE mLSE,
    NADim window_size,
    NADim stride,
    NADim dilation,
    Causal is_causal,
    QKVLayout qkv_layout,
    float attn_scale,
    int num_additional_kv) {
  using namespace cute;

  auto attention_mask =
      mask::NeighborhoodAttentionReferenceMask<NADim, Causal, QKVLayout>(
          window_size, stride, dilation, qkv_layout, num_additional_kv);

  using Element = typename TensorO::value_type;
  using ElementAccumulator = typename TensorLSE::value_type;

  extern __shared__ char mS_mem[];
  ElementAccumulator* mS = reinterpret_cast<ElementAccumulator*>(mS_mem);

  cutlass::Array<ElementAccumulator, DimPerThread> final_acc;

  auto id = make_identity_tensor(make_shape(1, 1));
  for (int idx_L = blockIdx.y; idx_L < size<3>(problem_shape);
       idx_L += gridDim.y) {
    for (int idx_Q = blockIdx.x; idx_Q < size<0>(problem_shape);
         idx_Q += gridDim.x) {
      final_acc.clear();

      auto coord_L = idx2crd(idx_L, shape<3>(problem_shape));
      auto coord = cute::make_tuple(idx_Q, _0{}, _0{}, coord_L);

      if (get<0, 0>(coord) >= get<0>(problem_shape))
        continue;

      int offset_Q = 0;
      if constexpr (rank<0>(decltype(coord){}) == 2) {
        offset_Q = get<0, 1>(coord);
      }

      int offset_K = 0;
      if constexpr (rank<1>(decltype(coord){}) == 2) {
        offset_K = get<1, 1>(coord);
      }

      auto num_kv_tiles = ceil_div(size<1>(problem_shape), KVTileSize);

      ElementAccumulator sum = 0;
      ElementAccumulator maxS_p =
          -std::numeric_limits<ElementAccumulator>::infinity();
      ElementAccumulator maxS =
          -std::numeric_limits<ElementAccumulator>::infinity();

      for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        auto kv_tile_offset = kv_tile * KVTileSize;
        auto kv_tile_max = kv_tile_offset + KVTileSize;
        auto K_max = std::min(kv_tile_max, size<1>(problem_shape));
        auto mS_max = K_max - kv_tile_offset;

        for (int idx_K = kv_tile_offset + threadIdx.x; idx_K < K_max;
             idx_K += blockDim.x) {
          ElementAccumulator acc = 0;
          for (int idx_D = 0; idx_D < size<2>(problem_shape); idx_D++) {
            ElementAccumulator eQ = mQ(idx_Q + offset_Q, idx_D, idx_L);
            ElementAccumulator eK = mK(idx_K + offset_K, idx_D, idx_L);
            acc += eQ * eK;
          }
          acc = acc * attn_scale;
          auto frag = make_tensor<ElementAccumulator>(Shape<_1, _1>{});
          frag(0) = acc;
          attention_mask.apply_mask(
              frag,
              make_tensor(
                  id.data() + make_arithmetic_tuple(idx_Q, idx_K),
                  id.layout()));
          mS[idx_K - kv_tile_offset] = frag(0);
        }

        maxS_p = maxS;
        __syncthreads();

        for (int i = 0; i < mS_max; i++) {
          maxS = std::max<ElementAccumulator>(maxS, mS[i]);
        }

        __syncthreads();

        // If everything so far is masked, just skip to the next tile.
        if (maxS == -std::numeric_limits<ElementAccumulator>::infinity())
          continue;

        // Scale accumulator so far
        if (maxS_p != -std::numeric_limits<ElementAccumulator>::infinity()) {
          ElementAccumulator correction = expf(maxS_p - maxS);
          sum = correction * sum;
          for (int i = 0; i < DimPerThread; ++i) {
            final_acc[i] = static_cast<Element>(
                static_cast<ElementAccumulator>(final_acc[i]) * correction);
          }
        }

        for (int idx_K = kv_tile_offset + threadIdx.x; idx_K < K_max;
             idx_K += blockDim.x) {
          mS[idx_K - kv_tile_offset] =
              expf((mS[idx_K - kv_tile_offset] - maxS));
        }

        __syncthreads();

        for (int i = 0; i < mS_max; i++) {
          sum += mS[i];
        }

        for (int i = 0; i < DimPerThread; ++i) {
          int idx_D = threadIdx.x + i * blockDim.x;
          if (idx_D < size<1>(mO)) {
            for (int j = 0; j < mS_max; j++) {
              int idx_K = j + kv_tile_offset;
              ElementAccumulator eV = mV(idx_K + offset_K, idx_D, idx_L);
              ElementAccumulator eK = static_cast<Element>(mS[j]);
              final_acc[i] += eK * eV;
            }
          }
        }

        __syncthreads();
      }

      for (int i = 0; i < DimPerThread; ++i) {
        int idx_D = threadIdx.x + i * blockDim.x;
        if (idx_D < size<1>(mO)) {
          ElementAccumulator scale = 1.0f / sum;
          mO(idx_Q + offset_Q, idx_D, idx_L) =
              static_cast<typename TensorO::value_type>(final_acc[i] * scale);
        }
      }

      if (threadIdx.x == 0) {
        mLSE(idx_Q + offset_Q, idx_L) = log(sum) + maxS;
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class Element, class ElementLSE, class NADim, class Causal>
void fna_reference_forward(
    Element* ptr_Q,
    Element* ptr_K,
    Element* ptr_V,
    Element* ptr_O,
    ElementLSE* ptr_LSE,
    int batch,
    int seqlen,
    int heads,
    int dim,
    int dim_value,
    int num_additional_kv,
    NADim qkv_shape,
    NADim window_size,
    NADim stride,
    NADim dilation,
    Causal is_causal,
    float attn_scale,
    cudaStream_t stream) {
  using namespace cute;

  // Only so that we don't oversubscribe shmem when seqlen is large.
  static constexpr int KVTileSize = 2048;
  static constexpr int MaxDimSupported = 1024;
  static constexpr int NumThreads = 256;
  static_assert(KVTileSize % NumThreads == 0);
  static_assert(MaxDimSupported % NumThreads == 0);
  static constexpr int DimPerThread = MaxDimSupported / NumThreads;

  // No GQA/MQA for now
  auto problem_shape = cute::make_tuple(
      seqlen,
      seqlen + num_additional_kv,
      dim,
      cute::make_tuple(cute::make_tuple(1, heads), batch),
      dim_value);

  int SQ = size<0>(problem_shape);
  int SK = size<1>(problem_shape);
  int D = size<2>(problem_shape);
  int DV = size<4>(problem_shape);
  int H = size<3, 0>(problem_shape);
  int H_K = size<3, 0, 1>(problem_shape);
  int H_Q = size<3, 0, 0>(problem_shape);
  int B = size<3, 1>(problem_shape);

  // heads last profile, with torch's "contiguous layout"
  // shape: (batch, seqlen, heads, dim)
  // stride: (dim*heads*seqlen, dim*heads, dim, 1)
  auto stride_Q = make_stride(
      H * D, _1{}, make_stride(make_stride(D, H_Q * D), H * D * SQ));
  auto stride_O = make_stride(
      H * DV, _1{}, make_stride(make_stride(DV, H_Q * DV), H * DV * SQ));
  auto stride_K = make_stride(
      H_K * D, _1{}, make_stride(make_stride(_0{}, D), H_K * D * SK));
  auto stride_V = make_stride(
      H_K * DV, _1{}, make_stride(make_stride(_0{}, DV), H_K * DV * SK));
  auto stride_LSE = make_stride(H, make_stride(make_stride(_1{}, H_Q), SQ * H));

  auto mQ = make_tensor(
      make_gmem_ptr(ptr_Q), select<0, 2, 3>(problem_shape), stride_Q);

  auto mK = make_tensor(
      make_gmem_ptr(ptr_K), select<1, 2, 3>(problem_shape), stride_K);

  auto mV = make_tensor(
      make_gmem_ptr(ptr_V), select<1, 4, 3>(problem_shape), stride_V);

  auto mO = make_tensor(
      make_gmem_ptr(ptr_O), select<0, 4, 3>(problem_shape), stride_O);

  auto mLSE = make_tensor(
      make_gmem_ptr(ptr_LSE), select<0, 3>(problem_shape), stride_LSE);

  // For compatibility with pytorch's default contiguous layout
  auto qkv_layout = make_layout(qkv_shape, make_qkv_stride(qkv_shape));

  NATTEN_CHECK(
      size<1>(mO) <= MaxDimSupported,
      "Reference kernel only supports up to head dim 1024.");

  dim3 grid(size<0>(mO), size<2>(mO), 1);
  dim3 block(NumThreads);
  int shared_mem =
      KVTileSize * int(sizeof(typename decltype(mLSE)::value_type));

  NATTEN_CHECK(
      shared_mem <= 0xc000,
      "Reference kernel oversubscribed shared memory. Please open an issue.");

  fna_reference_kernel<KVTileSize, DimPerThread>
      <<<grid, block, shared_mem, stream>>>(
          problem_shape,
          mQ,
          mK,
          mV,
          mO,
          mLSE,
          window_size,
          stride,
          dilation,
          Causal{},
          qkv_layout,
          attn_scale,
          num_additional_kv);

  NATTEN_CUDA_CHECK(cudaGetLastError());
}

} // namespace reference
} // namespace cuda
} // namespace natten

/////////////////////////////////////////////////////////////////////////////////////////////////
