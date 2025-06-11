/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "cute/tensor.hpp"

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
    class TensorDO,
    class TensorDQ, /* class TensorDK, class TensorDV, */
    class NADim,
    class Causal,
    class QKVLayout>
void __global__ fna_bwd_reference_dQ_kernel(
    ProblemShape problem_shape,
    TensorQ mQ,
    TensorK mK,
    TensorV mV,
    TensorO mO,
    TensorLSE mLSE,
    TensorDO mDO,
    TensorDQ mDQ, /* TensorDK mDK, TensorDV mDV, */
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
  Element* mS = reinterpret_cast<Element*>(mS_mem);

  cutlass::Array<ElementAccumulator, DimPerThread> final_acc;

  for (int idx_L = blockIdx.y; idx_L < size<2>(mDQ); idx_L += gridDim.y) {
    for (int idx_Q = blockIdx.x; idx_Q < size<0>(mDQ); idx_Q += gridDim.x) {
      final_acc.clear();
      auto num_kv_tiles = ceil_div(size<0>(mK), KVTileSize);
      for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        auto kv_tile_offset = kv_tile * KVTileSize;
        auto kv_tile_max = kv_tile_offset + KVTileSize;
        auto K_max = std::min(kv_tile_max, size<0>(mK));
        auto mS_max = K_max - kv_tile_offset;

        for (int idx_K = kv_tile_offset + threadIdx.x; idx_K < K_max;
             idx_K += blockDim.x) {
          ElementAccumulator acc_qk = 0;
          ElementAccumulator acc_dov = 0;
          ElementAccumulator acc_doo = 0;
          for (int idx_D0 = 0; idx_D0 < size<1>(mK); idx_D0++) {
            acc_qk += mQ(idx_Q, idx_D0, idx_L) * mK(idx_K, idx_D0, idx_L);
          } // for idx_D0
          for (int idx_D1 = 0; idx_D1 < size<1>(mV); idx_D1++) {
            acc_dov += mDO(idx_Q, idx_D1, idx_L) * mV(idx_K, idx_D1, idx_L);
            acc_doo += mDO(idx_Q, idx_D1, idx_L) * mO(idx_Q, idx_D1, idx_L);
          } // for idx_D1
          acc_qk *= attn_scale;
          acc_dov *= attn_scale;
          acc_doo *= attn_scale;

          auto id = make_identity_tensor(make_shape(1, 1));
          auto frag = make_tensor<ElementAccumulator>(Shape<_1, _1>{});
          frag(0) = acc_qk;
          attention_mask.apply_mask(
              frag,
              make_tensor(
                  id.data() + make_arithmetic_tuple(idx_Q, idx_K),
                  id.layout()));
          acc_qk = frag(0);

          mS[idx_K - kv_tile_offset] = static_cast<Element>(
              exp(acc_qk - mLSE(idx_Q, idx_L)) * (acc_dov - acc_doo));
        } // for idx_K

        __syncthreads();

        for (int i = 0; i < DimPerThread; ++i) {
          int idx_D = threadIdx.x + i * blockDim.x;
          if (idx_D < size<1>(mDQ)) {
            for (int j = 0; j < mS_max; j++) {
              int idx_K = j + kv_tile_offset;
              final_acc[i] += mS[j] * mK(idx_K, idx_D, idx_L);
            }
          }
        } // for idx_D

        __syncthreads();
      }

      for (int i = 0; i < DimPerThread; ++i) {
        int idx_D = threadIdx.x + i * blockDim.x;
        if (idx_D < size<1>(mDQ)) {
          mDQ(idx_Q, idx_D, idx_L) =
              static_cast<typename TensorDQ::value_type>(final_acc[i]);
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int QTileSize,
    int DimPerThread,
    class ProblemShape,
    class TensorQ,
    class TensorK,
    class TensorV,
    class TensorO,
    class TensorLSE,
    class TensorDO,
    /* class TensorDQ, */ class TensorDK, /* class TensorDV, */
    class NADim,
    class Causal,
    class QKVLayout>
void __global__ fna_bwd_reference_dK_kernel(
    ProblemShape problem_shape,
    TensorQ mQ,
    TensorK mK,
    TensorV mV,
    TensorO mO,
    TensorLSE mLSE,
    TensorDO mDO,
    /* TensorDQ mDQ, */ TensorDK mDK, /* TensorDV mDV, */
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
  Element* mS = reinterpret_cast<Element*>(mS_mem);

  cutlass::Array<ElementAccumulator, DimPerThread> final_acc;

  for (int idx_L = blockIdx.y; idx_L < size<2>(mDK); idx_L += gridDim.y) {
    for (int idx_K = blockIdx.x; idx_K < size<0>(mDK); idx_K += gridDim.x) {
      final_acc.clear();
      auto num_q_tiles = ceil_div(size<0>(mDO), QTileSize);
      for (int q_tile = 0; q_tile < num_q_tiles; ++q_tile) {
        auto q_tile_offset = q_tile * QTileSize;
        auto q_tile_max = q_tile_offset + QTileSize;
        auto Q_max = std::min(q_tile_max, size<0>(mDO));
        auto mS_max = Q_max - q_tile_offset;

        for (int idx_Q = q_tile_offset + threadIdx.x; idx_Q < Q_max;
             idx_Q += blockDim.x) {
          ElementAccumulator acc_qk = 0;
          ElementAccumulator acc_dov = 0;
          ElementAccumulator acc_doo = 0;
          for (int idx_D0 = 0; idx_D0 < size<1>(mK); idx_D0++) {
            acc_qk += mQ(idx_Q, idx_D0, idx_L) * mK(idx_K, idx_D0, idx_L);
          } // for idx_D0
          for (int idx_D1 = 0; idx_D1 < size<1>(mV); idx_D1++) {
            acc_dov += mDO(idx_Q, idx_D1, idx_L) * mV(idx_K, idx_D1, idx_L);
            acc_doo += mDO(idx_Q, idx_D1, idx_L) * mO(idx_Q, idx_D1, idx_L);
          } // for idx_D1
          acc_qk *= attn_scale;
          acc_dov *= attn_scale;
          acc_doo *= attn_scale;

          auto id = make_identity_tensor(make_shape(1, 1));
          auto frag = make_tensor<ElementAccumulator>(Shape<_1, _1>{});
          frag(0) = acc_qk;
          attention_mask.apply_mask(
              frag,
              make_tensor(
                  id.data() + make_arithmetic_tuple(idx_Q, idx_K),
                  id.layout()));
          acc_qk = frag(0);

          mS[idx_Q - q_tile_offset] = static_cast<Element>(
              exp(acc_qk - mLSE(idx_Q, idx_L)) * (acc_dov - acc_doo));
        } // for idx_Q

        __syncthreads();

        for (int i = 0; i < DimPerThread; ++i) {
          int idx_D = threadIdx.x + i * blockDim.x;
          if (idx_D < size<1>(mDK)) {
            for (int j = 0; j < mS_max; j++) {
              int idx_Q = j + q_tile_offset;
              final_acc[i] += mS[j] * mQ(idx_Q, idx_D, idx_L);
            }
          }
        }

        __syncthreads();
      }

      for (int i = 0; i < DimPerThread; ++i) {
        int idx_D = threadIdx.x + i * blockDim.x;
        if (idx_D < size<1>(mDK)) {
          mDK(idx_K, idx_D, idx_L) =
              static_cast<typename TensorDK::value_type>(final_acc[i]);
        }
      }

    } // for idx_K
  } // for idx_L
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int QTileSize,
    int DimPerThread,
    class ProblemShape,
    class TensorQ,
    class TensorK,
    class TensorV,
    class TensorO,
    class TensorLSE,
    class TensorDO,
    /* class TensorDQ, class TensorDK, */ class TensorDV,
    class NADim,
    class Causal,
    class QKVLayout>
void __global__ fna_bwd_reference_dV_kernel(
    ProblemShape problem_shape,
    TensorQ mQ,
    TensorK mK,
    TensorV mV,
    TensorO mO,
    TensorLSE mLSE,
    TensorDO mDO,
    /* TensorDQ mDQ, TensorDK mDK, */ TensorDV mDV,
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
  Element* mS = reinterpret_cast<Element*>(mS_mem);

  cutlass::Array<ElementAccumulator, DimPerThread> final_acc;

  for (int idx_L = blockIdx.y; idx_L < size<2>(mDV); idx_L += gridDim.y) {
    for (int idx_K = blockIdx.x; idx_K < size<0>(mDV); idx_K += gridDim.x) {
      final_acc.clear();
      auto num_q_tiles = ceil_div(size<0>(mDO), QTileSize);
      for (int q_tile = 0; q_tile < num_q_tiles; ++q_tile) {
        auto q_tile_offset = q_tile * QTileSize;
        auto q_tile_max = q_tile_offset + QTileSize;
        auto Q_max = std::min(q_tile_max, size<0>(mDO));
        auto mS_max = Q_max - q_tile_offset;

        for (int idx_Q = q_tile_offset + threadIdx.x; idx_Q < Q_max;
             idx_Q += blockDim.x) {
          ElementAccumulator acc_qk = 0;

          for (int idx_D0 = 0; idx_D0 < size<1>(mK); idx_D0++) {
            ElementAccumulator rQ = mQ(idx_Q, idx_D0, idx_L);
            ElementAccumulator rK = mK(idx_K, idx_D0, idx_L);
            acc_qk += rQ * rK;
          } // for idx_D0
          acc_qk *= attn_scale;

          auto id = make_identity_tensor(make_shape(1, 1));
          auto frag = make_tensor<ElementAccumulator>(Shape<_1, _1>{});
          frag(0) = acc_qk;
          attention_mask.apply_mask(
              frag,
              make_tensor(
                  id.data() + make_arithmetic_tuple(idx_Q, idx_K),
                  id.layout()));
          acc_qk = frag(0);

          mS[idx_Q - q_tile_offset] =
              static_cast<Element>(exp(acc_qk - mLSE(idx_Q, idx_L)));
        } // for idx_Q

        __syncthreads();

        for (int i = 0; i < DimPerThread; ++i) {
          int idx_D = threadIdx.x + i * blockDim.x;
          if (idx_D < size<1>(mDV)) {
            for (int j = 0; j < mS_max; j++) {
              int idx_Q = j + q_tile_offset;
              ElementAccumulator rS = mS[/*idx_Q*/ j];
              ElementAccumulator rDO = mDO(idx_Q, idx_D, idx_L);
              final_acc[i] += rS * rDO;
            }
          }
        }

        __syncthreads();
      }

      for (int i = 0; i < DimPerThread; ++i) {
        int idx_D = threadIdx.x + i * blockDim.x;
        if (idx_D < size<1>(mDV)) {
          mDV(idx_K, idx_D, idx_L) =
              static_cast<typename TensorDV::value_type>(final_acc[i]);
        }
      }

    } // for idx_K
  } // for idx_L
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class ProblemShape,
    class TensorQ,
    class TensorK,
    class TensorV,
    class TensorO,
    class TensorLSE,
    class TensorDO,
    /**/ class TensorDQ, /** / class TensorDK, / ** / class TensorDV, / **/
    class NADim,
    class Causal,
    class QKVLayout>
void fna_bwd_reference_dQ(
    ProblemShape problem_shape,
    TensorQ mQ,
    TensorK mK,
    TensorV mV,
    TensorO mO,
    TensorLSE mLSE,
    TensorDO mDO,
    /**/ TensorDQ mDQ, /** / TensorDK mDK, / ** / TensorDV mDV, / **/
    NADim window_size,
    NADim stride,
    NADim dilation,
    Causal is_causal,
    QKVLayout qkv_layout,
    float attn_scale,
    int num_additional_kv,
    cudaStream_t stream) {
  using namespace cute;

  // Only so that we don't oversubscribe shmem when seqlen is large.
  static constexpr int KVTileSize = 2048;
  static constexpr int MaxDimSupported = 1024;
  static constexpr int NumThreads = 256;
  static_assert(KVTileSize % NumThreads == 0);
  static_assert(MaxDimSupported % NumThreads == 0);
  static constexpr int DimPerThread = MaxDimSupported / NumThreads;

  NATTEN_CHECK(
      size<1>(mDQ) <= MaxDimSupported,
      "Reference kernel only supports up to head dim 1024.");

  dim3 grid(size<0>(mDQ), size<2>(mDQ), 1);
  dim3 block(NumThreads);
  int shared_mem = KVTileSize * sizeof(typename TensorO::value_type);

  NATTEN_CHECK(
      shared_mem <= 0xc000,
      "Reference kernel oversubscribed shared memory. Please open an issue.");

  fna_bwd_reference_dQ_kernel<KVTileSize, DimPerThread>
      <<<grid, block, shared_mem, stream>>>(
          problem_shape,
          mQ,
          mK,
          mV,
          mO,
          mLSE,
          mDO,
          mDQ,
          window_size,
          stride,
          dilation,
          Causal{},
          qkv_layout,
          attn_scale,
          num_additional_kv);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class ProblemShape,
    class TensorQ,
    class TensorK,
    class TensorV,
    class TensorO,
    class TensorLSE,
    class TensorDO,
    /** / class TensorDQ, / **/ class TensorDK, /** / class TensorDV, / **/
    class NADim,
    class Causal,
    class QKVLayout>
void fna_bwd_reference_dK(
    ProblemShape problem_shape,
    TensorQ mQ,
    TensorK mK,
    TensorV mV,
    TensorO mO,
    TensorLSE mLSE,
    TensorDO mDO,
    /** / TensorDQ mDQ, / **/ TensorDK mDK, /** / TensorDV mDV, / **/
    NADim window_size,
    NADim stride,
    NADim dilation,
    Causal is_causal,
    QKVLayout qkv_layout,
    float attn_scale,
    int num_additional_kv,
    cudaStream_t stream) {
  using namespace cute;

  // Only so that we don't oversubscribe shmem when seqlen is large.
  static constexpr int QTileSize = 2048;
  static constexpr int MaxDimSupported = 1024;
  static constexpr int NumThreads = 256;
  static_assert(QTileSize % NumThreads == 0);
  static_assert(MaxDimSupported % NumThreads == 0);
  static constexpr int DimPerThread = MaxDimSupported / NumThreads;

  NATTEN_CHECK(
      size<1>(mDK) <= MaxDimSupported,
      "Reference kernel only supports up to head dim 1024.");

  dim3 grid(size<0>(mDK), size<2>(mDK), 1);
  dim3 block(NumThreads);
  int shared_mem = QTileSize * sizeof(typename TensorO::value_type);

  NATTEN_CHECK(
      shared_mem <= 0xc000,
      "Reference kernel oversubscribed shared memory. Please open an issue.");

  fna_bwd_reference_dK_kernel<QTileSize, DimPerThread>
      <<<grid, block, shared_mem, stream>>>(
          problem_shape,
          mQ,
          mK,
          mV,
          mO,
          mLSE,
          mDO,
          mDK,
          window_size,
          stride,
          dilation,
          Causal{},
          qkv_layout,
          attn_scale,
          num_additional_kv);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class ProblemShape,
    class TensorQ,
    class TensorK,
    class TensorV,
    class TensorO,
    class TensorLSE,
    class TensorDO,
    /** / class TensorDQ, / ** / class TensorDK, / **/ class TensorDV, /**/
    class NADim,
    class Causal,
    class QKVLayout>
void fna_bwd_reference_dV(
    ProblemShape problem_shape,
    TensorQ mQ,
    TensorK mK,
    TensorV mV,
    TensorO mO,
    TensorLSE mLSE,
    TensorDO mDO,
    /** / TensorDQ mDQ, / ** / TensorDK mDK, / **/ TensorDV mDV, /**/
    NADim window_size,
    NADim stride,
    NADim dilation,
    Causal is_causal,
    QKVLayout qkv_layout,
    float attn_scale,
    int num_additional_kv,
    cudaStream_t stream) {
  using namespace cute;

  // Only so that we don't oversubscribe shmem when seqlen is large.
  static constexpr int QTileSize = 2048;
  static constexpr int MaxDimSupported = 1024;
  static constexpr int NumThreads = 256;
  static_assert(QTileSize % NumThreads == 0);
  static_assert(MaxDimSupported % NumThreads == 0);
  static constexpr int DimPerThread = MaxDimSupported / NumThreads;

  NATTEN_CHECK(
      size<1>(mDV) <= MaxDimSupported,
      "Reference kernel only supports up to head dim 1024.");

  dim3 grid(size<0>(mDV), size<2>(mDV), 1);
  dim3 block(NumThreads);
  int shared_mem = QTileSize * sizeof(typename TensorO::value_type);

  NATTEN_CHECK(
      shared_mem <= 0xc000,
      "Reference kernel oversubscribed shared memory. Please open an issue.");

  fna_bwd_reference_dV_kernel<QTileSize, DimPerThread>
      <<<grid, block, shared_mem, stream>>>(
          problem_shape,
          mQ,
          mK,
          mV,
          mO,
          mLSE,
          mDO,
          mDV,
          window_size,
          stride,
          dilation,
          Causal{},
          qkv_layout,
          attn_scale,
          num_additional_kv);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class Element, class ElementLSE, class NADim, class Causal>
void fna_reference_backward(
    Element* ptr_Q,
    Element* ptr_K,
    Element* ptr_V,
    Element* ptr_O,
    Element* ptr_DO,
    Element* ptr_DQ,
    Element* ptr_DK,
    Element* ptr_DV,
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
  auto mDQ = make_tensor(
      make_gmem_ptr(ptr_DQ), select<0, 2, 3>(problem_shape), stride_Q);

  auto mK = make_tensor(
      make_gmem_ptr(ptr_K), select<1, 2, 3>(problem_shape), stride_K);
  auto mDK = make_tensor(
      make_gmem_ptr(ptr_DK), select<1, 2, 3>(problem_shape), stride_K);

  auto mV = make_tensor(
      make_gmem_ptr(ptr_V), select<1, 4, 3>(problem_shape), stride_V);
  auto mDV = make_tensor(
      make_gmem_ptr(ptr_DV), select<1, 4, 3>(problem_shape), stride_V);

  auto mO = make_tensor(
      make_gmem_ptr(ptr_O), select<0, 4, 3>(problem_shape), stride_O);
  auto mDO = make_tensor(
      make_gmem_ptr(ptr_DO), select<0, 4, 3>(problem_shape), stride_O);

  auto mLSE = make_tensor(
      make_gmem_ptr(ptr_LSE), select<0, 3>(problem_shape), stride_LSE);

  // For compatibility with pytorch's default contiguous layout
  auto qkv_layout = make_layout(qkv_shape, make_qkv_stride(qkv_shape));

  fna_bwd_reference_dQ(
      problem_shape,
      mQ,
      mK,
      mV,
      mO,
      mLSE,
      mDO,
      mDQ,
      window_size,
      stride,
      dilation,
      Causal{},
      qkv_layout,
      attn_scale,
      num_additional_kv,
      stream);
  fna_bwd_reference_dK(
      problem_shape,
      mQ,
      mK,
      mV,
      mO,
      mLSE,
      mDO,
      mDK,
      window_size,
      stride,
      dilation,
      Causal{},
      qkv_layout,
      attn_scale,
      num_additional_kv,
      stream);
  fna_bwd_reference_dV(
      problem_shape,
      mQ,
      mK,
      mV,
      mO,
      mLSE,
      mDO,
      mDV,
      window_size,
      stride,
      dilation,
      Causal{},
      qkv_layout,
      attn_scale,
      num_additional_kv,
      stream);
}

} // namespace reference
} // namespace cuda
} // namespace natten

/////////////////////////////////////////////////////////////////////////////////////////////////
