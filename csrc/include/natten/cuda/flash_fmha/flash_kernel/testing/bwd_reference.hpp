/***************************************************************************************************
 * Copyright (c) 2025 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/


#pragma once

#include <cuda.h>
#include <iostream>
#include "cute/tensor.hpp"
#include "cuda_check.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    int KVTileSize,
    int DimPerThread,
    class ProblemShape,
    class TensorQ, class TensorK, class TensorV,
    class TensorO, class TensorLSE, class TensorDO,
    class TensorDQ, /* class TensorDK, class TensorDV, */
    class Fusion
>
void __global__ fmha_bwd_reference_dQ_kernel(
    ProblemShape problem_shape,
    TensorQ mQ, TensorK mK, TensorV mV,
    TensorO mO, TensorLSE mLSE, TensorDO mDO,
    TensorDQ mDQ, /* TensorDK mDK, TensorDV mDV, */
    Fusion fusion
) {
  using namespace cute;

  using Element = typename TensorO::value_type;
  using ElementAccumulator = typename TensorLSE::value_type;
  
  extern __shared__ char mS_mem[];
  Element* mS = reinterpret_cast<Element*>(mS_mem);

  Element softmax_scale = static_cast<Element>(1.0 / sqrt(1.0 * size<1>(mO)));

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
          }
          for (int idx_D1 = 0; idx_D1 < size<1>(mV); idx_D1++) {
            acc_dov += mDO(idx_Q, idx_D1, idx_L) * mV(idx_K, idx_D1, idx_L);
            acc_doo += mDO(idx_Q, idx_D1, idx_L) * mO(idx_Q, idx_D1, idx_L);
          } // for idx_D1

          auto id = make_identity_tensor(make_shape(1, 1));
          auto frag = make_tensor<ElementAccumulator>(Shape<_1, _1>{});
          frag(0) = acc_qk;
          fusion.before_softmax(frag, make_tensor(id.data() + make_arithmetic_tuple(idx_Q, idx_K), id.layout()), problem_shape);
          acc_qk = frag(0);

          mS[idx_K - kv_tile_offset] = static_cast<Element>(exp(softmax_scale * acc_qk - mLSE(idx_Q, idx_L)) * softmax_scale * (acc_dov - acc_doo));
        }

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

template<
    int QTileSize, int DimPerThread,
    class ProblemShape,
    class TensorQ, class TensorK, class TensorV,
    class TensorO, class TensorLSE, class TensorDO,
    /* class TensorDQ, */ class TensorDK, /* class TensorDV, */
    class Fusion
>
void __global__ fmha_bwd_reference_dK_kernel(
    ProblemShape problem_shape,
    TensorQ mQ, TensorK mK, TensorV mV,
    TensorO mO, TensorLSE mLSE, TensorDO mDO,
    /* TensorDQ mDQ, */ TensorDK mDK, /* TensorDV mDV, */
    Fusion fusion
) {
  using namespace cute;

  using Element = typename TensorO::value_type;
  using ElementAccumulator = typename TensorLSE::value_type;
  
  extern __shared__ char mS_mem[];
  Element* mS = reinterpret_cast<Element*>(mS_mem);

  Element softmax_scale = static_cast<Element>(1.0 / sqrt(1.0 * size<1>(mO)));
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
          }
          for (int idx_D1 = 0; idx_D1 < size<1>(mV); idx_D1++) {
            acc_dov += mDO(idx_Q, idx_D1, idx_L) * mV(idx_K, idx_D1, idx_L);
            acc_doo += mDO(idx_Q, idx_D1, idx_L) * mO(idx_Q, idx_D1, idx_L);
          } // for idx_D1
          acc_qk *= softmax_scale;
          acc_dov *= softmax_scale;
          acc_doo *= softmax_scale;
          
          auto id = make_identity_tensor(make_shape(1, 1));
          auto frag = make_tensor<ElementAccumulator>(Shape<_1, _1>{});
          frag(0) = acc_qk;
          fusion.before_softmax(frag, make_tensor(id.data() + make_arithmetic_tuple(idx_Q, idx_K), id.layout()), problem_shape);
          acc_qk = frag(0);

          mS[idx_Q - q_tile_offset] = static_cast<Element>(
              exp(acc_qk - mLSE(idx_Q, idx_L)) * (acc_dov - acc_doo));
        }

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

template<
    int QTileSize, int DimPerThread,
    class ProblemShape,
    class TensorQ, class TensorK, class TensorV,
    class TensorO, class TensorLSE, class TensorDO,
    /* class TensorDQ, class TensorDK, */ class TensorDV,
    class Fusion
>
void __global__ fmha_bwd_reference_dV_kernel(
    ProblemShape problem_shape,
    TensorQ mQ, TensorK mK, TensorV mV,
    TensorO mO, TensorLSE mLSE, TensorDO mDO,
    /* TensorDQ mDQ, TensorDK mDK, */ TensorDV mDV,
    Fusion fusion
) {
  using namespace cute;

  using Element = typename TensorO::value_type;
  using ElementAccumulator = typename TensorLSE::value_type;
  
  extern __shared__ char mS_mem[];
  Element* mS = reinterpret_cast<Element*>(mS_mem);

  Element softmax_scale = static_cast<Element>(1.0 / sqrt(1.0 * size<1>(mO)));

  cutlass::Array<ElementAccumulator, DimPerThread> final_acc;

  // mDV.shape = (K, D, (B, H))
  // mDO.shape = (Q, D, (B, H))
  //  mQ.shape = (Q, D, (B, H))

  for (int idx_L = blockIdx.y; idx_L < size<2>(mDV); idx_L += gridDim.y) {
    for (int idx_K = blockIdx.x; idx_K < size<0>(mDV); idx_K += gridDim.x) {

      final_acc.clear();
      auto num_q_tiles = ceil_div(size<0>(mDO), QTileSize); 

      for (int q_tile_idx = 0; q_tile_idx < num_q_tiles; q_tile_idx++){

        auto q_tile_offset = q_tile_idx * QTileSize; 
        auto q_tile_max = q_tile_offset + QTileSize;
        auto Q_max = std::min(q_tile_max, size<0>(mDO));
        auto mS_max = Q_max - q_tile_offset;

        for (int idx_Q = q_tile_offset + threadIdx.x; idx_Q < Q_max; idx_Q += blockDim.x) {
          ElementAccumulator acc_qk = 0;
          for (int idx_D0 = 0; idx_D0 < size<1>(mK); idx_D0++) {
            acc_qk += mQ(idx_Q, idx_D0, idx_L) * mK(idx_K, idx_D0, idx_L);
          }

          auto id = make_identity_tensor(make_shape(1, 1));
          auto frag = make_tensor<ElementAccumulator>(Shape<_1, _1>{});
          frag(0) = acc_qk;
          fusion.before_softmax(frag, make_tensor(id.data() + make_arithmetic_tuple(idx_Q, idx_K), id.layout()), problem_shape);
          acc_qk = frag(0);

          mS[idx_Q - q_tile_offset] = static_cast<Element>(exp(softmax_scale * acc_qk - mLSE(idx_Q, idx_L)));
        }

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

    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    class ProblemShape,
    class TensorQ, class TensorK, class TensorV,
    class TensorO, class TensorLSE, class TensorDO,
    /**/ class TensorDQ, /** / class TensorDK, / ** / class TensorDV, / **/
    class Fusion
>
void fmha_bwd_reference_dQ(
    ProblemShape problem_shape,
    TensorQ mQ, TensorK mK, TensorV mV,
    TensorO mO, TensorLSE mLSE, TensorDO mDO,
    /**/ TensorDQ mDQ, /** / TensorDK mDK, / ** / TensorDV mDV, / **/
    Fusion fusion
) {
  using namespace cute;

  static constexpr int KVTileSize = 2048;
  static constexpr int MaxDimSupported = 1024;
  static constexpr int NumThreads = 256;
  static_assert(KVTileSize % NumThreads == 0);
  static_assert(MaxDimSupported % NumThreads == 0);
  static constexpr int DimPerThread = MaxDimSupported / NumThreads;

  dim3 grid(size<0>(mDQ), size<2>(mDQ), 1);
  dim3 block(256);
  int shared_mem = KVTileSize * sizeof(typename TensorO::value_type);

  if (shared_mem >= (48 << 10)) {
    CUTLASS_TRACE_HOST("  Setting smem size to " << shared_mem);
    auto result = cudaFuncSetAttribute(
        fmha_bwd_reference_dQ_kernel<KVTileSize, DimPerThread, ProblemShape, TensorQ, TensorK, TensorV, TensorO, TensorLSE, TensorDO, TensorDQ, Fusion>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem);
    if (cudaSuccess != result) {
      result = cudaGetLastError(); // to clear the error bit
      CUTLASS_TRACE_HOST(
        "  cudaFuncSetAttribute() returned error: "
        << cudaGetErrorString(result));
      return;
    }
  }

  fmha_bwd_reference_dQ_kernel<KVTileSize, DimPerThread><<<grid, block, shared_mem>>>(problem_shape, mQ, mK, mV, mO, mLSE, mDO, mDQ, fusion);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    class ProblemShape,
    class TensorQ, class TensorK, class TensorV,
    class TensorO, class TensorLSE, class TensorDO,
    /** / class TensorDQ, / **/ class TensorDK, /** / class TensorDV, / **/
    class Fusion
>
void fmha_bwd_reference_dK(
    ProblemShape problem_shape,
    TensorQ mQ, TensorK mK, TensorV mV,
    TensorO mO, TensorLSE mLSE, TensorDO mDO,
    /** / TensorDQ mDQ, / **/ TensorDK mDK, /** / TensorDV mDV, / **/
    Fusion fusion
) {
  using namespace cute;

  // Only so that we don't oversubscribe shmem when seqlen is large.
  static constexpr int QTileSize = 2048;
  static constexpr int MaxDimSupported = 1024;
  static constexpr int NumThreads = 256;
  static_assert(QTileSize % NumThreads == 0);
  static_assert(MaxDimSupported % NumThreads == 0);
  static constexpr int DimPerThread = MaxDimSupported / NumThreads;

  dim3 grid(size<0>(mDK), size<2>(mDK), 1);
  dim3 block(NumThreads);
  int shared_mem = QTileSize * sizeof(typename TensorO::value_type);

  if (shared_mem >= (48 << 10)) {
    CUTLASS_TRACE_HOST("  Setting smem size to " << shared_mem);
    auto result = cudaFuncSetAttribute(
        fmha_bwd_reference_dK_kernel<QTileSize, DimPerThread, ProblemShape, TensorQ, TensorK, TensorV, TensorO, TensorLSE, TensorDO, TensorDK, Fusion>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem);
    if (cudaSuccess != result) {
      result = cudaGetLastError(); // to clear the error bit
      CUTLASS_TRACE_HOST(
        "  cudaFuncSetAttribute() returned error: "
        << cudaGetErrorString(result));
      return;
    }
  }

  fmha_bwd_reference_dK_kernel<QTileSize, DimPerThread><<<grid, block, shared_mem>>>(problem_shape, mQ, mK, mV, mO, mLSE, mDO, mDK, fusion);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    class ProblemShape,
    class TensorQ, class TensorK, class TensorV,
    class TensorO, class TensorLSE, class TensorDO,
    /** / class TensorDQ, / ** / class TensorDK, / **/ class TensorDV, /**/
    class Fusion
>
void fmha_bwd_reference_dV(
    ProblemShape problem_shape,
    TensorQ mQ, TensorK mK, TensorV mV,
    TensorO mO, TensorLSE mLSE, TensorDO mDO,
    /** / TensorDQ mDQ, / ** / TensorDK mDK, / **/ TensorDV mDV, /**/
    Fusion fusion
) {
  using namespace cute;

  static constexpr int QTileSize = 2048;
  static constexpr int MaxDimSupported = 1024;
  static constexpr int NumThreads = 256;
  static_assert(QTileSize % NumThreads == 0);
  static_assert(MaxDimSupported % NumThreads == 0);
  static constexpr int DimPerThread = MaxDimSupported / NumThreads;

  dim3 grid(size<0>(mDV), size<2>(mDV), 1);
  dim3 block(NumThreads);
  int shared_mem = QTileSize * sizeof(typename TensorO::value_type);

  if (shared_mem >= (48 << 10)) {
    CUTLASS_TRACE_HOST("  Setting smem size to " << shared_mem);
    auto result = cudaFuncSetAttribute(
        fmha_bwd_reference_dV_kernel<QTileSize, DimPerThread, ProblemShape, TensorQ, TensorK, TensorV, TensorO, TensorLSE, TensorDO, TensorDV, Fusion>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem);
    if (cudaSuccess != result) {
      result = cudaGetLastError(); // to clear the error bit
      CUTLASS_TRACE_HOST(
        "  cudaFuncSetAttribute() returned error: "
        << cudaGetErrorString(result));
      return;
    }
  }

  fmha_bwd_reference_dV_kernel<QTileSize, DimPerThread><<<grid, block, shared_mem>>>(problem_shape, mQ, mK, mV, mO, mLSE, mDO, mDV, fusion);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
    class ProblemShape,
    class TensorQ, class TensorK, class TensorV,
    class TensorO, class TensorLSE, class TensorDO,
    class TensorDQ, class TensorDK, class TensorDV,
    class Fusion
>
void fmha_bwd_reference(
    ProblemShape problem_shape,
    TensorQ mQ, TensorK mK, TensorV mV,
    TensorO mO, TensorLSE mLSE, TensorDO mDO,
    TensorDQ mDQ, TensorDK mDK, TensorDV mDV,
    Fusion fusion
) {
  fmha_bwd_reference_dQ(problem_shape, mQ, mK, mV, mO, mLSE, mDO, mDQ, fusion);
  CHECK_CUDA(cudaGetLastError());
  std::cout << " Finished computing dQ reference!" << std::endl;
  cudaError_t result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "Reference kernel failed. Last CUDA error: "
              << cudaGetErrorString(result) << std::endl;
  }
  fmha_bwd_reference_dK(problem_shape, mQ, mK, mV, mO, mLSE, mDO, mDK, fusion);
  CHECK_CUDA(cudaGetLastError());
  result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "Reference kernel failed. Last CUDA error: "
              << cudaGetErrorString(result) << std::endl;
  }
  std::cout << " Finished computing dK reference!" << std::endl;
  fmha_bwd_reference_dV(problem_shape, mQ, mK, mV, mO, mLSE, mDO, mDV, fusion);
  CHECK_CUDA(cudaGetLastError());
  result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "Reference kernel failed. Last CUDA error: "
              << cudaGetErrorString(result) << std::endl;
  }
  std::cout << " Finished computing dV reference!" << std::endl;
  std::cout << "======= Finished computing all bwd references!" << std::endl;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
