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
    \brief Holds common templates/functions shared between naive CUDA kernels.
*/

#pragma once

#ifndef __CUDA_ARCH__ 
#include <iostream>
#endif

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CUDA_NUM_THREADS 1024

// CUDA: number of blocks for threads.
inline int GET_BLOCKS(
    const int64_t N,
    const int64_t max_threads_per_block = CUDA_NUM_THREADS) {
  auto block_num = (N - 1) / max_threads_per_block + 1;
  return static_cast<int>(block_num);
}

namespace natten {
namespace cuda {
namespace naive {

struct LaunchParams {
  dim3 grid;
  dim3 block;

  LaunchParams(dim3 grid, dim3 block) : grid(grid), block(block) {}
};

template <typename KernelTemplate>
__global__ void launch_cuda_kernel(typename KernelTemplate::Params params) {
#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ > 300)
#if (__CUDA_ARCH__ < 600)
  // Half kernels are not supported in CC < 60,
  // Partial FP16 support was added in SM54, but
  // we use atomics, which were only introduced in
  // SM60, so disabling half kernels for CC < 60.
  // Also disabling tiled kernels, because older
  // architectures might not have enough shared memory
  // and the tiled kernels heavily rely on the assumed
  // amount of shared memory.
  if constexpr (KernelTemplate::IsHalfKernel || KernelTemplate::UsesSmem) {
    return;
  } else {
#elif (__CUDA_ARCH__ < 800)
  if constexpr (KernelTemplate::IsBF16Kernel) {
    return;
  } else {
#endif
    KernelTemplate kernel;
    kernel.launch(params);
#if (__CUDA_ARCH__ < 800)
  }
#endif
#else
  printf("Kernel not supported on this device / CUDA version.\n");
  asm volatile("brkpt;\n");
#endif
}

template <typename ElementScalar_>
struct IsBF16 {
  static constexpr bool value = false;
};

template <>
struct IsBF16<natten::bfloat16> {
  static constexpr bool value = true;
};

template <typename ElementScalar_>
struct HalfArray;

template <typename ElementScalar_, typename ElementVector_>
struct HalfArrayBase;

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 600)

template <>
struct HalfArrayBase<natten::float16, __half2> {
  using ElementNatten = natten::float16;
  using ElementScalar = __half;
  using ElementVector = __half2;

  __device__ __inline__ static ElementVector* typecast(
      ElementScalar* ptr_scalar) {
    ElementVector* ptr_vector = reinterpret_cast<ElementVector*>(ptr_scalar);
    return ptr_vector;
  }

  __device__ __inline__ static ElementNatten cast_back(ElementScalar s) {
    return s;
  }

  __device__ __inline__ static float to_float(ElementScalar s) {
    return __half2float(s);
  }

  __device__ __inline__ static ElementVector zero() {
    return __float2half2_rn(0.f);
  }
  __device__ __inline__ static ElementVector fma(
      ElementVector a,
      ElementVector b,
      ElementVector c) {
    return __hfma2(a, b, c);
  }
  __device__ __inline__ static ElementVector fma(
      ElementVector a,
      ElementScalar b,
      ElementVector c) {
    return __hfma2(a, __halves2half2(b, b), c);
  }
  __device__ __inline__ static ElementScalar add(
      ElementScalar a,
      ElementScalar b) {
    return __hadd(a, b);
  }
};

template <>
struct HalfArray<natten::float16> {
  using Base = HalfArrayBase<natten::float16, __half2>;
};
#else
struct float162 {};

template <>
struct HalfArrayBase<natten::float16, float162> {
  using ElementNatten = natten::float16;
  using ElementScalar = natten::float16;
  using ElementVector = float162;

  __device__ __inline__ static ElementVector* typecast(
      ElementScalar* ptr_scalar) {
    asm volatile("brkpt;\n");
  }

  __device__ __inline__ static ElementNatten cast_back(ElementScalar s) {
    asm volatile("brkpt;\n");
  }

  __device__ __inline__ static float to_float(ElementScalar s) {
    asm volatile("brkpt;\n");
  }

  __device__ __inline__ static ElementVector zero() {
    asm volatile("brkpt;\n");
  }
  __device__ __inline__ static ElementVector fma(
      ElementVector a,
      ElementVector b,
      ElementVector c) {
    asm volatile("brkpt;\n");
  }
  __device__ __inline__ static ElementVector fma(
      ElementVector a,
      ElementScalar b,
      ElementVector c) {
    asm volatile("brkpt;\n");
  }
  __device__ __inline__ static ElementScalar add(
      ElementScalar a,
      ElementScalar b) {
    asm volatile("brkpt;\n");
  }
};

template <>
struct HalfArray<natten::float16> {
  using Base = HalfArrayBase<natten::float16, float162>;
};
#endif

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 800)
template <>
struct HalfArrayBase<natten::bfloat16, __nv_bfloat162> {
  using ElementNatten = natten::bfloat16;
  using ElementScalar = __nv_bfloat16;
  using ElementVector = __nv_bfloat162;

  __device__ __inline__ static ElementVector* typecast(
      ElementScalar* ptr_scalar) {
    ElementVector* ptr_vector = reinterpret_cast<ElementVector*>(ptr_scalar);
    return ptr_vector;
  }

  __device__ __inline__ static ElementNatten cast_back(ElementScalar s) {
    return s;
  }

  __device__ __inline__ static float to_float(ElementScalar s) {
    return __bfloat162float(s);
  }

  __device__ __inline__ static ElementVector zero() {
    return __float2bfloat162_rn(0.f);
  }
  __device__ __inline__ static ElementVector fma(
      ElementVector a,
      ElementVector b,
      ElementVector c) {
    return __hfma2(a, b, c);
  }
  __device__ __inline__ static ElementVector fma(
      ElementVector a,
      ElementScalar b,
      ElementVector c) {
    return __hfma2(a, __halves2bfloat162(b, b), c);
  }
  __device__ __inline__ static ElementScalar add(
      ElementScalar a,
      ElementScalar b) {
    return __hadd(a, b);
  }
};

template <>
struct HalfArray<natten::bfloat16> {
  using Base = HalfArrayBase<natten::bfloat16, __nv_bfloat162>;
};
#else
struct bfloat162 {};

template <>
struct HalfArrayBase<natten::bfloat16, bfloat162> {
  using ElementNatten = natten::bfloat16;
  using ElementScalar = natten::bfloat16;
  using ElementVector = bfloat162;

  __device__ __inline__ static ElementVector* typecast(
      ElementScalar* ptr_scalar) {
    asm volatile("brkpt;\n");
  }

  __device__ __inline__ static ElementNatten cast_back(ElementScalar s) {
    asm volatile("brkpt;\n");
  }

  __device__ __inline__ static float to_float(ElementScalar s) {
    asm volatile("brkpt;\n");
  }

  __device__ __inline__ static ElementVector zero() {
    asm volatile("brkpt;\n");
  }
  __device__ __inline__ static ElementVector fma(
      ElementVector a,
      ElementVector b,
      ElementVector c) {
    asm volatile("brkpt;\n");
  }
  __device__ __inline__ static ElementVector fma(
      ElementVector a,
      ElementScalar b,
      ElementVector c) {
    asm volatile("brkpt;\n");
  }
  __device__ __inline__ static ElementScalar add(
      ElementScalar a,
      ElementScalar b) {
    asm volatile("brkpt;\n");
  }
};

template <>
struct HalfArray<natten::bfloat16> {
  using Base = HalfArrayBase<natten::bfloat16, bfloat162>;
};
#endif

inline __host__ __device__ int get_backward_window_start(
    const int index,
    const int KERNEL_SIZE,
    const int NEIGHBORHOOD_SIZE,
    const int dilation) {
  return (index < KERNEL_SIZE * dilation)
      ? (index % dilation)
      : index - NEIGHBORHOOD_SIZE * dilation;
}

inline __host__ __device__ int get_backward_window_end(
    const int index,
    const int length,
    const int KERNEL_SIZE,
    const int NEIGHBORHOOD_SIZE,
    const int dilation) {
  return (index >= length - KERNEL_SIZE * dilation)
      ? (length)
      : (index + (NEIGHBORHOOD_SIZE + 1) * dilation);
}

inline __host__ __device__ int get_window_start(
    const int index,
    const int length,
    const int KERNEL_SIZE,
    const int NEIGHBORHOOD_SIZE,
    const int dilation) {
  if (dilation <= 1)
    return max(index - NEIGHBORHOOD_SIZE, 0) +
        (index + NEIGHBORHOOD_SIZE >= length) *
        (length - index - NEIGHBORHOOD_SIZE - 1);
  int ni = index - NEIGHBORHOOD_SIZE * dilation;
  if (ni < 0)
    return index % dilation;
  if (index + NEIGHBORHOOD_SIZE * dilation >= length) {
    const int imodd = index % dilation;
    const int a = int(length / dilation) * dilation;
    const int b = length - a;
    if (imodd < b)
      return length - b + imodd - 2 * NEIGHBORHOOD_SIZE * dilation;
    return a + imodd - KERNEL_SIZE * dilation;
  }
  return ni;
}

inline __host__ __device__ int get_pb_start(
    const int index,
    const int length,
    const int KERNEL_SIZE,
    const int NEIGHBORHOOD_SIZE,
    const int dilation) {
  if (dilation <= 1)
    return NEIGHBORHOOD_SIZE +
        (index < NEIGHBORHOOD_SIZE) * (NEIGHBORHOOD_SIZE - index) +
        (index + NEIGHBORHOOD_SIZE >= length) *
        (length - index - 1 - NEIGHBORHOOD_SIZE);
  if (index - NEIGHBORHOOD_SIZE * dilation < 0)
    return KERNEL_SIZE - 1 - (index / dilation);
  if (index + NEIGHBORHOOD_SIZE * dilation >= length)
    return (length - index - 1) / dilation;
  return NEIGHBORHOOD_SIZE;
}

} // namespace naive
} // namespace cuda
} // namespace natten
