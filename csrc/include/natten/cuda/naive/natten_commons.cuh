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
inline int32_t GET_BLOCKS(
    int64_t N,
    int64_t max_threads_per_block = CUDA_NUM_THREADS) {
  auto block_num = (N - 1) / max_threads_per_block + 1;
  return static_cast<int32_t>(block_num);
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
  // Partial FP16 support was added in SM50, but ours need the half2 type
  // and ops, which aren't defined for SM50.
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

  __device__ __inline__ static ElementScalar from_float(float s) {
    return __float2half(s);
  }

  __device__ __inline__ static ElementScalar zero() {
    return __float2half(0.f);
  }

  __device__ __inline__ static ElementVector zeros() {
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

  __device__ __inline__ static ElementScalar from_float(float s) {
    return __float2bfloat16(s);
  }

  __device__ __inline__ static ElementScalar zero() {
    return __float2bfloat16(0.f);
  }

  __device__ __inline__ static ElementVector zeros() {
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
};

template <>
struct HalfArray<natten::bfloat16> {
  using Base = HalfArrayBase<natten::bfloat16, bfloat162>;
};
#endif

template <bool IsCausal_>
struct NeighborhoodMask {
  static constexpr bool IsCausal = IsCausal_;

  int32_t length;
  int32_t kernel_size;
  int32_t neighborhood_size;
  int32_t dilation;

  inline __device__ NeighborhoodMask(
      int32_t length,
      int32_t kernel_size,
      int32_t dilation)
      : length(length),
        kernel_size(kernel_size),
        neighborhood_size(kernel_size / 2),
        dilation(dilation) {}

  inline __device__ int32_t get_backward_window_start(int32_t index) {
    if constexpr (!IsCausal) {
      return (index < kernel_size * dilation)
          ? (index % dilation)
          : index - neighborhood_size * dilation;
    } else {
      return index;
    }
  }

  inline __device__ int32_t get_backward_window_end(int32_t index) {
    if constexpr (!IsCausal) {
      return (index >= length - kernel_size * dilation)
          ? (length)
          : (index + (neighborhood_size + 1) * dilation);
    } else {
      return min(index + kernel_size * dilation, length);
    }
  }

  inline __device__ int32_t get_window_start(int32_t index) {
    if constexpr (!IsCausal) {
      if (dilation <= 1)
        return max(index - neighborhood_size, 0) +
            (index + neighborhood_size >= length) *
            (length - index - neighborhood_size - 1);
      int32_t ni = index - neighborhood_size * dilation;
      if (ni < 0)
        return index % dilation;
      if (index + neighborhood_size * dilation >= length) {
        int32_t imodd = index % dilation;
        int32_t a = int32_t(length / dilation) * dilation;
        int32_t b = length - a;
        if (imodd < b)
          return length - b + imodd - 2 * neighborhood_size * dilation;
        return a + imodd - kernel_size * dilation;
      }
      return ni;
    } else {
      auto dilation_idx = index % dilation;
      auto index_pdp = index / dilation;
      return max(index_pdp - kernel_size + 1, 0) * dilation + dilation_idx;
    }
  }

  inline __device__ int32_t get_window_end(int32_t index, int32_t start_index) {
    if constexpr (!IsCausal) {
      return min(length, start_index + kernel_size * dilation);
    } else {
      return min(length, index + dilation);
    }
  }

  inline __device__ int32_t get_pb_start(int32_t index) {
    if constexpr (!IsCausal) {
      if (dilation <= 1)
        return neighborhood_size +
            (index < neighborhood_size) * (neighborhood_size - index) +
            (index + neighborhood_size >= length) *
            (length - index - 1 - neighborhood_size);
      if (index - neighborhood_size * dilation < 0)
        return kernel_size - 1 - (index / dilation);
      if (index + neighborhood_size * dilation >= length)
        return (length - index - 1) / dilation;
      return neighborhood_size;
    } else {
      printf(
          "FATAL: RPB kernels do not support causal masking. Please open an issue.");
      asm volatile("brkpt;\n");
    }
  }
};

inline __device__ int32_t get_window_start(
    int32_t index,
    int32_t length,
    int32_t KERNEL_SIZE,
    int32_t NEIGHBORHOOD_SIZE,
    int32_t dilation) {
  if (dilation <= 1)
    return max(index - NEIGHBORHOOD_SIZE, 0) +
        (index + NEIGHBORHOOD_SIZE >= length) *
        (length - index - NEIGHBORHOOD_SIZE - 1);
  int32_t ni = index - NEIGHBORHOOD_SIZE * dilation;
  if (ni < 0)
    return index % dilation;
  if (index + NEIGHBORHOOD_SIZE * dilation >= length) {
    int32_t imodd = index % dilation;
    int32_t a = int32_t(length / dilation) * dilation;
    int32_t b = length - a;
    if (imodd < b)
      return length - b + imodd - 2 * NEIGHBORHOOD_SIZE * dilation;
    return a + imodd - KERNEL_SIZE * dilation;
  }
  return ni;
}

inline __device__ int32_t get_pb_start(
    int32_t index,
    int32_t length,
    int32_t KERNEL_SIZE,
    int32_t NEIGHBORHOOD_SIZE,
    int32_t dilation) {
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

template <typename T>
struct AttnMask;

#ifdef __CUDA_ARCH__

// TODO: bitcast into a constexpr instead of evaluating at runtime

template <>
struct AttnMask<double> {
  static __device__ auto value(bool is_grad) {
    return is_grad ? 0.0 : -__longlong_as_double(0x7ff0000000000000ULL);
  }
};

template <>
struct AttnMask<float> {
  static __device__ auto value(bool is_grad) {
    return is_grad ? 0.f : -__int_as_float(0x7f800000U);
  }
};

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 600)
template <>
struct AttnMask<natten::float16> {
  static __device__ auto value(bool is_grad) {
    return is_grad ? __float2half(0.f)
                   : __hneg(__ushort_as_half((unsigned short)0x7C00U));
  }
};
#endif

#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 800)
template <>
struct AttnMask<natten::bfloat16> {
  static __device__ auto value(bool is_grad) {
    return is_grad ? __float2bfloat16(0.f)
                   : __hneg(__ushort_as_bfloat16((unsigned short)0x7F80U));
  }
};
#endif

#endif

//////////////////////////////////////////////////
/// Atomics for older architectures
//////////////////////////////////////////////////

#if defined(__CUDA_ARCH__)

static inline __device__ float floatOrDoubleAtomicAdd(
    float* address,
    float val) {
  return atomicAdd(address, val);
}

static inline __device__ double floatOrDoubleAtomicAdd(
    double* address,
    double val) {
#if (__CUDA_ARCH__ >= 600)
  return atomicAdd(address, val);
#else
    // Taken from pytorch;
    // ATen/cuda/Atomic.cuh
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do {
      assumed = old;
      old = atomicCAS(
          address_as_ull,
          assumed,
          __double_as_longlong(val + __longlong_as_double(assumed)));
      // Note: uses integer comparison to avoid hang in case of NaN (since NaN
      // != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
#endif
}

#endif

} // namespace naive
} // namespace cuda
} // namespace natten
