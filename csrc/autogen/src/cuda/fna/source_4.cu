#include <cuda_runtime.h>
#include <iostream>
#include <natten/dtypes.cuh>
#include <natten/cuda/fna/kernel_forward.h>
#include <natten_autogen/cuda/fna/kernels.h>
namespace natten { 
namespace cuda { 
namespace fna { 



///////////////////////////////////////////////////////////////////
// FNA-2D / float32 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 32, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 32, 128, 32, false>::kMinBlocksPerSm)
fna2_32x128x32_sm70_float32_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 32, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 32, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x32_sm70_float32_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 32, false>::kMinBlocksPerSm)
fna2_64x64x32_sm70_float32_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x32_sm70_float32_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 128, 32, false>::kMinBlocksPerSm)
fna2_64x128x32_sm70_float32_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x32_sm70_float32_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 32, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 32, 128, 64, false>::kMinBlocksPerSm)
fna2_32x128x64_sm70_float32_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 32, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 32, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x64_sm70_float32_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 64, false>::kMinBlocksPerSm)
fna2_64x64x64_sm70_float32_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x64_sm70_float32_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 128, 64, false>::kMinBlocksPerSm)
fna2_64x128x64_sm70_float32_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x64_sm70_float32_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 32, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 32, 128, 128, false>::kMinBlocksPerSm)
fna2_32x128x128_sm70_float32_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 32, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 32, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x128_sm70_float32_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 128, false>::kMinBlocksPerSm)
fna2_64x64x128_sm70_float32_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x128_sm70_float32_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 128, 128, false>::kMinBlocksPerSm)
fna2_64x128x128_sm70_float32_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x128_sm70_float32_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 32, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 32, 128, 65536, false>::kMinBlocksPerSm)
fna2_32x128x65536_sm70_float32_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 32, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 32, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x65536_sm70_float32_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 128, 65536, false>::kMinBlocksPerSm)
fna2_64x128x65536_sm70_float32_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x65536_sm70_float32_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fna2_64x64x65536_sm70_float32_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x65536_sm70_float32_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / float32 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 32, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 32, 128, 32, false>::kMinBlocksPerSm)
fna2_32x128x32_sm70_float32_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 32, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 32, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x32_sm70_float32_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 32, false>::kMinBlocksPerSm)
fna2_64x64x32_sm70_float32_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x32_sm70_float32_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 128, 32, false>::kMinBlocksPerSm)
fna2_64x128x32_sm70_float32_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x32_sm70_float32_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 32, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 32, 128, 64, false>::kMinBlocksPerSm)
fna2_32x128x64_sm70_float32_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 32, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 32, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x64_sm70_float32_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 64, false>::kMinBlocksPerSm)
fna2_64x64x64_sm70_float32_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x64_sm70_float32_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 128, 64, false>::kMinBlocksPerSm)
fna2_64x128x64_sm70_float32_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x64_sm70_float32_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 32, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 32, 128, 128, false>::kMinBlocksPerSm)
fna2_32x128x128_sm70_float32_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 32, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 32, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x128_sm70_float32_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 128, false>::kMinBlocksPerSm)
fna2_64x64x128_sm70_float32_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x128_sm70_float32_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 128, 128, false>::kMinBlocksPerSm)
fna2_64x128x128_sm70_float32_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x128_sm70_float32_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 32, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 32, 128, 65536, false>::kMinBlocksPerSm)
fna2_32x128x65536_sm70_float32_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 32, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 32, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x65536_sm70_float32_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 128, 65536, false>::kMinBlocksPerSm)
fna2_64x128x65536_sm70_float32_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x65536_sm70_float32_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fna2_64x64x65536_sm70_float32_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x65536_sm70_float32_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / float16 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::kMinBlocksPerSm)
fna2_32x128x32_sm70_float16_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x32_sm70_float16_cm_0_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::kMinBlocksPerSm)
fna2_64x64x32_sm70_float16_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x32_sm70_float16_cm_0_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::kMinBlocksPerSm)
fna2_64x128x32_sm70_float16_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x32_sm70_float16_cm_0_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::kMinBlocksPerSm)
fna2_32x128x64_sm70_float16_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x64_sm70_float16_cm_0_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::kMinBlocksPerSm)
fna2_64x64x64_sm70_float16_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x64_sm70_float16_cm_0_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::kMinBlocksPerSm)
fna2_64x128x64_sm70_float16_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x64_sm70_float16_cm_0_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::kMinBlocksPerSm)
fna2_32x128x128_sm70_float16_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x128_sm70_float16_cm_0_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::kMinBlocksPerSm)
fna2_64x64x128_sm70_float16_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x128_sm70_float16_cm_0_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::kMinBlocksPerSm)
fna2_64x128x128_sm70_float16_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x128_sm70_float16_cm_0_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::kMinBlocksPerSm)
fna2_32x128x65536_sm70_float16_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x65536_sm70_float16_cm_0_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::kMinBlocksPerSm)
fna2_64x128x65536_sm70_float16_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x65536_sm70_float16_cm_0_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fna2_64x64x65536_sm70_float16_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x65536_sm70_float16_cm_0_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / float16 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, true>::kMinBlocksPerSm)
fna2_32x128x32_sm70_float16_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x32_sm70_float16_cm_0_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, true>::kMinBlocksPerSm)
fna2_64x64x32_sm70_float16_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x32_sm70_float16_cm_0_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, true>::kMinBlocksPerSm)
fna2_64x128x32_sm70_float16_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x32_sm70_float16_cm_0_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, true>::kMinBlocksPerSm)
fna2_32x128x64_sm70_float16_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x64_sm70_float16_cm_0_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true>::kMinBlocksPerSm)
fna2_64x64x64_sm70_float16_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x64_sm70_float16_cm_0_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, true>::kMinBlocksPerSm)
fna2_64x128x64_sm70_float16_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x64_sm70_float16_cm_0_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true>::kMinBlocksPerSm)
fna2_32x128x128_sm70_float16_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x128_sm70_float16_cm_0_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, true>::kMinBlocksPerSm)
fna2_64x64x128_sm70_float16_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x128_sm70_float16_cm_0_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, true>::kMinBlocksPerSm)
fna2_64x128x128_sm70_float16_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x128_sm70_float16_cm_0_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true>::kMinBlocksPerSm)
fna2_32x128x65536_sm70_float16_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x65536_sm70_float16_cm_0_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, true>::kMinBlocksPerSm)
fna2_64x128x65536_sm70_float16_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x65536_sm70_float16_cm_0_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, true>::kMinBlocksPerSm)
fna2_64x64x65536_sm70_float16_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x65536_sm70_float16_cm_0_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / float16 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::kMinBlocksPerSm)
fna2_32x128x32_sm70_float16_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x32_sm70_float16_cm_0_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::kMinBlocksPerSm)
fna2_64x64x32_sm70_float16_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x32_sm70_float16_cm_0_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::kMinBlocksPerSm)
fna2_64x128x32_sm70_float16_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x32_sm70_float16_cm_0_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::kMinBlocksPerSm)
fna2_32x128x64_sm70_float16_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x64_sm70_float16_cm_0_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::kMinBlocksPerSm)
fna2_64x64x64_sm70_float16_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x64_sm70_float16_cm_0_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::kMinBlocksPerSm)
fna2_64x128x64_sm70_float16_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x64_sm70_float16_cm_0_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::kMinBlocksPerSm)
fna2_32x128x128_sm70_float16_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x128_sm70_float16_cm_0_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::kMinBlocksPerSm)
fna2_64x64x128_sm70_float16_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x128_sm70_float16_cm_0_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::kMinBlocksPerSm)
fna2_64x128x128_sm70_float16_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x128_sm70_float16_cm_0_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::kMinBlocksPerSm)
fna2_32x128x65536_sm70_float16_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x65536_sm70_float16_cm_0_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::kMinBlocksPerSm)
fna2_64x128x65536_sm70_float16_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x65536_sm70_float16_cm_0_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fna2_64x64x65536_sm70_float16_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x65536_sm70_float16_cm_0_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / float16 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::kMinBlocksPerSm)
fna2_32x128x32_sm70_float16_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x32_sm70_float16_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::kMinBlocksPerSm)
fna2_64x64x32_sm70_float16_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x32_sm70_float16_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::kMinBlocksPerSm)
fna2_64x128x32_sm70_float16_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x32_sm70_float16_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::kMinBlocksPerSm)
fna2_32x128x64_sm70_float16_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x64_sm70_float16_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::kMinBlocksPerSm)
fna2_64x64x64_sm70_float16_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x64_sm70_float16_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::kMinBlocksPerSm)
fna2_64x128x64_sm70_float16_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x64_sm70_float16_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::kMinBlocksPerSm)
fna2_32x128x128_sm70_float16_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x128_sm70_float16_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::kMinBlocksPerSm)
fna2_64x64x128_sm70_float16_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x128_sm70_float16_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::kMinBlocksPerSm)
fna2_64x128x128_sm70_float16_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x128_sm70_float16_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::kMinBlocksPerSm)
fna2_32x128x65536_sm70_float16_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x65536_sm70_float16_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::kMinBlocksPerSm)
fna2_64x128x65536_sm70_float16_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x65536_sm70_float16_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fna2_64x64x65536_sm70_float16_cm_1_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x65536_sm70_float16_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / float16 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::kMinBlocksPerSm)
fna2_32x128x32_sm70_float16_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x32_sm70_float16_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::kMinBlocksPerSm)
fna2_64x64x32_sm70_float16_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x32_sm70_float16_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::kMinBlocksPerSm)
fna2_64x128x32_sm70_float16_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x32_sm70_float16_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::kMinBlocksPerSm)
fna2_32x128x64_sm70_float16_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x64_sm70_float16_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::kMinBlocksPerSm)
fna2_64x64x64_sm70_float16_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x64_sm70_float16_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::kMinBlocksPerSm)
fna2_64x128x64_sm70_float16_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x64_sm70_float16_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::kMinBlocksPerSm)
fna2_32x128x128_sm70_float16_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x128_sm70_float16_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::kMinBlocksPerSm)
fna2_64x64x128_sm70_float16_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x128_sm70_float16_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::kMinBlocksPerSm)
fna2_64x128x128_sm70_float16_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x128_sm70_float16_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::kMinBlocksPerSm)
fna2_32x128x65536_sm70_float16_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x65536_sm70_float16_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::kMinBlocksPerSm)
fna2_64x128x65536_sm70_float16_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x65536_sm70_float16_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fna2_64x64x65536_sm70_float16_cm_1_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x65536_sm70_float16_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / float32 / SM75
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 32, false>::kMinBlocksPerSm)
fna2_32x128x32_sm75_float32_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x32_sm75_float32_cm_0_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 32, false>::kMinBlocksPerSm)
fna2_64x64x32_sm75_float32_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x32_sm75_float32_cm_0_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 32, false>::kMinBlocksPerSm)
fna2_64x128x32_sm75_float32_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x32_sm75_float32_cm_0_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 64, false>::kMinBlocksPerSm)
fna2_32x128x64_sm75_float32_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x64_sm75_float32_cm_0_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 64, false>::kMinBlocksPerSm)
fna2_64x64x64_sm75_float32_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x64_sm75_float32_cm_0_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 64, false>::kMinBlocksPerSm)
fna2_64x128x64_sm75_float32_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x64_sm75_float32_cm_0_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 128, false>::kMinBlocksPerSm)
fna2_32x128x128_sm75_float32_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x128_sm75_float32_cm_0_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 128, false>::kMinBlocksPerSm)
fna2_64x64x128_sm75_float32_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x128_sm75_float32_cm_0_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 128, false>::kMinBlocksPerSm)
fna2_64x128x128_sm75_float32_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x128_sm75_float32_cm_0_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 65536, false>::kMinBlocksPerSm)
fna2_32x128x65536_sm75_float32_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x65536_sm75_float32_cm_0_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 65536, false>::kMinBlocksPerSm)
fna2_64x128x65536_sm75_float32_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x65536_sm75_float32_cm_0_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fna2_64x64x65536_sm75_float32_cm_0_0(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x65536_sm75_float32_cm_0_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / float32 / SM75
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 32, true>::kMinBlocksPerSm)
fna2_32x128x32_sm75_float32_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x32_sm75_float32_cm_0_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 32, true>::kMinBlocksPerSm)
fna2_64x64x32_sm75_float32_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x32_sm75_float32_cm_0_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 32, true>::kMinBlocksPerSm)
fna2_64x128x32_sm75_float32_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x32_sm75_float32_cm_0_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 64, true>::kMinBlocksPerSm)
fna2_32x128x64_sm75_float32_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x64_sm75_float32_cm_0_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 64, true>::kMinBlocksPerSm)
fna2_64x64x64_sm75_float32_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x64_sm75_float32_cm_0_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 64, true>::kMinBlocksPerSm)
fna2_64x128x64_sm75_float32_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x64_sm75_float32_cm_0_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 128, true>::kMinBlocksPerSm)
fna2_32x128x128_sm75_float32_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x128_sm75_float32_cm_0_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 128, true>::kMinBlocksPerSm)
fna2_64x64x128_sm75_float32_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x128_sm75_float32_cm_0_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 128, true>::kMinBlocksPerSm)
fna2_64x128x128_sm75_float32_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x128_sm75_float32_cm_0_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 65536, true>::kMinBlocksPerSm)
fna2_32x128x65536_sm75_float32_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 32, 128, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x65536_sm75_float32_cm_0_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 65536, true>::kMinBlocksPerSm)
fna2_64x128x65536_sm75_float32_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 128, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x65536_sm75_float32_cm_0_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 65536, true>::kMinBlocksPerSm)
fna2_64x64x65536_sm75_float32_cm_0_0_rpb(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, false>, float, cutlass::arch::Sm75, true, 64, 64, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x65536_sm75_float32_cm_0_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / float32 / SM75
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 32, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 32, 128, 32, false>::kMinBlocksPerSm)
fna2_32x128x32_sm75_float32_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 32, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 32, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x32_sm75_float32_cm_0_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 64, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 64, 32, false>::kMinBlocksPerSm)
fna2_64x64x32_sm75_float32_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 64, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 64, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x32_sm75_float32_cm_0_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 128, 32, false>::kMinBlocksPerSm)
fna2_64x128x32_sm75_float32_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x32_sm75_float32_cm_0_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 32, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 32, 128, 64, false>::kMinBlocksPerSm)
fna2_32x128x64_sm75_float32_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 32, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 32, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x64_sm75_float32_cm_0_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 64, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 64, 64, false>::kMinBlocksPerSm)
fna2_64x64x64_sm75_float32_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 64, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 64, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x64_sm75_float32_cm_0_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 128, 64, false>::kMinBlocksPerSm)
fna2_64x128x64_sm75_float32_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x64_sm75_float32_cm_0_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 32, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 32, 128, 128, false>::kMinBlocksPerSm)
fna2_32x128x128_sm75_float32_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 32, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 32, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x128_sm75_float32_cm_0_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 64, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 64, 128, false>::kMinBlocksPerSm)
fna2_64x64x128_sm75_float32_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x128_sm75_float32_cm_0_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 128, 128, false>::kMinBlocksPerSm)
fna2_64x128x128_sm75_float32_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x128_sm75_float32_cm_0_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 32, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 32, 128, 65536, false>::kMinBlocksPerSm)
fna2_32x128x65536_sm75_float32_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 32, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 32, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_32x128x65536_sm75_float32_cm_0_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 128, 65536, false>::kMinBlocksPerSm)
fna2_64x128x65536_sm75_float32_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x128x65536_sm75_float32_cm_0_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fna2_64x64x65536_sm75_float32_cm_0_1(typename FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<2, CausalMask<false, true>, float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2_64x64x65536_sm75_float32_cm_0_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
} 
} 
} 

