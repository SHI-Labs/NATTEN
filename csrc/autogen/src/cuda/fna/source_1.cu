#include <cuda_runtime.h>
#include <iostream>
#include <natten/dtypes.cuh>
#include <natten/cuda/fna/kernel_forward.h>
#include <natten_autogen/cuda/fna/kernels.h>
namespace natten { 
namespace cuda { 
namespace fna { 



///////////////////////////////////////////////////////////////////
// FNA-1D / float16 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, true>::kMinBlocksPerSm)
fna1_32x128x32_sm70_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x32_sm70_float16_cm_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, true>::kMinBlocksPerSm)
fna1_64x64x32_sm70_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x32_sm70_float16_cm_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, true>::kMinBlocksPerSm)
fna1_64x128x32_sm70_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x32_sm70_float16_cm_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, true>::kMinBlocksPerSm)
fna1_32x128x64_sm70_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x64_sm70_float16_cm_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true>::kMinBlocksPerSm)
fna1_64x64x64_sm70_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x64_sm70_float16_cm_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, true>::kMinBlocksPerSm)
fna1_64x128x64_sm70_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x64_sm70_float16_cm_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true>::kMinBlocksPerSm)
fna1_32x128x128_sm70_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x128_sm70_float16_cm_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, true>::kMinBlocksPerSm)
fna1_64x64x128_sm70_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x128_sm70_float16_cm_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, true>::kMinBlocksPerSm)
fna1_64x128x128_sm70_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x128_sm70_float16_cm_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true>::kMinBlocksPerSm)
fna1_32x128x65536_sm70_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x65536_sm70_float16_cm_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, true>::kMinBlocksPerSm)
fna1_64x128x65536_sm70_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x65536_sm70_float16_cm_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, true>::kMinBlocksPerSm)
fna1_64x64x65536_sm70_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x65536_sm70_float16_cm_0_rpb` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float16 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::kMinBlocksPerSm)
fna1_32x128x32_sm70_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x32_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::kMinBlocksPerSm)
fna1_64x64x32_sm70_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x32_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::kMinBlocksPerSm)
fna1_64x128x32_sm70_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x32_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::kMinBlocksPerSm)
fna1_32x128x64_sm70_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x64_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::kMinBlocksPerSm)
fna1_64x64x64_sm70_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x64_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::kMinBlocksPerSm)
fna1_64x128x64_sm70_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x64_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::kMinBlocksPerSm)
fna1_32x128x128_sm70_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x128_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::kMinBlocksPerSm)
fna1_64x64x128_sm70_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x128_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::kMinBlocksPerSm)
fna1_64x128x128_sm70_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x128_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::kMinBlocksPerSm)
fna1_32x128x65536_sm70_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x65536_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::kMinBlocksPerSm)
fna1_64x128x65536_sm70_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x65536_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fna1_64x64x65536_sm70_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x65536_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float32 / SM75
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 32, false>::kMinBlocksPerSm)
fna1_32x128x32_sm75_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x32_sm75_float32_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 32, false>::kMinBlocksPerSm)
fna1_64x64x32_sm75_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x32_sm75_float32_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 32, false>::kMinBlocksPerSm)
fna1_64x128x32_sm75_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x32_sm75_float32_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 64, false>::kMinBlocksPerSm)
fna1_32x128x64_sm75_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x64_sm75_float32_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 64, false>::kMinBlocksPerSm)
fna1_64x64x64_sm75_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x64_sm75_float32_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 64, false>::kMinBlocksPerSm)
fna1_64x128x64_sm75_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x64_sm75_float32_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 128, false>::kMinBlocksPerSm)
fna1_32x128x128_sm75_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x128_sm75_float32_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 128, false>::kMinBlocksPerSm)
fna1_64x64x128_sm75_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x128_sm75_float32_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 128, false>::kMinBlocksPerSm)
fna1_64x128x128_sm75_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x128_sm75_float32_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 65536, false>::kMinBlocksPerSm)
fna1_32x128x65536_sm75_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x65536_sm75_float32_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 65536, false>::kMinBlocksPerSm)
fna1_64x128x65536_sm75_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x65536_sm75_float32_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fna1_64x64x65536_sm75_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x65536_sm75_float32_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float32 / SM75
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 32, true>::kMinBlocksPerSm)
fna1_32x128x32_sm75_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x32_sm75_float32_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 32, true>::kMinBlocksPerSm)
fna1_64x64x32_sm75_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x32_sm75_float32_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 32, true>::kMinBlocksPerSm)
fna1_64x128x32_sm75_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x32_sm75_float32_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 64, true>::kMinBlocksPerSm)
fna1_32x128x64_sm75_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x64_sm75_float32_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 64, true>::kMinBlocksPerSm)
fna1_64x64x64_sm75_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x64_sm75_float32_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 64, true>::kMinBlocksPerSm)
fna1_64x128x64_sm75_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x64_sm75_float32_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 128, true>::kMinBlocksPerSm)
fna1_32x128x128_sm75_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x128_sm75_float32_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 128, true>::kMinBlocksPerSm)
fna1_64x64x128_sm75_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x128_sm75_float32_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 128, true>::kMinBlocksPerSm)
fna1_64x128x128_sm75_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x128_sm75_float32_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 65536, true>::kMinBlocksPerSm)
fna1_32x128x65536_sm75_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 32, 128, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x65536_sm75_float32_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 65536, true>::kMinBlocksPerSm)
fna1_64x128x65536_sm75_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 128, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x65536_sm75_float32_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 65536, true>::kMinBlocksPerSm)
fna1_64x64x65536_sm75_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x65536_sm75_float32_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float32 / SM75
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 32, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 32, 128, 32, false>::kMinBlocksPerSm)
fna1_32x128x32_sm75_float32_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 32, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 32, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x32_sm75_float32_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 32, false>::kMinBlocksPerSm)
fna1_64x64x32_sm75_float32_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x32_sm75_float32_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 128, 32, false>::kMinBlocksPerSm)
fna1_64x128x32_sm75_float32_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x32_sm75_float32_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 32, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 32, 128, 64, false>::kMinBlocksPerSm)
fna1_32x128x64_sm75_float32_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 32, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 32, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x64_sm75_float32_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 64, false>::kMinBlocksPerSm)
fna1_64x64x64_sm75_float32_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x64_sm75_float32_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 128, 64, false>::kMinBlocksPerSm)
fna1_64x128x64_sm75_float32_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x64_sm75_float32_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 32, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 32, 128, 128, false>::kMinBlocksPerSm)
fna1_32x128x128_sm75_float32_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 32, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 32, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x128_sm75_float32_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 128, false>::kMinBlocksPerSm)
fna1_64x64x128_sm75_float32_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x128_sm75_float32_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 128, 128, false>::kMinBlocksPerSm)
fna1_64x128x128_sm75_float32_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x128_sm75_float32_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 32, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 32, 128, 65536, false>::kMinBlocksPerSm)
fna1_32x128x65536_sm75_float32_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 32, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 32, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x65536_sm75_float32_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 128, 65536, false>::kMinBlocksPerSm)
fna1_64x128x65536_sm75_float32_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x65536_sm75_float32_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fna1_64x64x65536_sm75_float32_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x65536_sm75_float32_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float16 / SM75
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 32, false>::kMinBlocksPerSm)
fna1_32x128x32_sm75_float16_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x32_sm75_float16_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32, false>::kMinBlocksPerSm)
fna1_64x64x32_sm75_float16_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x32_sm75_float16_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 32, false>::kMinBlocksPerSm)
fna1_64x128x32_sm75_float16_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x32_sm75_float16_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 64, false>::kMinBlocksPerSm)
fna1_32x128x64_sm75_float16_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x64_sm75_float16_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, false>::kMinBlocksPerSm)
fna1_64x64x64_sm75_float16_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x64_sm75_float16_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 64, false>::kMinBlocksPerSm)
fna1_64x128x64_sm75_float16_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x64_sm75_float16_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, false>::kMinBlocksPerSm)
fna1_32x128x128_sm75_float16_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x128_sm75_float16_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128, false>::kMinBlocksPerSm)
fna1_64x64x128_sm75_float16_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x128_sm75_float16_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 128, false>::kMinBlocksPerSm)
fna1_64x128x128_sm75_float16_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x128_sm75_float16_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, false>::kMinBlocksPerSm)
fna1_32x128x65536_sm75_float16_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x65536_sm75_float16_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 65536, false>::kMinBlocksPerSm)
fna1_64x128x65536_sm75_float16_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x65536_sm75_float16_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fna1_64x64x65536_sm75_float16_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x65536_sm75_float16_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float16 / SM75
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 32, true>::kMinBlocksPerSm)
fna1_32x128x32_sm75_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x32_sm75_float16_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32, true>::kMinBlocksPerSm)
fna1_64x64x32_sm75_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x32_sm75_float16_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 32, true>::kMinBlocksPerSm)
fna1_64x128x32_sm75_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x32_sm75_float16_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 64, true>::kMinBlocksPerSm)
fna1_32x128x64_sm75_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x64_sm75_float16_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, true>::kMinBlocksPerSm)
fna1_64x64x64_sm75_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x64_sm75_float16_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 64, true>::kMinBlocksPerSm)
fna1_64x128x64_sm75_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x64_sm75_float16_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, true>::kMinBlocksPerSm)
fna1_32x128x128_sm75_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x128_sm75_float16_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128, true>::kMinBlocksPerSm)
fna1_64x64x128_sm75_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x128_sm75_float16_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 128, true>::kMinBlocksPerSm)
fna1_64x128x128_sm75_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x128_sm75_float16_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, true>::kMinBlocksPerSm)
fna1_32x128x65536_sm75_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x65536_sm75_float16_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 65536, true>::kMinBlocksPerSm)
fna1_64x128x65536_sm75_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x65536_sm75_float16_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536, true>::kMinBlocksPerSm)
fna1_64x64x65536_sm75_float16_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x65536_sm75_float16_cm_0_rpb` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float16 / SM75
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 32, false>::kMinBlocksPerSm)
fna1_32x128x32_sm75_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x32_sm75_float16_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32, false>::kMinBlocksPerSm)
fna1_64x64x32_sm75_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x32_sm75_float16_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 32, false>::kMinBlocksPerSm)
fna1_64x128x32_sm75_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x32_sm75_float16_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 64, false>::kMinBlocksPerSm)
fna1_32x128x64_sm75_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x64_sm75_float16_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, false>::kMinBlocksPerSm)
fna1_64x64x64_sm75_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x64_sm75_float16_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 64, false>::kMinBlocksPerSm)
fna1_64x128x64_sm75_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x64_sm75_float16_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, false>::kMinBlocksPerSm)
fna1_32x128x128_sm75_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x128_sm75_float16_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128, false>::kMinBlocksPerSm)
fna1_64x64x128_sm75_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x128_sm75_float16_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 128, false>::kMinBlocksPerSm)
fna1_64x128x128_sm75_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x128_sm75_float16_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, false>::kMinBlocksPerSm)
fna1_32x128x65536_sm75_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x65536_sm75_float16_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 65536, false>::kMinBlocksPerSm)
fna1_64x128x65536_sm75_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x65536_sm75_float16_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fna1_64x64x65536_sm75_float16_cm_1(typename FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x65536_sm75_float16_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float32 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 32, false>::kMinBlocksPerSm)
fna1_32x128x32_sm80_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x32_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 32, false>::kMinBlocksPerSm)
fna1_64x64x32_sm80_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x32_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 32, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 32, false>::kMinBlocksPerSm)
fna1_64x128x32_sm80_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x32_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 64, false>::kMinBlocksPerSm)
fna1_32x128x64_sm80_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x64_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 64, false>::kMinBlocksPerSm)
fna1_64x64x64_sm80_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x64_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 64, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 64, false>::kMinBlocksPerSm)
fna1_64x128x64_sm80_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x64_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 128, false>::kMinBlocksPerSm)
fna1_32x128x128_sm80_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x128_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 128, false>::kMinBlocksPerSm)
fna1_64x64x128_sm80_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x128_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 128, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 128, false>::kMinBlocksPerSm)
fna1_64x128x128_sm80_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x128_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 65536, false>::kMinBlocksPerSm)
fna1_32x128x65536_sm80_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x65536_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 65536, false>::kMinBlocksPerSm)
fna1_64x128x65536_sm80_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x65536_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 65536, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fna1_64x64x65536_sm80_float32_cm_0(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x65536_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float32 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 32, true>::kMinBlocksPerSm)
fna1_32x128x32_sm80_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x32_sm80_float32_cm_0_rpb` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 32, true>::kMinBlocksPerSm)
fna1_64x64x32_sm80_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x32_sm80_float32_cm_0_rpb` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 32, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 32, true>::kMinBlocksPerSm)
fna1_64x128x32_sm80_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 32, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 32, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x32_sm80_float32_cm_0_rpb` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 64, true>::kMinBlocksPerSm)
fna1_32x128x64_sm80_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x64_sm80_float32_cm_0_rpb` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 64, true>::kMinBlocksPerSm)
fna1_64x64x64_sm80_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x64_sm80_float32_cm_0_rpb` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 64, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 64, true>::kMinBlocksPerSm)
fna1_64x128x64_sm80_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 64, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 64, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x64_sm80_float32_cm_0_rpb` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 128, true>::kMinBlocksPerSm)
fna1_32x128x128_sm80_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x128_sm80_float32_cm_0_rpb` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 128, true>::kMinBlocksPerSm)
fna1_64x64x128_sm80_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x128_sm80_float32_cm_0_rpb` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 128, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 128, true>::kMinBlocksPerSm)
fna1_64x128x128_sm80_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 128, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 128, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x128_sm80_float32_cm_0_rpb` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 65536, true>::kMinBlocksPerSm)
fna1_32x128x65536_sm80_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 32, 128, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_32x128x65536_sm80_float32_cm_0_rpb` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 65536, true>::kMinBlocksPerSm)
fna1_64x128x65536_sm80_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 128, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x128x65536_sm80_float32_cm_0_rpb` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 65536, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 65536, true>::kMinBlocksPerSm)
fna1_64x64x65536_sm80_float32_cm_0_rpb(typename FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 65536, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 65536, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1_64x64x65536_sm80_float32_cm_0_rpb` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
} 
} 
} 

