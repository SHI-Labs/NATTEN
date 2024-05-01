#include <cuda_runtime.h>
#include <iostream>
#include <natten/dtypes.cuh>
#include <natten/cuda/fna/kernel_backward.h>
#include <natten_autogen/cuda/fna/kernels.h>
namespace natten { 
namespace cuda { 
namespace fna { 



///////////////////////////////////////////////////////////////////
// FNA-2D / float32 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna2d_backward_64x64x32_sm80_float32_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x32_sm80_float32_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna2d_backward_64x64x64_sm80_float32_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x64_sm80_float32_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna2d_backward_64x64x128_sm80_float32_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x128_sm80_float32_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 128, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 128, 64, 128>::kMinBlocksPerSm)
fna2d_backward_128x64x128_sm80_float32_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 128, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 128, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x64x128_sm80_float32_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_64x64x65536_sm80_float32_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x65536_sm80_float32_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_128x64x65536_sm80_float32_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x64x65536_sm80_float32_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / float32 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna2d_backward_64x64x32_sm80_float32_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x32_sm80_float32_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna2d_backward_64x64x64_sm80_float32_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x64_sm80_float32_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna2d_backward_64x64x128_sm80_float32_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x128_sm80_float32_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 128, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 128, 64, 128>::kMinBlocksPerSm)
fna2d_backward_128x64x128_sm80_float32_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 128, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 128, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x64x128_sm80_float32_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_64x64x65536_sm80_float32_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x65536_sm80_float32_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_128x64x65536_sm80_float32_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x64x65536_sm80_float32_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / float16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna2d_backward_64x64x32_sm80_float16_cm_0_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x32_sm80_float16_cm_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna2d_backward_64x64x64_sm80_float16_cm_0_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x64_sm80_float16_cm_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna2d_backward_64x64x128_sm80_float16_cm_0_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x128_sm80_float16_cm_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::kMinBlocksPerSm)
fna2d_backward_128x128x128_sm80_float16_cm_0_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x128x128_sm80_float16_cm_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_64x64x65536_sm80_float16_cm_0_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x65536_sm80_float16_cm_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_128x64x65536_sm80_float16_cm_0_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x64x65536_sm80_float16_cm_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / float16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna2d_backward_64x64x32_sm80_float16_cm_0_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x32_sm80_float16_cm_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna2d_backward_64x64x64_sm80_float16_cm_0_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x64_sm80_float16_cm_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna2d_backward_64x64x128_sm80_float16_cm_0_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x128_sm80_float16_cm_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::kMinBlocksPerSm)
fna2d_backward_128x128x128_sm80_float16_cm_0_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x128x128_sm80_float16_cm_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_64x64x65536_sm80_float16_cm_0_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x65536_sm80_float16_cm_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_128x64x65536_sm80_float16_cm_0_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x64x65536_sm80_float16_cm_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / float16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna2d_backward_64x64x32_sm80_float16_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x32_sm80_float16_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna2d_backward_64x64x64_sm80_float16_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x64_sm80_float16_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna2d_backward_64x64x128_sm80_float16_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x128_sm80_float16_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::kMinBlocksPerSm)
fna2d_backward_128x128x128_sm80_float16_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x128x128_sm80_float16_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_64x64x65536_sm80_float16_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x65536_sm80_float16_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_128x64x65536_sm80_float16_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x64x65536_sm80_float16_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / float16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna2d_backward_64x64x32_sm80_float16_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x32_sm80_float16_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna2d_backward_64x64x64_sm80_float16_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x64_sm80_float16_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna2d_backward_64x64x128_sm80_float16_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x128_sm80_float16_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::kMinBlocksPerSm)
fna2d_backward_128x128x128_sm80_float16_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x128x128_sm80_float16_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_64x64x65536_sm80_float16_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x65536_sm80_float16_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_128x64x65536_sm80_float16_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x64x65536_sm80_float16_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna2d_backward_64x64x32_sm80_bfloat16_cm_0_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x32_sm80_bfloat16_cm_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna2d_backward_64x64x64_sm80_bfloat16_cm_0_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x64_sm80_bfloat16_cm_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna2d_backward_64x64x128_sm80_bfloat16_cm_0_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x128_sm80_bfloat16_cm_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128>::kMinBlocksPerSm)
fna2d_backward_128x128x128_sm80_bfloat16_cm_0_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x128x128_sm80_bfloat16_cm_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_64x64x65536_sm80_bfloat16_cm_0_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x65536_sm80_bfloat16_cm_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_128x64x65536_sm80_bfloat16_cm_0_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x64x65536_sm80_bfloat16_cm_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna2d_backward_64x64x32_sm80_bfloat16_cm_0_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x32_sm80_bfloat16_cm_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna2d_backward_64x64x64_sm80_bfloat16_cm_0_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x64_sm80_bfloat16_cm_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna2d_backward_64x64x128_sm80_bfloat16_cm_0_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x128_sm80_bfloat16_cm_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128>::kMinBlocksPerSm)
fna2d_backward_128x128x128_sm80_bfloat16_cm_0_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x128x128_sm80_bfloat16_cm_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_64x64x65536_sm80_bfloat16_cm_0_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x65536_sm80_bfloat16_cm_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_128x64x65536_sm80_bfloat16_cm_0_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x64x65536_sm80_bfloat16_cm_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna2d_backward_64x64x32_sm80_bfloat16_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x32_sm80_bfloat16_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna2d_backward_64x64x64_sm80_bfloat16_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x64_sm80_bfloat16_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna2d_backward_64x64x128_sm80_bfloat16_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x128_sm80_bfloat16_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128>::kMinBlocksPerSm)
fna2d_backward_128x128x128_sm80_bfloat16_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x128x128_sm80_bfloat16_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_64x64x65536_sm80_bfloat16_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x65536_sm80_bfloat16_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_128x64x65536_sm80_bfloat16_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x64x65536_sm80_bfloat16_cm_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna2d_backward_64x64x32_sm80_bfloat16_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x32_sm80_bfloat16_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna2d_backward_64x64x64_sm80_bfloat16_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x64_sm80_bfloat16_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna2d_backward_64x64x128_sm80_bfloat16_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x128_sm80_bfloat16_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128>::kMinBlocksPerSm)
fna2d_backward_128x128x128_sm80_bfloat16_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x128x128_sm80_bfloat16_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_64x64x65536_sm80_bfloat16_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x65536_sm80_bfloat16_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_128x64x65536_sm80_bfloat16_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_128x64x65536_sm80_bfloat16_cm_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / float32 / SM50
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, false>, float, cutlass::arch::Sm50, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, false>, float, cutlass::arch::Sm50, true, 64, 64, 32>::kMinBlocksPerSm)
fna3d_backward_64x64x32_sm50_float32_cm_0_0_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, false>, float, cutlass::arch::Sm50, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, false>, float, cutlass::arch::Sm50, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x32_sm50_float32_cm_0_0_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, false>, float, cutlass::arch::Sm50, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, false>, float, cutlass::arch::Sm50, true, 64, 64, 64>::kMinBlocksPerSm)
fna3d_backward_64x64x64_sm50_float32_cm_0_0_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, false>, float, cutlass::arch::Sm50, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, false>, float, cutlass::arch::Sm50, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x64_sm50_float32_cm_0_0_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, false>, float, cutlass::arch::Sm50, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, false>, float, cutlass::arch::Sm50, true, 64, 64, 128>::kMinBlocksPerSm)
fna3d_backward_64x64x128_sm50_float32_cm_0_0_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, false>, float, cutlass::arch::Sm50, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, false>, float, cutlass::arch::Sm50, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x128_sm50_float32_cm_0_0_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, false>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, false>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::kMinBlocksPerSm)
fna3d_backward_64x64x65536_sm50_float32_cm_0_0_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, false>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, false>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x65536_sm50_float32_cm_0_0_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / float32 / SM50
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, true>, float, cutlass::arch::Sm50, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, true>, float, cutlass::arch::Sm50, true, 64, 64, 32>::kMinBlocksPerSm)
fna3d_backward_64x64x32_sm50_float32_cm_0_0_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, true>, float, cutlass::arch::Sm50, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, true>, float, cutlass::arch::Sm50, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x32_sm50_float32_cm_0_0_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, true>, float, cutlass::arch::Sm50, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, true>, float, cutlass::arch::Sm50, true, 64, 64, 64>::kMinBlocksPerSm)
fna3d_backward_64x64x64_sm50_float32_cm_0_0_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, true>, float, cutlass::arch::Sm50, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, true>, float, cutlass::arch::Sm50, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x64_sm50_float32_cm_0_0_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, true>, float, cutlass::arch::Sm50, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, true>, float, cutlass::arch::Sm50, true, 64, 64, 128>::kMinBlocksPerSm)
fna3d_backward_64x64x128_sm50_float32_cm_0_0_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, true>, float, cutlass::arch::Sm50, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, true>, float, cutlass::arch::Sm50, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x128_sm50_float32_cm_0_0_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, true>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, true>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::kMinBlocksPerSm)
fna3d_backward_64x64x65536_sm50_float32_cm_0_0_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, true>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, false, true>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x65536_sm50_float32_cm_0_0_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / float32 / SM50
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, false>, float, cutlass::arch::Sm50, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, false>, float, cutlass::arch::Sm50, true, 64, 64, 32>::kMinBlocksPerSm)
fna3d_backward_64x64x32_sm50_float32_cm_0_1_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, false>, float, cutlass::arch::Sm50, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, false>, float, cutlass::arch::Sm50, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x32_sm50_float32_cm_0_1_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, false>, float, cutlass::arch::Sm50, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, false>, float, cutlass::arch::Sm50, true, 64, 64, 64>::kMinBlocksPerSm)
fna3d_backward_64x64x64_sm50_float32_cm_0_1_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, false>, float, cutlass::arch::Sm50, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, false>, float, cutlass::arch::Sm50, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x64_sm50_float32_cm_0_1_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, false>, float, cutlass::arch::Sm50, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, false>, float, cutlass::arch::Sm50, true, 64, 64, 128>::kMinBlocksPerSm)
fna3d_backward_64x64x128_sm50_float32_cm_0_1_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, false>, float, cutlass::arch::Sm50, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, false>, float, cutlass::arch::Sm50, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x128_sm50_float32_cm_0_1_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, false>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, false>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::kMinBlocksPerSm)
fna3d_backward_64x64x65536_sm50_float32_cm_0_1_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, false>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, false>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x65536_sm50_float32_cm_0_1_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / float32 / SM50
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, true>, float, cutlass::arch::Sm50, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, true>, float, cutlass::arch::Sm50, true, 64, 64, 32>::kMinBlocksPerSm)
fna3d_backward_64x64x32_sm50_float32_cm_0_1_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, true>, float, cutlass::arch::Sm50, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, true>, float, cutlass::arch::Sm50, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x32_sm50_float32_cm_0_1_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, true>, float, cutlass::arch::Sm50, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, true>, float, cutlass::arch::Sm50, true, 64, 64, 64>::kMinBlocksPerSm)
fna3d_backward_64x64x64_sm50_float32_cm_0_1_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, true>, float, cutlass::arch::Sm50, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, true>, float, cutlass::arch::Sm50, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x64_sm50_float32_cm_0_1_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, true>, float, cutlass::arch::Sm50, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, true>, float, cutlass::arch::Sm50, true, 64, 64, 128>::kMinBlocksPerSm)
fna3d_backward_64x64x128_sm50_float32_cm_0_1_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, true>, float, cutlass::arch::Sm50, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, true>, float, cutlass::arch::Sm50, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x128_sm50_float32_cm_0_1_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, true>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, true>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::kMinBlocksPerSm)
fna3d_backward_64x64x65536_sm50_float32_cm_0_1_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, true>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<false, true, true>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x65536_sm50_float32_cm_0_1_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
} 
} 
} 

