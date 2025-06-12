#include <cuda_runtime.h>
#include <iostream>
#include <natten/cuda/fna/kernel_forward.h>
#include <natten_autogen/cuda/fna/kernels.h>
namespace natten { 
namespace cuda { 
namespace fna { 



///////////////////////////////////////////////////////////////////
// FNA-3D / float16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 32>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 32>::kMinBlocksPerSm)
fna3d_32x128x32_sm80_float16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x32_sm80_float16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna3d_64x64x32_sm80_float16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x32_sm80_float16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 32>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 32>::kMinBlocksPerSm)
fna3d_64x128x32_sm80_float16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x32_sm80_float16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64>::kMinBlocksPerSm)
fna3d_32x128x64_sm80_float16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x64_sm80_float16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna3d_64x64x64_sm80_float16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x64_sm80_float16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64>::kMinBlocksPerSm)
fna3d_64x128x64_sm80_float16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x64_sm80_float16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 128>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 128>::kMinBlocksPerSm)
fna3d_32x128x128_sm80_float16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x128_sm80_float16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna3d_64x64x128_sm80_float16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x128_sm80_float16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128>::kMinBlocksPerSm)
fna3d_64x128x128_sm80_float16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x128_sm80_float16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536>::kMinBlocksPerSm)
fna3d_32x128x65536_sm80_float16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x65536_sm80_float16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 65536>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 65536>::kMinBlocksPerSm)
fna3d_64x128x65536_sm80_float16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x65536_sm80_float16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna3d_64x64x65536_sm80_float16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x65536_sm80_float16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / float16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 32>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 32>::kMinBlocksPerSm)
fna3d_32x128x32_sm80_float16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x32_sm80_float16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna3d_64x64x32_sm80_float16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x32_sm80_float16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 32>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 32>::kMinBlocksPerSm)
fna3d_64x128x32_sm80_float16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x32_sm80_float16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64>::kMinBlocksPerSm)
fna3d_32x128x64_sm80_float16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x64_sm80_float16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna3d_64x64x64_sm80_float16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x64_sm80_float16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64>::kMinBlocksPerSm)
fna3d_64x128x64_sm80_float16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x64_sm80_float16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 128>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 128>::kMinBlocksPerSm)
fna3d_32x128x128_sm80_float16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x128_sm80_float16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna3d_64x64x128_sm80_float16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x128_sm80_float16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128>::kMinBlocksPerSm)
fna3d_64x128x128_sm80_float16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x128_sm80_float16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536>::kMinBlocksPerSm)
fna3d_32x128x65536_sm80_float16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x65536_sm80_float16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 65536>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 65536>::kMinBlocksPerSm)
fna3d_64x128x65536_sm80_float16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x65536_sm80_float16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna3d_64x64x65536_sm80_float16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x65536_sm80_float16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
} 
} 
} 

