#include <cuda_runtime.h>
#include <iostream>
#include <natten/dtypes.cuh>
#include <natten/cuda/fna/kernel_backward.h>
#include <natten/cuda/fna/kernel_forward.h>
#include <natten_autogen/cuda/fna/kernels.h>
namespace natten { 
namespace cuda { 
namespace fna { 



///////////////////////////////////////////////////////////////////
// FNA-3D / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::kMinBlocksPerSm)
fna3d_32x128x32_sm80_bfloat16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x32_sm80_bfloat16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::kMinBlocksPerSm)
fna3d_64x64x32_sm80_bfloat16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x32_sm80_bfloat16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::kMinBlocksPerSm)
fna3d_64x128x32_sm80_bfloat16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x32_sm80_bfloat16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::kMinBlocksPerSm)
fna3d_32x128x64_sm80_bfloat16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x64_sm80_bfloat16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::kMinBlocksPerSm)
fna3d_64x64x64_sm80_bfloat16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x64_sm80_bfloat16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::kMinBlocksPerSm)
fna3d_64x128x64_sm80_bfloat16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x64_sm80_bfloat16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::kMinBlocksPerSm)
fna3d_32x128x128_sm80_bfloat16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x128_sm80_bfloat16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::kMinBlocksPerSm)
fna3d_64x64x128_sm80_bfloat16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x128_sm80_bfloat16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::kMinBlocksPerSm)
fna3d_64x128x128_sm80_bfloat16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x128_sm80_bfloat16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::kMinBlocksPerSm)
fna3d_32x128x65536_sm80_bfloat16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x65536_sm80_bfloat16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::kMinBlocksPerSm)
fna3d_64x128x65536_sm80_bfloat16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x65536_sm80_bfloat16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::kMinBlocksPerSm)
fna3d_64x64x65536_sm80_bfloat16_cm_0_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x65536_sm80_bfloat16_cm_0_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::kMinBlocksPerSm)
fna3d_32x128x32_sm80_bfloat16_cm_0_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x32_sm80_bfloat16_cm_0_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::kMinBlocksPerSm)
fna3d_64x64x32_sm80_bfloat16_cm_0_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x32_sm80_bfloat16_cm_0_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::kMinBlocksPerSm)
fna3d_64x128x32_sm80_bfloat16_cm_0_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x32_sm80_bfloat16_cm_0_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::kMinBlocksPerSm)
fna3d_32x128x64_sm80_bfloat16_cm_0_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x64_sm80_bfloat16_cm_0_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::kMinBlocksPerSm)
fna3d_64x64x64_sm80_bfloat16_cm_0_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x64_sm80_bfloat16_cm_0_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::kMinBlocksPerSm)
fna3d_64x128x64_sm80_bfloat16_cm_0_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x64_sm80_bfloat16_cm_0_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::kMinBlocksPerSm)
fna3d_32x128x128_sm80_bfloat16_cm_0_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x128_sm80_bfloat16_cm_0_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::kMinBlocksPerSm)
fna3d_64x64x128_sm80_bfloat16_cm_0_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x128_sm80_bfloat16_cm_0_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::kMinBlocksPerSm)
fna3d_64x128x128_sm80_bfloat16_cm_0_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x128_sm80_bfloat16_cm_0_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::kMinBlocksPerSm)
fna3d_32x128x65536_sm80_bfloat16_cm_0_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x65536_sm80_bfloat16_cm_0_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::kMinBlocksPerSm)
fna3d_64x128x65536_sm80_bfloat16_cm_0_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x65536_sm80_bfloat16_cm_0_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::kMinBlocksPerSm)
fna3d_64x64x65536_sm80_bfloat16_cm_0_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x65536_sm80_bfloat16_cm_0_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::kMinBlocksPerSm)
fna3d_32x128x32_sm80_bfloat16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x32_sm80_bfloat16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::kMinBlocksPerSm)
fna3d_64x64x32_sm80_bfloat16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x32_sm80_bfloat16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::kMinBlocksPerSm)
fna3d_64x128x32_sm80_bfloat16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x32_sm80_bfloat16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::kMinBlocksPerSm)
fna3d_32x128x64_sm80_bfloat16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x64_sm80_bfloat16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::kMinBlocksPerSm)
fna3d_64x64x64_sm80_bfloat16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x64_sm80_bfloat16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::kMinBlocksPerSm)
fna3d_64x128x64_sm80_bfloat16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x64_sm80_bfloat16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::kMinBlocksPerSm)
fna3d_32x128x128_sm80_bfloat16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x128_sm80_bfloat16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::kMinBlocksPerSm)
fna3d_64x64x128_sm80_bfloat16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x128_sm80_bfloat16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::kMinBlocksPerSm)
fna3d_64x128x128_sm80_bfloat16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x128_sm80_bfloat16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::kMinBlocksPerSm)
fna3d_32x128x65536_sm80_bfloat16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x65536_sm80_bfloat16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::kMinBlocksPerSm)
fna3d_64x128x65536_sm80_bfloat16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x65536_sm80_bfloat16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::kMinBlocksPerSm)
fna3d_64x64x65536_sm80_bfloat16_cm_0_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x65536_sm80_bfloat16_cm_0_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::kMinBlocksPerSm)
fna3d_32x128x32_sm80_bfloat16_cm_0_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x32_sm80_bfloat16_cm_0_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::kMinBlocksPerSm)
fna3d_64x64x32_sm80_bfloat16_cm_0_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x32_sm80_bfloat16_cm_0_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::kMinBlocksPerSm)
fna3d_64x128x32_sm80_bfloat16_cm_0_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x32_sm80_bfloat16_cm_0_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::kMinBlocksPerSm)
fna3d_32x128x64_sm80_bfloat16_cm_0_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x64_sm80_bfloat16_cm_0_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::kMinBlocksPerSm)
fna3d_64x64x64_sm80_bfloat16_cm_0_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x64_sm80_bfloat16_cm_0_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::kMinBlocksPerSm)
fna3d_64x128x64_sm80_bfloat16_cm_0_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x64_sm80_bfloat16_cm_0_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::kMinBlocksPerSm)
fna3d_32x128x128_sm80_bfloat16_cm_0_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x128_sm80_bfloat16_cm_0_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::kMinBlocksPerSm)
fna3d_64x64x128_sm80_bfloat16_cm_0_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x128_sm80_bfloat16_cm_0_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::kMinBlocksPerSm)
fna3d_64x128x128_sm80_bfloat16_cm_0_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x128_sm80_bfloat16_cm_0_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::kMinBlocksPerSm)
fna3d_32x128x65536_sm80_bfloat16_cm_0_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x65536_sm80_bfloat16_cm_0_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::kMinBlocksPerSm)
fna3d_64x128x65536_sm80_bfloat16_cm_0_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x65536_sm80_bfloat16_cm_0_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::kMinBlocksPerSm)
fna3d_64x64x65536_sm80_bfloat16_cm_0_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<false, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x65536_sm80_bfloat16_cm_0_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::kMinBlocksPerSm)
fna3d_32x128x32_sm80_bfloat16_cm_1_0_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x32_sm80_bfloat16_cm_1_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::kMinBlocksPerSm)
fna3d_64x64x32_sm80_bfloat16_cm_1_0_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x32_sm80_bfloat16_cm_1_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::kMinBlocksPerSm)
fna3d_64x128x32_sm80_bfloat16_cm_1_0_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x32_sm80_bfloat16_cm_1_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::kMinBlocksPerSm)
fna3d_32x128x64_sm80_bfloat16_cm_1_0_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x64_sm80_bfloat16_cm_1_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::kMinBlocksPerSm)
fna3d_64x64x64_sm80_bfloat16_cm_1_0_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x64_sm80_bfloat16_cm_1_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::kMinBlocksPerSm)
fna3d_64x128x64_sm80_bfloat16_cm_1_0_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x64_sm80_bfloat16_cm_1_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::kMinBlocksPerSm)
fna3d_32x128x128_sm80_bfloat16_cm_1_0_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x128_sm80_bfloat16_cm_1_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::kMinBlocksPerSm)
fna3d_64x64x128_sm80_bfloat16_cm_1_0_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x128_sm80_bfloat16_cm_1_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::kMinBlocksPerSm)
fna3d_64x128x128_sm80_bfloat16_cm_1_0_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x128_sm80_bfloat16_cm_1_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::kMinBlocksPerSm)
fna3d_32x128x65536_sm80_bfloat16_cm_1_0_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x65536_sm80_bfloat16_cm_1_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::kMinBlocksPerSm)
fna3d_64x128x65536_sm80_bfloat16_cm_1_0_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x65536_sm80_bfloat16_cm_1_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::kMinBlocksPerSm)
fna3d_64x64x65536_sm80_bfloat16_cm_1_0_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x65536_sm80_bfloat16_cm_1_0_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::kMinBlocksPerSm)
fna3d_32x128x32_sm80_bfloat16_cm_1_0_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x32_sm80_bfloat16_cm_1_0_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::kMinBlocksPerSm)
fna3d_64x64x32_sm80_bfloat16_cm_1_0_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x32_sm80_bfloat16_cm_1_0_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::kMinBlocksPerSm)
fna3d_64x128x32_sm80_bfloat16_cm_1_0_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x32_sm80_bfloat16_cm_1_0_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::kMinBlocksPerSm)
fna3d_32x128x64_sm80_bfloat16_cm_1_0_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x64_sm80_bfloat16_cm_1_0_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::kMinBlocksPerSm)
fna3d_64x64x64_sm80_bfloat16_cm_1_0_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x64_sm80_bfloat16_cm_1_0_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::kMinBlocksPerSm)
fna3d_64x128x64_sm80_bfloat16_cm_1_0_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x64_sm80_bfloat16_cm_1_0_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::kMinBlocksPerSm)
fna3d_32x128x128_sm80_bfloat16_cm_1_0_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x128_sm80_bfloat16_cm_1_0_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::kMinBlocksPerSm)
fna3d_64x64x128_sm80_bfloat16_cm_1_0_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x128_sm80_bfloat16_cm_1_0_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::kMinBlocksPerSm)
fna3d_64x128x128_sm80_bfloat16_cm_1_0_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x128_sm80_bfloat16_cm_1_0_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::kMinBlocksPerSm)
fna3d_32x128x65536_sm80_bfloat16_cm_1_0_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x65536_sm80_bfloat16_cm_1_0_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::kMinBlocksPerSm)
fna3d_64x128x65536_sm80_bfloat16_cm_1_0_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x65536_sm80_bfloat16_cm_1_0_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::kMinBlocksPerSm)
fna3d_64x64x65536_sm80_bfloat16_cm_1_0_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x65536_sm80_bfloat16_cm_1_0_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::kMinBlocksPerSm)
fna3d_32x128x32_sm80_bfloat16_cm_1_0_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x32_sm80_bfloat16_cm_1_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::kMinBlocksPerSm)
fna3d_64x64x32_sm80_bfloat16_cm_1_0_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x32_sm80_bfloat16_cm_1_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::kMinBlocksPerSm)
fna3d_64x128x32_sm80_bfloat16_cm_1_0_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x32_sm80_bfloat16_cm_1_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::kMinBlocksPerSm)
fna3d_32x128x64_sm80_bfloat16_cm_1_0_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x64_sm80_bfloat16_cm_1_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::kMinBlocksPerSm)
fna3d_64x64x64_sm80_bfloat16_cm_1_0_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x64_sm80_bfloat16_cm_1_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::kMinBlocksPerSm)
fna3d_64x128x64_sm80_bfloat16_cm_1_0_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x64_sm80_bfloat16_cm_1_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::kMinBlocksPerSm)
fna3d_32x128x128_sm80_bfloat16_cm_1_0_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x128_sm80_bfloat16_cm_1_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::kMinBlocksPerSm)
fna3d_64x64x128_sm80_bfloat16_cm_1_0_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x128_sm80_bfloat16_cm_1_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::kMinBlocksPerSm)
fna3d_64x128x128_sm80_bfloat16_cm_1_0_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x128_sm80_bfloat16_cm_1_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::kMinBlocksPerSm)
fna3d_32x128x65536_sm80_bfloat16_cm_1_0_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x65536_sm80_bfloat16_cm_1_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::kMinBlocksPerSm)
fna3d_64x128x65536_sm80_bfloat16_cm_1_0_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x65536_sm80_bfloat16_cm_1_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::kMinBlocksPerSm)
fna3d_64x64x65536_sm80_bfloat16_cm_1_0_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x65536_sm80_bfloat16_cm_1_0_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::kMinBlocksPerSm)
fna3d_32x128x32_sm80_bfloat16_cm_1_0_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x32_sm80_bfloat16_cm_1_0_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::kMinBlocksPerSm)
fna3d_64x64x32_sm80_bfloat16_cm_1_0_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x32_sm80_bfloat16_cm_1_0_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::kMinBlocksPerSm)
fna3d_64x128x32_sm80_bfloat16_cm_1_0_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x32_sm80_bfloat16_cm_1_0_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::kMinBlocksPerSm)
fna3d_32x128x64_sm80_bfloat16_cm_1_0_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x64_sm80_bfloat16_cm_1_0_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::kMinBlocksPerSm)
fna3d_64x64x64_sm80_bfloat16_cm_1_0_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x64_sm80_bfloat16_cm_1_0_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::kMinBlocksPerSm)
fna3d_64x128x64_sm80_bfloat16_cm_1_0_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x64_sm80_bfloat16_cm_1_0_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::kMinBlocksPerSm)
fna3d_32x128x128_sm80_bfloat16_cm_1_0_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x128_sm80_bfloat16_cm_1_0_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::kMinBlocksPerSm)
fna3d_64x64x128_sm80_bfloat16_cm_1_0_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x128_sm80_bfloat16_cm_1_0_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::kMinBlocksPerSm)
fna3d_64x128x128_sm80_bfloat16_cm_1_0_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x128_sm80_bfloat16_cm_1_0_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::kMinBlocksPerSm)
fna3d_32x128x65536_sm80_bfloat16_cm_1_0_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x65536_sm80_bfloat16_cm_1_0_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::kMinBlocksPerSm)
fna3d_64x128x65536_sm80_bfloat16_cm_1_0_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x65536_sm80_bfloat16_cm_1_0_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::kMinBlocksPerSm)
fna3d_64x64x65536_sm80_bfloat16_cm_1_0_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, false, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x65536_sm80_bfloat16_cm_1_0_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::kMinBlocksPerSm)
fna3d_32x128x32_sm80_bfloat16_cm_1_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x32_sm80_bfloat16_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::kMinBlocksPerSm)
fna3d_64x64x32_sm80_bfloat16_cm_1_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x32_sm80_bfloat16_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::kMinBlocksPerSm)
fna3d_64x128x32_sm80_bfloat16_cm_1_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x32_sm80_bfloat16_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::kMinBlocksPerSm)
fna3d_32x128x64_sm80_bfloat16_cm_1_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x64_sm80_bfloat16_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::kMinBlocksPerSm)
fna3d_64x64x64_sm80_bfloat16_cm_1_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x64_sm80_bfloat16_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::kMinBlocksPerSm)
fna3d_64x128x64_sm80_bfloat16_cm_1_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x64_sm80_bfloat16_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::kMinBlocksPerSm)
fna3d_32x128x128_sm80_bfloat16_cm_1_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x128_sm80_bfloat16_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::kMinBlocksPerSm)
fna3d_64x64x128_sm80_bfloat16_cm_1_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x128_sm80_bfloat16_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::kMinBlocksPerSm)
fna3d_64x128x128_sm80_bfloat16_cm_1_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x128_sm80_bfloat16_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::kMinBlocksPerSm)
fna3d_32x128x65536_sm80_bfloat16_cm_1_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x65536_sm80_bfloat16_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::kMinBlocksPerSm)
fna3d_64x128x65536_sm80_bfloat16_cm_1_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x65536_sm80_bfloat16_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::kMinBlocksPerSm)
fna3d_64x64x65536_sm80_bfloat16_cm_1_1_0(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x65536_sm80_bfloat16_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::kMinBlocksPerSm)
fna3d_32x128x32_sm80_bfloat16_cm_1_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x32_sm80_bfloat16_cm_1_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::kMinBlocksPerSm)
fna3d_64x64x32_sm80_bfloat16_cm_1_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x32_sm80_bfloat16_cm_1_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::kMinBlocksPerSm)
fna3d_64x128x32_sm80_bfloat16_cm_1_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x32_sm80_bfloat16_cm_1_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::kMinBlocksPerSm)
fna3d_32x128x64_sm80_bfloat16_cm_1_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x64_sm80_bfloat16_cm_1_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::kMinBlocksPerSm)
fna3d_64x64x64_sm80_bfloat16_cm_1_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x64_sm80_bfloat16_cm_1_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::kMinBlocksPerSm)
fna3d_64x128x64_sm80_bfloat16_cm_1_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x64_sm80_bfloat16_cm_1_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::kMinBlocksPerSm)
fna3d_32x128x128_sm80_bfloat16_cm_1_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x128_sm80_bfloat16_cm_1_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::kMinBlocksPerSm)
fna3d_64x64x128_sm80_bfloat16_cm_1_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x128_sm80_bfloat16_cm_1_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::kMinBlocksPerSm)
fna3d_64x128x128_sm80_bfloat16_cm_1_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x128_sm80_bfloat16_cm_1_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::kMinBlocksPerSm)
fna3d_32x128x65536_sm80_bfloat16_cm_1_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x65536_sm80_bfloat16_cm_1_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::kMinBlocksPerSm)
fna3d_64x128x65536_sm80_bfloat16_cm_1_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x65536_sm80_bfloat16_cm_1_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::kMinBlocksPerSm)
fna3d_64x64x65536_sm80_bfloat16_cm_1_1_0_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, false>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x65536_sm80_bfloat16_cm_1_1_0_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::kMinBlocksPerSm)
fna3d_32x128x32_sm80_bfloat16_cm_1_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x32_sm80_bfloat16_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::kMinBlocksPerSm)
fna3d_64x64x32_sm80_bfloat16_cm_1_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x32_sm80_bfloat16_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::kMinBlocksPerSm)
fna3d_64x128x32_sm80_bfloat16_cm_1_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x32_sm80_bfloat16_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::kMinBlocksPerSm)
fna3d_32x128x64_sm80_bfloat16_cm_1_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x64_sm80_bfloat16_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::kMinBlocksPerSm)
fna3d_64x64x64_sm80_bfloat16_cm_1_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x64_sm80_bfloat16_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::kMinBlocksPerSm)
fna3d_64x128x64_sm80_bfloat16_cm_1_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x64_sm80_bfloat16_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::kMinBlocksPerSm)
fna3d_32x128x128_sm80_bfloat16_cm_1_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x128_sm80_bfloat16_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::kMinBlocksPerSm)
fna3d_64x64x128_sm80_bfloat16_cm_1_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x128_sm80_bfloat16_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::kMinBlocksPerSm)
fna3d_64x128x128_sm80_bfloat16_cm_1_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x128_sm80_bfloat16_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::kMinBlocksPerSm)
fna3d_32x128x65536_sm80_bfloat16_cm_1_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x65536_sm80_bfloat16_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::kMinBlocksPerSm)
fna3d_64x128x65536_sm80_bfloat16_cm_1_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x65536_sm80_bfloat16_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::kMinBlocksPerSm)
fna3d_64x64x65536_sm80_bfloat16_cm_1_1_1(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x65536_sm80_bfloat16_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::kMinBlocksPerSm)
fna3d_32x128x32_sm80_bfloat16_cm_1_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x32_sm80_bfloat16_cm_1_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::kMinBlocksPerSm)
fna3d_64x64x32_sm80_bfloat16_cm_1_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x32_sm80_bfloat16_cm_1_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::kMinBlocksPerSm)
fna3d_64x128x32_sm80_bfloat16_cm_1_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x32_sm80_bfloat16_cm_1_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::kMinBlocksPerSm)
fna3d_32x128x64_sm80_bfloat16_cm_1_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x64_sm80_bfloat16_cm_1_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::kMinBlocksPerSm)
fna3d_64x64x64_sm80_bfloat16_cm_1_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x64_sm80_bfloat16_cm_1_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::kMinBlocksPerSm)
fna3d_64x128x64_sm80_bfloat16_cm_1_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x64_sm80_bfloat16_cm_1_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::kMinBlocksPerSm)
fna3d_32x128x128_sm80_bfloat16_cm_1_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x128_sm80_bfloat16_cm_1_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::kMinBlocksPerSm)
fna3d_64x64x128_sm80_bfloat16_cm_1_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x128_sm80_bfloat16_cm_1_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::kMinBlocksPerSm)
fna3d_64x128x128_sm80_bfloat16_cm_1_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x128_sm80_bfloat16_cm_1_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::kMinBlocksPerSm)
fna3d_32x128x65536_sm80_bfloat16_cm_1_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_32x128x65536_sm80_bfloat16_cm_1_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::kMinBlocksPerSm)
fna3d_64x128x65536_sm80_bfloat16_cm_1_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x128x65536_sm80_bfloat16_cm_1_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::kNumThreads,
    FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::kMinBlocksPerSm)
fna3d_64x64x65536_sm80_bfloat16_cm_1_1_1_lse(typename FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionKernel<3, CausalMask<true, true, true>, cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false, true>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_64x64x65536_sm80_bfloat16_cm_1_1_1_lse` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float32 / SM50
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm50, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm50, true, 64, 64, 32>::kMinBlocksPerSm)
fna1d_backward_64x64x32_sm50_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm50, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm50, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x32_sm50_float32_cm_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm50, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm50, true, 64, 64, 64>::kMinBlocksPerSm)
fna1d_backward_64x64x64_sm50_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm50, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm50, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x64_sm50_float32_cm_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm50, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm50, true, 64, 64, 128>::kMinBlocksPerSm)
fna1d_backward_64x64x128_sm50_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm50, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm50, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x128_sm50_float32_cm_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_64x64x65536_sm50_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x65536_sm50_float32_cm_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float32 / SM50
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm50, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm50, true, 64, 64, 32>::kMinBlocksPerSm)
fna1d_backward_64x64x32_sm50_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm50, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm50, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x32_sm50_float32_cm_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm50, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm50, true, 64, 64, 64>::kMinBlocksPerSm)
fna1d_backward_64x64x64_sm50_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm50, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm50, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x64_sm50_float32_cm_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm50, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm50, true, 64, 64, 128>::kMinBlocksPerSm)
fna1d_backward_64x64x128_sm50_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm50, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm50, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x128_sm50_float32_cm_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_64x64x65536_sm50_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm50, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x65536_sm50_float32_cm_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
} 
} 
} 

