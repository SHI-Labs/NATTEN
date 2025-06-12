#include <cuda_runtime.h>
#include <iostream>
#include <natten/cuda/fna/kernel_backward.h>
#include <natten_autogen/cuda/fna/kernels.h>
namespace natten { 
namespace cuda { 
namespace fna { 



///////////////////////////////////////////////////////////////////
// FNA-3D / float32 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna3d_backward_64x64x32_sm80_float32_cm_1_1_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x32_sm80_float32_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna3d_backward_64x64x64_sm80_float32_cm_1_1_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x64_sm80_float32_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna3d_backward_64x64x128_sm80_float32_cm_1_1_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x128_sm80_float32_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 128, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 128, 64, 128>::kMinBlocksPerSm)
fna3d_backward_128x64x128_sm80_float32_cm_1_1_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 128, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 128, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_128x64x128_sm80_float32_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna3d_backward_64x64x65536_sm80_float32_cm_1_1_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x65536_sm80_float32_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::kMinBlocksPerSm)
fna3d_backward_128x64x65536_sm80_float32_cm_1_1_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, false>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_128x64x65536_sm80_float32_cm_1_1_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / float32 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna3d_backward_64x64x32_sm80_float32_cm_1_1_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x32_sm80_float32_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna3d_backward_64x64x64_sm80_float32_cm_1_1_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x64_sm80_float32_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna3d_backward_64x64x128_sm80_float32_cm_1_1_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x128_sm80_float32_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 128, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 128, 64, 128>::kMinBlocksPerSm)
fna3d_backward_128x64x128_sm80_float32_cm_1_1_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 128, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 128, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_128x64x128_sm80_float32_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna3d_backward_64x64x65536_sm80_float32_cm_1_1_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x65536_sm80_float32_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::kMinBlocksPerSm)
fna3d_backward_128x64x65536_sm80_float32_cm_1_1_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, true, true>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_128x64x65536_sm80_float32_cm_1_1_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
} 
} 
} 

