#include <cuda_runtime.h>
#include <iostream>
#include <natten/cuda/fna/kernel_backward.h>
#include <natten_autogen/cuda/fna/kernels.h>
namespace natten { 
namespace cuda { 
namespace fna { 



///////////////////////////////////////////////////////////////////
// FNA-2D / float32 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 32>::kMinBlocksPerSm)
fna2d_backward_64x64x32_sm70_float32_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x32_sm70_float32_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 64>::kMinBlocksPerSm)
fna2d_backward_64x64x64_sm70_float32_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x64_sm70_float32_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 128>::kMinBlocksPerSm)
fna2d_backward_64x64x128_sm70_float32_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x128_sm70_float32_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_64x64x65536_sm70_float32_cm_1_0(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, false>, float, cutlass::arch::Sm70, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x65536_sm70_float32_cm_1_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-2D / float32 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 32>::kMinBlocksPerSm)
fna2d_backward_64x64x32_sm70_float32_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x32_sm70_float32_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 64>::kMinBlocksPerSm)
fna2d_backward_64x64x64_sm70_float32_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x64_sm70_float32_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 128>::kMinBlocksPerSm)
fna2d_backward_64x64x128_sm70_float32_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x128_sm70_float32_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 65536>::kMinBlocksPerSm)
fna2d_backward_64x64x65536_sm70_float32_cm_1_1(typename FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<2, CausalMask<true, true>, float, cutlass::arch::Sm70, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna2d_backward_64x64x65536_sm70_float32_cm_1_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
} 
} 
} 

