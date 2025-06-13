#include <cuda_runtime.h>
#include <iostream>
#include <natten/cuda/fna/kernel_backward.h>
#include <natten_autogen/cuda/fna/kernels.h>
namespace natten { 
namespace cuda { 
namespace fna { 



///////////////////////////////////////////////////////////////////
// FNA-3D / float16 / SM50
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::kMinBlocksPerSm)
fna3d_backward_64x64x32_sm50_float16_cm_1_0_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x32_sm50_float16_cm_1_0_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::kMinBlocksPerSm)
fna3d_backward_64x64x64_sm50_float16_cm_1_0_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x64_sm50_float16_cm_1_0_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::kMinBlocksPerSm)
fna3d_backward_64x64x128_sm50_float16_cm_1_0_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x128_sm50_float16_cm_1_0_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::kMinBlocksPerSm)
fna3d_backward_64x64x65536_sm50_float16_cm_1_0_0(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x65536_sm50_float16_cm_1_0_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-3D / float16 / SM50
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::kMinBlocksPerSm)
fna3d_backward_64x64x32_sm50_float16_cm_1_0_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x32_sm50_float16_cm_1_0_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::kMinBlocksPerSm)
fna3d_backward_64x64x64_sm50_float16_cm_1_0_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x64_sm50_float16_cm_1_0_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::kMinBlocksPerSm)
fna3d_backward_64x64x128_sm50_float16_cm_1_0_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x128_sm50_float16_cm_1_0_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::kMinBlocksPerSm)
fna3d_backward_64x64x65536_sm50_float16_cm_1_0_1(typename FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<3, CausalMask<true, false, true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna3d_backward_64x64x65536_sm50_float16_cm_1_0_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
} 
} 
} 

