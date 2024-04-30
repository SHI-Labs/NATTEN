#include <cuda_runtime.h>
#include <iostream>
#include <natten/dtypes.cuh>
#include <natten/cuda/fna/kernel_backward.h>
#include <natten_autogen/cuda/fna/kernels.h>
namespace natten { 
namespace cuda { 
namespace fna { 



///////////////////////////////////////////////////////////////////
// FNA-1D / float16 / SM50
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::kMinBlocksPerSm)
fna1d_backward_64x64x32_sm50_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x32_sm50_float16_cm_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::kMinBlocksPerSm)
fna1d_backward_64x64x64_sm50_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x64_sm50_float16_cm_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::kMinBlocksPerSm)
fna1d_backward_64x64x128_sm50_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x128_sm50_float16_cm_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_64x64x65536_sm50_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x65536_sm50_float16_cm_0` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float16 / SM50
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::kMinBlocksPerSm)
fna1d_backward_64x64x32_sm50_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x32_sm50_float16_cm_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::kMinBlocksPerSm)
fna1d_backward_64x64x64_sm50_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x64_sm50_float16_cm_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::kMinBlocksPerSm)
fna1d_backward_64x64x128_sm50_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x128_sm50_float16_cm_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_64x64x65536_sm50_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x65536_sm50_float16_cm_1` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float32 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm70, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm70, true, 64, 64, 32>::kMinBlocksPerSm)
fna1d_backward_64x64x32_sm70_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm70, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm70, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x32_sm70_float32_cm_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm70, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm70, true, 64, 64, 64>::kMinBlocksPerSm)
fna1d_backward_64x64x64_sm70_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm70, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm70, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x64_sm70_float32_cm_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm70, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm70, true, 64, 64, 128>::kMinBlocksPerSm)
fna1d_backward_64x64x128_sm70_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm70, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm70, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x128_sm70_float32_cm_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm70, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm70, true, 64, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_64x64x65536_sm70_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm70, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm70, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x65536_sm70_float32_cm_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float32 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm70, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm70, true, 64, 64, 32>::kMinBlocksPerSm)
fna1d_backward_64x64x32_sm70_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm70, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm70, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x32_sm70_float32_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm70, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm70, true, 64, 64, 64>::kMinBlocksPerSm)
fna1d_backward_64x64x64_sm70_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm70, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm70, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x64_sm70_float32_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm70, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm70, true, 64, 64, 128>::kMinBlocksPerSm)
fna1d_backward_64x64x128_sm70_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm70, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm70, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x128_sm70_float32_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm70, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm70, true, 64, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_64x64x65536_sm70_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm70, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm70, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x65536_sm70_float32_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float16 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32>::kMinBlocksPerSm)
fna1d_backward_64x64x32_sm70_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x32_sm70_float16_cm_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64>::kMinBlocksPerSm)
fna1d_backward_64x64x64_sm70_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x64_sm70_float16_cm_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128>::kMinBlocksPerSm)
fna1d_backward_64x64x128_sm70_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x128_sm70_float16_cm_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 128>::kMinBlocksPerSm)
fna1d_backward_128x64x128_sm70_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_128x64x128_sm70_float16_cm_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_64x64x65536_sm70_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x65536_sm70_float16_cm_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_128x64x65536_sm70_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_128x64x65536_sm70_float16_cm_0` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float16 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32>::kMinBlocksPerSm)
fna1d_backward_64x64x32_sm70_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x32_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64>::kMinBlocksPerSm)
fna1d_backward_64x64x64_sm70_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x64_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128>::kMinBlocksPerSm)
fna1d_backward_64x64x128_sm70_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x128_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 128>::kMinBlocksPerSm)
fna1d_backward_128x64x128_sm70_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_128x64x128_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_64x64x65536_sm70_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x65536_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_128x64x65536_sm70_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_128x64x65536_sm70_float16_cm_1` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float32 / SM75
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 32>::kMinBlocksPerSm)
fna1d_backward_64x64x32_sm75_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x32_sm75_float32_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 64>::kMinBlocksPerSm)
fna1d_backward_64x64x64_sm75_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x64_sm75_float32_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 128>::kMinBlocksPerSm)
fna1d_backward_64x64x128_sm75_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x128_sm75_float32_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_64x64x65536_sm75_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm75, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x65536_sm75_float32_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float32 / SM75
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 32>::kMinBlocksPerSm)
fna1d_backward_64x64x32_sm75_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x32_sm75_float32_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 64>::kMinBlocksPerSm)
fna1d_backward_64x64x64_sm75_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x64_sm75_float32_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 128>::kMinBlocksPerSm)
fna1d_backward_64x64x128_sm75_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x128_sm75_float32_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_64x64x65536_sm75_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm75, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x65536_sm75_float32_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float16 / SM75
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32>::kMinBlocksPerSm)
fna1d_backward_64x64x32_sm75_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x32_sm75_float16_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64>::kMinBlocksPerSm)
fna1d_backward_64x64x64_sm75_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x64_sm75_float16_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128>::kMinBlocksPerSm)
fna1d_backward_64x64x128_sm75_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x128_sm75_float16_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_64x64x65536_sm75_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x65536_sm75_float16_cm_0` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float16 / SM75
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32>::kMinBlocksPerSm)
fna1d_backward_64x64x32_sm75_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x32_sm75_float16_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64>::kMinBlocksPerSm)
fna1d_backward_64x64x64_sm75_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x64_sm75_float16_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128>::kMinBlocksPerSm)
fna1d_backward_64x64x128_sm75_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x128_sm75_float16_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_64x64x65536_sm75_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x65536_sm75_float16_cm_1` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float32 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna1d_backward_64x64x32_sm80_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x32_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna1d_backward_64x64x64_sm80_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x64_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna1d_backward_64x64x128_sm80_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x128_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 128, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 128, 64, 128>::kMinBlocksPerSm)
fna1d_backward_128x64x128_sm80_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 128, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 128, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_128x64x128_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_64x64x65536_sm80_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x65536_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_128x64x65536_sm80_float32_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_128x64x65536_sm80_float32_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float32 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna1d_backward_64x64x32_sm80_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x32_sm80_float32_cm_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna1d_backward_64x64x64_sm80_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x64_sm80_float32_cm_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna1d_backward_64x64x128_sm80_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x128_sm80_float32_cm_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 128, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 128, 64, 128>::kMinBlocksPerSm)
fna1d_backward_128x64x128_sm80_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 128, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 128, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_128x64x128_sm80_float32_cm_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_64x64x65536_sm80_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x65536_sm80_float32_cm_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_128x64x65536_sm80_float32_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, float, cutlass::arch::Sm80, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_128x64x65536_sm80_float32_cm_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna1d_backward_64x64x32_sm80_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x32_sm80_float16_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna1d_backward_64x64x64_sm80_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x64_sm80_float16_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna1d_backward_64x64x128_sm80_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x128_sm80_float16_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::kMinBlocksPerSm)
fna1d_backward_128x128x128_sm80_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_128x128x128_sm80_float16_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_64x64x65536_sm80_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x65536_sm80_float16_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_128x64x65536_sm80_float16_cm_0(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<false>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_128x64x65536_sm80_float16_cm_0` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FNA-1D / float16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fna1d_backward_64x64x32_sm80_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x32_sm80_float16_cm_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fna1d_backward_64x64x64_sm80_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x64_sm80_float16_cm_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fna1d_backward_64x64x128_sm80_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x128_sm80_float16_cm_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::kMinBlocksPerSm)
fna1d_backward_128x128x128_sm80_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_128x128x128_sm80_float16_cm_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_64x64x65536_sm80_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_64x64x65536_sm80_float16_cm_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kNumThreads,
    FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::kMinBlocksPerSm)
fna1d_backward_128x64x65536_sm80_float16_cm_1(typename FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1000
  if (!p.advance_to_block()) {
    return;
  }
  FusedNeighborhoodAttentionBackwardKernel<1, CausalMask<true>, cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `fna1d_backward_128x64x65536_sm80_float16_cm_1` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
} 
} 
} 

