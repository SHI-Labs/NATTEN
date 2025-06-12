#include <cuda_runtime.h>
#include <iostream>
#include <natten/cuda/fmha/kernel_backward.h>
#include <natten_autogen/cuda/fmha/kernels.h>
namespace natten { 
namespace cuda { 
namespace fmha { 



///////////////////////////////////////////////////////////////////
// FMHA / float16 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::kMinBlocksPerSm)
fmha_backward_64x64x32_sm70_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_backward_64x64x32_sm70_float16` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::kMinBlocksPerSm)
fmha_backward_64x64x64_sm70_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_backward_64x64x64_sm70_float16` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::kMinBlocksPerSm)
fmha_backward_64x64x128_sm70_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_backward_64x64x128_sm70_float16` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 128, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 128, false>::kMinBlocksPerSm)
fmha_backward_128x64x128_sm70_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_backward_128x64x128_sm70_float16` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fmha_backward_64x64x65536_sm70_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_backward_64x64x65536_sm70_float16` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 65536, false>::kMinBlocksPerSm)
fmha_backward_128x64x65536_sm70_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 700
#if __CUDA_ARCH__ < 750
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_backward_128x64x65536_sm70_float16` was built for SM70, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FMHA / float32 / SM75
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 32, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 32, false>::kMinBlocksPerSm)
fmha_backward_64x64x32_sm75_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_backward_64x64x32_sm75_float32` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, false>::kMinBlocksPerSm)
fmha_backward_64x64x64_sm75_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_backward_64x64x64_sm75_float32` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 128, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 128, false>::kMinBlocksPerSm)
fmha_backward_64x64x128_sm75_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_backward_64x64x128_sm75_float32` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fmha_backward_64x64x65536_sm75_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 750
#if __CUDA_ARCH__ < 800
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_backward_64x64x65536_sm75_float32` was built for SM75, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
} 
} 
} 

