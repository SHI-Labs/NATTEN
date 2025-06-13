#include <cuda_runtime.h>
#include <iostream>
#include <natten/cuda/fmha/kernel_backward.h>
#include <natten/cuda/fmha/kernel_forward.h>
#include <natten_autogen/cuda/fmha/kernels.h>
namespace natten { 
namespace cuda { 
namespace fmha { 



///////////////////////////////////////////////////////////////////
// FMHA / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32>::kMinBlocksPerSm)
fmha_32x128x32_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_32x128x32_sm80_bfloat16` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_64x64x32_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_64x64x32_sm80_bfloat16` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32>::kMinBlocksPerSm)
fmha_64x128x32_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_64x128x32_sm80_bfloat16` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64>::kMinBlocksPerSm)
fmha_32x128x64_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_32x128x64_sm80_bfloat16` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_64x64x64_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_64x64x64_sm80_bfloat16` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64>::kMinBlocksPerSm)
fmha_64x128x64_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_64x128x64_sm80_bfloat16` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128>::kMinBlocksPerSm)
fmha_32x128x128_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_32x128x128_sm80_bfloat16` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fmha_64x64x128_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_64x64x128_sm80_bfloat16` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128>::kMinBlocksPerSm)
fmha_64x128x128_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_64x128x128_sm80_bfloat16` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536>::kMinBlocksPerSm)
fmha_32x128x65536_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_32x128x65536_sm80_bfloat16` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536>::kMinBlocksPerSm)
fmha_64x128x65536_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_64x128x65536_sm80_bfloat16` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fmha_64x64x65536_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800
#if __CUDA_ARCH__ < 1300
  if (!p.advance_to_block()) {
    return;
  }
  AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_64x64x65536_sm80_bfloat16` was built for SM80, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


///////////////////////////////////////////////////////////////////
// FMHA / float32 / SM50
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 32, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 32, false>::kMinBlocksPerSm)
fmha_backward_64x64x32_sm50_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 32, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 32, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_backward_64x64x32_sm50_float32` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, false>::kMinBlocksPerSm)
fmha_backward_64x64x64_sm50_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_backward_64x64x64_sm50_float32` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 128, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 128, false>::kMinBlocksPerSm)
fmha_backward_64x64x128_sm50_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 128, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 128, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_backward_64x64x128_sm50_float32` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 65536, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fmha_backward_64x64x65536_sm50_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 65536, false>::Params p) {
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 500
#if __CUDA_ARCH__ < 700
  if (!p.advance_to_block()) {
    return;
  }
  AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 65536, false>::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: fmha kernel `fmha_backward_64x64x65536_sm50_float32` was built for SM50, but attempted to launch from SM%d\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}
} 
} 
} 

