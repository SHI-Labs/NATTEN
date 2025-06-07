#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/natten.h> 
#include <natten/cuda/fmha/kernel_forward.h> 
#include <natten/cuda/fmha/kernel_backward.h> 
namespace natten { 
namespace cuda { 
namespace fmha { 


///////////////////////////////////////////////////////////////////
// FMHA / float32 / SM50
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 32>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 32>::kMinBlocksPerSm)
fmha_32x128x32_sm50_float32(typename AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 32>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_64x64x32_sm50_float32(typename AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 128, 32>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 128, 32>::kMinBlocksPerSm)
fmha_64x128x32_sm50_float32(typename AttentionKernel<float, cutlass::arch::Sm50, true, 64, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 64>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 64>::kMinBlocksPerSm)
fmha_32x128x64_sm50_float32(typename AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 64>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_64x64x64_sm50_float32(typename AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 128, 64>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 128, 64>::kMinBlocksPerSm)
fmha_64x128x64_sm50_float32(typename AttentionKernel<float, cutlass::arch::Sm50, true, 64, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 128>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 128>::kMinBlocksPerSm)
fmha_32x128x128_sm50_float32(typename AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 128>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 128>::kMinBlocksPerSm)
fmha_64x64x128_sm50_float32(typename AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 128, 128>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 128, 128>::kMinBlocksPerSm)
fmha_64x128x128_sm50_float32(typename AttentionKernel<float, cutlass::arch::Sm50, true, 64, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 65536>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 65536>::kMinBlocksPerSm)
fmha_32x128x65536_sm50_float32(typename AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 128, 65536>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 128, 65536>::kMinBlocksPerSm)
fmha_64x128x65536_sm50_float32(typename AttentionKernel<float, cutlass::arch::Sm50, true, 64, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 65536>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 65536>::kMinBlocksPerSm)
fmha_64x64x65536_sm50_float32(typename AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 65536>::Params p);


template <typename T>
void fmha_sm50_float32(T cb) {
  cb(AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 32>(), fmha_32x128x32_sm50_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 32>(), fmha_64x64x32_sm50_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm50, true, 64, 128, 32>(), fmha_64x128x32_sm50_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 64>(), fmha_32x128x64_sm50_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 64>(), fmha_64x64x64_sm50_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm50, true, 64, 128, 64>(), fmha_64x128x64_sm50_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 128>(), fmha_32x128x128_sm50_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 128>(), fmha_64x64x128_sm50_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm50, true, 64, 128, 128>(), fmha_64x128x128_sm50_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm50, true, 32, 128, 65536>(), fmha_32x128x65536_sm50_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm50, true, 64, 128, 65536>(), fmha_64x128x65536_sm50_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm50, true, 64, 64, 65536>(), fmha_64x64x65536_sm50_float32);
}

///////////////////////////////////////////////////////////////////
// FMHA / float16 / SM50
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 32>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 32>::kMinBlocksPerSm)
fmha_32x128x32_sm50_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_64x64x32_sm50_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 128, 32>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 128, 32>::kMinBlocksPerSm)
fmha_64x128x32_sm50_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 64>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 64>::kMinBlocksPerSm)
fmha_32x128x64_sm50_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_64x64x64_sm50_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 128, 64>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 128, 64>::kMinBlocksPerSm)
fmha_64x128x64_sm50_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 128>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 128>::kMinBlocksPerSm)
fmha_32x128x128_sm50_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::kMinBlocksPerSm)
fmha_64x64x128_sm50_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 128, 128>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 128, 128>::kMinBlocksPerSm)
fmha_64x128x128_sm50_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 65536>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 65536>::kMinBlocksPerSm)
fmha_32x128x65536_sm50_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 128, 65536>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 128, 65536>::kMinBlocksPerSm)
fmha_64x128x65536_sm50_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::kMinBlocksPerSm)
fmha_64x64x65536_sm50_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>::Params p);


template <typename T>
void fmha_sm50_float16(T cb) {
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 32>(), fmha_32x128x32_sm50_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32>(), fmha_64x64x32_sm50_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 128, 32>(), fmha_64x128x32_sm50_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 64>(), fmha_32x128x64_sm50_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64>(), fmha_64x64x64_sm50_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 128, 64>(), fmha_64x128x64_sm50_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 128>(), fmha_32x128x128_sm50_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128>(), fmha_64x64x128_sm50_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 128, 128>(), fmha_64x128x128_sm50_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 32, 128, 65536>(), fmha_32x128x65536_sm50_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 128, 65536>(), fmha_64x128x65536_sm50_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536>(), fmha_64x64x65536_sm50_float16);
}

///////////////////////////////////////////////////////////////////
// FMHA / float32 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 32>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 32>::kMinBlocksPerSm)
fmha_32x128x32_sm70_float32(typename AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 32>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_64x64x32_sm70_float32(typename AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 128, 32>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 128, 32>::kMinBlocksPerSm)
fmha_64x128x32_sm70_float32(typename AttentionKernel<float, cutlass::arch::Sm70, true, 64, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 64>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 64>::kMinBlocksPerSm)
fmha_32x128x64_sm70_float32(typename AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 64>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_64x64x64_sm70_float32(typename AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 128, 64>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 128, 64>::kMinBlocksPerSm)
fmha_64x128x64_sm70_float32(typename AttentionKernel<float, cutlass::arch::Sm70, true, 64, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 128>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 128>::kMinBlocksPerSm)
fmha_32x128x128_sm70_float32(typename AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 128>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 128>::kMinBlocksPerSm)
fmha_64x64x128_sm70_float32(typename AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 128, 128>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 128, 128>::kMinBlocksPerSm)
fmha_64x128x128_sm70_float32(typename AttentionKernel<float, cutlass::arch::Sm70, true, 64, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 65536>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 65536>::kMinBlocksPerSm)
fmha_32x128x65536_sm70_float32(typename AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 128, 65536>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 128, 65536>::kMinBlocksPerSm)
fmha_64x128x65536_sm70_float32(typename AttentionKernel<float, cutlass::arch::Sm70, true, 64, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 65536>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 65536>::kMinBlocksPerSm)
fmha_64x64x65536_sm70_float32(typename AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 65536>::Params p);


template <typename T>
void fmha_sm70_float32(T cb) {
  cb(AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 32>(), fmha_32x128x32_sm70_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 32>(), fmha_64x64x32_sm70_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm70, true, 64, 128, 32>(), fmha_64x128x32_sm70_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 64>(), fmha_32x128x64_sm70_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 64>(), fmha_64x64x64_sm70_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm70, true, 64, 128, 64>(), fmha_64x128x64_sm70_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 128>(), fmha_32x128x128_sm70_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 128>(), fmha_64x64x128_sm70_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm70, true, 64, 128, 128>(), fmha_64x128x128_sm70_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm70, true, 32, 128, 65536>(), fmha_32x128x65536_sm70_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm70, true, 64, 128, 65536>(), fmha_64x128x65536_sm70_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm70, true, 64, 64, 65536>(), fmha_64x64x65536_sm70_float32);
}

///////////////////////////////////////////////////////////////////
// FMHA / float16 / SM70
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32>::kMinBlocksPerSm)
fmha_32x128x32_sm70_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_64x64x32_sm70_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32>::kMinBlocksPerSm)
fmha_64x128x32_sm70_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64>::kMinBlocksPerSm)
fmha_32x128x64_sm70_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_64x64x64_sm70_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64>::kMinBlocksPerSm)
fmha_64x128x64_sm70_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128>::kMinBlocksPerSm)
fmha_32x128x128_sm70_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128>::kMinBlocksPerSm)
fmha_64x64x128_sm70_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128>::kMinBlocksPerSm)
fmha_64x128x128_sm70_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536>::kMinBlocksPerSm)
fmha_32x128x65536_sm70_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536>::kMinBlocksPerSm)
fmha_64x128x65536_sm70_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536>::kMinBlocksPerSm)
fmha_64x64x65536_sm70_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536>::Params p);


template <typename T>
void fmha_sm70_float16(T cb) {
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 32>(), fmha_32x128x32_sm70_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32>(), fmha_64x64x32_sm70_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 32>(), fmha_64x128x32_sm70_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 64>(), fmha_32x128x64_sm70_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64>(), fmha_64x64x64_sm70_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 64>(), fmha_64x128x64_sm70_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 128>(), fmha_32x128x128_sm70_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128>(), fmha_64x64x128_sm70_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 128>(), fmha_64x128x128_sm70_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 32, 128, 65536>(), fmha_32x128x65536_sm70_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 128, 65536>(), fmha_64x128x65536_sm70_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536>(), fmha_64x64x65536_sm70_float16);
}

///////////////////////////////////////////////////////////////////
// FMHA / float32 / SM75
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 32>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 32>::kMinBlocksPerSm)
fmha_32x128x32_sm75_float32(typename AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 32>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_64x64x32_sm75_float32(typename AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 128, 32>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 128, 32>::kMinBlocksPerSm)
fmha_64x128x32_sm75_float32(typename AttentionKernel<float, cutlass::arch::Sm75, true, 64, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 64>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 64>::kMinBlocksPerSm)
fmha_32x128x64_sm75_float32(typename AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_64x64x64_sm75_float32(typename AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 128, 64>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 128, 64>::kMinBlocksPerSm)
fmha_64x128x64_sm75_float32(typename AttentionKernel<float, cutlass::arch::Sm75, true, 64, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128>::kMinBlocksPerSm)
fmha_32x128x128_sm75_float32(typename AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 128>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 128>::kMinBlocksPerSm)
fmha_64x64x128_sm75_float32(typename AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 128, 128>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 128, 128>::kMinBlocksPerSm)
fmha_64x128x128_sm75_float32(typename AttentionKernel<float, cutlass::arch::Sm75, true, 64, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 65536>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 65536>::kMinBlocksPerSm)
fmha_32x128x65536_sm75_float32(typename AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 128, 65536>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 128, 65536>::kMinBlocksPerSm)
fmha_64x128x65536_sm75_float32(typename AttentionKernel<float, cutlass::arch::Sm75, true, 64, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 65536>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 65536>::kMinBlocksPerSm)
fmha_64x64x65536_sm75_float32(typename AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 65536>::Params p);


template <typename T>
void fmha_sm75_float32(T cb) {
  cb(AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 32>(), fmha_32x128x32_sm75_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 32>(), fmha_64x64x32_sm75_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm75, true, 64, 128, 32>(), fmha_64x128x32_sm75_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 64>(), fmha_32x128x64_sm75_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 64>(), fmha_64x64x64_sm75_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm75, true, 64, 128, 64>(), fmha_64x128x64_sm75_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 128>(), fmha_32x128x128_sm75_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 128>(), fmha_64x64x128_sm75_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm75, true, 64, 128, 128>(), fmha_64x128x128_sm75_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm75, true, 32, 128, 65536>(), fmha_32x128x65536_sm75_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm75, true, 64, 128, 65536>(), fmha_64x128x65536_sm75_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm75, true, 64, 64, 65536>(), fmha_64x64x65536_sm75_float32);
}

///////////////////////////////////////////////////////////////////
// FMHA / float16 / SM75
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 32>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 32>::kMinBlocksPerSm)
fmha_32x128x32_sm75_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_64x64x32_sm75_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 32>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 32>::kMinBlocksPerSm)
fmha_64x128x32_sm75_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 64>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 64>::kMinBlocksPerSm)
fmha_32x128x64_sm75_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_64x64x64_sm75_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 64>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 64>::kMinBlocksPerSm)
fmha_64x128x64_sm75_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128>::kMinBlocksPerSm)
fmha_32x128x128_sm75_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128>::kMinBlocksPerSm)
fmha_64x64x128_sm75_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 128>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 128>::kMinBlocksPerSm)
fmha_64x128x128_sm75_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536>::kMinBlocksPerSm)
fmha_32x128x65536_sm75_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 65536>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 65536>::kMinBlocksPerSm)
fmha_64x128x65536_sm75_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536>::kMinBlocksPerSm)
fmha_64x64x65536_sm75_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536>::Params p);


template <typename T>
void fmha_sm75_float16(T cb) {
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 32>(), fmha_32x128x32_sm75_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32>(), fmha_64x64x32_sm75_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 32>(), fmha_64x128x32_sm75_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 64>(), fmha_32x128x64_sm75_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64>(), fmha_64x64x64_sm75_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 64>(), fmha_64x128x64_sm75_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 128>(), fmha_32x128x128_sm75_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128>(), fmha_64x64x128_sm75_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 128>(), fmha_64x128x128_sm75_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 32, 128, 65536>(), fmha_32x128x65536_sm75_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 128, 65536>(), fmha_64x128x65536_sm75_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536>(), fmha_64x64x65536_sm75_float16);
}

///////////////////////////////////////////////////////////////////
// FMHA / float32 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 32>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 32>::kMinBlocksPerSm)
fmha_32x128x32_sm80_float32(typename AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_64x64x32_sm80_float32(typename AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 32>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 32>::kMinBlocksPerSm)
fmha_64x128x32_sm80_float32(typename AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 64>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 64>::kMinBlocksPerSm)
fmha_32x128x64_sm80_float32(typename AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_64x64x64_sm80_float32(typename AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 64>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 64>::kMinBlocksPerSm)
fmha_64x128x64_sm80_float32(typename AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 128>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 128>::kMinBlocksPerSm)
fmha_32x128x128_sm80_float32(typename AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fmha_64x64x128_sm80_float32(typename AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128>::kMinBlocksPerSm)
fmha_64x128x128_sm80_float32(typename AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 65536>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 65536>::kMinBlocksPerSm)
fmha_32x128x65536_sm80_float32(typename AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 65536>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 65536>::kMinBlocksPerSm)
fmha_64x128x65536_sm80_float32(typename AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fmha_64x64x65536_sm80_float32(typename AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p);


template <typename T>
void fmha_sm80_float32(T cb) {
  cb(AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 32>(), fmha_32x128x32_sm80_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 32>(), fmha_64x64x32_sm80_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 32>(), fmha_64x128x32_sm80_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 64>(), fmha_32x128x64_sm80_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 64>(), fmha_64x64x64_sm80_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 64>(), fmha_64x128x64_sm80_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 128>(), fmha_32x128x128_sm80_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 128>(), fmha_64x64x128_sm80_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 128>(), fmha_64x128x128_sm80_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm80, true, 32, 128, 65536>(), fmha_32x128x65536_sm80_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm80, true, 64, 128, 65536>(), fmha_64x128x65536_sm80_float32);
  cb(AttentionKernel<float, cutlass::arch::Sm80, true, 64, 64, 65536>(), fmha_64x64x65536_sm80_float32);
}

///////////////////////////////////////////////////////////////////
// FMHA / float16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 32>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 32>::kMinBlocksPerSm)
fmha_32x128x32_sm80_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_64x64x32_sm80_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 32>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 32>::kMinBlocksPerSm)
fmha_64x128x32_sm80_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64>::kMinBlocksPerSm)
fmha_32x128x64_sm80_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_64x64x64_sm80_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64>::kMinBlocksPerSm)
fmha_64x128x64_sm80_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 128>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 128>::kMinBlocksPerSm)
fmha_32x128x128_sm80_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fmha_64x64x128_sm80_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128>::kMinBlocksPerSm)
fmha_64x128x128_sm80_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536>::kMinBlocksPerSm)
fmha_32x128x65536_sm80_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 65536>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 65536>::kMinBlocksPerSm)
fmha_64x128x65536_sm80_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fmha_64x64x65536_sm80_float16(typename AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p);


template <typename T>
void fmha_sm80_float16(T cb) {
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 32>(), fmha_32x128x32_sm80_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32>(), fmha_64x64x32_sm80_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 32>(), fmha_64x128x32_sm80_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 64>(), fmha_32x128x64_sm80_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64>(), fmha_64x64x64_sm80_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 64>(), fmha_64x128x64_sm80_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 128>(), fmha_32x128x128_sm80_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128>(), fmha_64x64x128_sm80_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 128>(), fmha_64x128x128_sm80_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 32, 128, 65536>(), fmha_32x128x65536_sm80_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 128, 65536>(), fmha_64x128x65536_sm80_float16);
  cb(AttentionKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536>(), fmha_64x64x65536_sm80_float16);
}

///////////////////////////////////////////////////////////////////
// FMHA / bfloat16 / SM80
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32>::kMinBlocksPerSm)
fmha_32x128x32_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::kMinBlocksPerSm)
fmha_64x64x32_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32>::kMinBlocksPerSm)
fmha_64x128x32_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64>::kMinBlocksPerSm)
fmha_32x128x64_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::kMinBlocksPerSm)
fmha_64x64x64_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64>::kMinBlocksPerSm)
fmha_64x128x64_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128>::kMinBlocksPerSm)
fmha_32x128x128_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::kMinBlocksPerSm)
fmha_64x64x128_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128>::kMinBlocksPerSm)
fmha_64x128x128_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536>::kMinBlocksPerSm)
fmha_32x128x65536_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536>::kMinBlocksPerSm)
fmha_64x128x65536_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536>::Params p);


__global__ void __launch_bounds__(
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kNumThreads,
    AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::kMinBlocksPerSm)
fmha_64x64x65536_sm80_bfloat16(typename AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>::Params p);


template <typename T>
void fmha_sm80_bfloat16(T cb) {
  cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 32>(), fmha_32x128x32_sm80_bfloat16);
  cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32>(), fmha_64x64x32_sm80_bfloat16);
  cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 32>(), fmha_64x128x32_sm80_bfloat16);
  cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 64>(), fmha_32x128x64_sm80_bfloat16);
  cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64>(), fmha_64x64x64_sm80_bfloat16);
  cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 64>(), fmha_64x128x64_sm80_bfloat16);
  cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 128>(), fmha_32x128x128_sm80_bfloat16);
  cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128>(), fmha_64x64x128_sm80_bfloat16);
  cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 128>(), fmha_64x128x128_sm80_bfloat16);
  cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 32, 128, 65536>(), fmha_32x128x65536_sm80_bfloat16);
  cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 128, 65536>(), fmha_64x128x65536_sm80_bfloat16);
  cb(AttentionKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536>(), fmha_64x64x65536_sm80_bfloat16);
}

///////////////////////////////////////////////////////////////////
// FMHA / float32 / SM50Backward Kernel
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 32, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 32, false>::kMinBlocksPerSm)
fmha_backward_64x64x32_sm50_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 32, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, false>::kMinBlocksPerSm)
fmha_backward_64x64x64_sm50_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 128, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 128, false>::kMinBlocksPerSm)
fmha_backward_64x64x128_sm50_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 128, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 65536, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fmha_backward_64x64x65536_sm50_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 65536, false>::Params p);


template <typename T>
void fmha_backward_sm50_float32(T cb) {
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 32, false>(), fmha_backward_64x64x32_sm50_float32);
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 64, false>(), fmha_backward_64x64x64_sm50_float32);
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 128, false>(), fmha_backward_64x64x128_sm50_float32);
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm50, true, 64, 64, 65536, false>(), fmha_backward_64x64x65536_sm50_float32);
}

///////////////////////////////////////////////////////////////////
// FMHA / float16 / SM50Backward Kernel
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32, false>::kMinBlocksPerSm)
fmha_backward_64x64x32_sm50_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64, false>::kMinBlocksPerSm)
fmha_backward_64x64x64_sm50_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128, false>::kMinBlocksPerSm)
fmha_backward_64x64x128_sm50_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fmha_backward_64x64x65536_sm50_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536, false>::Params p);


template <typename T>
void fmha_backward_sm50_float16(T cb) {
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 32, false>(), fmha_backward_64x64x32_sm50_float16);
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 64, false>(), fmha_backward_64x64x64_sm50_float16);
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 128, false>(), fmha_backward_64x64x128_sm50_float16);
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm50, true, 64, 64, 65536, false>(), fmha_backward_64x64x65536_sm50_float16);
}

///////////////////////////////////////////////////////////////////
// FMHA / float32 / SM70Backward Kernel
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm70, true, 64, 64, 32, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm70, true, 64, 64, 32, false>::kMinBlocksPerSm)
fmha_backward_64x64x32_sm70_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm70, true, 64, 64, 32, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm70, true, 64, 64, 64, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm70, true, 64, 64, 64, false>::kMinBlocksPerSm)
fmha_backward_64x64x64_sm70_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm70, true, 64, 64, 64, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm70, true, 64, 64, 128, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm70, true, 64, 64, 128, false>::kMinBlocksPerSm)
fmha_backward_64x64x128_sm70_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm70, true, 64, 64, 128, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fmha_backward_64x64x65536_sm70_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm70, true, 64, 64, 65536, false>::Params p);


template <typename T>
void fmha_backward_sm70_float32(T cb) {
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm70, true, 64, 64, 32, false>(), fmha_backward_64x64x32_sm70_float32);
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm70, true, 64, 64, 64, false>(), fmha_backward_64x64x64_sm70_float32);
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm70, true, 64, 64, 128, false>(), fmha_backward_64x64x128_sm70_float32);
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm70, true, 64, 64, 65536, false>(), fmha_backward_64x64x65536_sm70_float32);
}

///////////////////////////////////////////////////////////////////
// FMHA / float16 / SM70Backward Kernel
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::kMinBlocksPerSm)
fmha_backward_64x64x32_sm70_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::kMinBlocksPerSm)
fmha_backward_64x64x64_sm70_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::kMinBlocksPerSm)
fmha_backward_64x64x128_sm70_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 128, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 128, false>::kMinBlocksPerSm)
fmha_backward_128x64x128_sm70_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 128, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fmha_backward_64x64x65536_sm70_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 65536, false>::kMinBlocksPerSm)
fmha_backward_128x64x65536_sm70_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 65536, false>::Params p);


template <typename T>
void fmha_backward_sm70_float16(T cb) {
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 32, false>(), fmha_backward_64x64x32_sm70_float16);
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 64, false>(), fmha_backward_64x64x64_sm70_float16);
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 128, false>(), fmha_backward_64x64x128_sm70_float16);
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 128, false>(), fmha_backward_128x64x128_sm70_float16);
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 64, 64, 65536, false>(), fmha_backward_64x64x65536_sm70_float16);
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm70, true, 128, 64, 65536, false>(), fmha_backward_128x64x65536_sm70_float16);
}

///////////////////////////////////////////////////////////////////
// FMHA / float32 / SM75Backward Kernel
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 32, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 32, false>::kMinBlocksPerSm)
fmha_backward_64x64x32_sm75_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 32, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, false>::kMinBlocksPerSm)
fmha_backward_64x64x64_sm75_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 128, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 128, false>::kMinBlocksPerSm)
fmha_backward_64x64x128_sm75_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 128, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fmha_backward_64x64x65536_sm75_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 65536, false>::Params p);


template <typename T>
void fmha_backward_sm75_float32(T cb) {
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 32, false>(), fmha_backward_64x64x32_sm75_float32);
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 64, false>(), fmha_backward_64x64x64_sm75_float32);
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 128, false>(), fmha_backward_64x64x128_sm75_float32);
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm75, true, 64, 64, 65536, false>(), fmha_backward_64x64x65536_sm75_float32);
}

///////////////////////////////////////////////////////////////////
// FMHA / float16 / SM75Backward Kernel
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32, false>::kMinBlocksPerSm)
fmha_backward_64x64x32_sm75_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, false>::kMinBlocksPerSm)
fmha_backward_64x64x64_sm75_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128, false>::kMinBlocksPerSm)
fmha_backward_64x64x128_sm75_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fmha_backward_64x64x65536_sm75_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536, false>::Params p);


template <typename T>
void fmha_backward_sm75_float16(T cb) {
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 32, false>(), fmha_backward_64x64x32_sm75_float16);
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 64, false>(), fmha_backward_64x64x64_sm75_float16);
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 128, false>(), fmha_backward_64x64x128_sm75_float16);
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm75, true, 64, 64, 65536, false>(), fmha_backward_64x64x65536_sm75_float16);
}

///////////////////////////////////////////////////////////////////
// FMHA / float32 / SM80Backward Kernel
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 64, 64, 32, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 64, 64, 32, false>::kMinBlocksPerSm)
fmha_backward_64x64x32_sm80_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 64, 64, 32, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, false>::kMinBlocksPerSm)
fmha_backward_64x64x64_sm80_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 64, 64, 128, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 64, 64, 128, false>::kMinBlocksPerSm)
fmha_backward_64x64x128_sm80_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 64, 64, 128, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 128, 64, 128, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 128, 64, 128, false>::kMinBlocksPerSm)
fmha_backward_128x64x128_sm80_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 128, 64, 128, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 64, 64, 65536, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fmha_backward_64x64x65536_sm80_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 64, 64, 65536, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 128, 64, 65536, false>::kNumThreads,
    AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 128, 64, 65536, false>::kMinBlocksPerSm)
fmha_backward_128x64x65536_sm80_float32(typename AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 128, 64, 65536, false>::Params p);


template <typename T>
void fmha_backward_sm80_float32(T cb) {
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 64, 64, 32, false>(), fmha_backward_64x64x32_sm80_float32);
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 64, 64, 64, false>(), fmha_backward_64x64x64_sm80_float32);
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 64, 64, 128, false>(), fmha_backward_64x64x128_sm80_float32);
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 128, 64, 128, false>(), fmha_backward_128x64x128_sm80_float32);
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 64, 64, 65536, false>(), fmha_backward_64x64x65536_sm80_float32);
  cb(AttentionBackwardKernel<float, cutlass::arch::Sm80, true, 128, 64, 65536, false>(), fmha_backward_128x64x65536_sm80_float32);
}

///////////////////////////////////////////////////////////////////
// FMHA / float16 / SM80Backward Kernel
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32, false>::kMinBlocksPerSm)
fmha_backward_64x64x32_sm80_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, false>::kMinBlocksPerSm)
fmha_backward_64x64x64_sm80_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128, false>::kMinBlocksPerSm)
fmha_backward_64x64x128_sm80_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128, false>::kMinBlocksPerSm)
fmha_backward_128x128x128_sm80_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fmha_backward_64x64x65536_sm80_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536, false>::kMinBlocksPerSm)
fmha_backward_128x64x65536_sm80_float16(typename AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536, false>::Params p);


template <typename T>
void fmha_backward_sm80_float16(T cb) {
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 32, false>(), fmha_backward_64x64x32_sm80_float16);
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 64, false>(), fmha_backward_64x64x64_sm80_float16);
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 128, false>(), fmha_backward_64x64x128_sm80_float16);
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 128, 128, 128, false>(), fmha_backward_128x128x128_sm80_float16);
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 64, 64, 65536, false>(), fmha_backward_64x64x65536_sm80_float16);
  cb(AttentionBackwardKernel<cutlass::half_t, cutlass::arch::Sm80, true, 128, 64, 65536, false>(), fmha_backward_128x64x65536_sm80_float16);
}

///////////////////////////////////////////////////////////////////
// FMHA / bfloat16 / SM80Backward Kernel
///////////////////////////////////////////////////////////////////

__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false>::kMinBlocksPerSm)
fmha_backward_64x64x32_sm80_bfloat16(typename AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false>::kMinBlocksPerSm)
fmha_backward_64x64x64_sm80_bfloat16(typename AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false>::kMinBlocksPerSm)
fmha_backward_64x64x128_sm80_bfloat16(typename AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128, false>::kMinBlocksPerSm)
fmha_backward_128x128x128_sm80_bfloat16(typename AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false>::kMinBlocksPerSm)
fmha_backward_64x64x65536_sm80_bfloat16(typename AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false>::Params p);


__global__ void __launch_bounds__(
    AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536, false>::kNumThreads,
    AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536, false>::kMinBlocksPerSm)
fmha_backward_128x64x65536_sm80_bfloat16(typename AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536, false>::Params p);


template <typename T>
void fmha_backward_sm80_bfloat16(T cb) {
  cb(AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 32, false>(), fmha_backward_64x64x32_sm80_bfloat16);
  cb(AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 64, false>(), fmha_backward_64x64x64_sm80_bfloat16);
  cb(AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 128, false>(), fmha_backward_64x64x128_sm80_bfloat16);
  cb(AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 128, 128, false>(), fmha_backward_128x128x128_sm80_bfloat16);
  cb(AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 64, 64, 65536, false>(), fmha_backward_64x64x65536_sm80_bfloat16);
  cb(AttentionBackwardKernel<cutlass::bfloat16_t, cutlass::arch::Sm80, true, 128, 64, 65536, false>(), fmha_backward_128x64x65536_sm80_bfloat16);
}

} // namespace natten 
} // namespace cuda 
} // namespace fmha 

