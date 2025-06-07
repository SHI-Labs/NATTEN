#pragma once


#include <iostream> 
#include <type_traits> 
#ifdef NATTEN_WITH_CUTLASS
#ifdef NATTEN_WITH_BLACKWELL_FNA
#include <natten/natten.h> 
#include <ATen/ATen.h> 
#include <ATen/cuda/CUDAContext.h> 
#include <c10/cuda/CUDAGuard.h> 
#include <c10/cuda/CUDAStream.h> 
#include <torch/extension.h> 
#include <natten/natten.h> 
#include <natten/helpers.h> 
#include <natten/cuda/fmha_blackwell/fmha_forward.cuh> 
namespace natten { 
namespace cuda { 
namespace fmha_blackwell { 

void blackwell_fmha_float16_256x128x32(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads,
      int dim,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_float16_256x128x32_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads,
      int dim,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_float16_256x128x64(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads,
      int dim,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_float16_256x128x64_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads,
      int dim,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_float16_256x128x128(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads,
      int dim,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_float16_256x128x128_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads,
      int dim,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_bfloat16_256x128x32(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads,
      int dim,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_bfloat16_256x128x32_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads,
      int dim,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_bfloat16_256x128x64(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads,
      int dim,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_bfloat16_256x128x64_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads,
      int dim,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_bfloat16_256x128x128(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads,
      int dim,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options);

void blackwell_fmha_bfloat16_256x128x128_persistent(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads,
      int dim,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options);


} // namespace natten 
} // namespace cuda 
} // namespace fmha_blackwell 
#endif 
#endif 

