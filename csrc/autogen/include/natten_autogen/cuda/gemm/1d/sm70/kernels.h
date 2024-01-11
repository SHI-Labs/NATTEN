#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.cuh> 
#include <natten/gemm_argpack.cuh> 
#include <natten/config.h> 
namespace natten { 
namespace cuda { 
namespace gemm { 
void na1d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream);

void na1d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream);

void na1d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream);

void na1d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream);

void na1d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream);

void na1d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream);

void na1d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream);

void na1d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream);

void na1d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream);



} // namespace natten 
} // namespace cuda 
} // namespace gemm 

