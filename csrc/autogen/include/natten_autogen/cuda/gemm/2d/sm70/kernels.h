#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.cuh> 
#include <natten/gemm_argpack.cuh> 
#include <natten/config.h> 
namespace natten { 
namespace cuda { 
namespace gemm { 
void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks15_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks15_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks15_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks17_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks17_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks17_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks19_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks19_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks19_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks21_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks21_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks21_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks23_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks23_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks23_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks25_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks25_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks25_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks27_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks27_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks27_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks29_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks29_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks29_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks31_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks31_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks31_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks33_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks33_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks33_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks21_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks21_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks21_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks23_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks23_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks23_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks25_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks25_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks25_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks27_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks27_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks27_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks29_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks29_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks29_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks31_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks31_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks31_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks33_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks33_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks33_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks21_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks21_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks21_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks23_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks23_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks23_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks25_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks25_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks25_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks27_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks27_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks27_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks29_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks29_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks29_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks31_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks31_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks31_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks33_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks33_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks33_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale);



} // namespace natten 
} // namespace cuda 
} // namespace gemm 

