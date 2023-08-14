#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.cuh> 
namespace natten { 
namespace cuda { 
namespace naive { 
void na1d_pn_cuda_naive_double_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_double_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_double_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_double_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_double_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_double_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_double_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_double_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_double_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_double_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_double_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_double_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_double_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_double_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_float_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_float_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_float_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_float_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_float_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_float_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_float_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_float_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_float_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_float_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_float_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_float_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_float_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_float_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_half_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_half_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_half_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_half_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_half_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_half_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_half_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_half_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_half_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_half_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_half_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_half_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_half_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_half_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_bfloat16_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_bfloat16_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_bfloat16_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_bfloat16_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_bfloat16_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_bfloat16_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_bfloat16_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_bfloat16_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_bfloat16_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_bfloat16_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_bfloat16_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_bfloat16_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_bfloat16_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_cuda_naive_bfloat16_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_double_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_double_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_double_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_double_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_double_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_double_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_double_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_double_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_double_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_double_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_double_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_double_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_double_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_double_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_float_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_float_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_float_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_float_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_float_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_float_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_float_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_float_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_float_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_float_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_float_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_float_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_float_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_float_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_half_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_half_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_half_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_half_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_half_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_half_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_half_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_half_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_half_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_half_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_half_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_half_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_half_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_half_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_bfloat16_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_bfloat16_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_bfloat16_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_bfloat16_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_bfloat16_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_bfloat16_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_bfloat16_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_bfloat16_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_bfloat16_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_bfloat16_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_bfloat16_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_bfloat16_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_bfloat16_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_cuda_naive_bfloat16_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na3d_pn_cuda_naive_double_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_double_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_double_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_double_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_double_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_double_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_double_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_double_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_double_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_double_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_double_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_double_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_double_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_double_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_float_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_float_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_float_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_float_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_float_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_float_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_float_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_float_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_float_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_float_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_float_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_float_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_float_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_float_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_half_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_half_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_half_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_half_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_half_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_half_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_half_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_half_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_half_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_half_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_half_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_half_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_half_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_half_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_bfloat16_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_bfloat16_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_bfloat16_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_bfloat16_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_bfloat16_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_bfloat16_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_bfloat16_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_bfloat16_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_bfloat16_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_bfloat16_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_bfloat16_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_bfloat16_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_bfloat16_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_cuda_naive_bfloat16_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na1d_pn_bias_cuda_naive_double_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_double_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_double_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_double_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_double_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_double_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_double_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_double_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_double_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_double_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_double_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_double_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_double_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_double_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_float_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_float_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_float_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_float_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_float_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_float_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_float_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_float_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_float_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_float_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_float_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_float_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_float_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_float_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_half_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_half_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_half_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_half_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_half_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_half_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_half_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_half_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_half_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_half_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_half_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_half_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_half_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_half_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_bfloat16_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_bfloat16_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_bfloat16_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_bfloat16_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_bfloat16_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_bfloat16_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_bfloat16_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_bfloat16_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_bfloat16_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_bfloat16_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_bfloat16_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_bfloat16_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_bfloat16_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_pn_bias_cuda_naive_bfloat16_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_double_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_double_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_double_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_double_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_double_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_double_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_double_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_double_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_double_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_double_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_double_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_double_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_double_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_double_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_float_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_float_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_float_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_float_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_float_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_float_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_float_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_float_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_float_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_float_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_float_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_float_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_float_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_float_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_half_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_half_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_half_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_half_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_half_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_half_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_half_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_half_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_half_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_half_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_half_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_half_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_half_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_half_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_bfloat16_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_bfloat16_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_bfloat16_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_bfloat16_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_bfloat16_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_bfloat16_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_bfloat16_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_bfloat16_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_bfloat16_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_bfloat16_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_bfloat16_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_bfloat16_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_bfloat16_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_pn_bias_cuda_naive_bfloat16_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na3d_pn_bias_cuda_naive_double_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_double_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_double_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_double_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_double_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_double_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_double_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_double_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_double_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_double_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_double_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_double_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_double_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_double_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_float_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_float_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_float_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_float_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_float_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_float_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_float_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_float_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_float_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_float_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_float_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_float_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_float_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_float_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_half_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_half_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_half_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_half_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_half_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_half_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_half_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_half_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_half_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_half_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_half_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_half_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_half_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_half_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_bfloat16_ks_any_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_bfloat16_ks_any_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_bfloat16_ks_3_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_bfloat16_ks_3_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_bfloat16_ks_5_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_bfloat16_ks_5_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_bfloat16_ks_7_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_bfloat16_ks_7_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_bfloat16_ks_9_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_bfloat16_ks_9_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_bfloat16_ks_11_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_bfloat16_ks_11_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_bfloat16_ks_13_di_any(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_pn_bias_cuda_naive_bfloat16_ks_13_di_1(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na1d_nn_cuda_naive_double_ks_any_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_double_ks_any_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_double_ks_3_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_double_ks_3_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_double_ks_5_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_double_ks_5_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_double_ks_7_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_double_ks_7_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_double_ks_9_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_double_ks_9_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_double_ks_11_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_double_ks_11_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_double_ks_13_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_double_ks_13_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_float_ks_any_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_float_ks_any_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_float_ks_3_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_float_ks_3_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_float_ks_5_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_float_ks_5_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_float_ks_7_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_float_ks_7_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_float_ks_9_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_float_ks_9_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_float_ks_11_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_float_ks_11_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_float_ks_13_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_float_ks_13_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_half_ks_any_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_half_ks_any_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_half_ks_3_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_half_ks_3_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_half_ks_5_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_half_ks_5_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_half_ks_7_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_half_ks_7_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_half_ks_9_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_half_ks_9_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_half_ks_11_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_half_ks_11_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_half_ks_13_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_half_ks_13_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_bfloat16_ks_any_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_bfloat16_ks_any_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_bfloat16_ks_3_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_bfloat16_ks_3_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_bfloat16_ks_5_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_bfloat16_ks_5_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_bfloat16_ks_7_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_bfloat16_ks_7_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_bfloat16_ks_9_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_bfloat16_ks_9_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_bfloat16_ks_11_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_bfloat16_ks_11_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_bfloat16_ks_13_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_nn_cuda_naive_bfloat16_ks_13_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_double_ks_any_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_double_ks_any_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_double_ks_3_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_double_ks_3_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_double_ks_5_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_double_ks_5_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_double_ks_7_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_double_ks_7_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_double_ks_9_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_double_ks_9_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_double_ks_11_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_double_ks_11_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_double_ks_13_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_double_ks_13_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_float_ks_any_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_float_ks_any_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_float_ks_3_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_float_ks_3_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_float_ks_5_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_float_ks_5_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_float_ks_7_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_float_ks_7_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_float_ks_9_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_float_ks_9_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_float_ks_11_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_float_ks_11_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_float_ks_13_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_float_ks_13_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_half_ks_any_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_half_ks_any_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_half_ks_3_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_half_ks_3_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_half_ks_5_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_half_ks_5_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_half_ks_7_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_half_ks_7_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_half_ks_9_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_half_ks_9_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_half_ks_11_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_half_ks_11_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_half_ks_13_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_half_ks_13_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_bfloat16_ks_any_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_bfloat16_ks_any_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_bfloat16_ks_3_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_bfloat16_ks_3_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_bfloat16_ks_5_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_bfloat16_ks_5_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_bfloat16_ks_7_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_bfloat16_ks_7_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_bfloat16_ks_9_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_bfloat16_ks_9_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_bfloat16_ks_11_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_bfloat16_ks_11_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_bfloat16_ks_13_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_nn_cuda_naive_bfloat16_ks_13_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na3d_nn_cuda_naive_double_ks_any_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_double_ks_any_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_double_ks_3_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_double_ks_3_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_double_ks_5_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_double_ks_5_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_double_ks_7_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_double_ks_7_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_double_ks_9_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_double_ks_9_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_double_ks_11_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_double_ks_11_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_double_ks_13_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_double_ks_13_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_float_ks_any_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_float_ks_any_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_float_ks_3_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_float_ks_3_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_float_ks_5_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_float_ks_5_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_float_ks_7_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_float_ks_7_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_float_ks_9_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_float_ks_9_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_float_ks_11_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_float_ks_11_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_float_ks_13_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_float_ks_13_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_half_ks_any_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_half_ks_any_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_half_ks_3_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_half_ks_3_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_half_ks_5_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_half_ks_5_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_half_ks_7_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_half_ks_7_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_half_ks_9_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_half_ks_9_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_half_ks_11_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_half_ks_11_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_half_ks_13_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_half_ks_13_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_bfloat16_ks_any_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_bfloat16_ks_any_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_bfloat16_ks_3_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_bfloat16_ks_3_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_bfloat16_ks_5_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_bfloat16_ks_5_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_bfloat16_ks_7_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_bfloat16_ks_7_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_bfloat16_ks_9_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_bfloat16_ks_9_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_bfloat16_ks_11_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_bfloat16_ks_11_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_bfloat16_ks_13_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_nn_cuda_naive_bfloat16_ks_13_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na1d_in_cuda_naive_double_ks_any_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_double_ks_any_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_double_ks_3_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_double_ks_3_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_double_ks_5_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_double_ks_5_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_double_ks_7_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_double_ks_7_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_double_ks_9_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_double_ks_9_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_double_ks_11_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_double_ks_11_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_double_ks_13_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_double_ks_13_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_float_ks_any_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_float_ks_any_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_float_ks_3_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_float_ks_3_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_float_ks_5_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_float_ks_5_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_float_ks_7_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_float_ks_7_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_float_ks_9_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_float_ks_9_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_float_ks_11_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_float_ks_11_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_float_ks_13_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_float_ks_13_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_half_ks_any_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_half_ks_any_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_half_ks_3_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_half_ks_3_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_half_ks_5_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_half_ks_5_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_half_ks_7_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_half_ks_7_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_half_ks_9_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_half_ks_9_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_half_ks_11_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_half_ks_11_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_half_ks_13_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_half_ks_13_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_bfloat16_ks_any_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_bfloat16_ks_any_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_bfloat16_ks_3_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_bfloat16_ks_3_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_bfloat16_ks_5_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_bfloat16_ks_5_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_bfloat16_ks_7_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_bfloat16_ks_7_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_bfloat16_ks_9_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_bfloat16_ks_9_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_bfloat16_ks_11_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_bfloat16_ks_11_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_bfloat16_ks_13_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_in_cuda_naive_bfloat16_ks_13_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_double_ks_any_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_double_ks_any_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_double_ks_3_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_double_ks_3_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_double_ks_5_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_double_ks_5_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_double_ks_7_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_double_ks_7_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_double_ks_9_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_double_ks_9_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_double_ks_11_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_double_ks_11_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_double_ks_13_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_double_ks_13_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_float_ks_any_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_float_ks_any_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_float_ks_3_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_float_ks_3_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_float_ks_5_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_float_ks_5_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_float_ks_7_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_float_ks_7_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_float_ks_9_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_float_ks_9_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_float_ks_11_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_float_ks_11_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_float_ks_13_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_float_ks_13_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_half_ks_any_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_half_ks_any_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_half_ks_3_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_half_ks_3_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_half_ks_5_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_half_ks_5_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_half_ks_7_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_half_ks_7_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_half_ks_9_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_half_ks_9_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_half_ks_11_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_half_ks_11_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_half_ks_13_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_half_ks_13_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_bfloat16_ks_any_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_bfloat16_ks_any_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_bfloat16_ks_3_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_bfloat16_ks_3_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_bfloat16_ks_5_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_bfloat16_ks_5_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_bfloat16_ks_7_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_bfloat16_ks_7_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_bfloat16_ks_9_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_bfloat16_ks_9_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_bfloat16_ks_11_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_bfloat16_ks_11_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_bfloat16_ks_13_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_in_cuda_naive_bfloat16_ks_13_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na3d_in_cuda_naive_double_ks_any_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_double_ks_any_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_double_ks_3_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_double_ks_3_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_double_ks_5_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_double_ks_5_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_double_ks_7_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_double_ks_7_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_double_ks_9_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_double_ks_9_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_double_ks_11_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_double_ks_11_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_double_ks_13_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_double_ks_13_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_float_ks_any_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_float_ks_any_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_float_ks_3_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_float_ks_3_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_float_ks_5_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_float_ks_5_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_float_ks_7_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_float_ks_7_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_float_ks_9_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_float_ks_9_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_float_ks_11_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_float_ks_11_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_float_ks_13_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_float_ks_13_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_half_ks_any_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_half_ks_any_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_half_ks_3_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_half_ks_3_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_half_ks_5_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_half_ks_5_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_half_ks_7_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_half_ks_7_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_half_ks_9_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_half_ks_9_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_half_ks_11_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_half_ks_11_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_half_ks_13_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_half_ks_13_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_bfloat16_ks_any_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_bfloat16_ks_any_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_bfloat16_ks_3_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_bfloat16_ks_3_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_bfloat16_ks_5_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_bfloat16_ks_5_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_bfloat16_ks_7_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_bfloat16_ks_7_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_bfloat16_ks_9_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_bfloat16_ks_9_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_bfloat16_ks_11_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_bfloat16_ks_11_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_bfloat16_ks_13_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_in_cuda_naive_bfloat16_ks_13_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na1d_rpbgrad_cuda_naive_double_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_double_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_double_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_double_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_double_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_double_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_double_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_double_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_double_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_double_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_double_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_double_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_double_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_double_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_float_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_float_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_float_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_float_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_float_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_float_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_float_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_float_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_float_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_float_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_float_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_float_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_float_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_float_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_half_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_half_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_half_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_half_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_half_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_half_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_half_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_half_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_half_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_half_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_half_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_half_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_half_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_half_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_bfloat16_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_bfloat16_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_bfloat16_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_bfloat16_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_bfloat16_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_bfloat16_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_bfloat16_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_bfloat16_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_bfloat16_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_bfloat16_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_bfloat16_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_bfloat16_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_bfloat16_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na1d_rpbgrad_cuda_naive_bfloat16_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_double_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_double_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_double_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_double_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_double_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_double_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_double_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_double_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_double_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_double_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_double_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_double_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_double_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_double_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_float_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_float_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_float_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_float_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_float_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_float_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_float_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_float_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_float_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_float_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_float_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_float_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_float_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_float_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_half_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_half_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_half_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_half_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_half_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_half_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_half_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_half_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_half_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_half_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_half_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_half_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_half_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_half_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_bfloat16_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_bfloat16_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_bfloat16_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_bfloat16_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_bfloat16_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_bfloat16_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_bfloat16_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_bfloat16_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_bfloat16_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_bfloat16_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_bfloat16_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_bfloat16_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_bfloat16_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na2d_rpbgrad_cuda_naive_bfloat16_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation);

void na3d_rpbgrad_cuda_naive_double_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_double_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_double_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_double_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_double_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_double_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_double_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_double_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_double_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_double_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_double_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_double_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_double_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_double_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_float_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_float_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_float_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_float_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_float_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_float_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_float_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_float_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_float_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_float_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_float_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_float_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_float_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_float_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_half_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_half_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_half_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_half_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_half_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_half_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_half_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_half_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_half_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_half_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_half_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_half_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_half_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_half_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_bfloat16_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_bfloat16_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_bfloat16_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_bfloat16_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_bfloat16_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_bfloat16_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_bfloat16_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_bfloat16_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_bfloat16_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_bfloat16_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_bfloat16_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_bfloat16_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_bfloat16_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);

void na3d_rpbgrad_cuda_naive_bfloat16_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d);



} // namespace natten 
} // namespace cuda 
} // namespace naive 

