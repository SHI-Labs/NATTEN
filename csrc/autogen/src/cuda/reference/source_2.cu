#ifdef NATTEN_WITH_CUTLASS
#include <cuda_runtime.h>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <natten/natten.h>
#include <natten/helpers.h>
#include <natten/cuda/reference/fna_reference_forward.hpp>
#include <natten/cuda/reference/fna_reference_backward.hpp>
#include <natten_autogen/cuda/reference/kernels.h>
namespace natten { 
namespace cuda { 
namespace reference { 




void reference_fna3d_float32_causal1x1x0(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::true_type, cute::true_type, cute::false_type>;

  fna_reference_forward(
    static_cast<float*>(ptr_Q),
    static_cast<float*>(ptr_K),
    static_cast<float*>(ptr_V),
    static_cast<float*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_float32_causal1x1x1(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::true_type, cute::true_type, cute::true_type>;

  fna_reference_forward(
    static_cast<float*>(ptr_Q),
    static_cast<float*>(ptr_K),
    static_cast<float*>(ptr_V),
    static_cast<float*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_float16_causal0x0x0(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::false_type, cute::false_type, cute::false_type>;

  fna_reference_forward(
    static_cast<cutlass::half_t*>(ptr_Q),
    static_cast<cutlass::half_t*>(ptr_K),
    static_cast<cutlass::half_t*>(ptr_V),
    static_cast<cutlass::half_t*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_float16_causal0x0x1(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::false_type, cute::false_type, cute::true_type>;

  fna_reference_forward(
    static_cast<cutlass::half_t*>(ptr_Q),
    static_cast<cutlass::half_t*>(ptr_K),
    static_cast<cutlass::half_t*>(ptr_V),
    static_cast<cutlass::half_t*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_float16_causal0x1x0(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::false_type, cute::true_type, cute::false_type>;

  fna_reference_forward(
    static_cast<cutlass::half_t*>(ptr_Q),
    static_cast<cutlass::half_t*>(ptr_K),
    static_cast<cutlass::half_t*>(ptr_V),
    static_cast<cutlass::half_t*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_float16_causal0x1x1(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::false_type, cute::true_type, cute::true_type>;

  fna_reference_forward(
    static_cast<cutlass::half_t*>(ptr_Q),
    static_cast<cutlass::half_t*>(ptr_K),
    static_cast<cutlass::half_t*>(ptr_V),
    static_cast<cutlass::half_t*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_float16_causal1x0x0(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::true_type, cute::false_type, cute::false_type>;

  fna_reference_forward(
    static_cast<cutlass::half_t*>(ptr_Q),
    static_cast<cutlass::half_t*>(ptr_K),
    static_cast<cutlass::half_t*>(ptr_V),
    static_cast<cutlass::half_t*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_float16_causal1x0x1(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::true_type, cute::false_type, cute::true_type>;

  fna_reference_forward(
    static_cast<cutlass::half_t*>(ptr_Q),
    static_cast<cutlass::half_t*>(ptr_K),
    static_cast<cutlass::half_t*>(ptr_V),
    static_cast<cutlass::half_t*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_float16_causal1x1x0(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::true_type, cute::true_type, cute::false_type>;

  fna_reference_forward(
    static_cast<cutlass::half_t*>(ptr_Q),
    static_cast<cutlass::half_t*>(ptr_K),
    static_cast<cutlass::half_t*>(ptr_V),
    static_cast<cutlass::half_t*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_float16_causal1x1x1(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::true_type, cute::true_type, cute::true_type>;

  fna_reference_forward(
    static_cast<cutlass::half_t*>(ptr_Q),
    static_cast<cutlass::half_t*>(ptr_K),
    static_cast<cutlass::half_t*>(ptr_V),
    static_cast<cutlass::half_t*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_bfloat16_causal0x0x0(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::false_type, cute::false_type, cute::false_type>;

  fna_reference_forward(
    static_cast<cutlass::bfloat16_t*>(ptr_Q),
    static_cast<cutlass::bfloat16_t*>(ptr_K),
    static_cast<cutlass::bfloat16_t*>(ptr_V),
    static_cast<cutlass::bfloat16_t*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_bfloat16_causal0x0x1(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::false_type, cute::false_type, cute::true_type>;

  fna_reference_forward(
    static_cast<cutlass::bfloat16_t*>(ptr_Q),
    static_cast<cutlass::bfloat16_t*>(ptr_K),
    static_cast<cutlass::bfloat16_t*>(ptr_V),
    static_cast<cutlass::bfloat16_t*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_bfloat16_causal0x1x0(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::false_type, cute::true_type, cute::false_type>;

  fna_reference_forward(
    static_cast<cutlass::bfloat16_t*>(ptr_Q),
    static_cast<cutlass::bfloat16_t*>(ptr_K),
    static_cast<cutlass::bfloat16_t*>(ptr_V),
    static_cast<cutlass::bfloat16_t*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_bfloat16_causal0x1x1(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::false_type, cute::true_type, cute::true_type>;

  fna_reference_forward(
    static_cast<cutlass::bfloat16_t*>(ptr_Q),
    static_cast<cutlass::bfloat16_t*>(ptr_K),
    static_cast<cutlass::bfloat16_t*>(ptr_V),
    static_cast<cutlass::bfloat16_t*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_bfloat16_causal1x0x0(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::true_type, cute::false_type, cute::false_type>;

  fna_reference_forward(
    static_cast<cutlass::bfloat16_t*>(ptr_Q),
    static_cast<cutlass::bfloat16_t*>(ptr_K),
    static_cast<cutlass::bfloat16_t*>(ptr_V),
    static_cast<cutlass::bfloat16_t*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_bfloat16_causal1x0x1(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::true_type, cute::false_type, cute::true_type>;

  fna_reference_forward(
    static_cast<cutlass::bfloat16_t*>(ptr_Q),
    static_cast<cutlass::bfloat16_t*>(ptr_K),
    static_cast<cutlass::bfloat16_t*>(ptr_V),
    static_cast<cutlass::bfloat16_t*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_bfloat16_causal1x1x0(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::true_type, cute::true_type, cute::false_type>;

  fna_reference_forward(
    static_cast<cutlass::bfloat16_t*>(ptr_Q),
    static_cast<cutlass::bfloat16_t*>(ptr_K),
    static_cast<cutlass::bfloat16_t*>(ptr_V),
    static_cast<cutlass::bfloat16_t*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_bfloat16_causal1x1x1(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::true_type, cute::true_type, cute::true_type>;

  fna_reference_forward(
    static_cast<cutlass::bfloat16_t*>(ptr_Q),
    static_cast<cutlass::bfloat16_t*>(ptr_K),
    static_cast<cutlass::bfloat16_t*>(ptr_V),
    static_cast<cutlass::bfloat16_t*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_backward_float32_causal0x0x0(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_DO,
      void* ptr_DQ,
      void* ptr_DK,
      void* ptr_DV,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::false_type, cute::false_type, cute::false_type>;

  fna_reference_backward(
    static_cast<float*>(ptr_Q),
    static_cast<float*>(ptr_K),
    static_cast<float*>(ptr_V),
    static_cast<float*>(ptr_O),
    static_cast<float*>(ptr_DO),
    static_cast<float*>(ptr_DQ),
    static_cast<float*>(ptr_DK),
    static_cast<float*>(ptr_DV),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_backward_float32_causal0x0x1(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_DO,
      void* ptr_DQ,
      void* ptr_DK,
      void* ptr_DV,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::false_type, cute::false_type, cute::true_type>;

  fna_reference_backward(
    static_cast<float*>(ptr_Q),
    static_cast<float*>(ptr_K),
    static_cast<float*>(ptr_V),
    static_cast<float*>(ptr_O),
    static_cast<float*>(ptr_DO),
    static_cast<float*>(ptr_DQ),
    static_cast<float*>(ptr_DK),
    static_cast<float*>(ptr_DV),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}





void reference_fna3d_backward_float32_causal0x1x0(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_DO,
      void* ptr_DQ,
      void* ptr_DK,
      void* ptr_DV,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      float attn_scale,
      cudaStream_t stream) {

  using Causal = cute::tuple<cute::false_type, cute::true_type, cute::false_type>;

  fna_reference_backward(
    static_cast<float*>(ptr_Q),
    static_cast<float*>(ptr_K),
    static_cast<float*>(ptr_V),
    static_cast<float*>(ptr_O),
    static_cast<float*>(ptr_DO),
    static_cast<float*>(ptr_DQ),
    static_cast<float*>(ptr_DK),
    static_cast<float*>(ptr_DV),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{},
    attn_scale,
    stream);
}


} // namespace reference 
} // namespace cuda 
} // namespace natten 
#endif 

