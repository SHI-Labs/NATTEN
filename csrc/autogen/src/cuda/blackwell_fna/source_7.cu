#ifdef NATTEN_WITH_CUTLASS
#ifdef NATTEN_WITH_BLACKWELL_FNA
#include <cuda_runtime.h>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <natten/natten.h>
#include <natten/helpers.h>
#include <natten/cuda/fna_blackwell/fna_forward.cuh>
#include <natten_autogen/cuda/blackwell_fna/kernels.h>
namespace natten { 
namespace cuda { 
namespace fna_blackwell { 




void blackwell_fna2d_float16_256x128x128_Q16x16_KV16x8_causal1x1(
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
      int num_extra_kv,
      cute::tuple<int, int> q_shape,
      cute::tuple<int, int> kv_shape,
      cute::tuple<int, int> qkv_shape,
      cute::tuple<int, int> window_size,
      cute::tuple<int, int> stride,
      cute::tuple<int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::true_type, cute::true_type>;
  using QTileShape = cute::tuple<cute::Int<16>, cute::Int<16>>;
  using KVTileShape = cute::tuple<cute::Int<16>, cute::Int<8>>;
  using Config = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<128>>;
  using Kernel = natten::cuda::fna_blackwell::KernelForward<
    cutlass::half_t, Causal, QTileShape, KVTileShape, Config, false>;

  Kernel kernel;
  auto args = kernel.initialize(
      ptr_Q,
      ptr_K,
      ptr_V,
      ptr_O,
      ptr_LSE,
      batch_size,
      seqlen_q,
      seqlen_k,
      heads,
      dim,
      num_extra_kv,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale);

  auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
  auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  kernel.run(args, workspace_ptr, stream);
}





void blackwell_fna2d_float16_256x128x128_Q16x16_KV16x8_causal1x1_persistent(
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
      int num_extra_kv,
      cute::tuple<int, int> q_shape,
      cute::tuple<int, int> kv_shape,
      cute::tuple<int, int> qkv_shape,
      cute::tuple<int, int> window_size,
      cute::tuple<int, int> stride,
      cute::tuple<int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::true_type, cute::true_type>;
  using QTileShape = cute::tuple<cute::Int<16>, cute::Int<16>>;
  using KVTileShape = cute::tuple<cute::Int<16>, cute::Int<8>>;
  using Config = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<128>>;
  using Kernel = natten::cuda::fna_blackwell::KernelForward<
    cutlass::half_t, Causal, QTileShape, KVTileShape, Config, true>;

  Kernel kernel;
  auto args = kernel.initialize(
      ptr_Q,
      ptr_K,
      ptr_V,
      ptr_O,
      ptr_LSE,
      batch_size,
      seqlen_q,
      seqlen_k,
      heads,
      dim,
      num_extra_kv,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale);

  auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
  auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  kernel.run(args, workspace_ptr, stream);
}





void blackwell_fna2d_float16_256x128x128_Q16x16_KV8x16_causal1x1(
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
      int num_extra_kv,
      cute::tuple<int, int> q_shape,
      cute::tuple<int, int> kv_shape,
      cute::tuple<int, int> qkv_shape,
      cute::tuple<int, int> window_size,
      cute::tuple<int, int> stride,
      cute::tuple<int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::true_type, cute::true_type>;
  using QTileShape = cute::tuple<cute::Int<16>, cute::Int<16>>;
  using KVTileShape = cute::tuple<cute::Int<8>, cute::Int<16>>;
  using Config = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<128>>;
  using Kernel = natten::cuda::fna_blackwell::KernelForward<
    cutlass::half_t, Causal, QTileShape, KVTileShape, Config, false>;

  Kernel kernel;
  auto args = kernel.initialize(
      ptr_Q,
      ptr_K,
      ptr_V,
      ptr_O,
      ptr_LSE,
      batch_size,
      seqlen_q,
      seqlen_k,
      heads,
      dim,
      num_extra_kv,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale);

  auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
  auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  kernel.run(args, workspace_ptr, stream);
}





void blackwell_fna2d_float16_256x128x128_Q16x16_KV8x16_causal1x1_persistent(
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
      int num_extra_kv,
      cute::tuple<int, int> q_shape,
      cute::tuple<int, int> kv_shape,
      cute::tuple<int, int> qkv_shape,
      cute::tuple<int, int> window_size,
      cute::tuple<int, int> stride,
      cute::tuple<int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::true_type, cute::true_type>;
  using QTileShape = cute::tuple<cute::Int<16>, cute::Int<16>>;
  using KVTileShape = cute::tuple<cute::Int<8>, cute::Int<16>>;
  using Config = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<128>>;
  using Kernel = natten::cuda::fna_blackwell::KernelForward<
    cutlass::half_t, Causal, QTileShape, KVTileShape, Config, true>;

  Kernel kernel;
  auto args = kernel.initialize(
      ptr_Q,
      ptr_K,
      ptr_V,
      ptr_O,
      ptr_LSE,
      batch_size,
      seqlen_q,
      seqlen_k,
      heads,
      dim,
      num_extra_kv,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale);

  auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
  auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  kernel.run(args, workspace_ptr, stream);
}





void blackwell_fna2d_float16_256x128x128_Q8x32_KV8x16_causal1x1(
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
      int num_extra_kv,
      cute::tuple<int, int> q_shape,
      cute::tuple<int, int> kv_shape,
      cute::tuple<int, int> qkv_shape,
      cute::tuple<int, int> window_size,
      cute::tuple<int, int> stride,
      cute::tuple<int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::true_type, cute::true_type>;
  using QTileShape = cute::tuple<cute::Int<8>, cute::Int<32>>;
  using KVTileShape = cute::tuple<cute::Int<8>, cute::Int<16>>;
  using Config = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<128>>;
  using Kernel = natten::cuda::fna_blackwell::KernelForward<
    cutlass::half_t, Causal, QTileShape, KVTileShape, Config, false>;

  Kernel kernel;
  auto args = kernel.initialize(
      ptr_Q,
      ptr_K,
      ptr_V,
      ptr_O,
      ptr_LSE,
      batch_size,
      seqlen_q,
      seqlen_k,
      heads,
      dim,
      num_extra_kv,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale);

  auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
  auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  kernel.run(args, workspace_ptr, stream);
}





void blackwell_fna2d_float16_256x128x128_Q8x32_KV8x16_causal1x1_persistent(
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
      int num_extra_kv,
      cute::tuple<int, int> q_shape,
      cute::tuple<int, int> kv_shape,
      cute::tuple<int, int> qkv_shape,
      cute::tuple<int, int> window_size,
      cute::tuple<int, int> stride,
      cute::tuple<int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::true_type, cute::true_type>;
  using QTileShape = cute::tuple<cute::Int<8>, cute::Int<32>>;
  using KVTileShape = cute::tuple<cute::Int<8>, cute::Int<16>>;
  using Config = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<128>>;
  using Kernel = natten::cuda::fna_blackwell::KernelForward<
    cutlass::half_t, Causal, QTileShape, KVTileShape, Config, true>;

  Kernel kernel;
  auto args = kernel.initialize(
      ptr_Q,
      ptr_K,
      ptr_V,
      ptr_O,
      ptr_LSE,
      batch_size,
      seqlen_q,
      seqlen_k,
      heads,
      dim,
      num_extra_kv,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale);

  auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
  auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  kernel.run(args, workspace_ptr, stream);
}





void blackwell_fna2d_float16_256x128x128_Q8x32_KV4x32_causal1x1(
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
      int num_extra_kv,
      cute::tuple<int, int> q_shape,
      cute::tuple<int, int> kv_shape,
      cute::tuple<int, int> qkv_shape,
      cute::tuple<int, int> window_size,
      cute::tuple<int, int> stride,
      cute::tuple<int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::true_type, cute::true_type>;
  using QTileShape = cute::tuple<cute::Int<8>, cute::Int<32>>;
  using KVTileShape = cute::tuple<cute::Int<4>, cute::Int<32>>;
  using Config = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<128>>;
  using Kernel = natten::cuda::fna_blackwell::KernelForward<
    cutlass::half_t, Causal, QTileShape, KVTileShape, Config, false>;

  Kernel kernel;
  auto args = kernel.initialize(
      ptr_Q,
      ptr_K,
      ptr_V,
      ptr_O,
      ptr_LSE,
      batch_size,
      seqlen_q,
      seqlen_k,
      heads,
      dim,
      num_extra_kv,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale);

  auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
  auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  kernel.run(args, workspace_ptr, stream);
}





void blackwell_fna2d_float16_256x128x128_Q8x32_KV4x32_causal1x1_persistent(
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
      int num_extra_kv,
      cute::tuple<int, int> q_shape,
      cute::tuple<int, int> kv_shape,
      cute::tuple<int, int> qkv_shape,
      cute::tuple<int, int> window_size,
      cute::tuple<int, int> stride,
      cute::tuple<int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::true_type, cute::true_type>;
  using QTileShape = cute::tuple<cute::Int<8>, cute::Int<32>>;
  using KVTileShape = cute::tuple<cute::Int<4>, cute::Int<32>>;
  using Config = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<128>>;
  using Kernel = natten::cuda::fna_blackwell::KernelForward<
    cutlass::half_t, Causal, QTileShape, KVTileShape, Config, true>;

  Kernel kernel;
  auto args = kernel.initialize(
      ptr_Q,
      ptr_K,
      ptr_V,
      ptr_O,
      ptr_LSE,
      batch_size,
      seqlen_q,
      seqlen_k,
      heads,
      dim,
      num_extra_kv,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale);

  auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
  auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  kernel.run(args, workspace_ptr, stream);
}





void blackwell_fna2d_bfloat16_256x128x32_Q16x16_KV16x8_causal0x0(
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
      int num_extra_kv,
      cute::tuple<int, int> q_shape,
      cute::tuple<int, int> kv_shape,
      cute::tuple<int, int> qkv_shape,
      cute::tuple<int, int> window_size,
      cute::tuple<int, int> stride,
      cute::tuple<int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::false_type, cute::false_type>;
  using QTileShape = cute::tuple<cute::Int<16>, cute::Int<16>>;
  using KVTileShape = cute::tuple<cute::Int<16>, cute::Int<8>>;
  using Config = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_blackwell::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, Config, false>;

  Kernel kernel;
  auto args = kernel.initialize(
      ptr_Q,
      ptr_K,
      ptr_V,
      ptr_O,
      ptr_LSE,
      batch_size,
      seqlen_q,
      seqlen_k,
      heads,
      dim,
      num_extra_kv,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale);

  auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
  auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  kernel.run(args, workspace_ptr, stream);
}





void blackwell_fna2d_bfloat16_256x128x32_Q16x16_KV16x8_causal0x0_persistent(
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
      int num_extra_kv,
      cute::tuple<int, int> q_shape,
      cute::tuple<int, int> kv_shape,
      cute::tuple<int, int> qkv_shape,
      cute::tuple<int, int> window_size,
      cute::tuple<int, int> stride,
      cute::tuple<int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::false_type, cute::false_type>;
  using QTileShape = cute::tuple<cute::Int<16>, cute::Int<16>>;
  using KVTileShape = cute::tuple<cute::Int<16>, cute::Int<8>>;
  using Config = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_blackwell::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, Config, true>;

  Kernel kernel;
  auto args = kernel.initialize(
      ptr_Q,
      ptr_K,
      ptr_V,
      ptr_O,
      ptr_LSE,
      batch_size,
      seqlen_q,
      seqlen_k,
      heads,
      dim,
      num_extra_kv,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale);

  auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
  auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  kernel.run(args, workspace_ptr, stream);
}





void blackwell_fna2d_bfloat16_256x128x32_Q16x16_KV8x16_causal0x0(
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
      int num_extra_kv,
      cute::tuple<int, int> q_shape,
      cute::tuple<int, int> kv_shape,
      cute::tuple<int, int> qkv_shape,
      cute::tuple<int, int> window_size,
      cute::tuple<int, int> stride,
      cute::tuple<int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::false_type, cute::false_type>;
  using QTileShape = cute::tuple<cute::Int<16>, cute::Int<16>>;
  using KVTileShape = cute::tuple<cute::Int<8>, cute::Int<16>>;
  using Config = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_blackwell::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, Config, false>;

  Kernel kernel;
  auto args = kernel.initialize(
      ptr_Q,
      ptr_K,
      ptr_V,
      ptr_O,
      ptr_LSE,
      batch_size,
      seqlen_q,
      seqlen_k,
      heads,
      dim,
      num_extra_kv,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale);

  auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
  auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  kernel.run(args, workspace_ptr, stream);
}





void blackwell_fna2d_bfloat16_256x128x32_Q16x16_KV8x16_causal0x0_persistent(
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
      int num_extra_kv,
      cute::tuple<int, int> q_shape,
      cute::tuple<int, int> kv_shape,
      cute::tuple<int, int> qkv_shape,
      cute::tuple<int, int> window_size,
      cute::tuple<int, int> stride,
      cute::tuple<int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::false_type, cute::false_type>;
  using QTileShape = cute::tuple<cute::Int<16>, cute::Int<16>>;
  using KVTileShape = cute::tuple<cute::Int<8>, cute::Int<16>>;
  using Config = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_blackwell::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, Config, true>;

  Kernel kernel;
  auto args = kernel.initialize(
      ptr_Q,
      ptr_K,
      ptr_V,
      ptr_O,
      ptr_LSE,
      batch_size,
      seqlen_q,
      seqlen_k,
      heads,
      dim,
      num_extra_kv,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale);

  auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
  auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  kernel.run(args, workspace_ptr, stream);
}





void blackwell_fna2d_bfloat16_256x128x32_Q8x32_KV8x16_causal0x0(
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
      int num_extra_kv,
      cute::tuple<int, int> q_shape,
      cute::tuple<int, int> kv_shape,
      cute::tuple<int, int> qkv_shape,
      cute::tuple<int, int> window_size,
      cute::tuple<int, int> stride,
      cute::tuple<int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::false_type, cute::false_type>;
  using QTileShape = cute::tuple<cute::Int<8>, cute::Int<32>>;
  using KVTileShape = cute::tuple<cute::Int<8>, cute::Int<16>>;
  using Config = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_blackwell::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, Config, false>;

  Kernel kernel;
  auto args = kernel.initialize(
      ptr_Q,
      ptr_K,
      ptr_V,
      ptr_O,
      ptr_LSE,
      batch_size,
      seqlen_q,
      seqlen_k,
      heads,
      dim,
      num_extra_kv,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale);

  auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
  auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  kernel.run(args, workspace_ptr, stream);
}





void blackwell_fna2d_bfloat16_256x128x32_Q8x32_KV8x16_causal0x0_persistent(
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
      int num_extra_kv,
      cute::tuple<int, int> q_shape,
      cute::tuple<int, int> kv_shape,
      cute::tuple<int, int> qkv_shape,
      cute::tuple<int, int> window_size,
      cute::tuple<int, int> stride,
      cute::tuple<int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::false_type, cute::false_type>;
  using QTileShape = cute::tuple<cute::Int<8>, cute::Int<32>>;
  using KVTileShape = cute::tuple<cute::Int<8>, cute::Int<16>>;
  using Config = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_blackwell::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, Config, true>;

  Kernel kernel;
  auto args = kernel.initialize(
      ptr_Q,
      ptr_K,
      ptr_V,
      ptr_O,
      ptr_LSE,
      batch_size,
      seqlen_q,
      seqlen_k,
      heads,
      dim,
      num_extra_kv,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale);

  auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
  auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  kernel.run(args, workspace_ptr, stream);
}





void blackwell_fna2d_bfloat16_256x128x32_Q8x32_KV4x32_causal0x0(
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
      int num_extra_kv,
      cute::tuple<int, int> q_shape,
      cute::tuple<int, int> kv_shape,
      cute::tuple<int, int> qkv_shape,
      cute::tuple<int, int> window_size,
      cute::tuple<int, int> stride,
      cute::tuple<int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::false_type, cute::false_type>;
  using QTileShape = cute::tuple<cute::Int<8>, cute::Int<32>>;
  using KVTileShape = cute::tuple<cute::Int<4>, cute::Int<32>>;
  using Config = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_blackwell::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, Config, false>;

  Kernel kernel;
  auto args = kernel.initialize(
      ptr_Q,
      ptr_K,
      ptr_V,
      ptr_O,
      ptr_LSE,
      batch_size,
      seqlen_q,
      seqlen_k,
      heads,
      dim,
      num_extra_kv,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale);

  auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
  auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  kernel.run(args, workspace_ptr, stream);
}





void blackwell_fna2d_bfloat16_256x128x32_Q8x32_KV4x32_causal0x0_persistent(
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
      int num_extra_kv,
      cute::tuple<int, int> q_shape,
      cute::tuple<int, int> kv_shape,
      cute::tuple<int, int> qkv_shape,
      cute::tuple<int, int> window_size,
      cute::tuple<int, int> stride,
      cute::tuple<int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::false_type, cute::false_type>;
  using QTileShape = cute::tuple<cute::Int<8>, cute::Int<32>>;
  using KVTileShape = cute::tuple<cute::Int<4>, cute::Int<32>>;
  using Config = cute::tuple<cute::Int<256>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_blackwell::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, Config, true>;

  Kernel kernel;
  auto args = kernel.initialize(
      ptr_Q,
      ptr_K,
      ptr_V,
      ptr_O,
      ptr_LSE,
      batch_size,
      seqlen_q,
      seqlen_k,
      heads,
      dim,
      num_extra_kv,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale);

  auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
  auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  kernel.run(args, workspace_ptr, stream);
}


} // namespace fna_blackwell 
} // namespace cuda 
} // namespace natten 
#endif 
#endif 

