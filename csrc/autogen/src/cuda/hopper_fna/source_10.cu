#ifdef NATTEN_WITH_CUTLASS
#ifdef NATTEN_WITH_HOPPER_FNA
#include <cuda_runtime.h>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <natten/natten.h>
#include <natten/helpers.h>
#include <natten/cuda/fna_hopper/fna_forward.cuh>
#include <natten_autogen/cuda/hopper_fna/kernels.h>
namespace natten { 
namespace cuda { 
namespace fna_hopper { 




void hopper_fna3d_float16_128x64x256_coop_Q4x4x8_KV4x4x4_causal1x1x0(
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
      cute::tuple<int, int, int> q_shape,
      cute::tuple<int, int, int> kv_shape,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::true_type, cute::true_type, cute::false_type>;
  using QTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<8>>;
  using KVTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<4>>;
  using GemmShape = cute::tuple<cute::Int<128>, cute::Int<64>, cute::Int<256>>;
  using Kernel = natten::cuda::fna_hopper::KernelForward<
    cutlass::half_t, Causal, QTileShape, KVTileShape, GemmShape, natten::cuda::hopper::HopperKernelSchedule::WSCooperative>;

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





void hopper_fna3d_float16_128x64x256_coop_Q2x8x8_KV4x4x4_causal1x1x0(
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
      cute::tuple<int, int, int> q_shape,
      cute::tuple<int, int, int> kv_shape,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::true_type, cute::true_type, cute::false_type>;
  using QTileShape = cute::tuple<cute::Int<2>, cute::Int<8>, cute::Int<8>>;
  using KVTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<4>>;
  using GemmShape = cute::tuple<cute::Int<128>, cute::Int<64>, cute::Int<256>>;
  using Kernel = natten::cuda::fna_hopper::KernelForward<
    cutlass::half_t, Causal, QTileShape, KVTileShape, GemmShape, natten::cuda::hopper::HopperKernelSchedule::WSCooperative>;

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





void hopper_fna3d_float16_128x64x256_coop_Q4x4x8_KV4x4x4_causal1x1x1(
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
      cute::tuple<int, int, int> q_shape,
      cute::tuple<int, int, int> kv_shape,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::true_type, cute::true_type, cute::true_type>;
  using QTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<8>>;
  using KVTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<4>>;
  using GemmShape = cute::tuple<cute::Int<128>, cute::Int<64>, cute::Int<256>>;
  using Kernel = natten::cuda::fna_hopper::KernelForward<
    cutlass::half_t, Causal, QTileShape, KVTileShape, GemmShape, natten::cuda::hopper::HopperKernelSchedule::WSCooperative>;

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





void hopper_fna3d_float16_128x64x256_coop_Q2x8x8_KV4x4x4_causal1x1x1(
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
      cute::tuple<int, int, int> q_shape,
      cute::tuple<int, int, int> kv_shape,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::true_type, cute::true_type, cute::true_type>;
  using QTileShape = cute::tuple<cute::Int<2>, cute::Int<8>, cute::Int<8>>;
  using KVTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<4>>;
  using GemmShape = cute::tuple<cute::Int<128>, cute::Int<64>, cute::Int<256>>;
  using Kernel = natten::cuda::fna_hopper::KernelForward<
    cutlass::half_t, Causal, QTileShape, KVTileShape, GemmShape, natten::cuda::hopper::HopperKernelSchedule::WSCooperative>;

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





void hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal0x0x0(
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
      cute::tuple<int, int, int> q_shape,
      cute::tuple<int, int, int> kv_shape,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::false_type, cute::false_type, cute::false_type>;
  using QTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<4>>;
  using KVTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<8>>;
  using GemmShape = cute::tuple<cute::Int<64>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_hopper::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent>;

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





void hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal0x0x0(
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
      cute::tuple<int, int, int> q_shape,
      cute::tuple<int, int, int> kv_shape,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::false_type, cute::false_type, cute::false_type>;
  using QTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<4>>;
  using KVTileShape = cute::tuple<cute::Int<2>, cute::Int<8>, cute::Int<8>>;
  using GemmShape = cute::tuple<cute::Int<64>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_hopper::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent>;

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





void hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal0x0x1(
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
      cute::tuple<int, int, int> q_shape,
      cute::tuple<int, int, int> kv_shape,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::false_type, cute::false_type, cute::true_type>;
  using QTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<4>>;
  using KVTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<8>>;
  using GemmShape = cute::tuple<cute::Int<64>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_hopper::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent>;

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





void hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal0x0x1(
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
      cute::tuple<int, int, int> q_shape,
      cute::tuple<int, int, int> kv_shape,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::false_type, cute::false_type, cute::true_type>;
  using QTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<4>>;
  using KVTileShape = cute::tuple<cute::Int<2>, cute::Int<8>, cute::Int<8>>;
  using GemmShape = cute::tuple<cute::Int<64>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_hopper::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent>;

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





void hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal0x1x0(
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
      cute::tuple<int, int, int> q_shape,
      cute::tuple<int, int, int> kv_shape,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::false_type, cute::true_type, cute::false_type>;
  using QTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<4>>;
  using KVTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<8>>;
  using GemmShape = cute::tuple<cute::Int<64>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_hopper::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent>;

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





void hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal0x1x0(
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
      cute::tuple<int, int, int> q_shape,
      cute::tuple<int, int, int> kv_shape,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::false_type, cute::true_type, cute::false_type>;
  using QTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<4>>;
  using KVTileShape = cute::tuple<cute::Int<2>, cute::Int<8>, cute::Int<8>>;
  using GemmShape = cute::tuple<cute::Int<64>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_hopper::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent>;

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





void hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal0x1x1(
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
      cute::tuple<int, int, int> q_shape,
      cute::tuple<int, int, int> kv_shape,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::false_type, cute::true_type, cute::true_type>;
  using QTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<4>>;
  using KVTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<8>>;
  using GemmShape = cute::tuple<cute::Int<64>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_hopper::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent>;

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





void hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal0x1x1(
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
      cute::tuple<int, int, int> q_shape,
      cute::tuple<int, int, int> kv_shape,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::false_type, cute::true_type, cute::true_type>;
  using QTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<4>>;
  using KVTileShape = cute::tuple<cute::Int<2>, cute::Int<8>, cute::Int<8>>;
  using GemmShape = cute::tuple<cute::Int<64>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_hopper::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent>;

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





void hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal1x0x0(
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
      cute::tuple<int, int, int> q_shape,
      cute::tuple<int, int, int> kv_shape,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::true_type, cute::false_type, cute::false_type>;
  using QTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<4>>;
  using KVTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<8>>;
  using GemmShape = cute::tuple<cute::Int<64>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_hopper::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent>;

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





void hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal1x0x0(
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
      cute::tuple<int, int, int> q_shape,
      cute::tuple<int, int, int> kv_shape,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::true_type, cute::false_type, cute::false_type>;
  using QTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<4>>;
  using KVTileShape = cute::tuple<cute::Int<2>, cute::Int<8>, cute::Int<8>>;
  using GemmShape = cute::tuple<cute::Int<64>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_hopper::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent>;

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





void hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal1x0x1(
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
      cute::tuple<int, int, int> q_shape,
      cute::tuple<int, int, int> kv_shape,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::true_type, cute::false_type, cute::true_type>;
  using QTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<4>>;
  using KVTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<8>>;
  using GemmShape = cute::tuple<cute::Int<64>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_hopper::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent>;

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





void hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal1x0x1(
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
      cute::tuple<int, int, int> q_shape,
      cute::tuple<int, int, int> kv_shape,
      cute::tuple<int, int, int> qkv_shape,
      cute::tuple<int, int, int> window_size,
      cute::tuple<int, int, int> stride,
      cute::tuple<int, int, int> dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {

  using Causal = cute::tuple<cute::true_type, cute::false_type, cute::true_type>;
  using QTileShape = cute::tuple<cute::Int<4>, cute::Int<4>, cute::Int<4>>;
  using KVTileShape = cute::tuple<cute::Int<2>, cute::Int<8>, cute::Int<8>>;
  using GemmShape = cute::tuple<cute::Int<64>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fna_hopper::KernelForward<
    cutlass::bfloat16_t, Causal, QTileShape, KVTileShape, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent>;

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


} // namespace fna_hopper 
} // namespace cuda 
} // namespace natten 
#endif 
#endif 

