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
#include <natten/cuda/fmha_hopper/fmha_forward.cuh>
#include <natten_autogen/cuda/hopper_fmha/kernels.h>
namespace natten { 
namespace cuda { 
namespace fmha_hopper { 




void hopper_fmha_float16_64x128x32(
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
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<64>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fmha_hopper::KernelForward<
    cutlass::half_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent, false>;
  using KernelWithResidualMask = natten::cuda::fmha_hopper::KernelForward<
    cutlass::half_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent, true>;

  bool no_mask_required = seqlen_k % get<1>(GemmShape{}) == 0;
  if (no_mask_required) {
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  } else {
    KernelWithResidualMask kernel;
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  }
}





void hopper_fmha_float16_64x128x64(
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
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<64>, cute::Int<128>, cute::Int<64>>;
  using Kernel = natten::cuda::fmha_hopper::KernelForward<
    cutlass::half_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent, false>;
  using KernelWithResidualMask = natten::cuda::fmha_hopper::KernelForward<
    cutlass::half_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent, true>;

  bool no_mask_required = seqlen_k % get<1>(GemmShape{}) == 0;
  if (no_mask_required) {
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  } else {
    KernelWithResidualMask kernel;
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  }
}





void hopper_fmha_float16_128x128x128_coop(
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
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
  using Kernel = natten::cuda::fmha_hopper::KernelForward<
    cutlass::half_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::WSCooperative, false>;
  using KernelWithResidualMask = natten::cuda::fmha_hopper::KernelForward<
    cutlass::half_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::WSCooperative, true>;

  bool no_mask_required = seqlen_k % get<1>(GemmShape{}) == 0;
  if (no_mask_required) {
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  } else {
    KernelWithResidualMask kernel;
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  }
}





void hopper_fmha_float16_128x128x128_pp(
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
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
  using Kernel = natten::cuda::fmha_hopper::KernelForward<
    cutlass::half_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::WSPingpong, false>;
  using KernelWithResidualMask = natten::cuda::fmha_hopper::KernelForward<
    cutlass::half_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::WSPingpong, true>;

  bool no_mask_required = seqlen_k % get<1>(GemmShape{}) == 0;
  if (no_mask_required) {
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  } else {
    KernelWithResidualMask kernel;
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  }
}





void hopper_fmha_float16_128x64x256_coop(
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
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<128>, cute::Int<64>, cute::Int<256>>;
  using Kernel = natten::cuda::fmha_hopper::KernelForward<
    cutlass::half_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::WSCooperative, false>;
  using KernelWithResidualMask = natten::cuda::fmha_hopper::KernelForward<
    cutlass::half_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::WSCooperative, true>;

  bool no_mask_required = seqlen_k % get<1>(GemmShape{}) == 0;
  if (no_mask_required) {
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  } else {
    KernelWithResidualMask kernel;
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  }
}





void hopper_fmha_bfloat16_64x128x32(
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
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<64>, cute::Int<128>, cute::Int<32>>;
  using Kernel = natten::cuda::fmha_hopper::KernelForward<
    cutlass::bfloat16_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent, false>;
  using KernelWithResidualMask = natten::cuda::fmha_hopper::KernelForward<
    cutlass::bfloat16_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent, true>;

  bool no_mask_required = seqlen_k % get<1>(GemmShape{}) == 0;
  if (no_mask_required) {
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  } else {
    KernelWithResidualMask kernel;
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  }
}





void hopper_fmha_bfloat16_64x128x64(
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
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<64>, cute::Int<128>, cute::Int<64>>;
  using Kernel = natten::cuda::fmha_hopper::KernelForward<
    cutlass::bfloat16_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent, false>;
  using KernelWithResidualMask = natten::cuda::fmha_hopper::KernelForward<
    cutlass::bfloat16_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::NonPersistent, true>;

  bool no_mask_required = seqlen_k % get<1>(GemmShape{}) == 0;
  if (no_mask_required) {
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  } else {
    KernelWithResidualMask kernel;
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  }
}





void hopper_fmha_bfloat16_128x128x128_coop(
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
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
  using Kernel = natten::cuda::fmha_hopper::KernelForward<
    cutlass::bfloat16_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::WSCooperative, false>;
  using KernelWithResidualMask = natten::cuda::fmha_hopper::KernelForward<
    cutlass::bfloat16_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::WSCooperative, true>;

  bool no_mask_required = seqlen_k % get<1>(GemmShape{}) == 0;
  if (no_mask_required) {
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  } else {
    KernelWithResidualMask kernel;
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  }
}





void hopper_fmha_bfloat16_128x128x128_pp(
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
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<128>, cute::Int<128>, cute::Int<128>>;
  using Kernel = natten::cuda::fmha_hopper::KernelForward<
    cutlass::bfloat16_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::WSPingpong, false>;
  using KernelWithResidualMask = natten::cuda::fmha_hopper::KernelForward<
    cutlass::bfloat16_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::WSPingpong, true>;

  bool no_mask_required = seqlen_k % get<1>(GemmShape{}) == 0;
  if (no_mask_required) {
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  } else {
    KernelWithResidualMask kernel;
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  }
}





void hopper_fmha_bfloat16_128x64x256_coop(
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
      at::TensorOptions tensor_options) {

  using GemmShape = cute::tuple<cute::Int<128>, cute::Int<64>, cute::Int<256>>;
  using Kernel = natten::cuda::fmha_hopper::KernelForward<
    cutlass::bfloat16_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::WSCooperative, false>;
  using KernelWithResidualMask = natten::cuda::fmha_hopper::KernelForward<
    cutlass::bfloat16_t, GemmShape, natten::cuda::hopper::HopperKernelSchedule::WSCooperative, true>;

  bool no_mask_required = seqlen_k % get<1>(GemmShape{}) == 0;
  if (no_mask_required) {
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  } else {
    KernelWithResidualMask kernel;
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
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({bytes}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  }
}


} // namespace fmha_hopper 
} // namespace cuda 
} // namespace natten 
#endif 
#endif 

