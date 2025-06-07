/***************************************************************************************************
 * Copyright (c) 2022-2025 Ali Hassani.
 *
 * Fused Neighborhood Attention kernels are heavily based on the
 * memory-efficient attention kernels from the xFormers project by Meta
 * Platforms, Inc.
 *
 * Copyright (c) Facebook, Inc. and its affiliates
 *
 * BSD 3-Clause License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC
 * Laboratories America and IDIAP Research Institute nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

#include "natten/cuda/fmha_hopper/collective/fmha_fusion.hpp"
#include "natten/cuda/fmha_hopper/device/fmha_sm90.hpp"
#include "natten/cuda/fmha_hopper/kernel/fmha_kernel_builder.hpp"

namespace natten {
namespace cuda {
namespace fmha_hopper {

enum class HopperFmhaKernelType {
  NonPersistent,
  WSCooperative,
  WSPingpong,
  Invalid
};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha;

template <
    typename Element,
    class TileShape,
    HopperFmhaKernelType KernelSchedule,
    bool kIsResidual>
struct KernelForward {
  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;
  using ElementOut = Element;

  // B H Q K D
  using ProblemShapeType = cute::tuple<int, int, int, int, int>;

  using StrideQ = cute::tuple<int, _1, cute::tuple<int, int>>; // Q D (B H)
  using StrideK = cute::tuple<int, _1, cute::tuple<int, int>>; // K D (B H)

  using StrideV =
      StrideK; // NOTE: StrideV is different for FP8 due to transpose
  using StrideO = StrideQ;

  using StrideLSE = cute::tuple<int, cute::tuple<int, _1>>; // Q (B H)

  static_assert(
      KernelSchedule == HopperFmhaKernelType::NonPersistent ||
      KernelSchedule == HopperFmhaKernelType::WSCooperative ||
      KernelSchedule == HopperFmhaKernelType::WSPingpong);

  using DispatchPolicy = std::conditional_t<
      KernelSchedule == HopperFmhaKernelType::NonPersistent,
      cutlass::gemm::KernelTma,
      std::conditional_t<
          KernelSchedule == HopperFmhaKernelType::WSCooperative,
          cutlass::gemm::KernelTmaWarpSpecializedCooperative,
          cutlass::gemm::KernelTmaWarpSpecializedPingpong>>;

  using ActiveFusion = std::conditional_t<
      kIsResidual,
      cutlass::fmha::collective::ResidualFusion,
      cutlass::fmha::collective::DefaultFusion>;

  using Operation = cutlass::fmha::device::FmhaSm90<
      typename cutlass::fmha::kernel::FmhaBuilder<
          Element,
          ElementAccumulatorQK,
          ElementAccumulatorPV,
          TileShape,
          StrideQ,
          StrideK,
          StrideV,
          ActiveFusion,
          DispatchPolicy>::Kernel>;

  using Arguments = typename Operation::Arguments;

  Operation op;

  Arguments initialize(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch,
      int seqlen_Q,
      int seqlen_KV,
      int heads,
      int dim,
      int device_id,
      float attn_scale) {
    auto dim_aligned = cutlass::round_up(dim, 8); // alignment
    ProblemShapeType problem_size = ProblemShapeType{
        batch,
        heads,
        seqlen_Q,
        seqlen_KV,
        dim_aligned,
    };

    // heads last profile, with torch's "contiguous layout"
    // shape: (batch, heads, seqlen, dim)
    // stride: (dim*heads*seqlen, dim*heads, dim, 1)
    auto stride_Q = make_stride(
        heads * dim_aligned,
        _1{},
        make_stride(heads * seqlen_Q * dim_aligned, dim_aligned));
    auto stride_O = stride_Q;
    auto stride_K = make_stride(
        heads * dim_aligned,
        _1{},
        make_stride(heads * seqlen_KV * dim_aligned, dim_aligned));
    auto stride_V = stride_K;
    auto stride_LSE = make_stride(heads, make_stride(heads * seqlen_Q, _1{}));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);

    Arguments arguments{
        problem_size,
        {reinterpret_cast<Element*>(ptr_Q),
         stride_Q,
         reinterpret_cast<Element*>(ptr_K),
         stride_K,
         reinterpret_cast<Element*>(ptr_V),
         stride_V,
         attn_scale},
        {reinterpret_cast<Element*>(ptr_O),
         stride_O,
         reinterpret_cast<ElementAccumulatorPV*>(ptr_LSE),
         stride_LSE},
        hw_info};

    return arguments;
  }

  size_t get_workspace_size(Arguments const& arguments) {
    return Operation::get_workspace_size(arguments);
  }

  void run(
      Arguments const& arguments,
      void* workspace_ptr,
      cudaStream_t stream) {
    cutlass::Status status = cutlass::Status::kSuccess;
    status = op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "This kernel is not supported. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }

    status = op.initialize(arguments, workspace_ptr, stream);
    if (status != cutlass::Status::kSuccess) {
      std::cerr
          << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
          << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }

    // Run
    status = op.run(stream);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }

#if 0
    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error running the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
    }
#endif
  }
};

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

} // namespace fmha_hopper
} // namespace cuda
} // namespace natten
