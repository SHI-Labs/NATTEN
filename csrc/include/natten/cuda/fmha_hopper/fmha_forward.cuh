/***************************************************************************************************
 * Copyright (c) 2022-2025 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 **************************************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"

#include "natten/cuda/fmha_hopper/collective/fmha_fusion.hpp"
#include "natten/cuda/fmha_hopper/device/fmha_sm90.hpp"
#include "natten/cuda/fmha_hopper/kernel/fmha_kernel_builder.hpp"

#include "natten/cuda/hopper_fmha_fna.h"

namespace natten {
namespace cuda {
namespace fmha_hopper {

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha;

template <
    typename Element,
    class TileShape,
    natten::cuda::hopper::HopperKernelSchedule KernelSchedule,
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
      KernelSchedule ==
          natten::cuda::hopper::HopperKernelSchedule::NonPersistent ||
      KernelSchedule ==
          natten::cuda::hopper::HopperKernelSchedule::WSCooperative ||
      KernelSchedule == natten::cuda::hopper::HopperKernelSchedule::WSPingpong);

  using DispatchPolicy = std::conditional_t<
      KernelSchedule ==
          natten::cuda::hopper::HopperKernelSchedule::NonPersistent,
      cutlass::gemm::KernelTma,
      std::conditional_t<
          KernelSchedule ==
              natten::cuda::hopper::HopperKernelSchedule::WSCooperative,
          cutlass::gemm::KernelTmaWarpSpecializedCooperative,
          cutlass::gemm::KernelTmaWarpSpecializedPingpong>>;

  using ActiveFusion = std::conditional_t<
      kIsResidual,
      cutlass::fmha::collective::ResidualFusion,
      cutlass::fmha::collective::DefaultFusion>;

  using Operation = cutlass::fmha::device::Sm90DeviceKernel<
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
      return;
    }

    status = op.initialize(arguments, workspace_ptr, stream);
    if (status != cutlass::Status::kSuccess) {
      std::cerr
          << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
          << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return;
    }

    // Run
    status = op.run(stream);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
      return;
    }

#if 0
    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error running the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
      return;
    }
#endif
  }
};

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

} // namespace fmha_hopper
} // namespace cuda
} // namespace natten
