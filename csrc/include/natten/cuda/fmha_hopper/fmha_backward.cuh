/***************************************************************************************************
 * Copyright (c) 2022 - 2026 Ali Hassani.
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
#include "natten/cuda/fmha_hopper/device/fmha_bwd_sm90.hpp"

namespace natten {
namespace cuda {
namespace fmha_hopper {

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

using namespace cute;
using namespace cutlass::fmha;

template <typename Element, class TileShape, class Mask, bool kIsVarlen>
struct KernelBackward {
  using ElementAccumulator = float;
  using VariableLength = cutlass::fmha::collective::VariableLength;

  // B H Q K D
  using ProblemShapeRegular = cute::tuple<int, int, int, int, int>;
  using ProblemShapeVarlen =
      cute::tuple<int, int, VariableLength, VariableLength, int>;
  using ProblemShapeType =
      std::conditional_t<kIsVarlen, ProblemShapeVarlen, ProblemShapeRegular>;

  using Operation = cutlass::fmha::device::FmhaBwdSm90<
      ProblemShapeType,
      Element,
      ElementAccumulator,
      TileShape,
      Mask>;

  using Arguments = typename Operation::Arguments;

  Operation op;

  Arguments initialize(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      void* ptr_dQ,
      void* ptr_dK,
      void* ptr_dV,
      void* ptr_dO,
      int batch,
      int seqlen_Q,
      int seqlen_KV,
      int seqlen_LSE, // seqlen_LSE is usually = seqlen_Q, but it may require
                      // extra padding.
      int heads,
      int dim,
      float attn_scale,
      // varlen parameters
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      // init/launch params
      int device_id) {
    auto dim_aligned = cutlass::round_up(dim, 8); // alignment

    auto problem_shape_regular = ProblemShapeRegular{
        batch,
        heads,
        seqlen_Q,
        seqlen_KV,
        dim_aligned,
    };

    ProblemShapeType problem_shape_launch;
    decltype(problem_shape_regular) problem_shape_memory;

    if constexpr (kIsVarlen) {
      // Varlen mode: batch becomes 1 (packed sequence layout)
      // and Q/KV dimensions use VariableLength descriptors
      problem_shape_memory = problem_shape_regular;
      get<0>(problem_shape_memory) = 1; // batch becomes 1 in varlen mode
      get<1>(problem_shape_launch) = get<1>(problem_shape_regular);

      get<2>(problem_shape_launch) = VariableLength{
          max_seqlen_Q,
          reinterpret_cast<int*>(ptr_cumulative_seqlen_Q),
          seqlen_Q};
      get<3>(problem_shape_launch) = VariableLength{
          max_seqlen_KV,
          reinterpret_cast<int*>(ptr_cumulative_seqlen_KV),
          seqlen_KV};
      get<4>(problem_shape_launch) = get<4>(problem_shape_regular);
    } else {
      problem_shape_memory = problem_shape_regular;
      problem_shape_launch = problem_shape_regular;
    }

    int B = size<0>(problem_shape_memory);
    int H = size<1>(problem_shape_memory);
    int SQ = size<2>(problem_shape_memory);
    int SK = size<3>(problem_shape_memory);
    int D = size<4>(problem_shape_memory);

    // heads last profile, with torch's "contiguous layout"
    // shape: (batch, heads, seqlen, dim)
    // stride: (dim*heads*seqlen, dim*heads, dim, 1)
    auto stride_Q = make_stride(
        B == 1 ? 0
               : static_cast<int64_t>(dim_aligned * heads) *
                static_cast<int64_t>(seqlen_Q),
        dim_aligned,
        dim_aligned * heads,
        _1{});
    auto stride_O = stride_Q;
    auto stride_K = make_stride(
        B == 1 ? 0
               : static_cast<int64_t>(dim_aligned * heads) *
                static_cast<int64_t>(seqlen_KV),
        dim_aligned,
        dim_aligned * heads,
        _1{});
    auto stride_V = stride_K;
    auto stride_LSE = make_stride(
        B == 1 ? 0
               : static_cast<int64_t>(heads) * static_cast<int64_t>(seqlen_LSE),
        seqlen_LSE,
        _1{});

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);

    Arguments arguments{
        problem_shape_launch,
        reinterpret_cast<Element*>(ptr_Q),
        stride_Q,
        reinterpret_cast<Element*>(ptr_K),
        stride_K,
        reinterpret_cast<Element*>(ptr_V),
        stride_V,
        reinterpret_cast<Element*>(ptr_O),
        stride_O,
        reinterpret_cast<ElementAccumulator*>(ptr_LSE),
        stride_LSE,
        reinterpret_cast<Element*>(ptr_dO),
        stride_O,
        reinterpret_cast<Element*>(ptr_dQ),
        stride_Q,
        reinterpret_cast<Element*>(ptr_dK),
        stride_K,
        reinterpret_cast<Element*>(ptr_dV),
        stride_V,
        attn_scale,
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
  }
};

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

} // namespace fmha_hopper
} // namespace cuda
} // namespace natten
