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

#include "natten/cuda/fna_blackwell/collective/fna_fusion_bwd.hpp"
#include "natten/cuda/fna_blackwell/device/fna_bwd_sm100.hpp"

namespace natten {
namespace cuda {
namespace fna_blackwell {

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using namespace cute;
using namespace cutlass::fna;

template <
    typename Element,
    class Causal,
    class QTileShape,
    class KVTileShape,
    class TileShape,
    bool kIsVarlen>
struct KernelBackward {
  static_assert(
      rank(QTileShape{}) == 1 || rank(QTileShape{}) == 2 ||
      rank(QTileShape{}) == 3);
  static_assert(rank(QTileShape{}) == rank(KVTileShape{}));
  static_assert(rank(QTileShape{}) == rank(Causal{}));
  using NADim = std::conditional_t<
      rank(QTileShape{}) == 3,
      cute::tuple<int, int, int>,
      std::conditional_t<
          rank(QTileShape{}) == 2,
          cute::tuple<int, int>,
          cute::tuple<int>>>;

  using ElementAccumulator = float;
  using VariableLength = cutlass::fmha::collective::VariableLength;

  // Q K D D_VO ((H_R, H_K) B)
  using ProblemShapeRegular =
      cute::tuple<int, int, int, int, cute::tuple<cute::tuple<int, int>, int>>;
  using ProblemShapeVarlen = cute::tuple<
      VariableLength,
      VariableLength,
      int,
      int,
      cute::tuple<cute::tuple<int, int>, int>>;
  using ProblemShapeType =
      std::conditional_t<kIsVarlen, ProblemShapeVarlen, ProblemShapeRegular>;

  using Operation = cutlass::fna::device::FnaBwdSm100<
      ProblemShapeType,
      Element,
      ElementAccumulator,
      TileShape,
      collective::NeighborhoodAttentionBackwardMask<Causal>,
      QTileShape,
      KVTileShape,
      NADim>;

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
      int heads_q,
      int heads_kv,
      int dim,
      float attn_scale,
      // fna parameters
      NADim q_shape,
      NADim kv_shape,
      NADim qkv_shape,
      NADim window_size,
      NADim stride,
      NADim dilation,
      // varlen parameters
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      void* ptr_token_layouts,
      // init/launch params
      int device_id) {
    auto problem_shape_regular = cute::make_tuple(
        seqlen_Q,
        seqlen_KV,
        // head dim is always either 32, 64, or 128 in natten, but it should
        // always meet the 128-bit alignment constraint
        dim,
        dim, // dim_value -- if different from dim, needs the MLA kernel
        cute::make_tuple(make_tuple(heads_q / heads_kv, heads_kv), batch));

    ProblemShapeType problem_shape_launch;
    decltype(problem_shape_regular) problem_shape_memory;

    if constexpr (kIsVarlen) {
      problem_shape_memory = problem_shape_regular;
      get<4, 1>(problem_shape_memory) = 1;

      get<0>(problem_shape_launch) = VariableLength{
          max_seqlen_Q,
          reinterpret_cast<int*>(ptr_cumulative_seqlen_Q),
          seqlen_Q};
      get<1>(problem_shape_launch) = VariableLength{
          max_seqlen_KV,
          reinterpret_cast<int*>(ptr_cumulative_seqlen_KV),
          seqlen_KV};
      get<2>(problem_shape_launch) = get<2>(problem_shape_regular);
      get<3>(problem_shape_launch) = get<3>(problem_shape_regular);
      get<4>(problem_shape_launch) = get<4>(problem_shape_regular);
    } else {
      problem_shape_memory = problem_shape_regular;
      problem_shape_launch = problem_shape_regular;
    }

    int SQ = size<0>(problem_shape_memory);
    int SK = size<1>(problem_shape_memory);
    int D = size<2>(problem_shape_memory);
    int D_VO = size<3>(problem_shape_memory);
    auto HB = get<4, 0>(problem_shape_memory);
    auto [H_R, H_K] = HB;
    int B = size<4, 1>(problem_shape_memory);

    // heads last profile, with torch's "contiguous layout"
    // shape: (batch, seqlen, heads, dim)
    // stride: (dim*heads*seqlen, dim*heads, dim, 1)
    auto stride_Q = make_stride(
        H_R * H_K * D,
        _1{},
        make_stride(make_stride(D, D * H_R), B == 1 ? 0 : D * SQ * H_R * H_K));
    auto stride_K = make_stride(
        H_K * D,
        _1{},
        make_stride(make_stride(_0{}, D), B == 1 ? 0 : D * SK * H_K));
    auto stride_V = make_stride(
        H_K * D_VO,
        _1{},
        make_stride(make_stride(_0{}, D_VO), B == 1 ? 0 : D_VO * SK * H_K));
    auto stride_O = make_stride(
        H_R * H_K * D_VO,
        _1{},
        make_stride(
            make_stride(D_VO, D_VO * H_R), B == 1 ? 0 : D_VO * SQ * H_R * H_K));
    auto stride_LSE = make_stride(
        H_K * H_R,
        make_stride(make_stride(_1{}, H_R), B == 1 ? 0 : SQ * H_R * H_K));

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
        q_shape,
        kv_shape,
        qkv_shape,
        window_size,
        stride,
        dilation,
        reinterpret_cast<NADim*>(ptr_token_layouts), // varlen only
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

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

} // namespace fna_blackwell
} // namespace cuda
} // namespace natten
