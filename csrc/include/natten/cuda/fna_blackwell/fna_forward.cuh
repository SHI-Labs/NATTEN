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

#include "natten/cuda/fna_blackwell/collective/fna_fusion.hpp"
#include "natten/cuda/fna_blackwell/collective/sm100_fna_fwd_epilogue_tma_warpspecialized.hpp"
#include "natten/cuda/fna_blackwell/collective/sm100_fna_fwd_mainloop_tma_warpspecialized.hpp"
#include "natten/cuda/fna_blackwell/device/fna_sm100.hpp"
#include "natten/cuda/fna_blackwell/kernel/fna_tile_scheduler.hpp"
#include "natten/cuda/fna_blackwell/kernel/sm100_fna_fwd_kernel_tma_warpspecialized.hpp"

namespace natten {
namespace cuda {
namespace fna_blackwell {

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

using namespace cute;
using namespace cutlass::fna::kernel;
using namespace cutlass::fna::collective;
using namespace cutlass::fna;

template <
    typename Element,
    class Causal,
    class QTileShape,
    class KVTileShape,
    class TileShape,
    bool kIsPersistent>
struct KernelForward {
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

  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;
  using ElementOut = Element;

  // Q K D (B H)
  using ProblemShapeType =
      cute::tuple<int, int, int, cute::tuple<cute::tuple<int, int>, int>>;

  using StrideQ =
      cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>; // Q D (H_G
                                                                     // H_R B)
  using StrideK =
      cute::tuple<int, _1, cute::tuple<cute::tuple<_0, int>, int>>; // K D (H_G
                                                                    // H_R B)
  using StrideV = StrideK;
  using StrideO = StrideQ;
  using StrideLSE =
      cute::tuple<int, cute::tuple<cute::tuple<_1, int>, int>>; // Q   (H_G H_R
                                                                // B)

  using TileScheduler = std::conditional_t<
      kIsPersistent,
      cutlass::fna::kernel::PersistentTileScheduler,
      cutlass::fna::kernel::IndividualTileScheduler>;

  using Mainloop =
      cutlass::fna::collective::Sm100FnaFwdMainloopTmaWarpspecialized<
          Element,
          ElementAccumulatorQK,
          ElementAccumulatorPV,
          TileShape,
          StrideQ,
          StrideK,
          StrideV,
          NeighborhoodAttentionMask<Causal>,
          QTileShape,
          KVTileShape,
          NADim>;
  using Operation = cutlass::fna::device::FnaSm100<
      cutlass::fna::kernel::Sm100FnaFwdKernelTmaWarpspecialized<
          ProblemShapeType,
          Mainloop,
          cutlass::fna::collective::Sm100FnaFwdEpilogueTmaWarpspecialized<
              ElementOut,
              ElementAccumulatorPV,
              typename Mainloop::TileShapePV,
              StrideO,
              StrideLSE>,
          TileScheduler>>;

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
      int num_extra_kv,
      NADim q_shape,
      NADim kv_shape,
      NADim qkv_shape,
      NADim window_size,
      NADim stride,
      NADim dilation,
      int device_id,
      float attn_scale) {
    // No GQA/MQA for now
    auto problem_shape_in = cute::make_tuple(
        seqlen_Q,
        seqlen_KV,
        dim,
        cute::make_tuple(cute::make_tuple(1, heads), batch));

    ProblemShapeType problem_shape;
    decltype(problem_shape_in) problem_size;

    problem_size = problem_shape_in;
    problem_shape = problem_shape_in;

    get<2>(problem_size) =
        cutlass::round_up(get<2>(problem_size), 8); // alignment

    int SQ = size<0>(problem_size);
    int SK = size<1>(problem_size);
    int D = size<2>(problem_size);
    int H = size<3, 0>(problem_size);
    int H_K = size<3, 0, 1>(problem_size);
    int H_Q = size<3, 0, 0>(problem_size);
    int B = size<3, 1>(problem_size);

    int num_heads_actual = heads / size(dilation);

    // heads last profile, with torch's "contiguous layout"
    // shape: (batch, seqlen, heads, dim)
    // stride: (dim*heads*seqlen, dim*heads, dim, 1)
    auto stride_Q = make_stride(
        H * D, _1{}, make_stride(make_stride(D, H_Q * D), H * D * SQ));
    auto stride_O = stride_Q;
    auto stride_K = make_stride(
        H_K * D, _1{}, make_stride(make_stride(_0{}, D), H_K * D * SK));
    auto stride_V = stride_K;
    auto stride_LSE =
        make_stride(H, make_stride(make_stride(_1{}, H_Q), SQ * H));

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = device_id;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);

    Arguments arguments{
        problem_shape,
        {reinterpret_cast<Element*>(ptr_Q),
         stride_Q,
         reinterpret_cast<Element*>(ptr_K),
         stride_K,
         reinterpret_cast<Element*>(ptr_V),
         stride_V,
         q_shape,
         kv_shape,
         qkv_shape,
         window_size,
         stride,
         dilation,
         num_heads_actual,
         num_extra_kv,
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

#endif // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

} // namespace fna_blackwell
} // namespace cuda
} // namespace natten
