/***************************************************************************************************
 * Copyright (c) 2023 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
/*! \file
    \brief  GEMM-based Neighborhood Attention kernels.
    
      This header should be the only header that source files include.
      It introduces a single templated kernel, and source files will instantiate it.
*/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

#include "natten/cuda/gemm/utils.cuh"
#include "natten/cuda/gemm/neighborhood_attention.cuh"
#include "natten/cuda/gemm/kernel/default_na1d_pn.cuh"
#include "natten/cuda/gemm/kernel/default_na1d_nn.cuh"
#include "natten/cuda/gemm/kernel/default_na1d_in.cuh"
#include "natten/cuda/gemm/device/implicit_gemm_na1d.cuh"
#include "natten/cuda/gemm/threadblock/threadblock_swizzle.cuh"

namespace natten {
namespace cuda {
namespace gemm {

template <typename DeviceKernel> 
struct NA1DLauncher {
  using ElementOperand = typename DeviceKernel::ElementA; // TODO: add static assert on element types?
  using ElementOutput  = typename DeviceKernel::ElementC;
  using ElementCompute = typename DeviceKernel::ElementCompute;

  using RefOperand = typename cutlass::TensorRef<ElementOperand, cutlass::layout::TensorNHWC>;
  using RefOutput  = typename cutlass::TensorRef<ElementOutput,  cutlass::layout::TensorNHWC>;
  using RefBias    = typename cutlass::TensorRef<ElementOutput,  cutlass::layout::RowMajor>;

  private:
    void launch_cutlass_kernel(
      RefOperand const & ref_a,
      RefOperand const & ref_b,
      RefOutput const & ref_c,
      RefBias const & ref_bias,
      const int batch_size,
      const int heads,
      const int length,
      const int dim,
      const int kernel_size,
      const int dilation,
      const int tile_size,
      const int tile_stride,
      ElementCompute alpha,
      ElementCompute beta
      ) {

      NA1dProblemSize problem_size(
          batch_size,
          heads,
          length,
          dim,
          kernel_size,
          dilation,
          tile_size,
          tile_stride);

      typename DeviceKernel::Arguments arguments(
          problem_size,
          ref_a,
          ref_b,
          ref_c,
          ref_c,
          ref_bias,
          {alpha, beta});

      DeviceKernel gemm;

      cutlass::Status status = gemm.can_implement(arguments);
      CUTLASS_CHECK(status);

      status = gemm.initialize(arguments);
      CUTLASS_CHECK(status);

      status = gemm();
      CUTLASS_CHECK(status);
    }

  public:
    void launch_with_bias(
      RefOperand const & ref_a,
      RefOperand const & ref_b,
      RefOutput const & ref_c,
      RefBias const & ref_bias,
      const int batch_size,
      const int heads,
      const int length,
      const int dim,
      const int kernel_size,
      const int dilation,
      const int tile_size,
      const int tile_stride,
      ElementCompute scale
      ) {
      launch_cutlass_kernel(
        ref_a,
        ref_b,
        ref_c,
        ref_bias,
        batch_size, heads, length, dim,
        kernel_size, dilation, tile_size, tile_stride,
        scale, ElementCompute(1.0)
      );
    }

    void launch_without_bias(
      RefOperand const & ref_a,
      RefOperand const & ref_b,
      RefOutput const & ref_c,
      const int batch_size,
      const int heads,
      const int length,
      const int dim,
      const int kernel_size,
      const int dilation,
      const int tile_size,
      const int tile_stride,
      ElementCompute scale
      ) {
      RefBias ref_bias = RefBias();
      launch_cutlass_kernel(
        ref_a,
        ref_b,
        ref_c,
        ref_bias,
        batch_size, heads, length, dim,
        kernel_size, dilation, tile_size, tile_stride,
        scale, ElementCompute(0.0) // beta is zero because no bias
      );
    }
};

template <
    typename GemmConfig,
    typename AlignmentConfig,
    typename DTypeConfig,
    typename Arch> 
struct PointwiseNeighborhood1D {

  using ThreadblockShape = cutlass::gemm::GemmShape<GemmConfig::kM, GemmConfig::kN, GemmConfig::kK>;
  using WarpShape        = cutlass::gemm::GemmShape<GemmConfig::kWarpM, GemmConfig::kWarpN, GemmConfig::kWarpK>;
  using InstructionShape = cutlass::gemm::GemmShape<GemmConfig::kMathM, GemmConfig::kMathN, GemmConfig::kMathK>;
  static constexpr int Stages = GemmConfig::kStages;

  static constexpr int AlignmentA = AlignmentConfig::AlignmentA;
  static constexpr int AlignmentB = AlignmentConfig::AlignmentB;
  static constexpr int AlignmentC = AlignmentConfig::AlignmentC;

  using ElementOperand      = typename DTypeConfig::Element;
  using ElementOutput       = typename DTypeConfig::ElementOutput;
  using ElementAccumulator  = typename DTypeConfig::ElementAccumulator;
  using ElementCompute      = typename DTypeConfig::ElementCompute;

  using LayoutOperand = typename cutlass::layout::TensorNHWC; // N = batch, H = heads, W = length, C = dim_per_head
  using LayoutBias    = typename cutlass::layout::RowMajor;

  using RefOperand = typename cutlass::TensorRef<ElementOperand, LayoutOperand>;
  using RefOutput  = typename cutlass::TensorRef<ElementOutput,  LayoutOperand>;
  using RefBias    = typename cutlass::TensorRef<ElementOutput,  LayoutBias>;

  using UnderlyingKernel = typename natten::cuda::gemm::kernel::DefaultNA1dPN<
    ElementOperand, LayoutOperand,
    ElementOperand, LayoutOperand,
    ElementOutput, LayoutOperand,
    ElementAccumulator,
    typename Arch::OpClass,
    typename Arch::Tag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      AlignmentC,
      ElementAccumulator,
      ElementCompute
    >,
    natten::cuda::gemm::threadblock::NAIdentityThreadblockSwizzle,
    Stages,
    cutlass::arch::OpMultiplyAdd,
    AlignmentA,
    AlignmentB
  >::Kernel;

  using DeviceKernel = natten::cuda::gemm::device::ImplicitGemmNA1d<UnderlyingKernel>;

  void operator()(
    void * ptr_query,
    void * ptr_key,
    void * ptr_attn,
    void * ptr_bias,
    const int batch_size,
    const int heads,
    const int length,
    const int dim,
    const int kernel_size,
    const int dilation,
    const float scale = 1.0
    ) {

    // PN tile size
    int tile_stride = kernel_size / 2;
    int tile_size = int((kernel_size + ThreadblockShape::kM - 1) / ThreadblockShape::kM) * ThreadblockShape::kM - tile_stride;
    tile_stride += tile_size;

    // PN refs

    auto layout_ab = LayoutOperand(dim, dim * length, dim * length * heads);
    auto layout_c  = LayoutOperand(kernel_size, kernel_size * length, kernel_size * length * heads);
    auto ref_a = RefOperand(static_cast<ElementOperand*>(ptr_query), layout_ab);
    auto ref_b = RefOperand(static_cast<ElementOperand*>(ptr_key),   layout_ab);
    auto ref_c = RefOutput( static_cast<ElementOutput*>(ptr_attn),  layout_c);

    NA1DLauncher<DeviceKernel> l;

    // Optional bias
    if (ptr_bias != nullptr) {
      auto ref_bias = RefBias(static_cast<ElementOutput*>(ptr_bias), LayoutBias(kernel_size * 2 - 1));
      l.launch_with_bias(
        ref_a,
        ref_b,
        ref_c,
        ref_bias,
        batch_size, heads, length, dim,
        kernel_size, dilation, tile_size, tile_stride,
        ElementCompute(scale)
      );
      return;
    }
    l.launch_without_bias(
      ref_a,
      ref_b,
      ref_c,
      batch_size, heads, length, dim,
      kernel_size, dilation, tile_size, tile_stride,
      ElementCompute(scale)
    );
  }
};

template <
    typename GemmConfig,
    typename AlignmentConfig,
    typename DTypeConfig,
    typename Arch> 
struct NeighborhoodNeighborhood1D {

  using ThreadblockShape = cutlass::gemm::GemmShape<GemmConfig::kM, GemmConfig::kN, GemmConfig::kK>;
  using WarpShape        = cutlass::gemm::GemmShape<GemmConfig::kWarpM, GemmConfig::kWarpN, GemmConfig::kWarpK>;
  using InstructionShape = cutlass::gemm::GemmShape<GemmConfig::kMathM, GemmConfig::kMathN, GemmConfig::kMathK>;
  static constexpr int Stages = GemmConfig::kStages;

  static constexpr int AlignmentA = AlignmentConfig::AlignmentA;
  static constexpr int AlignmentB = AlignmentConfig::AlignmentB;
  static constexpr int AlignmentC = AlignmentConfig::AlignmentC;

  using ElementOperand      = typename DTypeConfig::Element;
  using ElementOutput       = typename DTypeConfig::ElementOutput;
  using ElementAccumulator  = typename DTypeConfig::ElementAccumulator;
  using ElementCompute      = typename DTypeConfig::ElementCompute;

  using LayoutOperand = typename cutlass::layout::TensorNHWC; // N = batch, H = heads, W = length, C = dim_per_head

  using RefOperand = typename cutlass::TensorRef<ElementOperand, LayoutOperand>;
  using RefOutput  = typename cutlass::TensorRef<ElementOutput,  LayoutOperand>;

  using UnderlyingKernel = typename natten::cuda::gemm::kernel::DefaultNA1dNN<
    ElementOperand, LayoutOperand,
    ElementOperand, LayoutOperand,
    ElementOutput, LayoutOperand,
    ElementAccumulator,
    typename Arch::OpClass,
    typename Arch::Tag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      AlignmentC,
      ElementAccumulator,
      ElementCompute
    >,
    natten::cuda::gemm::threadblock::NAIdentityThreadblockSwizzle,
    Stages,
    cutlass::arch::OpMultiplyAdd,
    AlignmentA,
    AlignmentB
  >::Kernel;

  using DeviceKernel = natten::cuda::gemm::device::ImplicitGemmNA1d<UnderlyingKernel>;

  void operator()(
    void * ptr_attn,
    void * ptr_value,
    void * ptr_out,
    const int batch_size,
    const int heads,
    const int length,
    const int dim,
    const int kernel_size,
    const int dilation,
    const float scale = 1.0
    ) {

    // NN tile size
    int tile_size = ThreadblockShape::kM;
    int tile_stride = int(kernel_size / 2) * 2 + tile_size;

    // NN refs
    auto layout_a  = LayoutOperand(kernel_size, kernel_size * length, kernel_size * length * heads);
    auto layout_bc = LayoutOperand(dim, dim * length, dim * length * heads);
    auto ref_a = RefOperand(static_cast<ElementOperand*>(ptr_attn),  layout_a);
    auto ref_b = RefOperand(static_cast<ElementOperand*>(ptr_value), layout_bc);
    auto ref_c = RefOutput( static_cast<ElementOutput* >(ptr_out),   layout_bc);

    NA1DLauncher<DeviceKernel> l;
    l.launch_without_bias(
      ref_a,
      ref_b,
      ref_c,
      batch_size, heads, length, dim,
      kernel_size, dilation, tile_size, tile_stride,
      ElementCompute(scale)
    );
  }
};

template <
    typename GemmConfig,
    typename AlignmentConfig,
    typename DTypeConfig,
    typename Arch> 
struct InverseNeighborhood1D {

  using ThreadblockShape = cutlass::gemm::GemmShape<GemmConfig::kM, GemmConfig::kN, GemmConfig::kK>;
  using WarpShape        = cutlass::gemm::GemmShape<GemmConfig::kWarpM, GemmConfig::kWarpN, GemmConfig::kWarpK>;
  using InstructionShape = cutlass::gemm::GemmShape<GemmConfig::kMathM, GemmConfig::kMathN, GemmConfig::kMathK>;
  static constexpr int Stages = GemmConfig::kStages;

  static constexpr int AlignmentA = AlignmentConfig::AlignmentA;
  static constexpr int AlignmentB = AlignmentConfig::AlignmentB;
  static constexpr int AlignmentC = AlignmentConfig::AlignmentC;

  using ElementOperand      = typename DTypeConfig::Element;
  using ElementOutput       = typename DTypeConfig::ElementOutput;
  using ElementAccumulator  = typename DTypeConfig::ElementAccumulator;
  using ElementCompute      = typename DTypeConfig::ElementCompute;

  using LayoutOperand = typename cutlass::layout::TensorNHWC; // N = batch, H = heads, W = length, C = dim_per_head

  using RefOperand = typename cutlass::TensorRef<ElementOperand, LayoutOperand>;
  using RefOutput  = typename cutlass::TensorRef<ElementOutput,  LayoutOperand>;

  using UnderlyingKernel = typename natten::cuda::gemm::kernel::DefaultNA1dIN<
    ElementOperand, LayoutOperand,
    ElementOperand, LayoutOperand,
    ElementOutput,  LayoutOperand,
    ElementAccumulator,
    typename Arch::OpClass,
    typename Arch::Tag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      AlignmentC,
      ElementAccumulator,
      ElementCompute
    >,
    natten::cuda::gemm::threadblock::NAIdentityThreadblockSwizzle,
    Stages,
    cutlass::arch::OpMultiplyAdd,
    AlignmentA,
    AlignmentB
  >::Kernel;

  using DeviceKernel = natten::cuda::gemm::device::ImplicitGemmNA1d<UnderlyingKernel>;

  void operator()(
    void * ptr_attn,
    void * ptr_value,
    void * ptr_out,
    const int batch_size,
    const int heads,
    const int length,
    const int dim,
    const int kernel_size,
    const int dilation,
    const float scale = 1.0
    ) {

    // IN tile size
    int tile_size = ThreadblockShape::kM;
    int tile_stride = int(kernel_size / 2) * 4 + tile_size;

    // IN refs
    auto layout_a  = LayoutOperand(kernel_size, kernel_size * length, kernel_size * length * heads);
    auto layout_bc = LayoutOperand(dim, dim * length, dim * length * heads);
    auto ref_a = RefOperand(static_cast<ElementOperand*>(ptr_attn),  layout_a);
    auto ref_b = RefOperand(static_cast<ElementOperand*>(ptr_value), layout_bc);
    auto ref_c = RefOutput( static_cast<ElementOutput* >(ptr_out),   layout_bc);

    NA1DLauncher<DeviceKernel> l;
    l.launch_without_bias(
      ref_a,
      ref_b,
      ref_c,
      batch_size, heads, length, dim,
      kernel_size, dilation, tile_size, tile_stride,
      ElementCompute(scale)
    );
  }
};


} // namespace gemm
} // namespace cuda
} // namespace natten

