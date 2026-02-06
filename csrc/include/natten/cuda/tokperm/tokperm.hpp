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

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>

#include <natten/cuda/utils/cutlass.cuh>

#include <natten/cuda/tokperm/layouts.hpp>
#include <natten/cuda/tokperm/token_permute_kernel.cuh>
#include <natten/cuda/tokperm/utils/permute.cuh>
#include <natten/cuda/tokperm/utils/stride.cuh>

namespace natten::tokperm {

using namespace cute;

template <typename ElementIn, typename ElementOut, typename CuteTuple>
bool token_permute_op(
    ElementIn* ptr_in,
    ElementOut* ptr_out,
    int batch,
    int heads,
    int seqlen_out,
    int dim,
    CuteTuple& token_layout,
    CuteTuple& tile_shape,
    CuteTuple& dilation,
    bool flip_tiled_dims,
    cudaStream_t stream) {
  static constexpr int NumDims = tuple_size_v<CuteTuple>;

  // too lazy to figure out how to generate these via templates
  using DimIn = cute::conditional_t<
      NumDims == 3,
      cute::tuple<int, int, int>,
      cute::
          conditional_t<NumDims == 2, cute::tuple<int, int>, cute::tuple<int>>>;

  using DimOut = cute::conditional_t<
      NumDims == 3,
      cute::tuple<int, int, int, int, int, int, int, int, int>,
      cute::conditional_t<
          NumDims == 2,
          cute::tuple<int, int, int, int, int, int>,
          cute::tuple<int, int, int>>>;

  auto rest = ceil_div(ceil_div(token_layout, tile_shape), dilation);

  auto problem_shape_in = cute::make_tuple(batch, token_layout, heads, dim);
  auto stride_in = utils::make_torch_contiguous_stride(problem_shape_in);

  auto layout_out = make_token_permuted_layout(
      rest, tile_shape, dilation, batch, heads, dim, flip_tiled_dims);

  auto problem_shape_out = layout_out.shape();
  auto stride_out = layout_out.stride();

  if (cute::size<1>(problem_shape_out) != seqlen_out) {
    std::cerr << "Token Permute output must have sequence length of exactly "
              << cute::size<1>(problem_shape_out)
              << ", got tensor with sequence length " << seqlen_out << "!"
              << std::endl;
    return false;
  }

  using OperationAlign1 = kernel::TokenPermuteKernel<
      DimIn,
      DimOut,
      ElementIn,
      ElementOut,
      /* IsUnpermute = */ false,
      1>;
  using OperationAlign4 = kernel::TokenPermuteKernel<
      DimIn,
      DimOut,
      ElementIn,
      ElementOut,
      /* IsUnpermute = */ false,
      4>;
  using OperationAlign8 = kernel::TokenPermuteKernel<
      DimIn,
      DimOut,
      ElementIn,
      ElementOut,
      /* IsUnpermute = */ false,
      8>;

  auto launch_kernel = [&](auto& op) {
    using Operation = std::remove_reference_t<decltype(op)>;
    using Arguments = typename Operation::Arguments;

    Arguments arguments{
        problem_shape_in,
        problem_shape_out,
        ptr_in,
        ptr_out,
        stride_in,
        stride_out,
        rest,
        tile_shape,
        dilation,
    };

    if (not op.can_implement(arguments)) {
      std::cerr << "Token Permute kernel is not supported." << std::endl;
      return false;
    }

    auto params = Operation::to_underlying_arguments(arguments);
    auto grid = Operation::get_grid_shape(params);
    auto block = Operation::get_block_shape();
    int smem_size = Operation::SharedStorageSize;
    cutlass::device_kernel<Operation>
        <<<grid, block, smem_size, stream>>>(params);

    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
      std::cerr << "Failed to launch Token Permute kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
      return false;
    }

    return true;
  };

  if (dim % 8 == 0) {
    OperationAlign8 op;
    return launch_kernel(op);
  } else if (dim % 4 == 0) {
    OperationAlign4 op;
    return launch_kernel(op);
  } else {
    OperationAlign1 op;
    return launch_kernel(op);
  }
}

template <typename ElementIn, typename ElementOut, typename CuteTuple>
bool token_unpermute_op(
    ElementIn* ptr_in,
    ElementOut* ptr_out,
    int batch,
    int heads,
    int dim,
    CuteTuple& token_layout,
    CuteTuple& tile_shape,
    CuteTuple& dilation,
    bool flip_tiled_dims,
    cudaStream_t stream) {
  static constexpr int NumDims = tuple_size_v<CuteTuple>;

  // too lazy to figure out how to generate these via templates
  using DimOut = cute::conditional_t<
      NumDims == 3,
      cute::tuple<int, int, int>,
      cute::
          conditional_t<NumDims == 2, cute::tuple<int, int>, cute::tuple<int>>>;

  using DimIn = cute::conditional_t<
      NumDims == 3,
      cute::tuple<int, int, int, int, int, int, int, int, int>,
      cute::conditional_t<
          NumDims == 2,
          cute::tuple<int, int, int, int, int, int>,
          cute::tuple<int, int, int>>>;

  auto rest = ceil_div(ceil_div(token_layout, tile_shape), dilation);

  auto problem_shape_out = cute::make_tuple(batch, token_layout, heads, dim);
  auto stride_out = utils::make_torch_contiguous_stride(problem_shape_out);

  auto layout_in = make_token_permuted_layout(
      rest, tile_shape, dilation, batch, heads, dim, flip_tiled_dims);

  auto problem_shape_in = layout_in.shape();
  auto stride_in = layout_in.stride();

  using OperationAlign1 = kernel::TokenPermuteKernel<
      DimIn,
      DimOut,
      ElementIn,
      ElementOut,
      /* IsUnpermute = */ true,
      1>;
  using OperationAlign4 = kernel::TokenPermuteKernel<
      DimIn,
      DimOut,
      ElementIn,
      ElementOut,
      /* IsUnpermute = */ true,
      4>;
  using OperationAlign8 = kernel::TokenPermuteKernel<
      DimIn,
      DimOut,
      ElementIn,
      ElementOut,
      /* IsUnpermute = */ true,
      8>;

  auto launch_kernel = [&](auto& op) {
    using Operation = std::remove_reference_t<decltype(op)>;
    using Arguments = typename Operation::Arguments;

    Arguments arguments{
        problem_shape_in,
        problem_shape_out,
        ptr_in,
        ptr_out,
        stride_in,
        stride_out,
        rest,
        tile_shape,
        dilation,
    };

    if (not op.can_implement(arguments)) {
      std::cerr << "Token UnPermute kernel is not supported." << std::endl;
      return false;
    }

    auto params = Operation::to_underlying_arguments(arguments);
    auto grid = Operation::get_grid_shape(params);
    auto block = Operation::get_block_shape();
    int smem_size = Operation::SharedStorageSize;
    cutlass::device_kernel<Operation>
        <<<grid, block, smem_size, stream>>>(params);

    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
      std::cerr
          << "Failed to launch Token UnPermute kernel. Last CUDA error is: "
          << cudaGetErrorString(result) << std::endl;
      return false;
    }

    return true;
  };

  if (dim % 8 == 0) {
    OperationAlign8 op;
    return launch_kernel(op);
  } else if (dim % 4 == 0) {
    OperationAlign4 op;
    return launch_kernel(op);
  } else {
    OperationAlign1 op;
    return launch_kernel(op);
  }
}

} // namespace natten::tokperm
