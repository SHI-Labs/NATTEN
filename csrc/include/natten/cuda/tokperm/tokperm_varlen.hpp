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

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>

#include <natten/cuda/tokperm/token_permute_varlen_kernel.cuh>
#include <natten/cuda/utils/cutlass.cuh>

namespace natten::tokperm {

using namespace cute;

template <
    typename ElementSrc,
    typename ElementDst,
    typename ElementOffset,
    typename CuteTuple>
bool token_permute_varlen_op(
    ElementSrc* ptr_src,
    ElementDst* ptr_dst,
    int batch,
    int seqlen_max,
    int heads,
    int dim,
    ElementOffset* ptr_offsets_pre_permute, // size: batch + 1
    ElementOffset* ptr_offsets_post_permute, // size: batch * size(dilation) + 1
    CuteTuple* ptr_token_layouts,
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

  using OperationAlign1 = kernel::TokenPermuteVarlenKernel<
      DimIn,
      DimOut,
      ElementSrc,
      ElementDst,
      ElementOffset,
      /* IsUnpermute = */ false,
      1>;
  using OperationAlign4 = kernel::TokenPermuteVarlenKernel<
      DimIn,
      DimOut,
      ElementSrc,
      ElementDst,
      ElementOffset,
      /* IsUnpermute = */ false,
      4>;
  using OperationAlign8 = kernel::TokenPermuteVarlenKernel<
      DimIn,
      DimOut,
      ElementSrc,
      ElementDst,
      ElementOffset,
      /* IsUnpermute = */ false,
      8>;

  auto launch_kernel = [&](auto& op) {
    using Operation = std::remove_reference_t<decltype(op)>;
    using Arguments = typename Operation::Arguments;

    Arguments arguments{
        batch,
        seqlen_max,
        heads,
        dim,
        ptr_token_layouts,
        ptr_offsets_pre_permute,
        ptr_offsets_post_permute,
        ptr_src,
        ptr_dst,
        tile_shape,
        dilation,
        flip_tiled_dims,
    };

    if (not op.can_implement(arguments)) {
      std::cerr << "Varlen Token Permute kernel is not supported." << std::endl;
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

template <
    typename ElementSrc,
    typename ElementDst,
    typename ElementOffset,
    typename CuteTuple>
bool token_unpermute_varlen_op(
    ElementSrc* ptr_src,
    ElementDst* ptr_dst,
    int batch,
    int seqlen_max,
    int heads,
    int dim,
    ElementOffset* ptr_offsets_pre_permute, // size: batch + 1
    ElementOffset* ptr_offsets_post_permute, // size: batch * size(dilation) + 1
    CuteTuple* ptr_token_layouts,
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

  using OperationAlign1 = kernel::TokenPermuteVarlenKernel<
      DimIn,
      DimOut,
      ElementSrc,
      ElementDst,
      ElementOffset,
      /* IsUnpermute = */ true,
      1>;
  using OperationAlign4 = kernel::TokenPermuteVarlenKernel<
      DimIn,
      DimOut,
      ElementSrc,
      ElementDst,
      ElementOffset,
      /* IsUnpermute = */ true,
      4>;
  using OperationAlign8 = kernel::TokenPermuteVarlenKernel<
      DimIn,
      DimOut,
      ElementSrc,
      ElementDst,
      ElementOffset,
      /* IsUnpermute = */ true,
      8>;

  auto launch_kernel = [&](auto& op) {
    using Operation = std::remove_reference_t<decltype(op)>;
    using Arguments = typename Operation::Arguments;

    Arguments arguments{
        batch,
        seqlen_max,
        heads,
        dim,
        ptr_token_layouts,
        ptr_offsets_pre_permute,
        ptr_offsets_post_permute,
        ptr_src,
        ptr_dst,
        tile_shape,
        dilation,
        flip_tiled_dims,
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
