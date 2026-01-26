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

#include <natten/cuda/tokperm/layouts.hpp>
#include <natten/cuda/tokperm/token_permute_kernel.cuh>
#include <natten/cuda/tokperm/utils/permute.cuh>
#include <natten/cuda/tokperm/utils/stride.cuh>

namespace natten::tokperm::kernel {

using namespace cute;

template <
    typename TokenLayoutSrc,
    typename TokenLayoutDst,
    class ElementSrc,
    class ElementDst,
    class ElementOffset,
    bool IsUnpermute = false,
    int kElementsPerLoad = 4>
struct TokenPermuteVarlenKernel {
  using OffsetType = ElementOffset;
  using OffsetTypeInternal = uint64_t;

  using TokenLayout =
      cute::conditional_t<IsUnpermute, TokenLayoutDst, TokenLayoutSrc>;

  // Varlen doesn't need the batch mode -- we offset to the correct batch,
  // and afterwards batch is just 1.
  // We also don't need to track heads and head_dim separately.
  using ProblemShapeSrc = cute::tuple<TokenLayoutSrc, int>;
  using ProblemShapeDst = cute::tuple<TokenLayoutDst, int>;

  using StrideSrc = decltype(make_torch_contiguous_stride(ProblemShapeSrc{}));
  using StrideDst = decltype(make_torch_contiguous_stride(ProblemShapeDst{}));

  using ClusterShape = Shape<_1, _1, _1>;
  static constexpr int SharedStorageSize = 0;

  struct Arguments {
    int batch; // batch before dilation
    int seqlen_max;
    int heads;
    int head_dim;

    TokenLayout* token_layout_array; // size: batch * size(dilation)
    TokenLayout* dilation_array; // size: batch * size(dilation)
    OffsetType* ptr_offsets_original; // size: batch + 1
    OffsetType* ptr_offsets_tokperm; // size: batch + 1

    ElementSrc* ptr_src;
    ElementDst* ptr_dst;

    TokenLayout tile_shape;
    TokenLayout dilation;

    bool flip_tiled_dims;
  };

  struct Params {
    int batch;
    int seqlen_max;
    int heads;
    int head_dim;

    TokenLayout* token_layout_array;
    TokenLayout* dilation_array;
    OffsetType* ptr_offsets_original;
    OffsetType* ptr_offsets_tokperm;

    ElementSrc* ptr_src;
    ElementDst* ptr_dst;

    TokenLayout tile_shape;
    TokenLayout dilation;
    bool flip_tiled_dims;

    bool has_variable_dilations;
    OffsetTypeInternal stride_seq =
        static_cast<OffsetTypeInternal>(heads * head_dim);
    OffsetTypeInternal stride_heads = static_cast<OffsetTypeInternal>(head_dim);
  };

  static const int MinBlocksPerMultiprocessor = 1;
  static const int MaxThreadsPerBlock = 128;

  static const int kBlockSeq = 8;

  static const int kNumThreadsD = 16;
  static const int kNumThreadsSeq = MaxThreadsPerBlock / kNumThreadsD;

  static const int kIterationsSeq = kBlockSeq / kNumThreadsSeq;

  static bool can_implement(Arguments const& args) {
    if (args.head_dim % kElementsPerLoad != 0) {
      std::cerr
          << "Varlen Token Permute/UnPermute requires head dim to be evenly divisible by load size ("
          << kElementsPerLoad << ")" << std::endl;
      return false;
    }

    return true;
  }

  static dim3 get_grid_shape(Params const& params) {
    dim3 grid(
        // never put seq in z, long seqs can easily exceed the 64K limit
        cute::ceil_div(params.seqlen_max, kBlockSeq),
        params.heads,
        params.batch);
    return grid;
  }

  static dim3 get_block_shape() {
    dim3 block(kNumThreadsSeq, kNumThreadsD, 1);
    return block;
  }

  static Params to_underlying_arguments(Arguments const& args) {
    return {
        args.batch,
        args.seqlen_max,
        args.heads,
        args.head_dim,
        //
        args.token_layout_array,
        args.dilation_array,
        args.ptr_offsets_original,
        args.ptr_offsets_tokperm,
        //
        args.ptr_src,
        args.ptr_dst,
        //
        args.tile_shape,
        args.dilation,
        args.flip_tiled_dims,
        //
        args.dilation_array != nullptr,
        static_cast<OffsetTypeInternal>(args.heads * args.head_dim),
        static_cast<OffsetTypeInternal>(args.head_dim)};
  }

  CUTLASS_DEVICE void operator()(const Params& params, char* smem) {
    OffsetTypeInternal batch_offset_src = 0;
    OffsetTypeInternal batch_offset_dst = 0;

    ProblemShapeSrc problem_shape_src;
    ProblemShapeDst problem_shape_dst;
    StrideSrc stride_src;
    StrideDst stride_dst;

    TokenLayout token_layout = params.token_layout_array[blockIdx.z];
    TokenLayout dilation = params.dilation;
    if (params.has_variable_dilations) {
      dilation = params.dilation_array[blockIdx.z];
    }
    TokenLayout rest =
        ceil_div(ceil_div(token_layout, params.tile_shape), dilation);

    if constexpr (IsUnpermute) {
      batch_offset_src = static_cast<OffsetTypeInternal>(
          params.ptr_offsets_tokperm[blockIdx.z]);
      batch_offset_dst = static_cast<OffsetTypeInternal>(
          params.ptr_offsets_original[blockIdx.z]);

      problem_shape_dst =
          cute::make_tuple(token_layout, params.heads * params.head_dim);
      stride_dst = utils::make_torch_contiguous_stride(problem_shape_dst);

      auto layout_src = make_token_permuted_layout_varlen(
          rest,
          params.tile_shape,
          dilation,
          params.flip_tiled_dims,
          get<1>(problem_shape_dst));

      problem_shape_src = layout_src.shape();
      stride_src = layout_src.stride();

    } else {
      batch_offset_src = static_cast<OffsetTypeInternal>(
          params.ptr_offsets_original[blockIdx.z]);
      batch_offset_dst = static_cast<OffsetTypeInternal>(
          params.ptr_offsets_tokperm[blockIdx.z]);

      problem_shape_src =
          cute::make_tuple(token_layout, params.heads * params.head_dim);
      stride_src = utils::make_torch_contiguous_stride(problem_shape_src);

      auto layout_dst = make_token_permuted_layout_varlen(
          rest,
          params.tile_shape,
          dilation,
          params.flip_tiled_dims,
          get<1>(problem_shape_src));

      problem_shape_dst = layout_dst.shape();
      stride_dst = layout_dst.stride();
    }

    auto ptr_src_bh = params.ptr_src +
        (params.stride_seq * batch_offset_src +
         params.stride_heads * static_cast<OffsetTypeInternal>(blockIdx.y));
    auto ptr_dst_bh = params.ptr_dst +
        (params.stride_seq * batch_offset_dst +
         params.stride_heads * static_cast<OffsetTypeInternal>(blockIdx.y));

    auto token_layout_src =
        make_layout(get<0>(problem_shape_src), get<0>(stride_src));
    auto token_layout_dst =
        make_layout(get<0>(problem_shape_dst), get<0>(stride_dst));

    auto src_shape = token_layout_src.shape();
    auto seqlen = size<0>(problem_shape_dst);

    for (int idx_s_t = threadIdx.x; idx_s_t < kBlockSeq;
         idx_s_t += kNumThreadsSeq) {
      int idx_s = idx_s_t + kBlockSeq * blockIdx.x;
      if (idx_s >= seqlen)
        continue;

      auto crd_dst = idx2crd(idx_s, token_layout_dst.shape());
      bool pred = false;

      TokenLayoutSrc crd_src;
      if constexpr (IsUnpermute) {
        crd_src = unperm2perm(crd_dst, rest, params.tile_shape, dilation);
      } else {
        crd_src = perm2unperm(crd_dst, rest, params.tile_shape, dilation);
      }

      pred = elem_less(crd_src, src_shape);

      auto ptr_src_bhs = ptr_src_bh + token_layout_src(crd_src);
      auto ptr_dst_bhs = ptr_dst_bh + token_layout_dst(crd_dst);

      for (int idx_d = threadIdx.y * kElementsPerLoad; idx_d < params.head_dim;
           idx_d += kElementsPerLoad * kNumThreadsD) {
        ElementSrc value_src[kElementsPerLoad];
        ElementDst value_dst[kElementsPerLoad];

        using VecSrc = uint_bit_t<sizeof_bits_v<ElementSrc> * kElementsPerLoad>;
        using VecDst = uint_bit_t<sizeof_bits_v<ElementDst> * kElementsPerLoad>;
        if (pred) {
          *reinterpret_cast<VecSrc*>(value_src) =
              *reinterpret_cast<VecSrc*>(&ptr_src_bhs[idx_d]);

          for (int v = 0; v < kElementsPerLoad; v++) {
            value_dst[v] = static_cast<ElementDst>(value_src[v]);
          }
        } else {
          for (int v = 0; v < kElementsPerLoad; v++) {
            value_dst[v] = static_cast<ElementDst>(0);
          }
        }

        *reinterpret_cast<VecDst*>(&ptr_dst_bhs[idx_d]) =
            *reinterpret_cast<VecDst*>(value_dst);
      }
    }
  }
};

} // namespace natten::tokperm::kernel
