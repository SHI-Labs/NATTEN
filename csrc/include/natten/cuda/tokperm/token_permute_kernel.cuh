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

#include <natten/cuda/tokperm/utils/permute.cuh>
#include <natten/cuda/tokperm/utils/stride.cuh>

namespace natten::tokperm::kernel {

using namespace cute;

template <class DstCrd, class Dim>
CUTE_HOST_DEVICE constexpr auto perm2unperm(
    DstCrd const& dstCrd,
    Dim const& rest,
    Dim const& tile,
    Dim const& dilation) {
  static_assert(
      rank(DstCrd{}) == 3 || rank(DstCrd{}) == 6 || rank(DstCrd{}) == 9);
  static_assert(rank(DstCrd{}) / 3 == rank(Dim{}));

  if constexpr (rank(Dim{}) == 1) {
    auto tiled_layout_0 = make_layout(
        make_shape(get<0>(rest), get<0>(tile), get<0>(dilation)),
        make_stride(get<0>(tile) * get<0>(dilation), get<0>(dilation), _1{}));

    auto crd_0 = select<0, 1, 2>(dstCrd);
    auto crd_src_0 = tiled_layout_0(crd_0);
    return make_tuple(crd_src_0);
  } else if constexpr (rank(Dim{}) == 2) {
    auto tiled_layout_0 = make_layout(
        make_shape(get<0>(rest), get<0>(tile), get<0>(dilation)),
        make_stride(get<0>(tile) * get<0>(dilation), get<0>(dilation), _1{}));
    auto tiled_layout_1 = make_layout(
        make_shape(get<1>(rest), get<1>(tile), get<1>(dilation)),
        make_stride(get<1>(tile) * get<1>(dilation), get<1>(dilation), _1{}));

    auto crd_0 = select<0, 1, 2>(dstCrd);
    auto crd_1 = select<3, 4, 5>(dstCrd);
    auto crd_src_0 = tiled_layout_0(crd_0);
    auto crd_src_1 = tiled_layout_1(crd_1);
    return make_tuple(crd_src_0, crd_src_1);
  } else if constexpr (rank(Dim{}) == 3) {
    auto tiled_layout_0 = make_layout(
        make_shape(get<0>(rest), get<0>(tile), get<0>(dilation)),
        make_stride(get<0>(tile) * get<0>(dilation), get<0>(dilation), _1{}));
    auto tiled_layout_1 = make_layout(
        make_shape(get<1>(rest), get<1>(tile), get<1>(dilation)),
        make_stride(get<1>(tile) * get<1>(dilation), get<1>(dilation), _1{}));
    auto tiled_layout_2 = make_layout(
        make_shape(get<2>(rest), get<2>(tile), get<2>(dilation)),
        make_stride(get<2>(tile) * get<2>(dilation), get<2>(dilation), _1{}));

    auto crd_0 = select<0, 1, 2>(dstCrd);
    auto crd_1 = select<3, 4, 5>(dstCrd);
    auto crd_2 = select<6, 7, 8>(dstCrd);
    auto crd_src_0 = tiled_layout_0(crd_0);
    auto crd_src_1 = tiled_layout_1(crd_1);
    auto crd_src_2 = tiled_layout_2(crd_2);
    return make_tuple(crd_src_0, crd_src_1, crd_src_2);
  }
}

template <class DstCrd, class Dim>
CUTE_HOST_DEVICE constexpr auto unperm2perm(
    DstCrd const& dstCrd,
    Dim const& rest,
    Dim const& tile,
    Dim const& dilation) {
  static_assert(rank(DstCrd{}) == rank(Dim{}));

  if constexpr (rank(Dim{}) == 1) {
    auto tiled_layout_0 = make_layout(
        make_shape(get<0>(rest), get<0>(tile), get<0>(dilation)),
        make_stride(get<0>(tile) * get<0>(dilation), get<0>(dilation), _1{}));

    auto [crd_0] = dstCrd;
    auto [crd_src_0_r, crd_src_0_t, crd_src_0_d] =
        idx2crd(crd_0, tiled_layout_0.shape(), tiled_layout_0.stride());
    return make_tuple(crd_src_0_r, crd_src_0_t, crd_src_0_d);
  } else if constexpr (rank(Dim{}) == 2) {
    auto tiled_layout_0 = make_layout(
        make_shape(get<0>(rest), get<0>(tile), get<0>(dilation)),
        make_stride(get<0>(tile) * get<0>(dilation), get<0>(dilation), _1{}));
    auto tiled_layout_1 = make_layout(
        make_shape(get<1>(rest), get<1>(tile), get<1>(dilation)),
        make_stride(get<1>(tile) * get<1>(dilation), get<1>(dilation), _1{}));

    auto [crd_0, crd_1] = dstCrd;
    auto [crd_src_0_r, crd_src_0_t, crd_src_0_d] =
        idx2crd(crd_0, tiled_layout_0.shape(), tiled_layout_0.stride());
    auto [crd_src_1_r, crd_src_1_t, crd_src_1_d] =
        idx2crd(crd_1, tiled_layout_1.shape(), tiled_layout_1.stride());
    return make_tuple(
        crd_src_0_r,
        crd_src_0_t,
        crd_src_0_d,
        crd_src_1_r,
        crd_src_1_t,
        crd_src_1_d);
  } else if constexpr (rank(Dim{}) == 3) {
    auto tiled_layout_0 = make_layout(
        make_shape(get<0>(rest), get<0>(tile), get<0>(dilation)),
        make_stride(get<0>(tile) * get<0>(dilation), get<0>(dilation), _1{}));
    auto tiled_layout_1 = make_layout(
        make_shape(get<1>(rest), get<1>(tile), get<1>(dilation)),
        make_stride(get<1>(tile) * get<1>(dilation), get<1>(dilation), _1{}));
    auto tiled_layout_2 = make_layout(
        make_shape(get<2>(rest), get<2>(tile), get<2>(dilation)),
        make_stride(get<2>(tile) * get<2>(dilation), get<2>(dilation), _1{}));

    auto [crd_0, crd_1, crd_2] = dstCrd;
    auto [crd_src_0_r, crd_src_0_t, crd_src_0_d] =
        idx2crd(crd_0, tiled_layout_0.shape(), tiled_layout_0.stride());
    auto [crd_src_1_r, crd_src_1_t, crd_src_1_d] =
        idx2crd(crd_1, tiled_layout_1.shape(), tiled_layout_1.stride());
    auto [crd_src_2_r, crd_src_2_t, crd_src_2_d] =
        idx2crd(crd_2, tiled_layout_2.shape(), tiled_layout_2.stride());
    return make_tuple(
        crd_src_0_r,
        crd_src_0_t,
        crd_src_0_d,
        crd_src_1_r,
        crd_src_1_t,
        crd_src_1_d,
        crd_src_2_r,
        crd_src_2_t,
        crd_src_2_d);
  }
}

template <
    typename TokenLayoutIn,
    typename TokenLayoutOut,
    class ElementIn,
    class ElementOut,
    bool IsUnpermute = false,
    int kElementsPerLoad = 4>
struct TokenPermuteKernel {
  using TokenLayout =
      cute::conditional_t<IsUnpermute, TokenLayoutOut, TokenLayoutIn>;

  // B, token layout shape, tile shape, dilation, H, D
  using ProblemShapeIn = cute::tuple<int, TokenLayoutIn, int, int>;
  using ProblemShapeOut = cute::tuple<int, TokenLayoutOut, int, int>;

  using StrideIn = cute::tuple<int, TokenLayoutIn, int, _1>;
  using StrideOut = cute::tuple<int, TokenLayoutOut, int, _1>;

  using ClusterShape = Shape<_1, _1, _1>;
  static constexpr int SharedStorageSize = 0;

  struct Arguments {
    ProblemShapeIn problem_shape_src;
    ProblemShapeOut problem_shape_dst;

    ElementIn* ptr_src;
    ElementOut* ptr_dst;

    StrideIn stride_src;
    StrideOut stride_dst;

    TokenLayout rest;
    TokenLayout tile;
    TokenLayout dilation;
  };

  using Params = Arguments;

  static const int MinBlocksPerMultiprocessor = 1;
  static const int MaxThreadsPerBlock = 128;

  static const int kBlockSeq = 8;

  static const int kNumThreadsD = 16;
  static const int kNumThreadsSeq = MaxThreadsPerBlock / kNumThreadsD;

  static const int kIterationsSeq = kBlockSeq / kNumThreadsSeq;

  static bool can_implement(Arguments const& args) {
    if (cute::size<0>(args.problem_shape_src) !=
        cute::size<0>(args.problem_shape_dst)) {
      std::cerr
          << "Token Permute/UnPermute requires input and output batch sizes to match."
          << std::endl;
      return false;
    }
    if (IsUnpermute) {
      if (cute::size<1>(args.problem_shape_src) <
          cute::size<1>(args.problem_shape_dst)) {
        std::cerr << "Token UnPermute requires more or equal input tokens."
                  << std::endl;
        return false;
      }
    } else {
      if (cute::size<1>(args.problem_shape_src) >
          cute::size<1>(args.problem_shape_dst)) {
        std::cerr
            << "Token Permute/UnPermute requires more or equal output tokens."
            << std::endl;
        return false;
      }
    }
    if (cute::size<2>(args.problem_shape_src) !=
        cute::size<2>(args.problem_shape_dst)) {
      std::cerr
          << "Token Permute/UnPermute requires input and output heads to match."
          << std::endl;
      return false;
    }
    if (cute::size<3>(args.problem_shape_src) !=
        cute::size<3>(args.problem_shape_dst)) {
      std::cerr
          << "Token Permute/UnPermute requires input and output head dims to match."
          << std::endl;
      return false;
    }
    if (cute::get<3>(args.stride_src) != _1{} ||
        cute::get<3>(args.stride_dst) != _1{}) {
      std::cerr
          << "Token Permute/UnPermute requires contiguous input and output (stride[-1] == 1)."
          << std::endl;
      return false;
    }
    if (cute::size<3>(args.problem_shape_src) % kElementsPerLoad != 0) {
      std::cerr
          << "Token Permute/UnPermute requires head dim to be evenly divisible by load size ("
          << kElementsPerLoad << ")" << std::endl;
      return false;
    }

    return true;
  }

  static dim3 get_grid_shape(Params const& params) {
    dim3 grid(
        // never put seq in z, long seqs can easily exceed the 64K limit
        cute::ceil_div(size<1>(params.problem_shape_dst), kBlockSeq),
        cute::size<2>(params.problem_shape_dst),
        cute::size<0>(params.problem_shape_dst));
    return grid;
  }

  static dim3 get_block_shape() {
    dim3 block(kNumThreadsSeq, kNumThreadsD, 1);
    return block;
  }

  static Params to_underlying_arguments(Arguments const& args) {
    return args;
  }

  CUTLASS_DEVICE void operator()(const Params& params, char* smem) {
    auto ptr_src_bh = params.ptr_src + get<0>(params.stride_src) * blockIdx.z +
        get<2>(params.stride_src) * blockIdx.y;
    auto ptr_dst_bh = params.ptr_dst + get<0>(params.stride_dst) * blockIdx.z +
        get<2>(params.stride_dst) * blockIdx.y;

    auto token_layout_src = make_layout(
        get<1>(params.problem_shape_src), get<1>(params.stride_src));
    auto token_layout_dst = make_layout(
        get<1>(params.problem_shape_dst), get<1>(params.stride_dst));

    auto src_shape = token_layout_src.shape();

    auto seqlen = size<1>(params.problem_shape_dst);

    for (int idx_s_t = threadIdx.x; idx_s_t < kBlockSeq;
         idx_s_t += kNumThreadsSeq) {
      int idx_s = idx_s_t + kBlockSeq * blockIdx.x;
      if (idx_s >= seqlen)
        continue;

      auto crd_dst = idx2crd(idx_s, token_layout_dst.shape());
      bool pred = false;

      TokenLayoutIn crd_src;
      if constexpr (IsUnpermute) {
        crd_src =
            unperm2perm(crd_dst, params.rest, params.tile, params.dilation);
      } else {
        crd_src =
            perm2unperm(crd_dst, params.rest, params.tile, params.dilation);
      }

      pred = elem_less(crd_src, src_shape);

      auto ptr_src_bhs = ptr_src_bh + token_layout_src(crd_src);
      auto ptr_dst_bhs = ptr_dst_bh + token_layout_dst(crd_dst);

      for (int idx_d = threadIdx.y * kElementsPerLoad;
           idx_d < get<3>(params.problem_shape_dst);
           idx_d += kElementsPerLoad * kNumThreadsD) {
        ElementIn value_src[kElementsPerLoad];
        ElementOut value_dst[kElementsPerLoad];

        using VecSrc = uint_bit_t<sizeof_bits_v<ElementIn> * kElementsPerLoad>;
        using VecDst = uint_bit_t<sizeof_bits_v<ElementOut> * kElementsPerLoad>;
        if (pred) {
          *reinterpret_cast<VecSrc*>(value_src) =
              *reinterpret_cast<VecSrc*>(&ptr_src_bhs[idx_d]);

          for (int v = 0; v < kElementsPerLoad; v++) {
            value_dst[v] = static_cast<ElementOut>(value_src[v]);
          }
        } else {
          for (int v = 0; v < kElementsPerLoad; v++) {
            value_dst[v] = static_cast<ElementOut>(0);
          }
        }

        *reinterpret_cast<VecDst*>(&ptr_dst_bhs[idx_d]) =
            *reinterpret_cast<VecDst*>(value_dst);
      }
    }
  }
};

} // namespace natten::tokperm::kernel
