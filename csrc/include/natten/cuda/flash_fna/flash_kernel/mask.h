/******************************************************************************
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
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>
#include "cute/util/debug.hpp"
#include "cute/util/print.hpp"

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

// #include "utils.h"
#include "natten/cuda/flash_fmha/flash_kernel/utils.h"

namespace natten {
namespace cuda {
namespace flash_fna {

using namespace cute;


template<int kBlockM, int kBlockN, class NADim, class QTileShape, class KVTileShape, class Causal,
  bool PackGQA, typename TiledMma, class IterMapType, bool SwapAB=false>
struct NAMask {
    static_assert(!(PackGQA && SwapAB), "Cannot be both PackGQA and SwapAB");

    int const thread_idx;
    int const seqlen_q, seqlen_k;

    NADim window_size;
    NADim window_left;
    NADim window_right;
    NADim stride;
    NADim qkv_shape;
    NADim q_shape;
    NADim kv_shape;
    NADim kv_blk_offset;
    NADim kv_diff_tiles;

    IterMapType iter_to_tile_map;
    
    cutlass::FastDivmod const qhead_per_khead_divmod;

    CUTLASS_DEVICE
    NAMask (const int thread_idx, const int seqlen_q, const int seqlen_k,
        cutlass::FastDivmod const &qhead_per_khead_divmod,
        NADim window_size, NADim window_left, NADim window_right, NADim stride,
        NADim qkv_shape, NADim q_shape, NADim kv_shape,
        NADim kv_blk_offset, NADim kv_diff_tiles, IterMapType iter_to_tile_map):
        thread_idx(thread_idx), 
        seqlen_q(seqlen_q), 
        seqlen_k(seqlen_k), 
        qhead_per_khead_divmod(qhead_per_khead_divmod),
        window_size(window_size),
        window_left(window_left),
        window_right(window_right),
        stride(stride),
        qkv_shape(qkv_shape),
        q_shape(q_shape),
        kv_shape(kv_shape),
        kv_blk_offset(kv_blk_offset),
        kv_diff_tiles(kv_diff_tiles),
        iter_to_tile_map(iter_to_tile_map) {}


    template <typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) {
        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);

        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);

        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));

        // TODO (aditya): This doesn't consider SwapAB, fix.
        tScS_rowcol.data() = tScS_rowcol.data() + E<0>{} * m_block * kBlockM;
        tScS_rowcol.data() = tScS_rowcol.data() + E<1>{} * n_block * kBlockN + E<1>{} * size(kv_blk_offset);

        // NOTE (aditya): Copied over with minor modifications from Hopper FNA
        auto q_tile_shape = QTileShape{};
        auto kv_tile_shape = KVTileShape{};

        auto q_tiled = ceil_div(q_shape, q_tile_shape);

        auto [q_idx_first, kv_idx_first] =
            tScS_rowcol(crd2idx(make_coord(0, 0), tSrS_rowcol.layout()));

        // Q coord remap
        int q_tile_idx = q_idx_first / size(q_tile_shape);
        int q_idx_first_in_tile = q_tile_idx * size(q_tile_shape);

        int kv_tile_idx = n_block;
        int kv_idx_first_in_tile = (kv_tile_idx * size(kv_tile_shape)) + size(kv_blk_offset);
        //int q_tile_res = q_idx_first % size(q_tile_shape);

        // KV coord remap
        // int kv_tile_idx = n_block;
        // //int kv_tile_res = kv_idx_first % size(kv_tile_shape);
        // int kv_tile_res = 0;

        // auto kv_tile_coord = idx2crd(kv_tile_idx, kv_diff_tiles);
        // auto kv_tile_offset = idx2crd(kv_tile_res, kv_tile_shape);
        // auto kv_thread_offset = tuple_add(
        //     kv_tile_offset,
        //     tuple_add(kv_blk_offset, tuple_mul(kv_tile_coord, kv_tile_shape)));

        // auto kv_ctr = make_identity_tensor(kv_tile_shape);
        // auto kv_ctr_offset = domain_offset(kv_thread_offset, kv_ctr);

        // auto kv_tile_offset = idx2crd(kv_tile_res, kv_tile_shape);
        auto q_tile_coord = idx2crd(q_tile_idx, q_tiled);
        //                              Offset from origin
        auto q_tile_offset = tuple_mul(q_tile_coord, q_tile_shape);
        auto q_ctr = make_identity_tensor(q_tile_shape);
        auto q_ctr_offset = domain_offset(q_tile_offset, q_ctr);

        auto kv_tile_coord = idx2crd(n_block, kv_diff_tiles);
        //                              Offset from origin + offset from start of KVs for this Q
        auto kv_tile_offset = tuple_add(kv_blk_offset, tuple_mul(kv_tile_coord, kv_tile_shape));
        auto kv_ctr = make_identity_tensor(kv_tile_shape);
        auto kv_ctr_offset = domain_offset(kv_tile_offset, kv_ctr);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<0>(tSrS_rowcol); i++) {
          // auto [q_idx, _] = tScS_rowcol(crd2idx(make_coord(i, 0), tSrS_rowcol.layout()));
          auto [q_idx, _] = tScS_rowcol(i, 0);

          auto q_coord = q_ctr_offset(q_idx - q_idx_first_in_tile);
          //auto q_coord = q_ctr_offset(q_idx);

          auto kv_start = get_window_start<Causal>(
              q_coord, window_left, window_right, stride, qkv_shape);
          auto kv_end =
              get_window_end<Causal>(q_coord, kv_start, window_size, qkv_shape);

          auto total_kv_tiles = ceil_div(kv_shape, KVTileShape{});


          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<1>(tSrS_rowcol); j++) {

            // auto [_, kv_idx] = tScS_rowcol(crd2idx(make_coord(i, j), tSrS_rowcol.layout()));
            auto [_, kv_idx] = tScS_rowcol(0, j);
            //auto kv_coord = kv_ctr_offset(kv_idx - kv_idx_first);
            auto kv_coord = kv_ctr_offset(kv_idx - kv_idx_first_in_tile);

            bool is_neigh = is_neighbor(kv_coord, kv_start, kv_end);

            if (not is_neigh) {
              tSrS_rowcol(i, j) = -INFINITY;
            }

#if 0
printf(
"\n[block: <%d, %d, %d>,  thrd: <%d, %d, %d>] thread_idx: %d, m_block: %d, n_block: %d, q_idx_first: %d, kv_idx_first: %d, i: %d, j: %d, q_idx: %d, kv_idx: %d, q_coord<%d, %d, %d>, kv_coord<%d, %d, %d>, kv_start<%d, %d, %d>, kv_end<%d, %d, %d>, is_neighbor: %d\n",
static_cast<int>(blockIdx.x),
static_cast<int>(blockIdx.y),
static_cast<int>(blockIdx.z),
static_cast<int>(threadIdx.x),
static_cast<int>(threadIdx.y),
static_cast<int>(threadIdx.z),
static_cast<int>(thread_idx),
static_cast<int>(m_block),
static_cast<int>(n_block),
static_cast<int>(q_idx_first),
static_cast<int>(kv_idx_first),
i,j,
static_cast<int>(q_idx),
static_cast<int>(kv_idx),
static_cast<int>(get<0>(q_coord)),
static_cast<int>(get<1>(q_coord)),
static_cast<int>(get<2>(q_coord)),
static_cast<int>(get<0>(kv_coord)),
static_cast<int>(get<1>(kv_coord)),
static_cast<int>(get<2>(kv_coord)),
static_cast<int>(get<0>(kv_start)),
static_cast<int>(get<1>(kv_start)),
static_cast<int>(get<2>(kv_start)),
static_cast<int>(get<0>(kv_end)),
static_cast<int>(get<1>(kv_end)),
static_cast<int>(get<2>(kv_end)),
static_cast<int>(is_neigh)
);
#endif
            //if (thread(1, 0)) {
            //bool neighbor = is_neighbor(kv_coord, kv_start, kv_end);
            //// One-liner print
            //  print("Q: "); print(q_coord); print(", KV: "); print(kv_coord);
            //  print(", neighbor? "); print(neighbor); print("\n");
            //}
          }
        }

    }

    template <typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply_padding(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) {

      auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);

      Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
      Tensor tScS = thread_mma.partition_C(cS);

      Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
      Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));

      // TODO (aditya): This doesn't consider SwapAB, fix.
      tScS_rowcol.data() = tScS_rowcol.data() + E<0>{} * m_block * kBlockM;
      tScS_rowcol.data() = tScS_rowcol.data() + E<1>{} * n_block * kBlockN + E<1>{} * size(kv_blk_offset);

      auto q_tile_shape = QTileShape{};
      auto kv_tile_shape = KVTileShape{};

      auto kv_tile_coord = idx2crd(n_block, kv_diff_tiles);
      auto kv_tile_offset = tuple_add(kv_blk_offset, tuple_mul(kv_tile_coord, kv_tile_shape));
      auto kv_ctr = make_identity_tensor(kv_tile_shape);
      auto kv_ctr_offset = domain_offset(kv_tile_offset, kv_ctr);

      int kv_tile_idx = n_block;
      int kv_idx_first_in_tile = (kv_tile_idx * size(kv_tile_shape)) + size(kv_blk_offset);

      // NOTE (aditya): Copied over with minor modifications from Hopper FNA

      // KV coord remap
      // auto [_, kv_idx_first] =
      //   tScS_rowcol(crd2idx(make_coord(0, 0), tSrS_rowcol.layout()));
      // int kv_tile_idx = kv_idx_first / size(kv_tile_shape);
      // int kv_tile_res = kv_idx_first % size(kv_tile_shape);

      // auto kv_tile_coord = idx2crd(kv_tile_idx, kv_diff_tiles);
      // auto kv_tile_offset = idx2crd(kv_tile_res, kv_tile_shape);
      // auto kv_thread_offset = tuple_add(
      //     kv_tile_offset,
      //     tuple_add(kv_blk_offset, tuple_mul(kv_tile_coord, kv_tile_shape)));

      // auto kv_ctr = make_identity_tensor(kv_tile_shape);
      // auto kv_ctr_offset = domain_offset(kv_thread_offset, kv_ctr);


      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < size<1>(tSrS_rowcol); j++) {
        auto [_, kv_idx] = tScS_rowcol(0, j);

        int iter_offset = kv_idx - kv_idx_first_in_tile; // % size(kv_tile_shape);
        auto kv_coord = kv_ctr_offset(iter_offset);

        if (not is_within_bounds(kv_coord, qkv_shape)) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size<0>(tSrS_rowcol); i++) {
            tSrS_rowcol(i, j) = -INFINITY;
          }
        }
      }
    }
};

template <int kBlockM, int kBlockN, bool PackGQA, typename TiledMma, bool SwapAB=false>
struct Mask {

    static_assert(!(PackGQA && SwapAB), "Cannot be both PackGQA and SwapAB");

    int const thread_idx;
    int const seqlen_q, seqlen_k;
    cutlass::FastDivmod const qhead_per_khead_divmod;

    CUTLASS_DEVICE
    Mask(const int thread_idx, const int seqlen_q, const int seqlen_k,
         cutlass::FastDivmod const &qhead_per_khead_divmod)
        : thread_idx(thread_idx)
        , seqlen_q(seqlen_q)
        , seqlen_k(seqlen_k)
        , qhead_per_khead_divmod(qhead_per_khead_divmod)
    {
    };

    template <bool Seqlenk_mask=false,
        typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) const {
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        if (!Seqlenk_mask) { return; }

        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

        static constexpr int Col = !SwapAB ? 1 : 0;
   
        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));
        // We want to use the col indices of thread0 to compare, since that is known at compile time.
        // So we subtract the limit by the first col index of this thread (get<Col>(tScS_rowcol(_0{}, _0{})))
        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;
        if constexpr (Seqlenk_mask) {  // Just masking based on col
            #pragma unroll
            for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= seqlenk_col_limit) {
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) { tSrS_rowcol(m, n) = -INFINITY; }
                }
            }
        }

    };

};

} // namespace flash_fna
} // namespace cuda
} // namespace natten
