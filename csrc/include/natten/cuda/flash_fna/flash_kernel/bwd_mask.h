

#pragma once

#include <cute/tensor.hpp>
#include "cute/util/debug.hpp"
#include "cute/util/print.hpp"

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "utils.h"
#include "na_utils.h"
// #include "natten/cuda/flash_fmha/utils.h"

namespace natten {
namespace cuda {
namespace flash_fna {

using namespace cute;

template<int kBlockM, int kBlockN, class NADim, class QTileShape, class KVTileShape, class Causal,
  bool PackGQA, typename TiledMma, class IterMapType, bool SwapAB=false>
struct BwdNAMask {
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
    NADim q_blk_offset;
    NADim q_diff_tiles;

    bool is_fully_block_sparse;
    bool has_q_padding;

    IterMapType iter_to_tile_map;
    cutlass::FastDivmod const qhead_per_khead_divmod;

    CUTLASS_DEVICE
    BwdNAMask (const int thread_idx, const int seqlen_q, const int seqlen_k,
        cutlass::FastDivmod const &qhead_per_khead_divmod,
        NADim window_size, NADim window_left, NADim window_right, NADim stride,
        NADim qkv_shape, NADim q_shape, NADim kv_shape,
        NADim q_blk_offset, NADim q_diff_tiles, IterMapType iter_to_tile_map,
        bool is_fully_block_sparse, bool has_q_padding):
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
        q_blk_offset(q_blk_offset),
        q_diff_tiles(q_diff_tiles),
        iter_to_tile_map(iter_to_tile_map),
        is_fully_block_sparse(is_fully_block_sparse),
        has_q_padding(has_q_padding) {}

    template <typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply_na_mask(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) {
        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);

        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);

        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash_fna::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash_fna::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));

        if constexpr (!SwapAB) {
          tScS_rowcol.data() = tScS_rowcol.data() + E<0>{} * m_block * kBlockM + E<0>{} * size(q_blk_offset);
          tScS_rowcol.data() = tScS_rowcol.data() + E<1>{} * n_block * kBlockN;
        }
        else {
          tScS_rowcol.data() = tScS_rowcol.data() + E<1>{} * m_block * kBlockM + E<1>{} * size(q_blk_offset);
          tScS_rowcol.data() = tScS_rowcol.data() + E<0>{} * n_block * kBlockN;
        }
        // tScS_rowcol.data() = tScS_rowcol.data() + E<0>{} * m_block * kBlockM + E<0>{} * size(q_blk_offset);
        // tScS_rowcol.data() = tScS_rowcol.data() + E<1>{} * n_block * kBlockN;

        auto q_tile_shape = QTileShape{};
        auto kv_tile_shape = KVTileShape{};

        auto stride_group_offset = get_bwd_stride_offset(stride);

        auto kv_tiled = ceil_div(kv_shape, kv_tile_shape);

        auto [q_idx_first, kv_idx_first] = tScS_rowcol(0);
        if constexpr (SwapAB) {
          auto tmp = q_idx_first;
          q_idx_first = kv_idx_first;
          kv_idx_first = tmp;
        }

        // KV coord remap
        int kv_tile_idx = kv_idx_first / size(kv_tile_shape);
        auto kv_tile_coord = idx2crd(kv_tile_idx, kv_tiled);
        auto kv_tile_offset = tuple_mul(kv_tile_coord, kv_tile_shape);
        int kv_idx_first_in_tile = kv_tile_idx * size(kv_tile_shape);
        auto kv_ctr = make_identity_tensor(kv_tile_shape);
        auto kv_ctr_offset = domain_offset(kv_tile_offset, kv_ctr);

        // Q coord remap
        int q_tile_idx = m_block;
        auto q_tile_coord = idx2crd(q_tile_idx, q_diff_tiles);
        auto q_tile_offset = tuple_add(q_blk_offset, tuple_mul(q_tile_coord, q_tile_shape));
        int q_idx_first_in_tile = (q_tile_idx * size(q_tile_shape)) + size(q_blk_offset);

        auto q_ctr = make_identity_tensor(q_tile_shape);
        auto q_ctr_offset = domain_offset(q_tile_offset, q_ctr);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tSrS_rowcol); i++) {
          auto [q_idx, kv_idx] = tScS_rowcol(i);

          if constexpr (SwapAB) {
            auto tmp = q_idx;
            q_idx = kv_idx;
            kv_idx = tmp;
          }

          auto q_coord = q_ctr_offset(q_idx - q_idx_first_in_tile);
          auto kv_coord = kv_ctr_offset(kv_idx - kv_idx_first_in_tile);

          auto q_start = get_bwd_window_start<Causal>(
              kv_coord,
              stride_group_offset,
              window_left,
              window_right,
              window_size,
              stride,
              qkv_shape);
          auto q_end = get_bwd_window_end<Causal>(
              kv_coord,
              stride_group_offset,
              window_left,
              window_right,
              window_size,
              stride,
              qkv_shape);

          bool is_neigh = is_neighbor(q_coord, q_start, q_end);
          if (not is_neighbor(q_coord, q_start, q_end)) {
            tSrS_rowcol(i) = -INFINITY;
          }
        }
  }

    template <typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply_padding(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) {

        // Q coord remap
        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);

        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);

        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash_fna::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash_fna::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));

        if constexpr (!SwapAB) {
          tScS_rowcol.data() = tScS_rowcol.data() + E<0>{} * m_block * kBlockM + E<0>{} * size(q_blk_offset);
          tScS_rowcol.data() = tScS_rowcol.data() + E<1>{} * n_block * kBlockN;
        }
        else {
          tScS_rowcol.data() = tScS_rowcol.data() + E<1>{} * m_block * kBlockM + E<1>{} * size(q_blk_offset);
          tScS_rowcol.data() = tScS_rowcol.data() + E<0>{} * n_block * kBlockN;
        }

        auto q_tile_shape = QTileShape{};

        auto stride_group_offset = get_bwd_stride_offset(stride);

        auto [q_idx_first, kv_idx_first] = tScS_rowcol(0);
        if constexpr (SwapAB) {
          auto tmp = q_idx_first;
          q_idx_first = kv_idx_first;
          kv_idx_first = tmp;
        }

        int q_tile_idx = m_block;
        auto q_tile_coord = idx2crd(q_tile_idx, q_diff_tiles);
        // auto q_tile_offset = idx2crd(q_tile_res, q_tile_shape);
        auto q_tile_offset = tuple_add(q_blk_offset, tuple_mul(q_tile_coord, q_tile_shape));
        int q_idx_first_in_tile = (q_tile_idx * size(q_tile_shape)) + size(q_blk_offset);

        auto q_ctr = make_identity_tensor(q_tile_shape);
        auto q_ctr_offset = domain_offset(q_tile_offset, q_ctr);

        // int q_tile_idx = q_idx_first / size(q_tile_shape);
        // int q_tile_res = q_idx_first % size(q_tile_shape);

        // auto q_tile_coord = idx2crd(q_tile_idx, q_diff_tiles);
        // auto q_tile_offset = idx2crd(q_tile_res, q_tile_shape);
        // auto q_thread_offset = tuple_add(
        //     q_tile_offset,
        //     tuple_add(q_blk_offset, tuple_mul(q_tile_coord, q_tile_shape)));

        // auto q_ctr = make_identity_tensor(q_tile_shape);
        // auto q_ctr_offset = domain_offset(q_thread_offset, q_ctr);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tSrS_rowcol); i++) {
          auto [q_idx, kv_idx] = tScS_rowcol(i);

          if constexpr (SwapAB) {
            auto tmp = q_idx;
            q_idx = kv_idx;
            kv_idx = tmp;
          }

          auto q_coord = q_ctr_offset(q_idx - q_idx_first_in_tile);

          if (not is_within_bounds(q_coord, qkv_shape)) {
            tSrS_rowcol(i) = -INFINITY;
          }
        }
    }

};

} // namespace flash_fna
} // namespace cuda
} // namespace natten
