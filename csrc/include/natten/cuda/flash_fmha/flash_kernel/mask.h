/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>
#include "cute/util/debug.hpp"
#include "cute/util/print.hpp"

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "utils.h"
// #include "natten/cuda/flash_fmha/utils.h"

namespace natten {
namespace cuda {
namespace flash {

using namespace cute;

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
        // if (thread0()) {
        //   print("================================================================================== "); print("\n");
        //   print("\t TiledMma{}=\n"); print(TiledMma{}); print("\n");
        //   print("\t thread_mma=\n"); print(thread_mma); print("\n");
        //   print("\t thread0_mma=\n"); print(thread0_mma); print("\n");
        //   print("\t cS= "); print(cS); print("\n");
        //   print("\t tSrS= "); print(tSrS); print("\n");
        //   print("\t tScS= "); print(tScS); print("\n");
        //   print("\t tSrS_rowcol= "); print(tSrS_rowcol); print("\n");
        //   print("\t tScS_rowcol= "); print(tScS_rowcol); print("\n");
        //   print("\t t0ScS= "); print(t0ScS); print("\n");
        //   print("\t t0ScS_rowcol= "); print(t0ScS_rowcol); print("\n");
        //   print("\t thread_col_offset= "); print(thread_col_offset); print("\n");
        //   print("\t seqlenk_col_limit= "); print(seqlenk_col_limit); print("\n");
        //   print("\t idk, some coord stuff="); print(get<Col>(t0ScS_rowcol(_0{}, 1))); print("\n");
        //   print("================================================================================== "); print("\n");
        // }

    };

};

} // namespace flash
} // namespace cuda
} // namespace natten
