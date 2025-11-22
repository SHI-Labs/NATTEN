/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include "cute/tensor.hpp"
#include "cute/util/print.hpp"

#include "seqlen.h"
#include "block.h"
#include "mask.h"
#include "pack_gqa.h"
#include "utils.h"
#include "na_utils.h"
// #include "natten/cuda/flash_fmha/seqlen.h"
// #include "natten/cuda/flash_fmha/block.h"
// #include "natten/cuda/flash_fmha/mask.h"
// #include "natten/cuda/flash_fmha/pack_gqa.h"
// #include "natten/cuda/flash_fmha/utils.h"

// #include "paged_kv.h"
// #include "rotary.h"

namespace natten {
namespace cuda {
namespace flash_fna {

using namespace cute;

template <int kNWarps, int Stages, bool Q_in_regs, class TileShape_MNK_, int kHeadDimV, class Element_, class ElementAccum_, class ArchTag_, bool PackGQA_,
         class NADim, class QTileShape, class KVTileShape, class Causal>
struct CollectiveMainloopFwdSm80 {

    static constexpr int kStages = Stages;
    static_assert(kStages > 0, "kStages must be greater than 0");
    using TileShape_MNK = TileShape_MNK_;
    using TileShape_MNK_PV = Shape<decltype(get<0>(TileShape_MNK{})), Int<kHeadDimV>, decltype(get<1>(TileShape_MNK{}))>;
    using Element = Element_;
    using ElementAccum = ElementAccum_;
    using ArchTag = ArchTag_;
    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;;
    static constexpr bool PackGQA = PackGQA_;
    static constexpr bool Transpose_V = Is_FP8;

    static_assert(ArchTag::kMinComputeCapability >= 80);

    static constexpr bool Has_cp_async = ArchTag::kMinComputeCapability >= 80;

    static constexpr int kBlockM = get<0>(TileShape_MNK{});
    static constexpr int kBlockN = get<1>(TileShape_MNK{});
    static constexpr int kHeadDim = get<2>(TileShape_MNK{});

    // using SeqlenInfo_t = flash::SeqlenInfoQKNewK<Varlen, AppendKV>;
    using SeqlenInfo_t = flash::SeqlenInfoQKNewK<false, false>;
    using BlockMN_t = flash::BlockMN<SeqlenInfo_t, kBlockM, kBlockN, PackGQA>;
    using NABlockMN_t = flash::NABlockMN<SeqlenInfo_t, kBlockM, kBlockN, NADim, QTileShape, KVTileShape, Causal, PackGQA>;

    using MMA_Atom_Arch = std::conditional_t<
        ArchTag::kMinComputeCapability >= 80,
        std::conditional_t<
            std::is_same_v<Element, cutlass::half_t>,
            MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
            MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>
        >,
        MMA_Atom<SM75_16x8x8_F32F16F16F32_TN>
    >;
    using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * kNWarps>, _16, _16>>; // 

    static constexpr int NumMmaThreads = size(TiledMma{});
    static constexpr int NumProducerThreads = NumMmaThreads;  // For compatibility with TileScheduler

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "Headdim must be a multiple of kGmemElemsPerLoad");
    // We want each "row" to have 64 elements (128 bytes, i.e. 1 cache line). E.g. if hdim=128, we want each
    // thread to have 4 loads in the M direction and 2 vectorized load in the K direction.
    static constexpr int kBytePerRow = kHeadDim * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);

    static constexpr int kSwizzle = kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
    static constexpr int kSwizzleBase = sizeof(Element) == 4 ? 2 : (sizeof(Element) == 2 ? 3 : 4);
    using SmemLayoutAtomQKV = decltype(
        composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
                    Layout<Shape<_8, Int<kBlockKGmem>>,
                           Stride<Int<kBlockKGmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQKV{}, select<0, 2>(TileShape_MNK{})));

    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomQKV{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomQKV{},
        make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutVt = decltype(
        composition(SmemLayoutV{},
                    make_ordered_layout(make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{}),
                                        Step<_2, _1, _3>{})));

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;

    // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
    // from the same address by the same threadblock. This is slightly faster.
    using GmemCopyAtom = Copy_Atom<std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>,
        AutoVectorizingCopyWithAssumedAlignment<128>
    >, Element>;

    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumMmaThreads % kGmemThreadsPerRow == 0, "NumMmaThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(GmemCopyAtom{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per read
    // So that we don't have to check if we overshot kBlockM when we load Q
    static_assert(kBlockM % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0);

    // For AppendKV, We want each thread to have at least 2 loads in the K direction since in the case of
    // non-interleaved rotary (combining elements at indices 0 and rotary_dim/2, 1 and rotary_dim/2+1, etc),
    // each thread will load twice from the same row.
    static constexpr int kBytePerHalfRow = kHeadDim / 2 * sizeof(Element);
    static constexpr int kBlockKGmemAppend = (kBytePerHalfRow % 128 == 0 ? 128 : (kBytePerHalfRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRowAppend = kBlockKGmemAppend / kGmemElemsPerLoad;
    static_assert(NumMmaThreads % kGmemThreadsPerRowAppend == 0, "NumMmaThreads must be a multiple of kGmemThreadsPerRowAppend");
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRowAppend == 0, "kGmemThreadsPerRowAppend must divide NumThreadsPerWarp");
    using GmemLayoutAtomAppend = Layout<Shape <Int<NumMmaThreads / kGmemThreadsPerRowAppend>, Int<kGmemThreadsPerRowAppend>>,
                                        Stride<Int<kGmemThreadsPerRowAppend>, _1>>;
    // If AppendKV, we'll be loading Q for rotary, and we assume divisibility to avoid predication
    using GmemTiledCopyAppendKV = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
                        GmemLayoutAtomAppend{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per store

    using ShapeQKV = cute::Shape<int32_t, int32_t, int32_t, int32_t>;  // (seqlen, d, head, batch)
    using StrideQK = cute::Stride<int64_t, _1, int64_t, int64_t>;
    using StrideV = StrideQK;
    // ((qhead_per_khead, seqlen_q), d, nheads_kv, batch, num_splits)
    using ShapeQPacked = std::conditional_t<!PackGQA, ShapeQKV, cute::Shape<cute::Shape<int32_t, int32_t>, int32_t, int32_t, int32_t>>;
    using StrideQPacked = std::conditional_t<!PackGQA, StrideQK, cute::Stride<cute::Stride<int64_t, int64_t>, _1, int64_t, int64_t>>;
    using ShapePageTable = cute::Shape<int32_t, int32_t>;  // (batch, max_num_pages_per_seq)
    using StridePageTable = cute::Stride<int64_t, _1>;
    using ShapeRotary = cute::Shape<int32_t, int32_t>;  // (seqlen_ro, rotary_dim // 2)
    using StrideRotary = cute::Stride<int64_t, _1>;
    using StrideDescale = cute::Stride<int64_t, int64_t>;

    static constexpr bool Share_QV_Smem = Q_in_regs;

    struct TensorStorageSharedQV : cute::aligned_struct<128> {
        union {
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
        };
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    };

    struct TensorStorageSeparateQV : cute::aligned_struct<128> {
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    };

    using TensorStorage = std::conditional_t<Share_QV_Smem, TensorStorageSharedQV, TensorStorageSeparateQV>;

    // Host side kernel arguments
    struct Arguments {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQK const stride_Q;
        Element* const ptr_K;  // Not Element const* since we might append to KV cache in-place
        ShapeQKV const shape_K;
        StrideQK const stride_K;
        Element* const ptr_V;
        int32_t const headdim_v;
        StrideV const stride_V;
        float const softmax_scale;
        float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
        StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
        int const num_splits;

        // NA Args
        NADim qkv_shape;
        NADim q_shape;
        NADim kv_shape;
        NADim window_size;
        NADim stride;
        NADim dilation;
        int num_heads_actual;
    };

    // Device side kernel params
    struct Params {
        Element const* const ptr_Q;
        ShapeQKV const shape_Q;
        StrideQK const stride_Q;
        ShapeQPacked const shape_Q_packed;
        StrideQPacked const stride_Q_packed;
        Element* const ptr_K;
        ShapeQKV const shape_K;
        StrideQK const stride_K;
        Element* const ptr_V;
        int32_t const headdim_v;
        StrideV const stride_V;
        cutlass::FastDivmod qhead_per_khead_divmod;
        float const softmax_scale_log2;
        float const* ptr_q_descale, *ptr_k_descale, *ptr_v_descale;
        StrideDescale const stride_q_descale, stride_k_descale, stride_v_descale;
        int const num_splits;

        // NA Args
        NADim qkv_shape;
        NADim q_shape;
        NADim kv_shape;
        NADim window_size;
        NADim window_left;
        NADim window_right;
        NADim stride;
        NADim dilation;
        int num_heads_actual;
        bool is_fully_block_sparse;
        bool has_kv_padding;
        bool requires_qkv_fixup;
        bool is_dilated;
    };

    static Params
    to_underlying_arguments(Arguments const& args) {
        // If PackGQA, reshape Q to be ((qhead_per_khead, seqlen_q), head_size, nhead_k, batch_size)
        int const qhead_per_khead = !PackGQA ? 1 : cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K));
        auto const shape_Q_packed = cute::conditional_return<!PackGQA>(
            args.shape_Q,
            make_shape(make_shape(qhead_per_khead, get<0>(args.shape_Q)), get<1>(args.shape_Q), get<2>(args.shape_K), get<3>(args.shape_Q))
        );
        auto const stride_Q_packed = cute::conditional_return<!PackGQA>(
            args.stride_Q,
            make_stride(make_stride(get<2>(args.stride_Q), get<0>(args.stride_Q)), get<1>(args.stride_Q), get<2>(args.stride_Q) * qhead_per_khead, get<3>(args.stride_Q))
        );
        // if (get<1>(args.shape_rotary) > 0) {
        //     assert(args.ptr_rotary_cos != nullptr && args.ptr_rotary_sin != nullptr);
        // }
        assert(args.num_splits >= 1);
        // Avoid dividing by zero
        // cutlass::FastDivmod attention_chunk_divmod(args.attention_chunk >= 1 ? args.attention_chunk : 1);
        // attention_chunk_divmod.divisor = args.attention_chunk;

        
        bool requires_qkv_fixup = not evenly_divides(args.qkv_shape, args.dilation);
        bool has_kv_padding = not evenly_divides(args.qkv_shape, KVTileShape{});
        bool is_dilated_ = is_dilated(args.dilation);
        bool is_fully_block_sparse = fully_block_sparse<Causal>(
            args.qkv_shape,
            args.window_size,
            args.stride,
            QTileShape{},
            KVTileShape{}
        );

        auto window_left = get_window_left(args.window_size);
        auto window_right = get_window_right(args.window_size);

        return {args.ptr_Q, args.shape_Q, args.stride_Q, shape_Q_packed, stride_Q_packed,
                args.ptr_K, args.shape_K, args.stride_K, args.ptr_V, args.headdim_v, args.stride_V,
                cutlass::FastDivmod(cute::ceil_div(get<2>(args.shape_Q), get<2>(args.shape_K))),
                float(args.softmax_scale * M_LOG2E),
                args.ptr_q_descale, args.ptr_k_descale, args.ptr_v_descale,
                args.stride_q_descale, args.stride_k_descale, args.stride_v_descale,
                1 /* args.num_splits */,
                args.qkv_shape, args.q_shape, args.kv_shape, args.window_size, window_left, window_right,
                args.stride, args.dilation, args.num_heads_actual,
                is_fully_block_sparse, has_kv_padding, requires_qkv_fixup, is_dilated_
               };
    }

    template <typename SharedStorage, typename FrgTensorO, typename Softmax>
    CUTLASS_DEVICE bool
    mma(Params const& params,
        FrgTensorO& tOrO,
        Softmax& softmax,
        int const thread_idx,
        SeqlenInfo_t const& seqlen_info,
        cute::tuple<int32_t, int32_t, int32_t, int32_t> block_coord,
        SharedStorage& shared_storage
        ) {

        static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");
        static constexpr int kBlockM = get<0>(TileShape_MNK{});
        static constexpr int kBlockN = get<1>(TileShape_MNK{});

        // can't use auto [m_block, ...] = block_coord since structured binding cannot be captured in lambda
        int const m_block = get<0>(block_coord);
        int const bidh = get<1>(block_coord);
        int const bidb = get<2>(block_coord);
        int const split_idx = get<3>(block_coord);
        int const bidh_kv = !PackGQA ? params.qhead_per_khead_divmod.divide(bidh) : bidh;

        int head_idx = bidh;
        auto qkv_shape = params.qkv_shape;
        bool is_fully_block_sparse = params.is_fully_block_sparse;
        bool has_kv_padding = params.has_kv_padding;
        if (params.requires_qkv_fixup) {
          qkv_shape = correct_qkv_shape(params.qkv_shape, head_idx, params.dilation, params.num_heads_actual);
          is_fully_block_sparse = fully_block_sparse<Causal>(
              qkv_shape,
              params.window_size,
              params.stride,
              QTileShape{},
              KVTileShape{});
          has_kv_padding = not evenly_divides(qkv_shape, KVTileShape{});
        } else if (params.is_dilated) {
          qkv_shape = ceil_div(params.qkv_shape, params.dilation);
          is_fully_block_sparse = fully_block_sparse<Causal>(
              qkv_shape,
              params.window_size,
              params.stride,
              QTileShape{},
              KVTileShape{});
          has_kv_padding = not evenly_divides(qkv_shape, KVTileShape{});
        }

        auto [kv_start_coord, num_kv_tiles] = NABlockMN_t::get_n_block_min_max(
            seqlen_info, m_block, bidb, split_idx, params.num_splits,
            params.qhead_per_khead_divmod,
            params.q_shape, qkv_shape, params.window_size, params.window_left,
            params.window_right, params.stride);


        // auto [_n_min, _n_max] = BlockMN_t::get_n_block_min_max(
        //     seqlen_info, m_block, bidb, split_idx, params.num_splits,
        //     params.qhead_per_khead_divmod
        //   );
        int const n_block_min = 0;
        int const n_block_max = size(num_kv_tiles);

        auto kv_start_tile = ceil_div(kv_start_coord, KVTileShape{});

        auto kv_tiled = ceil_div(params.kv_shape, KVTileShape{});
        auto ctr = make_identity_tensor(num_kv_tiles);
        auto ctr_offset = domain_offset(kv_start_tile, ctr);

        auto kv_tiled_layout = make_layout(kv_tiled);

        auto iter_to_tile_map = [&ctr_offset, &kv_tiled_layout](int iter) {
          return crd2idx(ctr_offset(iter), kv_tiled_layout);
        };

        
        // It's possible to have n_block_max <= n_block_min. We don't want to load Q or change any barrier

        Tensor sQ = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_q.data()), SmemLayoutQ{});
        Tensor sK = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_k.data()), SmemLayoutK{});
        Tensor sV = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutV{});
        Tensor sVt = make_tensor(make_smem_ptr(shared_storage.tensors.mainloop.smem_v.data()), SmemLayoutVt{});

        int const bidb_kv = bidb; 
        Tensor mQ = make_tensor(make_gmem_ptr(params.ptr_Q + seqlen_info.offset_q * get<0>(params.stride_Q)), params.shape_Q_packed, params.stride_Q_packed)(_, _, bidh, bidb);
        Tensor gQ = local_tile(mQ, select<0, 2>(TileShape_MNK{}), make_coord(m_block, _0{}));  // (M, K)
        Tensor mK = make_tensor(make_gmem_ptr(params.ptr_K + seqlen_info.offset_k * get<0>(params.stride_K)), params.shape_K, params.stride_K)(_, _, bidh_kv, bidb_kv);
        Tensor gK = local_tile(mK, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)
        Tensor mV = make_tensor(make_gmem_ptr(params.ptr_V + seqlen_info.offset_k * get<0>(params.stride_V)), params.shape_K, params.stride_V)(_, _, bidh_kv, bidb_kv);
        Tensor gV = local_tile(mV, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N, K, _)

        GmemTiledCopyQKV gmem_tiled_copy_QKV;
        auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(thread_idx);
        auto gmem_thr0_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(_0{});  // For index calculation

        Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
        Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
        Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
        Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

        TiledMma tiled_mma;
        auto thr_mma = tiled_mma.get_slice(thread_idx);

        // Allocate "fragments/descriptors"
        Tensor tSrQ = thr_mma.partition_fragment_A(sQ);

        // Copy Atom retiling
        auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
        auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
        auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(thread_idx);
        auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
        auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);
        Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
        Tensor tSsK = smem_thr_copy_K.partition_S(sK);
        Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

        // Predicates
        Tensor cKV = cute::make_identity_tensor(select<1, 2>(TileShape_MNK{}));
        Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
        Tensor t0KVcKV = gmem_thr0_copy_QKV.partition_S(cKV);
        Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(_0{}, _0{}, k)) < get<1>(params.shape_K); }

        int const seqlen_q = seqlen_info.seqlen_q;
        int const seqlen_k = seqlen_info.seqlen_k;
        int n_block = n_block_max - 1;

        // Prologue: load Q, K, V
        // If persistent, we don't need to wait for the previous work_idx to finish
        // since we assume that all MMA threads sync in the epilogue before writing to smem_o.
        // So any thread gets there, all threads must have finished the previous MMA and at least started
        // writing to smem_o.
        // If persistent, need to sync to make sure all threads have finished with smem_o before writing to smem_v
        if constexpr (Share_QV_Smem) { __syncthreads(); }

        // NOTE (aditya): We don't need to change any logic for loading Q tile. We only need to
        // modify the number of KV tiles and the mapping from kv tile itr idx to physical idx.
        if constexpr (!PackGQA) {
            Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
            Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
            Tensor cQ = cute::make_identity_tensor(select<0, 2>(TileShape_MNK{}));
            Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
            Tensor t0QcQ = gmem_thr0_copy_QKV.partition_S(cQ);
            Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
            #pragma unroll
            for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(_0{}, _0{}, k)) < get<1>(params.shape_Q); }
            // Instead of passing in tQcQ, we pass in t0QcQ and subtract the offset from the limit
            // (seqlen_q - m_block * kBlockM). This is because the entries of t0QcQ are known at compile time.
            // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
            flash::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/true>(
                gmem_tiled_copy_QKV, tQgQ, tQsQ, t0QcQ, tQpQ, seqlen_info.seqlen_q - m_block * kBlockM - get<0>(tQcQ(_0{}, _0{}, _0{}))
            );
        } else {
            using PackGQAt = flash::PackGQAManager<get<0>(TileShape_MNK{}), get<2>(TileShape_MNK{}), NumMmaThreads, Element>;
            PackGQAt::load_Q(mQ, sQ, params.qhead_per_khead_divmod, thread_idx, seqlen_q, m_block);
        }
        cute::cp_async_fence();

        // using PagedKVManager_t = PagedKVManager<get<1>(TileShape_MNK{}), get<2>(TileShape_MNK{}), get<1>(TileShape_MNK_PV{}), NumMmaThreads, Element, true /*KV_Same_Iter*/>;
        // PagedKVManager_t paged_kv_manager(
        //     params.ptr_pagetable, params.shape_pagetable, params.stride_pagetable,
        //     params.ptr_K, params.shape_K, params.stride_K,
        //     params.ptr_V, params.headdim_v, params.stride_V,
        //     params.page_size_divmod,
        //     params.page_size_divmod /*blockN_per_page_size_divmod, not used since we don't use TMA*/,
        //     bidb_kv, bidh_kv, thread_idx, seqlen_info.seqlen_k, seqlen_info.leftpad_k,
        //     0 /*bidb_kv_idx, not used since we don't use TMA for Sm8x*/
        // );


        auto load_K = [&] (int const n_block, int const smem_pipe_write, auto need_seqlenk_masking_type) {
            // NOTE (aditya): Add n_block mapping logic here
          
            int const kv_tile_idx = iter_to_tile_map(n_block);

            static constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
            // Do we need bound check to make sure the row doesn't go above kBlockN
            static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
            Tensor tKsK_cur = tKsK(_, _, _, smem_pipe_write);
            // Instead of passing in tKVcKV, we pass in t0KVcKV and subtract the offset from the limit
            // (seqlen_k - kv_tile_idx * kBlockN). This is because the entries of t0KVcKV are known at compile time.
            int const seqlenk_row_limit = -int(get<0>(tKVcKV(_0{}, _0{}, _0{}))) + (EvenN
                ? seqlen_info.seqlen_k - kv_tile_idx * kBlockN
                : (!Seqlenk_mask ? kBlockN : std::min(seqlen_info.seqlen_k - kv_tile_idx * kBlockN, kBlockN)));
            // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
            flash::copy</*Is_even_MN=*/!Seqlenk_mask && EvenN, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/true>(
                gmem_tiled_copy_QKV, tKgK(_, _, _, kv_tile_idx), tKsK_cur, t0KVcKV, tKVpKV, seqlenk_row_limit);
        };

        auto load_V = [&] (int const n_block, int const smem_pipe_write, auto need_seqlenk_masking_type) {
            // NOTE (aditya): Add n_block mapping logic here
            int const kv_tile_idx = iter_to_tile_map(n_block);
            static constexpr bool Seqlenk_mask = decltype(need_seqlenk_masking_type)::value;
            // Do we need bound check to make sure the row doesn't go above kBlockN
            static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtom{})) == 0;
            Tensor tVsV_cur = tVsV(_, _, _, smem_pipe_write);
            // We don't call flash::copy since it doesn't support bound checking
            // to not overshot kBlockN when writing to smem.
            Tensor tVgV_cur = tVgV(_, _, _, kv_tile_idx);
            int const seqlenk_row_limit = seqlen_info.seqlen_k - kv_tile_idx * kBlockN - get<0>(tKVcKV(_0{}, _0{}, _0{}));
            #pragma unroll
            for (int m = 0; m < size<1>(tVsV); ++m) {
                // If kBlockN doesn't evenly divide the tiled copy, only the last `m` needs to be checked
                if (EvenN || m < size<1>(tVsV) - 1 || get<0>(tKVcKV(_0{}, m, _0{})) < kBlockN) {
                    bool const predicate_n = !Seqlenk_mask || get<0>(t0KVcKV(_0{}, m, _0{})) < seqlenk_row_limit;
                    #pragma unroll
                    for (int k = 0; k < size<2>(tVsV); ++k) {
                        cute::copy(gmem_tiled_copy_QKV.with(tKVpKV(k) && predicate_n), tVgV_cur(_, m, k), tVsV_cur(_, m, k));
                    }
                }
            }
        };

        auto preprocess_Q = [&] {
            flash::cp_async_wait<Share_QV_Smem ? 1 : kStages * 2 - 1>();
            if constexpr (Q_in_regs) {
                __syncthreads();
                Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
                Tensor tSsQ_copy_view = smem_thr_copy_Q.partition_S(sQ);
                cute::copy(smem_tiled_copy_Q, tSsQ_copy_view, tSrQ_copy_view);
            }
        };

        // If Share_QV_Smem, we load Q, then load 1 stage of K, then (optionally) rotate Q and
        // read from smem_q to registers, then load V.
        // If !Share_QV, Smem, we load Q, load all stages of K & V, then (optionally) rotate Q.

        if constexpr (Share_QV_Smem) {
            load_K(n_block, 0, cute::true_type{} /*Seqlenk_mask*/);
            cute::cp_async_fence();
            preprocess_Q();
            __syncthreads();  // Make sure all threads have read smem_q before loading V
        }

        // For persistent, make sure all threads have finished reading smem_o
        if constexpr (!Share_QV_Smem) { __syncthreads(); }
        // Note, using the for_each() function here to ensure `stage` is of type Int<x>.
        for_each(make_int_sequence<kStages>{}, [&] (auto stage) {
            static constexpr bool Is_first_stage = CUTE_STATIC_V(stage) == 0;
            static constexpr bool Is_last_stage = CUTE_STATIC_V(stage) == kStages - 1;
            if constexpr (!Share_QV_Smem || !Is_first_stage) {
                if (Is_first_stage || n_block - stage >= n_block_min) {
                    load_K(n_block - stage, stage, cute::bool_constant<Is_first_stage>{} /*Seqlenk_mask*/);
                }
                // We want the fence outside the if statement to have a fixed number of cp.async commits.
                // so that we can wait with the correct number of outstanding commits.
                cute::cp_async_fence();
            }
            if constexpr (!Is_last_stage) {
                if (Is_first_stage || n_block - stage >= n_block_min) {
                    load_V(n_block - stage, stage, cute::bool_constant<Is_first_stage>{} /*Seqlenk_mask*/);
                }
                cute::cp_async_fence();
            }
        });

        if constexpr (!Share_QV_Smem) { preprocess_Q(); }

        // flash::Mask<kBlockM, kBlockN, PackGQA, TiledMma> mask(
        //     thread_idx, seqlen_q, seqlen_k, 
        //     // params.window_size_left, params.window_size_right, 0 /*sink_token_length*/, params.attention_chunk_divmod, 
        //     params.qhead_per_khead_divmod
        // );
        flash::NAMask<kBlockM, kBlockN, NADim, QTileShape, KVTileShape, Causal, PackGQA, TiledMma, decltype(iter_to_tile_map)>
          na_mask (thread_idx, seqlen_q, seqlen_k, params.qhead_per_khead_divmod, params.window_size, params.window_left,
              params.window_right, params.stride, qkv_shape, params.q_shape, params.kv_shape, kv_start_coord, num_kv_tiles, iter_to_tile_map);

        auto scoremod_premask_fn = [&](auto& tSrS) {};

        int smem_pipe_read = 0, smem_pipe_write = kStages - 1;

        auto load_K_next = [&] {
            if (n_block - kStages >= n_block_min) {
                load_K(n_block - kStages, kStages > 1 ? smem_pipe_write : 0, cute::false_type{} /*Seqlenk_mask*/);
            }
            cute::cp_async_fence();
        };

        auto sync = [&] {
            flash::cp_async_wait<kStages * 2 - 2>();
            __syncthreads();
        };

        clear(tOrO);

        auto fwd_step = [&](int const n_block, auto mask_fn, auto is_first_iter_type, auto check_inf_type) {

            static constexpr bool Is_first_iter = decltype(is_first_iter_type)::value;
            static constexpr bool Check_inf = decltype(check_inf_type)::value;
            Tensor tSrS = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
            clear(tSrS);
            sync();
            auto load_V_next = [&] {
                if (n_block - kStages + 1 >= n_block_min) {
                    load_V(n_block - kStages + 1, kStages > 1 ? smem_pipe_write : 0, cute::bool_constant<Is_first_iter && kStages == 1>{} /*Seqlenk_mask*/);
                }
                cute::cp_async_fence();
            };
            Tensor tSrQ_cur = cute::conditional_return<Q_in_regs>(tSrQ, thr_mma.partition_fragment_A(sQ));
            Tensor tSrK = thr_mma.partition_fragment_B(sK(_, _, _0{}));


            flash::gemm_sm80<Q_in_regs>(
                tSrS, tSrQ_cur, tSrK, tSsQ, tSsK(_, _, _, kStages > 1 ? smem_pipe_read : 0),
                tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K, load_V_next
            );
            smem_pipe_write = smem_pipe_write < kStages - 1 ? smem_pipe_write + 1 : 0;
            scoremod_premask_fn(tSrS);
            // Faster to load_K before gemm if we only have 1 stage
            if constexpr (kStages == 1) { sync(); load_K_next(); }
            mask_fn(tSrS, n_block);
            Tensor scores_scale = softmax.template max_get_scale</*Is_first=*/Is_first_iter, Check_inf>(tSrS);
            softmax.template online_softmax</*Is_first=*/Is_first_iter, Check_inf>(tSrS);
            if constexpr (Is_FP8) { flash::permute_Cregs_fp8(tSrS); }
            Tensor tOrP_acc = make_tensor(tSrS.data(), flash::convert_layout_acc_Aregs<TiledMma>(tSrS.layout()));
            Tensor tOrP = make_tensor_like<Element>(tOrP_acc);
            convert_type_out(tOrP_acc, tOrP);
            if constexpr (!Is_first_iter) { softmax.rescale_o(tOrO, scores_scale); }
            if constexpr (kStages > 1) { sync(); }
            Tensor tOrV = thr_mma.partition_fragment_B(sVt(_, _, _0{}));
            flash::gemm_rs_sm80(tOrO, tOrP, tOrV, tOsVt(_, _, _, kStages > 1 ? smem_pipe_read : 0), tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
            if constexpr (kStages > 1) { load_K_next(); }
            smem_pipe_read = smem_pipe_read < kStages - 1 ? smem_pipe_read + 1 : 0;
        };

        // auto first_iter_mask_fn = [&](auto& tSrS, int n_block) { mask.template apply<true /*Seqlenk_mask*/>(tSrS, m_block, n_block); };
        auto no_mask_fn = [&](auto& tSrS, int n_block) {};

        auto na_mask_fn = [&](auto& tSrS, int n_block) { 
          if (not is_fully_block_sparse) {
            na_mask.apply(tSrS, m_block, n_block);
          } else if (has_kv_padding) {
            na_mask.apply_padding(tSrS, m_block, n_block);
          } else {
          }
        };
        fwd_step(n_block, na_mask_fn, cute::true_type{} /*is_first_iter*/, cute::true_type{} /*check_inf*/);
        --n_block;

        // int const n_block_min_before_local_mask = BlockMN_t::get_n_block_min_before_local_mask(
        //     seqlen_info, m_block, n_block_min, 
        //     // params.window_size_left, params.attention_chunk_divmod,
        //     params.qhead_per_khead_divmod);
        // auto no_mask_fn = [](auto& tSrS, int n_block) { };

        #pragma unroll 1
        for (; n_block >= n_block_min; --n_block) {
            // NOTE (aditya): Always check for inf in all runs for NA mask
            fwd_step(n_block, na_mask_fn, cute::false_type{} /*is_first_iter*/, cute::true_type{} /*check_inf*/);
        }
        // Separate masking iterations on the left for local attention
        float const v_descale = !Is_FP8 || params.ptr_v_descale == nullptr ? 1.0f : params.ptr_v_descale[bidb * get<0>(params.stride_v_descale) + bidh_kv * get<1>(params.stride_v_descale)];
        Tensor scores_scale = softmax.finalize(v_descale);
        softmax.rescale_o(tOrO, scores_scale);
        if constexpr (Is_FP8) { flash::permute_output_fp8(tOrO); }
        return true;
    }
};

} // namespace flash_fna
} // namespace cuda
} // namespace natten
