/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <iostream>

#include "cute/tensor.hpp"

#include "cutlass/device_kernel.h"  // For device_kernel
#include "cutlass/kernel_launch.h"  // For kernel_launch
#include "cutlass/cluster_launch.hpp"  // For ClusterLauncher

#include "flash.h"
#include "flash_bwd_preprocess_kernel.h"
#include "flash_bwd_postprocess_kernel.h"
#include "tile_scheduler.hpp"
#include "mainloop_bwd_sm80.hpp"
#include "epilogue_bwd.hpp"
#include "flash_bwd_kernel_sm80.h"

// #include "natten/cuda/flash_fmha/flash.h"
// #include "natten/cuda/flash_fmha/flash_bwd_preprocess_kernel.h"
// #include "natten/cuda/flash_fmha/flash_bwd_postprocess_kernel.h"
// #include "natten/cuda/flash_fmha/tile_scheduler.hpp"
// #include "natten/cuda/flash_fmha/mainloop_bwd_sm80.hpp"
// #include "natten/cuda/flash_fmha/epilogue_bwd.hpp"
// #include "natten/cuda/flash_fmha/flash_bwd_kernel_sm80.h"

// #include "flash_bwd_kernel_sm90.h"
// #include "static_switch.h"
// #include "mainloop_bwd_sm90_tma_gmma_ws.hpp"
namespace natten {
namespace cuda {
namespace flash_fna {

using namespace cute;

template <int Arch, int kHeadDim, int kBlockM, int kBlockN, typename Element,
          class NADim, class QTileShape, class KVTileShape, class Causal,
          bool Deterministic, bool GQA,
          int Stages_dO=2, int Stages_dS_or_QSm80=2,
          bool SdP_swapAB=true, bool dKV_swapAB=false, bool dQ_swapAB=false,
          int NumMmaWarpGroups=2, int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
          bool V_in_regs=false>
void run_flash_bwd(Flash_fna_bwd_params<NADim> &params, cudaStream_t stream) {
    // static_assert(!(Is_causal && Is_local), "Is_causal and Is_local cannot be true at the same time.");
    using ElementAccum = float;
    using ArchTag = std::conditional_t<Arch >= 90, cutlass::arch::Sm90, cutlass::arch::Sm80>;

    int const total_q_padded_rounded = cute::round_up(params.total_q + params.b * kBlockM, kBlockM);
    int const total_k_padded_rounded = cute::round_up(params.total_k + params.b * kBlockN, kBlockN);
    int seqlen_q = params.seqlen_q;
    int seqlen_k = params.seqlen_k;
    int seqlen_q_rounded = params.seqlen_q_rounded;
    int seqlen_k_rounded = params.seqlen_k_rounded;
    int batch_q = params.b;
    int batch_k = params.b;

    using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;
    using PreprocessKernel = flash_fna::FlashAttnBwdPreprocess<TileShape_MK, Element, ElementAccum, ArchTag, /*Clear_dQaccum=*/true>;
    typename PreprocessKernel::Arguments preprocess_args {
        static_cast<Element const*>(params.o_ptr),
        {seqlen_q, params.dv, params.h, batch_q},  // shape_O
        {params.o_row_stride, _1{}, params.o_head_stride, params.o_batch_stride},  // stride_O
        static_cast<Element const*>(params.do_ptr),
        {params.do_row_stride, _1{}, params.do_head_stride, params.do_batch_stride},  // stride_dO
        static_cast<float*>(params.dsoftmax_sum),
        {seqlen_q_rounded, params.h, batch_q},  // shape_dPsum
        {_1{}, seqlen_q_rounded, seqlen_q_rounded * params.h},  // stride_dPsum
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, seqlen_q, seqlen_q * params.h},  // stride_LSE
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {_1{}, seqlen_q_rounded, params.h * params.seqlen_q_rounded},  // stride_LSE_log2
        params.dq_accum_ptr == nullptr ? nullptr : static_cast<ElementAccum*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded, params.h, batch_q},  // shape_dQaccum
        {_1{}, seqlen_q_rounded * params.d_rounded, params.d_rounded * seqlen_q_rounded * params.h},  // stride_dQaccum
        params.b,
        params.dq_semaphore
        // , nullptr, // params.cu_seqlens_q,
        // nullptr // params.seqused_q
    };
    typename PreprocessKernel::Params preprocess_params = PreprocessKernel::to_underlying_arguments(preprocess_args);
    int num_m_block = cute::ceil_div(params.seqlen_q, kBlockM);
    dim3 grid_m(num_m_block, params.h, params.b);
    cutlass::kernel_launch<PreprocessKernel>(grid_m, PreprocessKernel::MaxThreadsPerBlock, PreprocessKernel::SharedStorageSize, stream, preprocess_params, false /*launch_with_pdl*/);
    FLASH_CHECK_CUDA_KERNEL_LAUNCH();
    FLASH_CHECK_CUDA(cudaDeviceSynchronize());
    FLASH_CHECK_CUDA(cudaGetLastError());

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape = cute::Shape<_1, Int<1>, _1>;  // Currently doesn't not support cluster
    // Stages_dS_or_QSm80 is Stages_dS if Sm90 and Stages if Sm80
    static constexpr int Stages = Arch >= 90 ? 2 : Stages_dS_or_QSm80;
    // static constexpr int Stages_dS = Arch >= 90 ? Stages_dS_or_QSm80 : 1;
    // using CollectiveMainloop = std::conditional_t<
    //     Arch >= 90,
    //     flash_fna::CollectiveMainloopBwdSm90<Stages, Stages_dO, Stages_dS, ClusterShape, TileShape_MNK, Element, ElementAccum, cutlass::arch::Sm90,
    //         Is_causal, Is_local, Has_softcap, Varlen, Deterministic,
    //         SdP_swapAB, dKV_swapAB, dQ_swapAB, NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs>,
    //     flash_fna::CollectiveMainloopBwdSm80<Stages, Stages_dO, TileShape_MNK, Element, ElementAccum, cutlass::arch::Sm80,
    //         Is_causal, Is_local, Has_softcap, Varlen, Deterministic,
    //         SdP_swapAB, dKV_swapAB, dQ_swapAB, NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs>
    // >;
    using CollectiveMainloop = flash_fna::CollectiveMainloopBwdSm80<Stages, Stages_dO, TileShape_MNK, Element, ElementAccum, cutlass::arch::Sm80,
            Deterministic, NADim, QTileShape, KVTileShape, Causal,
            SdP_swapAB, dKV_swapAB, dQ_swapAB, NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs>;
    using CollectiveEpilogue = std::conditional_t<
        !GQA,
        flash_fna::CollectiveEpilogueBwd<TileShape_MNK, Element, ArchTag, CollectiveMainloop::NumMmaThreads, dKV_swapAB, NumMmaWarpGroups * (Arch >= 90 ? 1 : cutlass::NumWarpsPerWarpGroup) / AtomLayoutNdKV>,
        flash_fna::CollectiveEpilogueBwdGQA<TileShape_MNK, ElementAccum, ArchTag, CollectiveMainloop::NumMmaThreads, Deterministic>
    >;
    // using Scheduler = std::conditional_t<
    //     Is_causal && !Varlen,
    //     flash_fna::SingleTileBwdLPTScheduler,
    //     // flash_fna::SingleTileScheduler<Varlen, false /*Split*/, false /*PackGQA*/, kBlockN>
    //     flash_fna::SingleTileScheduler</* PackGQA= */ false, kBlockN>
    // >;
    using Scheduler = flash_fna::SingleTileScheduler</* PackGQA= */ false, kBlockN>;
    // using AttnKernel = std::conditional_t<
    //     Arch >= 90,
    //     flash_fna::enable_sm90_or_later<flash_fna::FlashAttnBwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>,
    //     flash_fna::enable_sm80_to_sm89<flash_fna::FlashAttnBwdSm80<CollectiveMainloop, CollectiveEpilogue, Scheduler>>
    // >;
    using AttnKernel = flash_fna::enable_sm80_to_sm89<flash_fna::FlashAttnBwdSm80<CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, batch_q},  // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride},  // stride_Q
        static_cast<Element const*>(params.k_ptr),
        {seqlen_k, params.d, params.h_k, batch_k},  // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride},  // stride_K
        static_cast<Element const*>(params.v_ptr),
        {seqlen_k, params.dv, params.h_k, batch_k},  // shape_V
        {params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride},  // stride_V
        static_cast<Element const*>(params.do_ptr),
        {seqlen_q, params.dv, params.h, batch_q},  // shape_dO
        {params.do_row_stride, _1{}, params.do_head_stride, params.do_batch_stride},  // stride_dO
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded, params.h, batch_q},  // shape_dQaccum
        {_1{}, seqlen_q_rounded * params.d_rounded, params.d_rounded * params.seqlen_q_rounded * params.h}, // stride_dQaccum
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {seqlen_q_rounded, params.h, batch_q},  // shape_LSE
        {_1{}, seqlen_q_rounded, params.h * params.seqlen_q_rounded},  // stride_LSE_log2
        static_cast<float*>(params.dsoftmax_sum),
        {_1{}, seqlen_q_rounded, params.h * params.seqlen_q_rounded},  // stride_dPsum
        params.scale_softmax,
        params.b,
        params.dq_semaphore,
        // NA Args
        params.qkv_shape,
        params.q_shape,
        params.kv_shape,
        params.window_size,
        params.stride,
        params.dilation,
        params.num_heads_actual
    };
    // The case work with GQA is ugly but idk how to fix it.
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<typename CollectiveEpilogue::Element*>(!GQA ? params.dk_ptr : params.dk_accum_ptr),
        [&] {
            if constexpr (!GQA) {
                return typename CollectiveEpilogue::ShapedKV {seqlen_k, params.d, params.h, batch_k};  // shape_dK
            } else {
                return typename CollectiveEpilogue::ShapedKV {seqlen_k_rounded * params.d_rounded, params.h_k, batch_k};  // shape_dKaccum
            }
        }(),
        [&] {
            if constexpr (!GQA) {
                return typename CollectiveEpilogue::StridedKV {params.dk_row_stride, _1{}, params.dk_head_stride, params.dk_batch_stride};  // stride_dK
            } else {
                return typename CollectiveEpilogue::StridedKV {_1{}, params.d_rounded * seqlen_k_rounded, params.h_k * params.d_rounded * params.seqlen_k_rounded};  // stride_dKaccum
            }
        }(),
        static_cast<typename CollectiveEpilogue::Element*>(!GQA ? params.dv_ptr : params.dv_accum_ptr),
        [&] {
            if constexpr (!GQA) {
                return typename CollectiveEpilogue::ShapedKV {seqlen_k, params.dv, params.h, batch_k};  // shape_dV
            } else {
                return typename CollectiveEpilogue::ShapedKV {seqlen_k_rounded * params.dv_rounded, params.h_k, batch_k};  // shape_dVaccum
            }
        }(),
        [&] {
            if constexpr (!GQA) {
                return typename CollectiveEpilogue::StridedKV {params.dv_row_stride, _1{}, params.dv_head_stride, params.dv_batch_stride};  // stride_dV
            } else {
                return typename CollectiveEpilogue::StridedKV {_1{}, params.dv_rounded * seqlen_k_rounded, params.h_k * params.dv_rounded * params.seqlen_k_rounded};  // stride_dVaccum
            }
        }(),
        params.h,
        params.dk_semaphore,
        params.dv_semaphore,
        // params.cu_seqlens_k,
        // params.seqused_k,
    };

    int num_blocks_n = cutlass::ceil_div(params.seqlen_k, get<1>(TileShape_MNK{}));
    num_blocks_n = cutlass::round_up(num_blocks_n, size<1>(ClusterShape{}));
    typename flash_fna::TileSchedulerArguments scheduler_args {
        num_blocks_n, params.h, params.b, 1 /*num_splits*/,
        params.h / params.h_k,
        params.seqlen_k,
        params.seqlen_q, params.d, params.dv, sizeof(Element),
        params.tile_count_semaphore // , params.cu_seqlens_k, params.seqused_k
    };

    int device;
    cudaGetDevice(&device);
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args
    });

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    // int smem_size_q = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_q));
    // int smem_size_do = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_do));
    // int smem_size_ds = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_ds));
    // int smem_size_dqacc = [&] {
    //     if constexpr (Arch >= 90) {
    //         return sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_dqacc));
    //     } else {
    //         return 0;
    //     }
    // }();
    // int smem_size_k = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_k));
    // int smem_size_v = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v));
    // int smem_size_lse = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_lse));
    // int smem_size_dpsum = sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_dpsum));
    // printf("smem_size = %d, q = %d, k = %d, v = %d, do = %d, ds = %d, dqacc = %d, lse = %d, dpsum = %d\n", smem_size, smem_size_q, smem_size_k, smem_size_v, smem_size_do, smem_size_ds, smem_size_dqacc, smem_size_lse, smem_size_dpsum);
    void const* kernel = (void const*) cutlass::device_kernel<AttnKernel>;
    if constexpr (size(ClusterShape{}) > 1) {
        if (smem_size >= 48 * 1024) {
            FLASH_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
        cutlass::ClusterLauncher::launch(
            grid_dims, cluster_dims, block_dims, smem_size, stream, kernel, kernel_params, false /*launch_with_pdl*/);
    } else {
        if (smem_size >= 48 * 1024) {
            FLASH_CHECK_CUDA(cudaGetLastError());
            int max_smem_size;
            FLASH_CHECK_CUDA(cudaDeviceSynchronize());
            cudaDeviceGetAttribute(&max_smem_size, cudaDevAttrMaxSharedMemoryPerBlockOptin, device); 
            FLASH_CHECK_CUDA(cudaGetLastError());
            FLASH_CHECK_CUDA(cudaDeviceSynchronize());
            FLASH_CHECK_CUDA(cudaGetLastError());
            FLASH_CHECK_CUDA(cudaDeviceSynchronize());
            auto err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem_size);
            if (cudaSuccess != err) {
              err = cudaGetLastError(); // to clear the error bit
              std::cout << "  cudaFuncSetAttribute() returned error: " << cudaGetErrorString(err) << "\n";
            }
            FLASH_CHECK_CUDA(cudaGetLastError());
        }
        cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, false /*launch_with_pdl*/);
    }
    FLASH_CHECK_CUDA_KERNEL_LAUNCH();

    using PostprocessKernel = flash_fna::FlashAttnBwdPostprocessConvertdQ<TileShape_MK, Element, ElementAccum, ArchTag,
        AttnKernel::CollectiveMainloop::NumMmaThreads,
        typename AttnKernel::CollectiveMainloop::TiledMmadQ,
        AttnKernel::CollectiveMainloop::dQ_swapAB
        >;
    typename PostprocessKernel::Arguments postprocess_args {
        static_cast<ElementAccum const*>(params.dq_accum_ptr),
        {seqlen_q_rounded * params.d_rounded, params.h, batch_q},  // shape_dQaccum
        {_1{}, seqlen_q_rounded * params.d_rounded, params.d_rounded * params.seqlen_q_rounded * params.h}, // stride_dQaccum
        static_cast<Element*>(params.dq_ptr),
        {seqlen_q, params.d, params.h, batch_q},  // shape_dQ
        {params.dq_row_stride, _1{}, params.dq_head_stride, params.dq_batch_stride},  // stride_dQ
        params.scale_softmax,
    };
    typename PostprocessKernel::Params postprocess_params = PostprocessKernel::to_underlying_arguments(postprocess_args);
    int num_m_block_postprocess = cute::ceil_div(params.seqlen_q, get<0>(TileShape_MK{}));
    dim3 grid_m_postprocess(num_m_block_postprocess, params.h, params.b);
    int smem_size_postprocess = PostprocessKernel::SharedStorageSize;
    if (smem_size_postprocess >= 48 * 1024) {
        FLASH_CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<PostprocessKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_postprocess));
    }
    cutlass::kernel_launch<PostprocessKernel>(grid_m_postprocess, PostprocessKernel::MaxThreadsPerBlock, smem_size_postprocess, stream, postprocess_params, false /*launch_with_pdl*/);
    FLASH_CHECK_CUDA_KERNEL_LAUNCH();

    if constexpr (GQA) {
        using TileShape_NK = cute::Shape<Int<kBlockN>, Int<kHeadDim>>;
        using PostprocessKerneldKV = flash_fna::FlashAttnBwdPostprocessConvertdQ<TileShape_NK, Element, ElementAccum, ArchTag,
            AttnKernel::CollectiveEpilogue::NumEpilogueThreads,
            typename AttnKernel::CollectiveMainloop::TiledMmadKV,
            AttnKernel::CollectiveMainloop::dKV_swapAB
            >;
        typename PostprocessKerneldKV::Arguments postprocess_dK_args {
            static_cast<ElementAccum const*>(params.dk_accum_ptr),
            {seqlen_k_rounded * params.d_rounded, params.h_k, batch_k},  // shape_dKaccum
            {_1{}, seqlen_k_rounded * params.d_rounded, params.d_rounded * params.seqlen_k_rounded * params.h_k},  // stride_dKaccum
            static_cast<Element*>(params.dk_ptr),
            {seqlen_k, params.d, params.h_k, batch_k},  // shape_dK
            {params.dk_row_stride, _1{}, params.dk_head_stride, params.dk_batch_stride},  // stride_dK
            1.f,
            // params.cu_seqlens_k,
            // params.seqused_k
        };
        typename PostprocessKerneldKV::Params postprocess_dK_params = PostprocessKerneldKV::to_underlying_arguments(postprocess_dK_args);
        typename PostprocessKerneldKV::Arguments postprocess_dV_args {
            static_cast<ElementAccum const*>(params.dv_accum_ptr),
            {seqlen_k_rounded * params.dv_rounded, params.h_k, batch_k},  // shape_dVaccum
            {_1{}, seqlen_k_rounded * params.dv_rounded, params.dv_rounded * params.seqlen_k_rounded * params.h_k},  // stride_dVaccum
            static_cast<Element*>(params.dv_ptr),
            {seqlen_k, params.dv, params.h_k, batch_k},  // shape_dV
            {params.dv_row_stride, _1{}, params.dv_head_stride, params.dv_batch_stride},  // stride_dV
            1.f,
            // params.cu_seqlens_k,
            // params.seqused_k
        };
        typename PostprocessKerneldKV::Params postprocess_dV_params = PostprocessKerneldKV::to_underlying_arguments(postprocess_dV_args);
        int num_n_block_postprocess = cute::ceil_div(params.seqlen_k, get<0>(TileShape_NK{}));
        dim3 grid_n_postprocess(num_n_block_postprocess, params.h_k, params.b);
        int smem_size_postprocess = PostprocessKerneldKV::SharedStorageSize;
        if (smem_size_postprocess >= 48 * 1024) {
            FLASH_CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<PostprocessKerneldKV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_postprocess));
        }
        cutlass::kernel_launch<PostprocessKerneldKV>(grid_n_postprocess, PostprocessKerneldKV::MaxThreadsPerBlock, smem_size_postprocess, stream, postprocess_dK_params, false /*launch_with_pdl*/);
        FLASH_CHECK_CUDA_KERNEL_LAUNCH();
        cutlass::kernel_launch<PostprocessKerneldKV>(grid_n_postprocess, PostprocessKerneldKV::MaxThreadsPerBlock, smem_size_postprocess, stream, postprocess_dV_params, false /*launch_with_pdl*/);
        FLASH_CHECK_CUDA_KERNEL_LAUNCH();
    }

}
} // namespace flash_fna
} // namespace cuda
} // namespace natten
