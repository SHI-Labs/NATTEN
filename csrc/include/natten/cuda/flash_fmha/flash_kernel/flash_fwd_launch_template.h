/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"  // For device_kernel
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/cluster_launch.hpp"
#include "cutlass/kernel_launch.h"

#include "flash.h"
#include "tile_size.h"
#include "tile_scheduler.hpp"
#include "flash_fwd_kernel_sm80.h"
#include "mainloop_fwd_sm80.hpp"
#include "epilogue_fwd.hpp"

// #include "natten/cuda/flash_fmha/flash.h"
// #include "natten/cuda/flash_fmha/tile_size.h"
// #include "natten/cuda/flash_fmha/tile_scheduler.hpp"
// #include "natten/cuda/flash_fmha/flash_fwd_kernel_sm80.h"
// #include "natten/cuda/flash_fmha/mainloop_fwd_sm80.hpp"
// #include "natten/cuda/flash_fmha/epilogue_fwd.hpp"

namespace natten {
namespace cuda {
namespace flash {

using namespace cute;

template <int Arch, int kHeadDim, int kHeadDimV, int kBlockM, int kBlockN, typename Element, typename ElementOut,
          bool PackGQA = false, bool V_colmajor = false>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;
    static constexpr bool FP8_TransposeV = Is_FP8 && !V_colmajor;
    using ArchTag = std::conditional_t<Arch >= 90, cutlass::arch::Sm90, cutlass::arch::Sm80>;

    // Can't use structured binding since it's not compatible with constexpr
    static constexpr std::tuple<int, int, bool, bool> kBlockMN_RS_IntraWGOverlap = tile_size_fwd_sm90(kHeadDim, kHeadDimV, sizeof(Element) /*element_size*/, V_colmajor);
    static constexpr std::tuple<int, int, int, int, bool> kBlockMN_kNWarps_Stages_RS = tile_size_fwd_sm8x(Arch == 86 || Arch == 89, kHeadDim, kHeadDimV, sizeof(Element), /* varlen_and_split= */ false);
    // static constexpr int kBlockM = Arch >= 90 ? std::get<0>(kBlockMN_RS_IntraWGOverlap) : std::get<0>(kBlockMN_kNWarps_Stages_RS);
    // static constexpr int kBlockN = Arch >= 90 ? std::get<1>(kBlockMN_RS_IntraWGOverlap) : std::get<1>(kBlockMN_kNWarps_Stages_RS);
    static constexpr bool MmaPV_is_RS = std::get<2>(kBlockMN_RS_IntraWGOverlap);
    static constexpr bool IntraWGOverlap = std::get<3>(kBlockMN_RS_IntraWGOverlap);
    static constexpr int kNWarps = std::get<2>(kBlockMN_kNWarps_Stages_RS);
    static constexpr int kStages = Arch >= 90 ? 2 : std::get<3>(kBlockMN_kNWarps_Stages_RS);
    static constexpr bool Q_in_regs = Arch >= 90 ? false : std::get<4>(kBlockMN_kNWarps_Stages_RS);

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using TileShape_MNK_PV = cute::Shape<Int<kBlockM>, Int<kHeadDimV>, Int<kBlockN>>;
    using ClusterShape = cute::Shape<_1, _1, _1>;
    using CollectiveMainloop = flash::CollectiveMainloopFwdSm80<kNWarps, kStages, Q_in_regs, TileShape_MNK, kHeadDimV, Element, float, cutlass::arch::Sm80, PackGQA>;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<TileShape_MNK_PV, ClusterShape, ElementOut, ArchTag, CollectiveMainloop::NumMmaThreads, PackGQA, FP8_TransposeV>;

    using SchedulerPersistent = flash::StaticPersistentTileScheduler;
    using SchedulerSingleTile = flash::SingleTileScheduler<PackGQA, kBlockM>;
    // If Split then we probably don't have enough work for PersistentScheduler to be useful.
    // However, if Varlen (e.g., during decode where we have max_seqlens), using PersistentScheduler is better
    // since we'll avoid launching a bunch of thread blocks that immediately exit.
    // On Sm80, noncausal persistent seems a bit slower.
    static constexpr bool UsePersistentScheduler = false;
    using Scheduler = std::conditional_t<!UsePersistentScheduler, SchedulerSingleTile, SchedulerPersistent>;
    using AttnKernel = flash::enable_sm80_to_sm89<flash::FlashAttnFwdSm80<CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    // bool const is_varlen_q = false; // params.cu_seqlens_q;
    // bool const is_varlen_k = false; // params.cu_seqlens_k;
    // bool const is_varlen_k_new = params.cu_seqlens_knew;
    int seqlen_q = params.seqlen_q; //  !is_varlen_q ? params.seqlen_q : params.total_q;
    int batch_q = params.b;         // !is_varlen_q ? params.b : 1;
    int batch_k = params.b;         //!is_varlen_k ? (params.kv_batch_idx ? params.b_k : params.b) : 1;
    typename CollectiveMainloop::StrideV v_strides =
        cute::conditional_return<!V_colmajor>(
            make_stride(params.v_row_stride, _1{}, params.v_head_stride, params.v_batch_stride),
            make_stride(_1{}, params.v_dim_stride, params.v_head_stride, params.v_batch_stride));
    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {seqlen_q, params.d, params.h, batch_q},  // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride, params.q_batch_stride},  // stride_Q
        static_cast<Element*>(params.k_ptr),
        {params.seqlen_k, params.d, params.h_k, batch_k},  // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride, params.k_batch_stride},  // stride_K
        static_cast<Element*>(params.v_ptr),
        params.dv,  // headdim_v
        v_strides,  // stride_V
        params.scale_softmax,
        params.q_descale_ptr, params.k_descale_ptr, params.v_descale_ptr,
        {params.q_descale_batch_stride, params.q_descale_head_stride},
        {params.k_descale_batch_stride, params.k_descale_head_stride},
        {params.v_descale_batch_stride, params.v_descale_head_stride},
        params.num_splits,
    };
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<ElementOut*>(params.o_ptr),
        {seqlen_q, params.dv, params.h, batch_q, params.num_splits},  // shape_O
        {params.o_row_stride, _1{}, params.o_head_stride, params.o_batch_stride, 0}, // stride_O
        static_cast<float*>(params.oaccum_ptr),
        {params.oaccum_row_stride, _1{}, params.oaccum_head_stride, params.oaccum_batch_stride, params.oaccum_split_stride}, // stride_O_partial
        static_cast<float*>(params.softmax_lse_ptr),
        // {params.h, _1{}, seqlen_q * params.h, 0},
        {_1{}, seqlen_q, params.h * seqlen_q, 0},  // stride_LSE
        static_cast<float*>(params.softmax_lseaccum_ptr),
        {_1{}, seqlen_q, params.h * seqlen_q, params.h * seqlen_q * batch_q},  // stride_LSE_partial
        params.h_k,
        // params.cu_seqlens_q, params.seqused_q // NOTE (aditya): These have default init to
                                                 // nullptr in Epilogue constructor
    };

    int qhead_per_khead = !PackGQA ? 1 : cutlass::ceil_div(params.h, params.h_k);
    int num_blocks_m = cutlass::ceil_div(params.seqlen_q * qhead_per_khead, get<0>(TileShape_MNK{}));
    num_blocks_m = cutlass::round_up(num_blocks_m, size<0>(ClusterShape{}));
    typename flash::TileSchedulerArguments scheduler_args {
        num_blocks_m, !PackGQA ? params.h : params.h_k, params.b, params.num_splits,
        params.h / params.h_k,
        params.seqlen_q,
        params.seqlen_k, params.d, params.dv, sizeof(Element), 
        params.tile_count_semaphore,
    };


    int device;
    FLASH_CHECK_CUDA(cudaGetDevice(&device));
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args
    });

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    // Get the ptr to kernel function.
    if constexpr (size(ClusterShape{}) > 1) {
        void const* kernel = (void const*) cutlass::device_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024) {
            FLASH_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        dim3 cluster_dims(size<0>(ClusterShape{}), size<1>(ClusterShape{}), size<2>(ClusterShape{}));
        cutlass::ClusterLaunchParams launch_params{grid_dims, block_dims, cluster_dims, smem_size, stream};
        cutlass::launch_kernel_on_cluster(launch_params, kernel, kernel_params);
    } else {
        auto kernel = cutlass::device_kernel<AttnKernel>;
        if (smem_size >= 48 * 1024) {
            FLASH_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        }
        // kernel<<<grid_dims, block_dims, smem_size, stream>>>(kernel_params);
        cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, false);
    }
    FLASH_CHECK_CUDA_KERNEL_LAUNCH();
}
} // namespace flash
} // namespace cuda
} // namespace natten
