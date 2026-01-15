/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <tuple>

namespace natten {
namespace cuda {
namespace flash_fna {

// Return {kBlockM, kBlockN, MmaPV_is_RS, IntraWGOverlap}
constexpr std::tuple<int, int, bool, bool> tile_size_fwd_sm90(
        int headdim, int headdim_v, int element_size=2, bool v_colmajor=false) {
    if (element_size == 2) {
        if (headdim <= 64) {
            // return {same_hdim ? 192 : 64, same_hdim ? 128 : 64, same_hdim, same_hdim};
            // With this workaround in Cutlass 3.8, tile size 192 x 128 got slower for non-causal, idk why
            // https://github.com/NVIDIA/cutlass/blob/833f6990e031b48b4cd2fcf55e0849c51ef6bac2/include/cute/container/tuple.hpp#L131
            if (headdim_v == 512) {
                return {64, 64, false, false};
            } else if (headdim_v == 256) {
                return {128, 96, true, false};
            } else {
                // Switch to tile size 192 x 192 for now
                return {192, 128, true, true};
            }
            // Good for long seqlen (>= 4k) but suffers from tile quantization at short seqlen
            // return {192, is_causal || is_local ? 192 : 176, true, false};
        } else if (headdim <= 96) {
            return {192, 128, false, true};
        } else if (headdim <= 128) {
            return {128, 128, true, true};
            // {128, 192, true, false} and {192, 128, false, true} are quite good too
            // 128 x 192 hits the limit of smem if MmaPV_is_RS, 128 x 144 hits the limit if !MmaPV_is_RS
        } else if (headdim <= 192) {
            return {128, 96, true, true};  // 128 x 112 hits the limit of smem
        } else {
            return {128, 80, true, true};  // 128 x 80 hits the limit of smem
        }
    } else {
        if (headdim <= 64) {
            return {192, 160, true, true};
        } else if (headdim <= 96) {
            return {192, 128, true, true};
        } else if (headdim <= 128) {
            return {128, 160, true, true};
        } else if (headdim <= 192) {
            return {128, 160, true, true};
        } else {
            return {128, 128, true, false};  // PagedKV uses more registers so we disabled IntraWGOverlap
        }
    }
}

// Return {kBlockM, kBlockN, kNWarps, kStages, Q_in_regs}
constexpr std::tuple<int, int, int, int, bool> tile_size_fwd_sm8x(
        bool sm86_or_89, int headdim, int headdim_v, int element_size=2,
        bool varlen_and_split=false) {
    if (element_size == 2) {
        if (headdim <= 64) {
            return {128, 112, 4, 1, false};
        } else if (headdim <= 96) {
            return {128, 64, 4, 1, false};
        } else if (headdim <= 128) {
            // return {128, sm86_or_89 ? 128 : 64, sm86_or_89 ? 8 : 4, 1, sm86_or_89};
            return {128, sm86_or_89 ? 128 : 64, 8, 1, sm86_or_89};
            // return {128, sm86_or_89 ? 128 : 64, 8, 1, true};
            // return {128, sm86_or_89 ? 128 : 64, 8, 1, true};
            // bool const use_8_warps = sm86_or_89;
            // return {128, use_8_warps ? 96 : 48, use_8_warps ? 8 : 4, 1, use_8_warps};
            // return {128, 64, 4, 1, false}; // 186 ms
            // return {128, 64, 8, 1, true}; // 24.185 ms
            // return {128, 64, 8, 1, false}; // 25.185 ms
        } else if (headdim <= 192) {
            return {128, 96, 8, sm86_or_89 ? 1 : 2, true};
        } else {
            return {128, sm86_or_89 ? 64: 96, 8, 1, sm86_or_89};
        }
    } else {
        // Placeholder for now
        return {128, 64, 8, 2, false};
    }
    // if (element_size == 2) {
    //     if (headdim <= 64) {
    //         return {128, varlen_and_split ? 80 : 112, 4, 1, false};
    //     } else if (headdim <= 96) {
    //         return {128, varlen_and_split ? 48 : 64, 4, 1, false};
    //     } else if (headdim <= 128) {
    //         bool const use_8_warps = sm86_or_89 | varlen_and_split;
    //         return {128, use_8_warps ? (varlen_and_split ? 112 : 128) : 64, use_8_warps ? 8 : 4, 1, use_8_warps};
    //     } else if (headdim <= 192) {
    //         bool const kBlockN_64 = varlen_and_split;
    //         return {128, kBlockN_64 ? 64 : 96, 8, sm86_or_89 ? 1 : 2, !kBlockN_64};
    //     } else {
    //         return {128, sm86_or_89 ? (varlen_and_split ? 48 : 64) : (varlen_and_split ? 64 : 96), 8, 1, sm86_or_89};
    //     }
    // } else {
    //     // Placeholder for now
    //     return {128, 64, 8, 2, false};
    // }
}
} // namespace flash_fna
} // namespace cuda
} // namespace natten
