/**
 * Metal Neighborhood Attention — Forward and Backward
 *
 * Tiled flash-attention forward and backward kernels for Apple Silicon (MPS).
 * Supports 1D, 2D, and 3D neighborhood attention with FP32/FP16/BF16.
 */

#include <torch/extension.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <mutex>
#include <atomic>
#include <tuple>

// =============================================================================
// NAParams must match the Metal shader struct exactly
// =============================================================================

struct NAParams {
    int32_t batch_size;
    int32_t seqlen_q;
    int32_t seqlen_kv;
    int32_t heads_q;
    int32_t heads_kv;
    int32_t dim;
    int32_t dim_value;
    int32_t num_additional_kv;
    float attn_scale;
    int32_t na_dim;
    int32_t qkv_shape[3];
    int32_t window_size[3];
    int32_t stride[3];
    int32_t dilation[3];
    int32_t is_causal[3];
};

// =============================================================================
// Metal Kernel Source (embedded inline)
// =============================================================================

static NSString* get_metal_source() {
    return @R"(
#include <metal_stdlib>
using namespace metal;

struct NAParams {
    int batch_size;
    int seqlen_q;
    int seqlen_kv;
    int heads_q;
    int heads_kv;
    int dim;
    int dim_value;
    int num_additional_kv;
    float attn_scale;
    int na_dim;
    int qkv_shape[3];
    int window_size[3];
    int stride[3];
    int dilation[3];
    int is_causal[3];
};

static inline int qkv_stride_fn(int na_dim, constant int* shape, int d) {
    int s = 1;
    for (int i = d + 1; i < na_dim; i++) {
        s *= shape[i];
    }
    return s;
}

static inline void idx_to_coord(int idx, int na_dim, constant int* shape, thread int* coord) {
    for (int d = 0; d < na_dim; d++) {
        int s = qkv_stride_fn(na_dim, shape, d);
        coord[d] = idx / s;
        idx = idx % s;
    }
}

static inline int qkv_fix_dilation(int qkv_shape, int dilation, int dilation_group) {
    int padding = 1 - ((dilation_group + (dilation - (qkv_shape % dilation))) / dilation);
    return (qkv_shape / dilation) + padding;
}

static inline int get_win_start_nc(int index, int window_left, int window_right, int stride_val, int length) {
    int stride_group_leader_idx = min((index / stride_val) * stride_val + (stride_val / 2), length - 1);
    return max(stride_group_leader_idx - window_left, 0) +
        ((stride_group_leader_idx + window_right >= length) *
         (length - window_right - stride_group_leader_idx - 1));
}

static inline int get_win_start_causal(int index, int window_left, int window_right, int stride_val, int length) {
    int stride_group_leader_idx = min((index / stride_val) * stride_val + stride_val - 1, length - 1);
    return max(stride_group_leader_idx - window_left - window_right, 0);
}

static inline int get_win_end_nc(int start, int window_size_val) {
    return start + window_size_val;
}

static inline int get_win_end_causal(int index, int length) {
    return min(index + 1, length);
}

static inline bool is_neighbor(int na_dim, thread int* kv_coord, thread int* win_start, thread int* win_end) {
    for (int d = 0; d < na_dim; d++) {
        if (kv_coord[d] < win_start[d] || kv_coord[d] >= win_end[d]) {
            return false;
        }
    }
    return true;
}

// ======== FP32 Kernel ========

kernel void na_forward_fp32(
    device const float* Q        [[buffer(0)]],
    device const float* K        [[buffer(1)]],
    device const float* V        [[buffer(2)]],
    device float* O              [[buffer(3)]],
    device float* LSE            [[buffer(4)]],
    constant NAParams& params    [[buffer(5)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]
) {
    int idx_Q = tgid.x;
    int idx_L = tgid.y;

    if (idx_Q >= params.seqlen_q) return;
    if (idx_L >= params.heads_q * params.batch_size) return;

    int batch_idx = idx_L / params.heads_q;
    int head_q_idx = idx_L % params.heads_q;
    int head_kv_idx = head_q_idx / (params.heads_q / params.heads_kv);

    int SQ = params.seqlen_q;
    int SK = params.seqlen_kv;
    int D = params.dim;
    int DV = params.dim_value;
    int H = params.heads_q;
    int HK = params.heads_kv;
    int na_dim = params.na_dim;
    int additional_kv_offset = SQ;

    int q_coord_global[3] = {0, 0, 0};
    idx_to_coord(idx_Q, na_dim, params.qkv_shape, q_coord_global);

    int q_di_group[3], q_coord[3];
    int corrected_shape[3];
    for (int d = 0; d < na_dim; d++) {
        q_di_group[d] = q_coord_global[d] % params.dilation[d];
        q_coord[d] = q_coord_global[d] / params.dilation[d];
        corrected_shape[d] = qkv_fix_dilation(params.qkv_shape[d], params.dilation[d], q_di_group[d]);
    }

    int win_start[3], win_end[3];
    for (int d = 0; d < na_dim; d++) {
        int wl = params.window_size[d] / 2;
        int wr = (params.window_size[d] / 2) + ((params.window_size[d] % 2) - 1);
        if (params.is_causal[d]) {
            win_start[d] = get_win_start_causal(q_coord[d], wl, wr, params.stride[d], corrected_shape[d]);
            win_end[d] = get_win_end_causal(q_coord[d], corrected_shape[d]);
        } else {
            win_start[d] = get_win_start_nc(q_coord[d], wl, wr, params.stride[d], corrected_shape[d]);
            win_end[d] = get_win_end_nc(win_start[d], params.window_size[d]);
        }
    }

    int q_base = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;
    int k_batch_offset = batch_idx * SK * HK * D;
    int k_head_offset = head_kv_idx * D;
    int v_batch_offset = batch_idx * SK * HK * DV;
    int v_head_offset = head_kv_idx * DV;

    constexpr int KV_TILE_SIZE = 2048;
    threadgroup float scores[KV_TILE_SIZE];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    constexpr int MAX_DIM = 1024;
    constexpr int NUM_THREADS = 256;
    constexpr int DIM_PER_THREAD = MAX_DIM / NUM_THREADS;
    float final_acc[DIM_PER_THREAD];
    for (int i = 0; i < DIM_PER_THREAD; i++) final_acc[i] = 0.0f;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    int num_kv_tiles = (SK + KV_TILE_SIZE - 1) / KV_TILE_SIZE;

    for (int tile = 0; tile < num_kv_tiles; tile++) {
        int tile_offset = tile * KV_TILE_SIZE;
        int tile_end = min(tile_offset + KV_TILE_SIZE, SK);
        int tile_len = tile_end - tile_offset;

        for (int idx_K = tile_offset + (int)tid; idx_K < tile_end; idx_K += (int)NUM_THREADS) {
            float acc = 0.0f;
            int k_offset = k_batch_offset + idx_K * HK * D + k_head_offset;
            // SIMD vectorized Q*K dot product
            int d4 = D / 4;
            for (int d = 0; d < d4; d++) {
                float4 q4 = *reinterpret_cast<device const float4*>(&Q[q_base + d * 4]);
                float4 k4 = *reinterpret_cast<device const float4*>(&K[k_offset + d * 4]);
                acc += dot(q4, k4);
            }
            for (int d = d4 * 4; d < D; d++) {
                acc += Q[q_base + d] * K[k_offset + d];
            }
            acc *= params.attn_scale;

            if (idx_K >= additional_kv_offset && idx_K - additional_kv_offset < params.num_additional_kv) {
                // Additional KV token - always visible
            } else if (idx_K >= additional_kv_offset) {
                acc = -INFINITY;
            } else {
                int kv_coord_global[3] = {0, 0, 0};
                idx_to_coord(idx_K, na_dim, params.qkv_shape, kv_coord_global);
                int kv_di_group[3], kv_coord[3];
                for (int d = 0; d < na_dim; d++) {
                    kv_di_group[d] = kv_coord_global[d] % params.dilation[d];
                    kv_coord[d] = kv_coord_global[d] / params.dilation[d];
                }
                bool di_match = true;
                for (int d = 0; d < na_dim; d++) {
                    if (q_di_group[d] != kv_di_group[d]) { di_match = false; break; }
                }
                if (!di_match || !is_neighbor(na_dim, kv_coord, win_start, win_end)) {
                    acc = -INFINITY;
                }
            }
            scores[idx_K - tile_offset] = acc;
        }

        float prev_max = running_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            for (int i = 0; i < tile_len; i++) {
                running_max = max(running_max, scores[i]);
            }
            shared_max = running_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        running_max = shared_max;

        if (running_max == -INFINITY) continue;

        if (prev_max != -INFINITY) {
            float correction = exp(prev_max - running_max);
            running_sum *= correction;
            for (int i = 0; i < DIM_PER_THREAD; i++) {
                final_acc[i] *= correction;
            }
        }

        for (int idx_K = tile_offset + (int)tid; idx_K < tile_end; idx_K += (int)NUM_THREADS) {
            scores[idx_K - tile_offset] = exp(scores[idx_K - tile_offset] - running_max);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            for (int i = 0; i < tile_len; i++) {
                running_sum += scores[i];
            }
            shared_sum = running_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        running_sum = shared_sum;

        for (int i = 0; i < DIM_PER_THREAD; i++) {
            int idx_D = tid + i * NUM_THREADS;
            if (idx_D < DV) {
                for (int j = 0; j < tile_len; j++) {
                    int idx_K = j + tile_offset;
                    int v_offset = v_batch_offset + idx_K * HK * DV + v_head_offset;
                    final_acc[i] += scores[j] * V[v_offset + idx_D];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    int o_base = batch_idx * SQ * H * DV + idx_Q * H * DV + head_q_idx * DV;
    float scale_val = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    for (int i = 0; i < DIM_PER_THREAD; i++) {
        int idx_D = tid + i * NUM_THREADS;
        if (idx_D < DV) {
            O[o_base + idx_D] = final_acc[i] * scale_val;
        }
    }

    if (tid == 0) {
        int lse_idx = batch_idx * SQ * H + idx_Q * H + head_q_idx;
        LSE[lse_idx] = log(running_sum) + running_max;
    }
}

// ======== FP16 Kernel ========

kernel void na_forward_fp16(
    device const half* Q         [[buffer(0)]],
    device const half* K         [[buffer(1)]],
    device const half* V         [[buffer(2)]],
    device half* O               [[buffer(3)]],
    device float* LSE            [[buffer(4)]],
    constant NAParams& params    [[buffer(5)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]
) {
    int idx_Q = tgid.x;
    int idx_L = tgid.y;

    if (idx_Q >= params.seqlen_q) return;
    if (idx_L >= params.heads_q * params.batch_size) return;

    int batch_idx = idx_L / params.heads_q;
    int head_q_idx = idx_L % params.heads_q;
    int head_kv_idx = head_q_idx / (params.heads_q / params.heads_kv);

    int SQ = params.seqlen_q;
    int SK = params.seqlen_kv;
    int D = params.dim;
    int DV = params.dim_value;
    int H = params.heads_q;
    int HK = params.heads_kv;
    int na_dim = params.na_dim;
    int additional_kv_offset = SQ;

    int q_coord_global[3] = {0, 0, 0};
    idx_to_coord(idx_Q, na_dim, params.qkv_shape, q_coord_global);

    int q_di_group[3], q_coord[3];
    int corrected_shape[3];
    for (int d = 0; d < na_dim; d++) {
        q_di_group[d] = q_coord_global[d] % params.dilation[d];
        q_coord[d] = q_coord_global[d] / params.dilation[d];
        corrected_shape[d] = qkv_fix_dilation(params.qkv_shape[d], params.dilation[d], q_di_group[d]);
    }

    int win_start[3], win_end[3];
    for (int d = 0; d < na_dim; d++) {
        int wl = params.window_size[d] / 2;
        int wr = (params.window_size[d] / 2) + ((params.window_size[d] % 2) - 1);
        if (params.is_causal[d]) {
            win_start[d] = get_win_start_causal(q_coord[d], wl, wr, params.stride[d], corrected_shape[d]);
            win_end[d] = get_win_end_causal(q_coord[d], corrected_shape[d]);
        } else {
            win_start[d] = get_win_start_nc(q_coord[d], wl, wr, params.stride[d], corrected_shape[d]);
            win_end[d] = get_win_end_nc(win_start[d], params.window_size[d]);
        }
    }

    int q_base = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;
    int k_batch_offset = batch_idx * SK * HK * D;
    int k_head_offset = head_kv_idx * D;
    int v_batch_offset = batch_idx * SK * HK * DV;
    int v_head_offset = head_kv_idx * DV;

    constexpr int KV_TILE_SIZE = 2048;
    threadgroup float scores[KV_TILE_SIZE];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    constexpr int MAX_DIM = 1024;
    constexpr int NUM_THREADS = 256;
    constexpr int DIM_PER_THREAD = MAX_DIM / NUM_THREADS;
    float final_acc[DIM_PER_THREAD];
    for (int i = 0; i < DIM_PER_THREAD; i++) final_acc[i] = 0.0f;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    int num_kv_tiles = (SK + KV_TILE_SIZE - 1) / KV_TILE_SIZE;

    for (int tile = 0; tile < num_kv_tiles; tile++) {
        int tile_offset = tile * KV_TILE_SIZE;
        int tile_end = min(tile_offset + KV_TILE_SIZE, SK);
        int tile_len = tile_end - tile_offset;

        for (int idx_K = tile_offset + (int)tid; idx_K < tile_end; idx_K += (int)NUM_THREADS) {
            float acc = 0.0f;
            int k_offset = k_batch_offset + idx_K * HK * D + k_head_offset;
            // SIMD vectorized Q*K dot product (half4 -> float4)
            int d4 = D / 4;
            for (int d = 0; d < d4; d++) {
                half4 q4 = *reinterpret_cast<device const half4*>(&Q[q_base + d * 4]);
                half4 k4 = *reinterpret_cast<device const half4*>(&K[k_offset + d * 4]);
                acc += dot(float4(q4), float4(k4));
            }
            for (int d = d4 * 4; d < D; d++) {
                acc += float(Q[q_base + d]) * float(K[k_offset + d]);
            }
            acc *= params.attn_scale;

            if (idx_K >= additional_kv_offset && idx_K - additional_kv_offset < params.num_additional_kv) {
            } else if (idx_K >= additional_kv_offset) {
                acc = -INFINITY;
            } else {
                int kv_coord_global[3] = {0, 0, 0};
                idx_to_coord(idx_K, na_dim, params.qkv_shape, kv_coord_global);
                int kv_di_group[3], kv_coord[3];
                for (int d = 0; d < na_dim; d++) {
                    kv_di_group[d] = kv_coord_global[d] % params.dilation[d];
                    kv_coord[d] = kv_coord_global[d] / params.dilation[d];
                }
                bool di_match = true;
                for (int d = 0; d < na_dim; d++) {
                    if (q_di_group[d] != kv_di_group[d]) { di_match = false; break; }
                }
                if (!di_match || !is_neighbor(na_dim, kv_coord, win_start, win_end)) {
                    acc = -INFINITY;
                }
            }
            scores[idx_K - tile_offset] = acc;
        }

        float prev_max = running_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            for (int i = 0; i < tile_len; i++) {
                running_max = max(running_max, scores[i]);
            }
            shared_max = running_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        running_max = shared_max;

        if (running_max == -INFINITY) continue;

        if (prev_max != -INFINITY) {
            float correction = exp(prev_max - running_max);
            running_sum *= correction;
            for (int i = 0; i < DIM_PER_THREAD; i++) {
                final_acc[i] *= correction;
            }
        }

        for (int idx_K = tile_offset + (int)tid; idx_K < tile_end; idx_K += (int)NUM_THREADS) {
            scores[idx_K - tile_offset] = exp(scores[idx_K - tile_offset] - running_max);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            for (int i = 0; i < tile_len; i++) {
                running_sum += scores[i];
            }
            shared_sum = running_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        running_sum = shared_sum;

        for (int i = 0; i < DIM_PER_THREAD; i++) {
            int idx_D = tid + i * NUM_THREADS;
            if (idx_D < DV) {
                for (int j = 0; j < tile_len; j++) {
                    int idx_K = j + tile_offset;
                    int v_offset = v_batch_offset + idx_K * HK * DV + v_head_offset;
                    final_acc[i] += scores[j] * float(V[v_offset + idx_D]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    int o_base = batch_idx * SQ * H * DV + idx_Q * H * DV + head_q_idx * DV;
    float scale_val = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    for (int i = 0; i < DIM_PER_THREAD; i++) {
        int idx_D = tid + i * NUM_THREADS;
        if (idx_D < DV) {
            O[o_base + idx_D] = half(final_acc[i] * scale_val);
        }
    }

    if (tid == 0) {
        int lse_idx = batch_idx * SQ * H + idx_Q * H + head_q_idx;
        LSE[lse_idx] = log(running_sum) + running_max;
    }
}

// ======== BF16 Kernel ========

kernel void na_forward_bf16(
    device const bfloat* Q       [[buffer(0)]],
    device const bfloat* K       [[buffer(1)]],
    device const bfloat* V       [[buffer(2)]],
    device bfloat* O             [[buffer(3)]],
    device float* LSE            [[buffer(4)]],
    constant NAParams& params    [[buffer(5)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]
) {
    int idx_Q = tgid.x;
    int idx_L = tgid.y;

    if (idx_Q >= params.seqlen_q) return;
    if (idx_L >= params.heads_q * params.batch_size) return;

    int batch_idx = idx_L / params.heads_q;
    int head_q_idx = idx_L % params.heads_q;
    int head_kv_idx = head_q_idx / (params.heads_q / params.heads_kv);

    int SQ = params.seqlen_q;
    int SK = params.seqlen_kv;
    int D = params.dim;
    int DV = params.dim_value;
    int H = params.heads_q;
    int HK = params.heads_kv;
    int na_dim = params.na_dim;
    int additional_kv_offset = SQ;

    int q_coord_global[3] = {0, 0, 0};
    idx_to_coord(idx_Q, na_dim, params.qkv_shape, q_coord_global);

    int q_di_group[3], q_coord[3];
    int corrected_shape[3];
    for (int d = 0; d < na_dim; d++) {
        q_di_group[d] = q_coord_global[d] % params.dilation[d];
        q_coord[d] = q_coord_global[d] / params.dilation[d];
        corrected_shape[d] = qkv_fix_dilation(params.qkv_shape[d], params.dilation[d], q_di_group[d]);
    }

    int win_start[3], win_end[3];
    for (int d = 0; d < na_dim; d++) {
        int wl = params.window_size[d] / 2;
        int wr = (params.window_size[d] / 2) + ((params.window_size[d] % 2) - 1);
        if (params.is_causal[d]) {
            win_start[d] = get_win_start_causal(q_coord[d], wl, wr, params.stride[d], corrected_shape[d]);
            win_end[d] = get_win_end_causal(q_coord[d], corrected_shape[d]);
        } else {
            win_start[d] = get_win_start_nc(q_coord[d], wl, wr, params.stride[d], corrected_shape[d]);
            win_end[d] = get_win_end_nc(win_start[d], params.window_size[d]);
        }
    }

    int q_base = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;
    int k_batch_offset = batch_idx * SK * HK * D;
    int k_head_offset = head_kv_idx * D;
    int v_batch_offset = batch_idx * SK * HK * DV;
    int v_head_offset = head_kv_idx * DV;

    constexpr int KV_TILE_SIZE = 2048;
    threadgroup float scores[KV_TILE_SIZE];
    threadgroup float shared_max;
    threadgroup float shared_sum;

    constexpr int MAX_DIM = 1024;
    constexpr int NUM_THREADS = 256;
    constexpr int DIM_PER_THREAD = MAX_DIM / NUM_THREADS;
    float final_acc[DIM_PER_THREAD];
    for (int i = 0; i < DIM_PER_THREAD; i++) final_acc[i] = 0.0f;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    int num_kv_tiles = (SK + KV_TILE_SIZE - 1) / KV_TILE_SIZE;

    for (int tile = 0; tile < num_kv_tiles; tile++) {
        int tile_offset = tile * KV_TILE_SIZE;
        int tile_end = min(tile_offset + KV_TILE_SIZE, SK);
        int tile_len = tile_end - tile_offset;

        for (int idx_K = tile_offset + (int)tid; idx_K < tile_end; idx_K += (int)NUM_THREADS) {
            float acc = 0.0f;
            int k_offset = k_batch_offset + idx_K * HK * D + k_head_offset;
            // SIMD vectorized Q*K dot product (bf16 -> float4)
            int d4 = D / 4;
            for (int d = 0; d < d4; d++) {
                int base = d * 4;
                float4 q4 = float4(float(Q[q_base + base]), float(Q[q_base + base + 1]),
                                   float(Q[q_base + base + 2]), float(Q[q_base + base + 3]));
                float4 k4 = float4(float(K[k_offset + base]), float(K[k_offset + base + 1]),
                                   float(K[k_offset + base + 2]), float(K[k_offset + base + 3]));
                acc += dot(q4, k4);
            }
            for (int d = d4 * 4; d < D; d++) {
                acc += float(Q[q_base + d]) * float(K[k_offset + d]);
            }
            acc *= params.attn_scale;

            if (idx_K >= additional_kv_offset && idx_K - additional_kv_offset < params.num_additional_kv) {
            } else if (idx_K >= additional_kv_offset) {
                acc = -INFINITY;
            } else {
                int kv_coord_global[3] = {0, 0, 0};
                idx_to_coord(idx_K, na_dim, params.qkv_shape, kv_coord_global);
                int kv_di_group[3], kv_coord[3];
                for (int d = 0; d < na_dim; d++) {
                    kv_di_group[d] = kv_coord_global[d] % params.dilation[d];
                    kv_coord[d] = kv_coord_global[d] / params.dilation[d];
                }
                bool di_match = true;
                for (int d = 0; d < na_dim; d++) {
                    if (q_di_group[d] != kv_di_group[d]) { di_match = false; break; }
                }
                if (!di_match || !is_neighbor(na_dim, kv_coord, win_start, win_end)) {
                    acc = -INFINITY;
                }
            }
            scores[idx_K - tile_offset] = acc;
        }

        float prev_max = running_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            for (int i = 0; i < tile_len; i++) {
                running_max = max(running_max, scores[i]);
            }
            shared_max = running_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        running_max = shared_max;

        if (running_max == -INFINITY) continue;

        if (prev_max != -INFINITY) {
            float correction = exp(prev_max - running_max);
            running_sum *= correction;
            for (int i = 0; i < DIM_PER_THREAD; i++) {
                final_acc[i] *= correction;
            }
        }

        for (int idx_K = tile_offset + (int)tid; idx_K < tile_end; idx_K += (int)NUM_THREADS) {
            scores[idx_K - tile_offset] = exp(scores[idx_K - tile_offset] - running_max);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid == 0) {
            for (int i = 0; i < tile_len; i++) {
                running_sum += scores[i];
            }
            shared_sum = running_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        running_sum = shared_sum;

        for (int i = 0; i < DIM_PER_THREAD; i++) {
            int idx_D = tid + i * NUM_THREADS;
            if (idx_D < DV) {
                for (int j = 0; j < tile_len; j++) {
                    int idx_K = j + tile_offset;
                    int v_offset = v_batch_offset + idx_K * HK * DV + v_head_offset;
                    final_acc[i] += scores[j] * float(V[v_offset + idx_D]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    int o_base = batch_idx * SQ * H * DV + idx_Q * H * DV + head_q_idx * DV;
    float scale_val = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;
    for (int i = 0; i < DIM_PER_THREAD; i++) {
        int idx_D = tid + i * NUM_THREADS;
        if (idx_D < DV) {
            O[o_base + idx_D] = bfloat(final_acc[i] * scale_val);
        }
    }

    if (tid == 0) {
        int lse_idx = batch_idx * SQ * H + idx_Q * H + head_q_idx;
        LSE[lse_idx] = log(running_sum) + running_max;
    }
}
)";
}

// =============================================================================
// Metal Tiled Forward Kernel Source (Flash-Attention style)
// =============================================================================

static NSString* get_metal_tiled_source() {
    return @R"(
#include <metal_stdlib>
using namespace metal;

struct NAParams {
    int batch_size;
    int seqlen_q;
    int seqlen_kv;
    int heads_q;
    int heads_kv;
    int dim;
    int dim_value;
    int num_additional_kv;
    float attn_scale;
    int na_dim;
    int qkv_shape[3];
    int window_size[3];
    int stride[3];
    int dilation[3];
    int is_causal[3];
};

// Tiling constants - two presets dispatched based on head dimension D
// Preset 0: D <= 64  -> Br=32, Bc=32
// Preset 1: D <= 128 -> Br=16, Bc=32
// D > 128 falls back to reference kernel (not dispatched here)
//
// Threadgroup: 256 threads = 8 simdgroups of 32
// Each simdgroup owns Br/8 Q rows
// Within a simdgroup, each lane handles 1 K position (32 lanes = Bc=32)

constant constexpr int NUM_THREADS = 256;
constant constexpr int SIMD_SIZE = 32;
constant constexpr int NUM_SIMDGROUPS = NUM_THREADS / SIMD_SIZE;  // 8

// Maximum head dim supported by tiled kernel
constant constexpr int MAX_D_TILED = 128;

// ---- Helper functions (same as reference kernel) ----

static inline int qkv_stride_fn(int na_dim, constant int* shape, int d) {
    int s = 1;
    for (int i = d + 1; i < na_dim; i++) {
        s *= shape[i];
    }
    return s;
}

static inline void idx_to_coord(int idx, int na_dim, constant int* shape, thread int* coord) {
    for (int d = 0; d < na_dim; d++) {
        int s = qkv_stride_fn(na_dim, shape, d);
        coord[d] = idx / s;
        idx = idx % s;
    }
}

static inline int qkv_fix_dilation(int qkv_shape, int dilation, int dilation_group) {
    int padding = 1 - ((dilation_group + (dilation - (qkv_shape % dilation))) / dilation);
    return (qkv_shape / dilation) + padding;
}

static inline int get_win_start_nc(int index, int window_left, int window_right, int stride_val, int length) {
    int stride_group_leader_idx = min((index / stride_val) * stride_val + (stride_val / 2), length - 1);
    return max(stride_group_leader_idx - window_left, 0) +
        ((stride_group_leader_idx + window_right >= length) *
         (length - window_right - stride_group_leader_idx - 1));
}

static inline int get_win_start_causal(int index, int window_left, int window_right, int stride_val, int length) {
    int stride_group_leader_idx = min((index / stride_val) * stride_val + stride_val - 1, length - 1);
    return max(stride_group_leader_idx - window_left - window_right, 0);
}

static inline int get_win_end_nc(int start, int window_size_val) {
    return start + window_size_val;
}

static inline int get_win_end_causal(int index, int length) {
    return min(index + 1, length);
}

// Compute NA mask for a single Q-K pair (returns true if K is a valid neighbor)
static inline bool compute_na_mask(
    int q_idx, int k_idx, int na_dim,
    constant int* qkv_shape, constant int* window_size,
    constant int* stride_arr, constant int* dilation_arr, constant int* is_causal
) {
    int q_coord_global[3] = {0, 0, 0};
    int k_coord_global[3] = {0, 0, 0};

    // idx_to_coord inline
    {
        int rem = q_idx;
        for (int d = 0; d < na_dim; d++) {
            int s = 1;
            for (int i = d + 1; i < na_dim; i++) s *= qkv_shape[i];
            q_coord_global[d] = rem / s;
            rem = rem % s;
        }
    }
    {
        int rem = k_idx;
        for (int d = 0; d < na_dim; d++) {
            int s = 1;
            for (int i = d + 1; i < na_dim; i++) s *= qkv_shape[i];
            k_coord_global[d] = rem / s;
            rem = rem % s;
        }
    }

    for (int d = 0; d < na_dim; d++) {
        int q_di = q_coord_global[d] % dilation_arr[d];
        int k_di = k_coord_global[d] % dilation_arr[d];
        if (q_di != k_di) return false;

        int q_c = q_coord_global[d] / dilation_arr[d];
        int k_c = k_coord_global[d] / dilation_arr[d];
        int eff_len = qkv_fix_dilation(qkv_shape[d], dilation_arr[d], q_di);

        int wl = window_size[d] / 2;
        int wr = (window_size[d] / 2) + ((window_size[d] % 2) - 1);

        int ws, we;
        if (is_causal[d]) {
            ws = get_win_start_causal(q_c, wl, wr, stride_arr[d], eff_len);
            we = get_win_end_causal(q_c, eff_len);
        } else {
            ws = get_win_start_nc(q_c, wl, wr, stride_arr[d], eff_len);
            we = get_win_end_nc(ws, window_size[d]);
        }

        if (k_c < ws || k_c >= we) return false;
    }
    return true;
}

// Compute flattened KV range bounds for a single Q position
// Returns (min_kv_flat, max_kv_flat) where max is exclusive
static inline void compute_kv_bounds(
    int q_idx, int na_dim,
    constant int* qkv_shape, constant int* window_size,
    constant int* stride_arr, constant int* dilation_arr, constant int* is_causal,
    thread int& min_kv, thread int& max_kv
) {
    int q_coord_global[3] = {0, 0, 0};
    {
        int rem = q_idx;
        for (int d = 0; d < na_dim; d++) {
            int s = 1;
            for (int i = d + 1; i < na_dim; i++) s *= qkv_shape[i];
            q_coord_global[d] = rem / s;
            rem = rem % s;
        }
    }

    // Compute per-dim window start/end in global coords, then flatten
    int kv_start_coord[3] = {0, 0, 0};
    int kv_end_coord[3] = {0, 0, 0};  // exclusive

    for (int d = 0; d < na_dim; d++) {
        int q_di = q_coord_global[d] % dilation_arr[d];
        int q_c = q_coord_global[d] / dilation_arr[d];
        int eff_len = qkv_fix_dilation(qkv_shape[d], dilation_arr[d], q_di);

        int wl = window_size[d] / 2;
        int wr = (window_size[d] / 2) + ((window_size[d] % 2) - 1);

        int ws, we;
        if (is_causal[d]) {
            ws = get_win_start_causal(q_c, wl, wr, stride_arr[d], eff_len);
            we = get_win_end_causal(q_c, eff_len);
        } else {
            ws = get_win_start_nc(q_c, wl, wr, stride_arr[d], eff_len);
            we = get_win_end_nc(ws, window_size[d]);
        }

        // Convert back to global coords
        kv_start_coord[d] = ws * dilation_arr[d] + q_di;
        // end is exclusive, last valid inner coord is (we-1), global = (we-1)*dil+di
        kv_end_coord[d] = min((we - 1) * dilation_arr[d] + q_di + 1, qkv_shape[d]);
    }

    // Flatten start coord -> min flat index
    min_kv = 0;
    max_kv = 0;
    for (int d = 0; d < na_dim; d++) {
        int s = 1;
        for (int i = d + 1; i < na_dim; i++) s *= qkv_shape[i];
        min_kv += kv_start_coord[d] * s;
        max_kv += (kv_end_coord[d] - 1) * s;
    }
    max_kv += 1;  // exclusive
}

// ---- Tiled Forward Kernel Macro ----
// Parameters: FNAME=kernel name, DTYPE=buffer type, BR=Q tile rows, MAX_D=max head dim
#define TILED_FWD_KERNEL(FNAME, DTYPE, BR, MAX_D) \
kernel void FNAME( \
    device const DTYPE* Q        [[buffer(0)]], \
    device const DTYPE* K        [[buffer(1)]], \
    device const DTYPE* V        [[buffer(2)]], \
    device DTYPE* O              [[buffer(3)]], \
    device float* LSE            [[buffer(4)]], \
    constant NAParams& params    [[buffer(5)]], \
    uint2 tgid                   [[threadgroup_position_in_grid]], \
    uint tid                     [[thread_index_in_threadgroup]], \
    uint simd_lane               [[thread_index_in_simdgroup]], \
    uint simdgroup_id            [[simdgroup_index_in_threadgroup]] \
) { \
    constexpr int Br = BR; \
    constexpr int Bc = 32; \
    constexpr int ROWS_PER_SIMD = Br / NUM_SIMDGROUPS; \
    \
    int tile_q_start = (int)tgid.x * Br; \
    int idx_L = (int)tgid.y; \
    if (idx_L >= params.heads_q * params.batch_size) return; \
    \
    int batch_idx = idx_L / params.heads_q; \
    int head_q_idx = idx_L % params.heads_q; \
    int head_kv_idx = head_q_idx / (params.heads_q / params.heads_kv); \
    \
    int SQ = params.seqlen_q; \
    int SK = params.seqlen_kv; \
    int D = params.dim; \
    int DV = params.dim_value; \
    int H = params.heads_q; \
    int HK = params.heads_kv; \
    int na_dim = params.na_dim; \
    \
    threadgroup float smem_K[Bc * MAX_D]; \
    threadgroup float smem_V[Bc * MAX_D]; \
    \
    float q_reg[ROWS_PER_SIMD][MAX_D]; \
    float o_acc[ROWS_PER_SIMD][MAX_D]; \
    float row_max[ROWS_PER_SIMD]; \
    float row_sum[ROWS_PER_SIMD]; \
    int q_indices[ROWS_PER_SIMD]; \
    \
    for (int r = 0; r < ROWS_PER_SIMD; r++) { \
        int q_row = tile_q_start + (int)simdgroup_id * ROWS_PER_SIMD + r; \
        q_indices[r] = q_row; \
        row_max[r] = -INFINITY; \
        row_sum[r] = 0.0f; \
        for (int dd = 0; dd < MAX_D; dd++) { o_acc[r][dd] = 0.0f; q_reg[r][dd] = 0.0f; } \
        if (q_row < SQ) { \
            int q_base = batch_idx * SQ * H * D + q_row * H * D + head_q_idx * D; \
            for (int dd = 0; dd < D; dd++) q_reg[r][dd] = float(Q[q_base + dd]); \
        } \
    } \
    \
    /* KV Bounding Box — as_type bit-cast through smem_K (no extra threadgroup memory) */ \
    int first_tile, last_tile; \
    { \
        int local_min_kv = SK, local_max_kv = 0; \
        for (int r = 0; r < ROWS_PER_SIMD; r++) { \
            int q_row = q_indices[r]; \
            if (q_row < SQ) { \
                int qmin, qmax; \
                compute_kv_bounds(q_row, na_dim, params.qkv_shape, params.window_size, \
                                  params.stride, params.dilation, params.is_causal, qmin, qmax); \
                local_min_kv = min(local_min_kv, qmin); \
                local_max_kv = max(local_max_kv, qmax); \
            } \
        } \
        int sg_mn = simd_min(local_min_kv), sg_mx = simd_max(local_max_kv); \
        if (simd_lane == 0) { \
            smem_K[simdgroup_id * 2 + 0] = as_type<float>(sg_mn); \
            smem_K[simdgroup_id * 2 + 1] = as_type<float>(sg_mx); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        int bbox_min_kv, bbox_max_kv; \
        if (tid == 0) { \
            bbox_min_kv = as_type<int>(smem_K[0]); \
            bbox_max_kv = as_type<int>(smem_K[1]); \
            for (int s = 1; s < NUM_SIMDGROUPS; s++) { \
                bbox_min_kv = min(bbox_min_kv, as_type<int>(smem_K[s * 2])); \
                bbox_max_kv = max(bbox_max_kv, as_type<int>(smem_K[s * 2 + 1])); \
            } \
            smem_K[0] = as_type<float>(bbox_min_kv); \
            smem_K[1] = as_type<float>(bbox_max_kv); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        bbox_min_kv = as_type<int>(smem_K[0]); \
        bbox_max_kv = as_type<int>(smem_K[1]); \
        first_tile = bbox_min_kv / Bc; \
        last_tile = min((bbox_max_kv + Bc - 1) / Bc, (SK + Bc - 1) / Bc); \
    } \
    int has_additional = params.num_additional_kv > 0 ? 1 : 0; \
    int additional_first_tile = SQ / Bc; \
    int additional_last_tile = (SQ + params.num_additional_kv + Bc - 1) / Bc; \
    \
    /* Main KV Tile Loop */ \
    for (int tile_idx = first_tile; tile_idx < last_tile; tile_idx++) { \
        int kv_start = tile_idx * Bc; \
        int kv_end = min(kv_start + Bc, SK); \
        int tile_len = kv_end - kv_start; \
        \
        /* Cooperative load K and V tiles */ \
        int total_k = tile_len * D; \
        for (int i = (int)tid; i < total_k; i += NUM_THREADS) { \
            int kk = i / D, dd = i % D; \
            smem_K[kk*D+dd] = float(K[batch_idx*SK*HK*D + (kv_start+kk)*HK*D + head_kv_idx*D + dd]); \
        } \
        int total_v = tile_len * DV; \
        for (int i = (int)tid; i < total_v; i += NUM_THREADS) { \
            int kk = i / DV, dd = i % DV; \
            smem_V[kk*DV+dd] = float(V[batch_idx*SK*HK*DV + (kv_start+kk)*HK*DV + head_kv_idx*DV + dd]); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        \
        for (int r = 0; r < ROWS_PER_SIMD; r++) { \
            int q_row = q_indices[r]; \
            if (q_row >= SQ) continue; \
            \
            float score = -INFINITY; \
            int k_pos = (int)simd_lane; \
            if (k_pos < tile_len) { \
                int global_k = kv_start + k_pos; \
                float acc = 0.0f; \
                for (int dd = 0; dd < D; dd += 4) { \
                    int rem = min(4, D - dd); \
                    if (rem == 4) { \
                        float4 q4 = float4(q_reg[r][dd], q_reg[r][dd+1], q_reg[r][dd+2], q_reg[r][dd+3]); \
                        float4 k4 = *reinterpret_cast<threadgroup const float4*>(&smem_K[k_pos*D+dd]); \
                        acc += dot(q4, k4); \
                    } else { for (int ddd = 0; ddd < rem; ddd++) acc += q_reg[r][dd+ddd] * smem_K[k_pos*D+dd+ddd]; } \
                } \
                acc *= params.attn_scale; \
                int additional_kv_offset = SQ; \
                if (global_k >= additional_kv_offset && global_k - additional_kv_offset < params.num_additional_kv) { \
                    score = acc; \
                } else if (global_k >= additional_kv_offset) { \
                    score = -INFINITY; \
                } else { \
                    bool is_nb = compute_na_mask(q_row, global_k, na_dim, \
                                                  params.qkv_shape, params.window_size, \
                                                  params.stride, params.dilation, params.is_causal); \
                    score = is_nb ? acc : -INFINITY; \
                } \
            } \
            \
            float tile_max = simd_max(score); \
            if (tile_max == -INFINITY) continue; \
            if (tile_max > row_max[r]) { \
                float correction = exp(row_max[r] - tile_max); \
                row_sum[r] *= correction; \
                for (int dd = 0; dd < DV; dd++) o_acc[r][dd] *= correction; \
                row_max[r] = tile_max; \
            } \
            float exp_score = (score != -INFINITY) ? exp(score - row_max[r]) : 0.0f; \
            row_sum[r] += simd_sum(exp_score); \
            \
            if (k_pos < tile_len) { \
                for (int dd = 0; dd < DV; dd += 4) { \
                    int rem = min(4, DV - dd); \
                    if (rem == 4) { \
                        float4 v4 = *reinterpret_cast<threadgroup const float4*>(&smem_V[k_pos*DV+dd]); \
                        float4 w = exp_score * v4; \
                        o_acc[r][dd] += simd_sum(w.x); o_acc[r][dd+1] += simd_sum(w.y); \
                        o_acc[r][dd+2] += simd_sum(w.z); o_acc[r][dd+3] += simd_sum(w.w); \
                    } else { for (int ddd = 0; ddd < rem; ddd++) o_acc[r][dd+ddd] += simd_sum(exp_score * smem_V[k_pos*DV+dd+ddd]); } \
                } \
            } else { \
                for (int dd = 0; dd < DV; dd += 4) { \
                    int rem = min(4, DV - dd); \
                    if (rem == 4) { simd_sum(0.0f); simd_sum(0.0f); simd_sum(0.0f); simd_sum(0.0f); } \
                    else { for (int ddd = 0; ddd < rem; ddd++) simd_sum(0.0f); } \
                } \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    \
    /* Additional KV tiles */ \
    if (has_additional) { \
        for (int tile_idx = additional_first_tile; tile_idx < additional_last_tile; tile_idx++) { \
            if (tile_idx >= first_tile && tile_idx < last_tile) continue; \
            int kv_start = tile_idx * Bc; \
            int kv_end = min(kv_start + Bc, SK); \
            int tile_len = kv_end - kv_start; \
            int total_k = tile_len * D; \
            for (int i = (int)tid; i < total_k; i += NUM_THREADS) { \
                int kk = i / D, dd = i % D; \
                smem_K[kk*D+dd] = float(K[batch_idx*SK*HK*D + (kv_start+kk)*HK*D + head_kv_idx*D + dd]); \
            } \
            int total_v = tile_len * DV; \
            for (int i = (int)tid; i < total_v; i += NUM_THREADS) { \
                int kk = i / DV, dd = i % DV; \
                smem_V[kk*DV+dd] = float(V[batch_idx*SK*HK*DV + (kv_start+kk)*HK*DV + head_kv_idx*DV + dd]); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            for (int r = 0; r < ROWS_PER_SIMD; r++) { \
                int q_row = q_indices[r]; \
                if (q_row >= SQ) continue; \
                float score = -INFINITY; \
                int k_pos = (int)simd_lane; \
                if (k_pos < tile_len) { \
                    int global_k = kv_start + k_pos; \
                    if (global_k >= SQ && global_k - SQ < params.num_additional_kv) { \
                        float acc = 0.0f; \
                        for (int dd = 0; dd < D; dd += 4) { \
                            int rem = min(4, D - dd); \
                            if (rem == 4) { \
                                float4 q4 = float4(q_reg[r][dd], q_reg[r][dd+1], q_reg[r][dd+2], q_reg[r][dd+3]); \
                                float4 k4 = *reinterpret_cast<threadgroup const float4*>(&smem_K[k_pos*D+dd]); \
                                acc += dot(q4, k4); \
                            } else { for (int ddd = 0; ddd < rem; ddd++) acc += q_reg[r][dd+ddd] * smem_K[k_pos*D+dd+ddd]; } \
                        } \
                        score = acc * params.attn_scale; \
                    } \
                } \
                float tile_max = simd_max(score); \
                if (tile_max == -INFINITY) continue; \
                if (tile_max > row_max[r]) { \
                    float correction = exp(row_max[r] - tile_max); \
                    row_sum[r] *= correction; \
                    for (int dd = 0; dd < DV; dd++) o_acc[r][dd] *= correction; \
                    row_max[r] = tile_max; \
                } \
                float exp_score = (score != -INFINITY) ? exp(score - row_max[r]) : 0.0f; \
                row_sum[r] += simd_sum(exp_score); \
                if (k_pos < tile_len) { \
                    for (int dd = 0; dd < DV; dd += 4) { \
                        int rem = min(4, DV - dd); \
                        if (rem == 4) { \
                            float4 v4 = *reinterpret_cast<threadgroup const float4*>(&smem_V[k_pos*DV+dd]); \
                            float4 w = exp_score * v4; \
                            o_acc[r][dd] += simd_sum(w.x); o_acc[r][dd+1] += simd_sum(w.y); \
                            o_acc[r][dd+2] += simd_sum(w.z); o_acc[r][dd+3] += simd_sum(w.w); \
                        } else { for (int ddd = 0; ddd < rem; ddd++) o_acc[r][dd+ddd] += simd_sum(exp_score * smem_V[k_pos*DV+dd+ddd]); } \
                    } \
                } else { \
                    for (int dd = 0; dd < DV; dd += 4) { \
                        int rem = min(4, DV - dd); \
                        if (rem == 4) { simd_sum(0.0f); simd_sum(0.0f); simd_sum(0.0f); simd_sum(0.0f); } \
                        else { for (int ddd = 0; ddd < rem; ddd++) simd_sum(0.0f); } \
                    } \
                } \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
    } \
    \
    /* Write output */ \
    if (simd_lane == 0) { \
        for (int r = 0; r < ROWS_PER_SIMD; r++) { \
            int q_row = q_indices[r]; \
            if (q_row >= SQ) continue; \
            int o_base = batch_idx * SQ * H * DV + q_row * H * DV + head_q_idx * DV; \
            float s = (row_sum[r] > 0.0f) ? (1.0f / row_sum[r]) : 0.0f; \
            for (int dd = 0; dd < DV; dd++) O[o_base + dd] = DTYPE(o_acc[r][dd] * s); \
            LSE[batch_idx * SQ * H + q_row * H + head_q_idx] = log(row_sum[r]) + row_max[r]; \
        } \
    } \
}

TILED_FWD_KERNEL(na_forward_tiled_fp32_br32_d32, float, 32, 32)
TILED_FWD_KERNEL(na_forward_tiled_fp32_br32, float, 32, 64)

TILED_FWD_KERNEL(na_forward_tiled_fp32_br16, float, 16, 128)
TILED_FWD_KERNEL(na_forward_tiled_fp16_br32_d32, half, 32, 32)
TILED_FWD_KERNEL(na_forward_tiled_fp16_br32, half, 32, 64)
TILED_FWD_KERNEL(na_forward_tiled_fp16_br16, half, 16, 128)
TILED_FWD_KERNEL(na_forward_tiled_bf16_br32_d32, bfloat, 32, 32)
TILED_FWD_KERNEL(na_forward_tiled_bf16_br32, bfloat, 32, 64)
TILED_FWD_KERNEL(na_forward_tiled_bf16_br16, bfloat, 16, 128)

)";
}

// =============================================================================
// Metal Backward Kernel Source (embedded inline)
// =============================================================================

static NSString* get_metal_backward_source() {
    return @R"(
#include <metal_stdlib>
using namespace metal;

struct NAParams {
    int batch_size;
    int seqlen_q;
    int seqlen_kv;
    int heads_q;
    int heads_kv;
    int dim;
    int dim_value;
    int num_additional_kv;
    float attn_scale;
    int na_dim;
    int qkv_shape[3];
    int window_size[3];
    int stride[3];
    int dilation[3];
    int is_causal[3];
};

static inline int qkv_stride_fn(int na_dim, constant int* shape, int d) {
    int s = 1;
    for (int i = d + 1; i < na_dim; i++) {
        s *= shape[i];
    }
    return s;
}

static inline void idx_to_coord(int idx, int na_dim, constant int* shape, thread int* coord) {
    for (int d = 0; d < na_dim; d++) {
        int s = qkv_stride_fn(na_dim, shape, d);
        coord[d] = idx / s;
        idx = idx % s;
    }
}

static inline int qkv_fix_dilation(int qkv_shape, int dilation, int dilation_group) {
    int padding = 1 - ((dilation_group + (dilation - (qkv_shape % dilation))) / dilation);
    return (qkv_shape / dilation) + padding;
}

static inline int get_win_start_nc(int index, int window_left, int window_right, int stride_val, int length) {
    int stride_group_leader_idx = min((index / stride_val) * stride_val + (stride_val / 2), length - 1);
    return max(stride_group_leader_idx - window_left, 0) +
        ((stride_group_leader_idx + window_right >= length) *
         (length - window_right - stride_group_leader_idx - 1));
}

static inline int get_win_start_causal(int index, int window_left, int window_right, int stride_val, int length) {
    int stride_group_leader_idx = min((index / stride_val) * stride_val + stride_val - 1, length - 1);
    return max(stride_group_leader_idx - window_left - window_right, 0);
}

static inline int get_win_end_nc(int start, int window_size_val) {
    return start + window_size_val;
}

static inline int get_win_end_causal(int index, int length) {
    return min(index + 1, length);
}

static inline bool is_neighbor(int na_dim, thread int* kv_coord, thread int* win_start, thread int* win_end) {
    for (int d = 0; d < na_dim; d++) {
        if (kv_coord[d] < win_start[d] || kv_coord[d] >= win_end[d]) {
            return false;
        }
    }
    return true;
}

// Compute NA mask: returns true if (idx_Q, idx_K) is a valid neighbor pair
static inline bool compute_na_mask(
    int idx_Q, int idx_K, constant NAParams& params
) {
    int na_dim = params.na_dim;
    int additional_kv_offset = params.seqlen_q;

    // Additional KV tokens are always visible
    if (idx_K >= additional_kv_offset && idx_K - additional_kv_offset < params.num_additional_kv) {
        return true;
    }
    // Beyond additional KV range = invalid
    if (idx_K >= additional_kv_offset) {
        return false;
    }

    int q_coord_global[3] = {0, 0, 0};
    idx_to_coord(idx_Q, na_dim, params.qkv_shape, q_coord_global);

    int kv_coord_global[3] = {0, 0, 0};
    idx_to_coord(idx_K, na_dim, params.qkv_shape, kv_coord_global);

    int q_di_group[3], q_coord[3], kv_di_group[3], kv_coord[3];
    int corrected_shape[3];
    for (int d = 0; d < na_dim; d++) {
        q_di_group[d] = q_coord_global[d] % params.dilation[d];
        q_coord[d] = q_coord_global[d] / params.dilation[d];
        kv_di_group[d] = kv_coord_global[d] % params.dilation[d];
        kv_coord[d] = kv_coord_global[d] / params.dilation[d];
        corrected_shape[d] = qkv_fix_dilation(params.qkv_shape[d], params.dilation[d], q_di_group[d]);
    }

    // Dilation group mismatch
    for (int d = 0; d < na_dim; d++) {
        if (q_di_group[d] != kv_di_group[d]) return false;
    }

    // Window check
    int win_start[3], win_end[3];
    for (int d = 0; d < na_dim; d++) {
        int wl = params.window_size[d] / 2;
        int wr = (params.window_size[d] / 2) + ((params.window_size[d] % 2) - 1);
        if (params.is_causal[d]) {
            win_start[d] = get_win_start_causal(q_coord[d], wl, wr, params.stride[d], corrected_shape[d]);
            win_end[d] = get_win_end_causal(q_coord[d], corrected_shape[d]);
        } else {
            win_start[d] = get_win_start_nc(q_coord[d], wl, wr, params.stride[d], corrected_shape[d]);
            win_end[d] = get_win_end_nc(win_start[d], params.window_size[d]);
        }
    }

    return is_neighbor(na_dim, kv_coord, win_start, win_end);
}

// =============================================================================
// dQ Backward Kernel — one threadgroup per Q position
// =============================================================================

kernel void na_backward_dQ_fp32(
    device const float* Q        [[buffer(0)]],
    device const float* K        [[buffer(1)]],
    device const float* V        [[buffer(2)]],
    device const float* O        [[buffer(3)]],
    device const float* dO       [[buffer(4)]],
    device const float* LSE      [[buffer(5)]],
    device float* dQ             [[buffer(6)]],
    constant NAParams& params    [[buffer(7)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]
) {
    int idx_Q = tgid.x;
    int idx_L = tgid.y;

    if (idx_Q >= params.seqlen_q) return;
    if (idx_L >= params.heads_q * params.batch_size) return;

    int batch_idx = idx_L / params.heads_q;
    int head_q_idx = idx_L % params.heads_q;
    int head_kv_idx = head_q_idx / (params.heads_q / params.heads_kv);

    int SQ = params.seqlen_q;
    int SK = params.seqlen_kv;
    int D = params.dim;
    int DV = params.dim_value;
    int H = params.heads_q;
    int HK = params.heads_kv;

    int q_base = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;
    int do_base = batch_idx * SQ * H * DV + idx_Q * H * DV + head_q_idx * DV;
    int o_base = do_base;
    int k_batch_offset = batch_idx * SK * HK * D;
    int k_head_offset = head_kv_idx * D;
    int v_batch_offset = batch_idx * SK * HK * DV;
    int v_head_offset = head_kv_idx * DV;

    float lse_val = LSE[batch_idx * SQ * H + idx_Q * H + head_q_idx];

    constexpr int KV_TILE_SIZE = 2048;
    threadgroup float scores[KV_TILE_SIZE];

    constexpr int MAX_DIM = 1024;
    constexpr int NUM_THREADS = 256;
    constexpr int DIM_PER_THREAD = MAX_DIM / NUM_THREADS;
    float final_acc[DIM_PER_THREAD];
    for (int i = 0; i < DIM_PER_THREAD; i++) final_acc[i] = 0.0f;

    int num_kv_tiles = (SK + KV_TILE_SIZE - 1) / KV_TILE_SIZE;

    for (int tile = 0; tile < num_kv_tiles; tile++) {
        int tile_offset = tile * KV_TILE_SIZE;
        int tile_end = min(tile_offset + KV_TILE_SIZE, SK);
        int tile_len = tile_end - tile_offset;

        for (int idx_K = tile_offset + (int)tid; idx_K < tile_end; idx_K += (int)NUM_THREADS) {
            // Compute Q*K dot product (SIMD vectorized)
            float acc_qk = 0.0f;
            int k_offset = k_batch_offset + idx_K * HK * D + k_head_offset;
            { int d4 = D / 4;
              for (int d = 0; d < d4; d++) {
                  float4 q4 = *reinterpret_cast<device const float4*>(&Q[q_base + d * 4]);
                  float4 k4 = *reinterpret_cast<device const float4*>(&K[k_offset + d * 4]);
                  acc_qk += dot(q4, k4);
              }
              for (int d = d4 * 4; d < D; d++) acc_qk += Q[q_base + d] * K[k_offset + d];
            }
            acc_qk *= params.attn_scale;

            // Compute dO*V dot product (SIMD vectorized)
            float acc_dov = 0.0f;
            int v_offset = v_batch_offset + idx_K * HK * DV + v_head_offset;
            { int d4 = DV / 4;
              for (int d = 0; d < d4; d++) {
                  float4 do4 = *reinterpret_cast<device const float4*>(&dO[do_base + d * 4]);
                  float4 v4 = *reinterpret_cast<device const float4*>(&V[v_offset + d * 4]);
                  acc_dov += dot(do4, v4);
              }
              for (int d = d4 * 4; d < DV; d++) acc_dov += dO[do_base + d] * V[v_offset + d];
            }
            acc_dov *= params.attn_scale;

            // Compute dO*O dot product (SIMD vectorized)
            float acc_doo = 0.0f;
            { int d4 = DV / 4;
              for (int d = 0; d < d4; d++) {
                  float4 do4 = *reinterpret_cast<device const float4*>(&dO[do_base + d * 4]);
                  float4 o4 = *reinterpret_cast<device const float4*>(&O[o_base + d * 4]);
                  acc_doo += dot(do4, o4);
              }
              for (int d = d4 * 4; d < DV; d++) acc_doo += dO[do_base + d] * O[o_base + d];
            }
            acc_doo *= params.attn_scale;

            // Apply NA mask
            if (!compute_na_mask(idx_Q, idx_K, params)) {
                acc_qk = -INFINITY;
            }

            scores[idx_K - tile_offset] = exp(min(acc_qk - lse_val, 0.0f)) * (acc_dov - acc_doo);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate: dQ += scores[j] * K[j]
        for (int i = 0; i < DIM_PER_THREAD; i++) {
            int idx_D = tid + i * NUM_THREADS;
            if (idx_D < D) {
                for (int j = 0; j < tile_len; j++) {
                    int idx_K = j + tile_offset;
                    int k_offset = k_batch_offset + idx_K * HK * D + k_head_offset;
                    final_acc[i] += scores[j] * K[k_offset + idx_D];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write dQ
    int dq_base = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;
    for (int i = 0; i < DIM_PER_THREAD; i++) {
        int idx_D = tid + i * NUM_THREADS;
        if (idx_D < D) {
            dQ[dq_base + idx_D] = final_acc[i];
        }
    }
}

// =============================================================================
// dK Backward Kernel — one threadgroup per K position
// =============================================================================

kernel void na_backward_dK_fp32(
    device const float* Q        [[buffer(0)]],
    device const float* K        [[buffer(1)]],
    device const float* V        [[buffer(2)]],
    device const float* O        [[buffer(3)]],
    device const float* dO       [[buffer(4)]],
    device const float* LSE      [[buffer(5)]],
    device float* dK             [[buffer(6)]],
    constant NAParams& params    [[buffer(7)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]
) {
    int idx_K = tgid.x;
    int idx_L = tgid.y;

    if (idx_K >= params.seqlen_kv) return;
    if (idx_L >= params.heads_kv * params.batch_size) return;

    int batch_idx = idx_L / params.heads_kv;
    int head_kv_idx = idx_L % params.heads_kv;
    int gqa_ratio = params.heads_q / params.heads_kv;

    int SQ = params.seqlen_q;
    int SK = params.seqlen_kv;
    int D = params.dim;
    int DV = params.dim_value;
    int H = params.heads_q;
    int HK = params.heads_kv;

    int k_base = batch_idx * SK * HK * D + idx_K * HK * D + head_kv_idx * D;
    int v_base = batch_idx * SK * HK * DV + idx_K * HK * DV + head_kv_idx * DV;

    constexpr int Q_TILE_SIZE = 2048;
    threadgroup float scores[Q_TILE_SIZE];

    constexpr int MAX_DIM = 1024;
    constexpr int NUM_THREADS = 256;
    constexpr int DIM_PER_THREAD = MAX_DIM / NUM_THREADS;
    float final_acc[DIM_PER_THREAD];

    int num_q_tiles = (SQ + Q_TILE_SIZE - 1) / Q_TILE_SIZE;

    // Iterate over all Q heads in the GQA group
    for (int gqa = 0; gqa < gqa_ratio; gqa++) {
        int head_q_idx = head_kv_idx * gqa_ratio + gqa;

        for (int i = 0; i < DIM_PER_THREAD; i++) final_acc[i] = 0.0f;

        for (int tile = 0; tile < num_q_tiles; tile++) {
            int tile_offset = tile * Q_TILE_SIZE;
            int tile_end = min(tile_offset + Q_TILE_SIZE, SQ);
            int tile_len = tile_end - tile_offset;

            for (int idx_Q = tile_offset + (int)tid; idx_Q < tile_end; idx_Q += (int)NUM_THREADS) {
                int q_base = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;
                int do_base = batch_idx * SQ * H * DV + idx_Q * H * DV + head_q_idx * DV;
                int o_base = do_base;

                // Q*K dot product (SIMD vectorized)
                float acc_qk = 0.0f;
                { int d4 = D / 4;
                  for (int d = 0; d < d4; d++) {
                      float4 q4 = *reinterpret_cast<device const float4*>(&Q[q_base + d * 4]);
                      float4 k4 = *reinterpret_cast<device const float4*>(&K[k_base + d * 4]);
                      acc_qk += dot(q4, k4);
                  }
                  for (int d = d4 * 4; d < D; d++) acc_qk += Q[q_base + d] * K[k_base + d];
                }
                acc_qk *= params.attn_scale;

                // dO*V dot product (SIMD vectorized)
                float acc_dov = 0.0f;
                { int d4 = DV / 4;
                  for (int d = 0; d < d4; d++) {
                      float4 do4 = *reinterpret_cast<device const float4*>(&dO[do_base + d * 4]);
                      float4 v4 = *reinterpret_cast<device const float4*>(&V[v_base + d * 4]);
                      acc_dov += dot(do4, v4);
                  }
                  for (int d = d4 * 4; d < DV; d++) acc_dov += dO[do_base + d] * V[v_base + d];
                }
                acc_dov *= params.attn_scale;

                // dO*O dot product (SIMD vectorized)
                float acc_doo = 0.0f;
                { int d4 = DV / 4;
                  for (int d = 0; d < d4; d++) {
                      float4 do4 = *reinterpret_cast<device const float4*>(&dO[do_base + d * 4]);
                      float4 o4 = *reinterpret_cast<device const float4*>(&O[o_base + d * 4]);
                      acc_doo += dot(do4, o4);
                  }
                  for (int d = d4 * 4; d < DV; d++) acc_doo += dO[do_base + d] * O[o_base + d];
                }
                acc_doo *= params.attn_scale;

                // Apply NA mask
                if (!compute_na_mask(idx_Q, idx_K, params)) {
                    acc_qk = -INFINITY;
                }

                float lse_val = LSE[batch_idx * SQ * H + idx_Q * H + head_q_idx];
                scores[idx_Q - tile_offset] = exp(min(acc_qk - lse_val, 0.0f)) * (acc_dov - acc_doo);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate: dK += scores[j] * Q[j]
            for (int i = 0; i < DIM_PER_THREAD; i++) {
                int idx_D = tid + i * NUM_THREADS;
                if (idx_D < D) {
                    for (int j = 0; j < tile_len; j++) {
                        int idx_Q = j + tile_offset;
                        int q_offset = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;
                        final_acc[i] += scores[j] * Q[q_offset + idx_D];
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Accumulate into dK (additive across GQA heads)
        for (int i = 0; i < DIM_PER_THREAD; i++) {
            int idx_D = tid + i * NUM_THREADS;
            if (idx_D < D) {
                if (gqa == 0) {
                    dK[k_base + idx_D] = final_acc[i];
                } else {
                    dK[k_base + idx_D] += final_acc[i];
                }
            }
        }
    }
}

// =============================================================================
// dV Backward Kernel — one threadgroup per K position
// =============================================================================

kernel void na_backward_dV_fp32(
    device const float* Q        [[buffer(0)]],
    device const float* K        [[buffer(1)]],
    device const float* V        [[buffer(2)]],
    device const float* O        [[buffer(3)]],
    device const float* dO       [[buffer(4)]],
    device const float* LSE      [[buffer(5)]],
    device float* dV             [[buffer(6)]],
    constant NAParams& params    [[buffer(7)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]
) {
    int idx_K = tgid.x;
    int idx_L = tgid.y;

    if (idx_K >= params.seqlen_kv) return;
    if (idx_L >= params.heads_kv * params.batch_size) return;

    int batch_idx = idx_L / params.heads_kv;
    int head_kv_idx = idx_L % params.heads_kv;
    int gqa_ratio = params.heads_q / params.heads_kv;

    int SQ = params.seqlen_q;
    int SK = params.seqlen_kv;
    int D = params.dim;
    int DV = params.dim_value;
    int H = params.heads_q;
    int HK = params.heads_kv;

    int k_base = batch_idx * SK * HK * D + idx_K * HK * D + head_kv_idx * D;

    constexpr int Q_TILE_SIZE = 2048;
    threadgroup float scores[Q_TILE_SIZE];

    constexpr int MAX_DIM = 1024;
    constexpr int NUM_THREADS = 256;
    constexpr int DIM_PER_THREAD = MAX_DIM / NUM_THREADS;
    float final_acc[DIM_PER_THREAD];

    int num_q_tiles = (SQ + Q_TILE_SIZE - 1) / Q_TILE_SIZE;

    // Iterate over all Q heads in the GQA group
    for (int gqa = 0; gqa < gqa_ratio; gqa++) {
        int head_q_idx = head_kv_idx * gqa_ratio + gqa;

        for (int i = 0; i < DIM_PER_THREAD; i++) final_acc[i] = 0.0f;

        for (int tile = 0; tile < num_q_tiles; tile++) {
            int tile_offset = tile * Q_TILE_SIZE;
            int tile_end = min(tile_offset + Q_TILE_SIZE, SQ);
            int tile_len = tile_end - tile_offset;

            for (int idx_Q = tile_offset + (int)tid; idx_Q < tile_end; idx_Q += (int)NUM_THREADS) {
                int q_base = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;

                // Q*K dot product (SIMD vectorized)
                float acc_qk = 0.0f;
                { int d4 = D / 4;
                  for (int d = 0; d < d4; d++) {
                      float4 q4 = *reinterpret_cast<device const float4*>(&Q[q_base + d * 4]);
                      float4 k4 = *reinterpret_cast<device const float4*>(&K[k_base + d * 4]);
                      acc_qk += dot(q4, k4);
                  }
                  for (int d = d4 * 4; d < D; d++) acc_qk += Q[q_base + d] * K[k_base + d];
                }
                acc_qk *= params.attn_scale;

                // Apply NA mask
                if (!compute_na_mask(idx_Q, idx_K, params)) {
                    acc_qk = -INFINITY;
                }

                float lse_val = LSE[batch_idx * SQ * H + idx_Q * H + head_q_idx];
                // NOTE: dV uses only exp(qk - lse), no delta correction
                scores[idx_Q - tile_offset] = exp(min(acc_qk - lse_val, 0.0f));
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Accumulate: dV += scores[j] * dO[j]
            for (int i = 0; i < DIM_PER_THREAD; i++) {
                int idx_D = tid + i * NUM_THREADS;
                if (idx_D < DV) {
                    for (int j = 0; j < tile_len; j++) {
                        int idx_Q = j + tile_offset;
                        int do_offset = batch_idx * SQ * H * DV + idx_Q * H * DV + head_q_idx * DV;
                        final_acc[i] += scores[j] * dO[do_offset + idx_D];
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Accumulate into dV (additive across GQA heads)
        int dv_base = batch_idx * SK * HK * DV + idx_K * HK * DV + head_kv_idx * DV;
        for (int i = 0; i < DIM_PER_THREAD; i++) {
            int idx_D = tid + i * NUM_THREADS;
            if (idx_D < DV) {
                if (gqa == 0) {
                    dV[dv_base + idx_D] = final_acc[i];
                } else {
                    dV[dv_base + idx_D] += final_acc[i];
                }
            }
        }
    }
}

// ======== FP16 Backward Kernels ========

kernel void na_backward_dQ_fp16(
    device const half* Q         [[buffer(0)]],
    device const half* K         [[buffer(1)]],
    device const half* V         [[buffer(2)]],
    device const half* O         [[buffer(3)]],
    device const half* dO        [[buffer(4)]],
    device const float* LSE      [[buffer(5)]],
    device half* dQ              [[buffer(6)]],
    constant NAParams& params    [[buffer(7)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]
) {
    int idx_Q = tgid.x;
    int idx_L = tgid.y;

    if (idx_Q >= params.seqlen_q) return;
    if (idx_L >= params.heads_q * params.batch_size) return;

    int batch_idx = idx_L / params.heads_q;
    int head_q_idx = idx_L % params.heads_q;
    int head_kv_idx = head_q_idx / (params.heads_q / params.heads_kv);

    int SQ = params.seqlen_q;
    int SK = params.seqlen_kv;
    int D = params.dim;
    int DV = params.dim_value;
    int H = params.heads_q;
    int HK = params.heads_kv;

    int q_base = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;
    int do_base = batch_idx * SQ * H * DV + idx_Q * H * DV + head_q_idx * DV;
    int o_base = do_base;
    int k_batch_offset = batch_idx * SK * HK * D;
    int k_head_offset = head_kv_idx * D;
    int v_batch_offset = batch_idx * SK * HK * DV;
    int v_head_offset = head_kv_idx * DV;

    float lse_val = LSE[batch_idx * SQ * H + idx_Q * H + head_q_idx];

    constexpr int KV_TILE_SIZE = 2048;
    threadgroup float scores[KV_TILE_SIZE];

    constexpr int MAX_DIM = 1024;
    constexpr int NUM_THREADS = 256;
    constexpr int DIM_PER_THREAD = MAX_DIM / NUM_THREADS;
    float final_acc[DIM_PER_THREAD];
    for (int i = 0; i < DIM_PER_THREAD; i++) final_acc[i] = 0.0f;

    int num_kv_tiles = (SK + KV_TILE_SIZE - 1) / KV_TILE_SIZE;

    for (int tile = 0; tile < num_kv_tiles; tile++) {
        int tile_offset = tile * KV_TILE_SIZE;
        int tile_end = min(tile_offset + KV_TILE_SIZE, SK);
        int tile_len = tile_end - tile_offset;

        for (int idx_K = tile_offset + (int)tid; idx_K < tile_end; idx_K += (int)NUM_THREADS) {
            // Q*K (SIMD vectorized half4 -> float4)
            float acc_qk = 0.0f;
            int k_offset = k_batch_offset + idx_K * HK * D + k_head_offset;
            { int d4 = D / 4;
              for (int d = 0; d < d4; d++) {
                  acc_qk += dot(float4(*reinterpret_cast<device const half4*>(&Q[q_base + d * 4])),
                                float4(*reinterpret_cast<device const half4*>(&K[k_offset + d * 4])));
              }
              for (int d = d4 * 4; d < D; d++) acc_qk += float(Q[q_base + d]) * float(K[k_offset + d]);
            }
            acc_qk *= params.attn_scale;

            // dO*V (SIMD vectorized)
            float acc_dov = 0.0f;
            int v_offset = v_batch_offset + idx_K * HK * DV + v_head_offset;
            { int d4 = DV / 4;
              for (int d = 0; d < d4; d++) {
                  acc_dov += dot(float4(*reinterpret_cast<device const half4*>(&dO[do_base + d * 4])),
                                 float4(*reinterpret_cast<device const half4*>(&V[v_offset + d * 4])));
              }
              for (int d = d4 * 4; d < DV; d++) acc_dov += float(dO[do_base + d]) * float(V[v_offset + d]);
            }
            acc_dov *= params.attn_scale;

            // dO*O (SIMD vectorized)
            float acc_doo = 0.0f;
            { int d4 = DV / 4;
              for (int d = 0; d < d4; d++) {
                  acc_doo += dot(float4(*reinterpret_cast<device const half4*>(&dO[do_base + d * 4])),
                                 float4(*reinterpret_cast<device const half4*>(&O[o_base + d * 4])));
              }
              for (int d = d4 * 4; d < DV; d++) acc_doo += float(dO[do_base + d]) * float(O[o_base + d]);
            }
            acc_doo *= params.attn_scale;

            if (!compute_na_mask(idx_Q, idx_K, params)) {
                acc_qk = -INFINITY;
            }

            scores[idx_K - tile_offset] = exp(min(acc_qk - lse_val, 0.0f)) * (acc_dov - acc_doo);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < DIM_PER_THREAD; i++) {
            int idx_D = tid + i * NUM_THREADS;
            if (idx_D < D) {
                for (int j = 0; j < tile_len; j++) {
                    int idx_K = j + tile_offset;
                    int k_offset = k_batch_offset + idx_K * HK * D + k_head_offset;
                    final_acc[i] += scores[j] * float(K[k_offset + idx_D]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    int dq_base = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;
    for (int i = 0; i < DIM_PER_THREAD; i++) {
        int idx_D = tid + i * NUM_THREADS;
        if (idx_D < D) {
            dQ[dq_base + idx_D] = half(final_acc[i]);
        }
    }
}

kernel void na_backward_dK_fp16(
    device const half* Q         [[buffer(0)]],
    device const half* K         [[buffer(1)]],
    device const half* V         [[buffer(2)]],
    device const half* O         [[buffer(3)]],
    device const half* dO        [[buffer(4)]],
    device const float* LSE      [[buffer(5)]],
    device half* dK              [[buffer(6)]],
    constant NAParams& params    [[buffer(7)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]
) {
    int idx_K = tgid.x;
    int idx_L = tgid.y;

    if (idx_K >= params.seqlen_kv) return;
    if (idx_L >= params.heads_kv * params.batch_size) return;

    int batch_idx = idx_L / params.heads_kv;
    int head_kv_idx = idx_L % params.heads_kv;
    int gqa_ratio = params.heads_q / params.heads_kv;

    int SQ = params.seqlen_q;
    int SK = params.seqlen_kv;
    int D = params.dim;
    int DV = params.dim_value;
    int H = params.heads_q;
    int HK = params.heads_kv;

    int k_base = batch_idx * SK * HK * D + idx_K * HK * D + head_kv_idx * D;
    int v_base = batch_idx * SK * HK * DV + idx_K * HK * DV + head_kv_idx * DV;

    constexpr int Q_TILE_SIZE = 2048;
    threadgroup float scores[Q_TILE_SIZE];

    constexpr int MAX_DIM = 1024;
    constexpr int NUM_THREADS = 256;
    constexpr int DIM_PER_THREAD = MAX_DIM / NUM_THREADS;
    float final_acc[DIM_PER_THREAD];

    int num_q_tiles = (SQ + Q_TILE_SIZE - 1) / Q_TILE_SIZE;

    for (int gqa = 0; gqa < gqa_ratio; gqa++) {
        int head_q_idx = head_kv_idx * gqa_ratio + gqa;

        for (int i = 0; i < DIM_PER_THREAD; i++) final_acc[i] = 0.0f;

        for (int tile = 0; tile < num_q_tiles; tile++) {
            int tile_offset = tile * Q_TILE_SIZE;
            int tile_end = min(tile_offset + Q_TILE_SIZE, SQ);
            int tile_len = tile_end - tile_offset;

            for (int idx_Q = tile_offset + (int)tid; idx_Q < tile_end; idx_Q += (int)NUM_THREADS) {
                int q_base = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;
                int do_base = batch_idx * SQ * H * DV + idx_Q * H * DV + head_q_idx * DV;
                int o_base = do_base;

                float acc_qk = 0.0f;
                { int d4 = D / 4;
                  for (int d = 0; d < d4; d++) {
                      acc_qk += dot(float4(*reinterpret_cast<device const half4*>(&Q[q_base + d * 4])),
                                    float4(*reinterpret_cast<device const half4*>(&K[k_base + d * 4])));
                  }
                  for (int d = d4 * 4; d < D; d++) acc_qk += float(Q[q_base + d]) * float(K[k_base + d]);
                }
                acc_qk *= params.attn_scale;

                float acc_dov = 0.0f;
                { int d4 = DV / 4;
                  for (int d = 0; d < d4; d++) {
                      acc_dov += dot(float4(*reinterpret_cast<device const half4*>(&dO[do_base + d * 4])),
                                     float4(*reinterpret_cast<device const half4*>(&V[v_base + d * 4])));
                  }
                  for (int d = d4 * 4; d < DV; d++) acc_dov += float(dO[do_base + d]) * float(V[v_base + d]);
                }
                acc_dov *= params.attn_scale;

                float acc_doo = 0.0f;
                { int d4 = DV / 4;
                  for (int d = 0; d < d4; d++) {
                      acc_doo += dot(float4(*reinterpret_cast<device const half4*>(&dO[do_base + d * 4])),
                                     float4(*reinterpret_cast<device const half4*>(&O[o_base + d * 4])));
                  }
                  for (int d = d4 * 4; d < DV; d++) acc_doo += float(dO[do_base + d]) * float(O[o_base + d]);
                }
                acc_doo *= params.attn_scale;

                if (!compute_na_mask(idx_Q, idx_K, params)) {
                    acc_qk = -INFINITY;
                }

                float lse_val = LSE[batch_idx * SQ * H + idx_Q * H + head_q_idx];
                scores[idx_Q - tile_offset] = exp(min(acc_qk - lse_val, 0.0f)) * (acc_dov - acc_doo);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int i = 0; i < DIM_PER_THREAD; i++) {
                int idx_D = tid + i * NUM_THREADS;
                if (idx_D < D) {
                    for (int j = 0; j < tile_len; j++) {
                        int idx_Q = j + tile_offset;
                        int q_offset = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;
                        final_acc[i] += scores[j] * float(Q[q_offset + idx_D]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (int i = 0; i < DIM_PER_THREAD; i++) {
            int idx_D = tid + i * NUM_THREADS;
            if (idx_D < D) {
                if (gqa == 0) {
                    dK[k_base + idx_D] = half(final_acc[i]);
                } else {
                    dK[k_base + idx_D] = half(float(dK[k_base + idx_D]) + final_acc[i]);
                }
            }
        }
    }
}

kernel void na_backward_dV_fp16(
    device const half* Q         [[buffer(0)]],
    device const half* K         [[buffer(1)]],
    device const half* V         [[buffer(2)]],
    device const half* O         [[buffer(3)]],
    device const half* dO        [[buffer(4)]],
    device const float* LSE      [[buffer(5)]],
    device half* dV              [[buffer(6)]],
    constant NAParams& params    [[buffer(7)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]
) {
    int idx_K = tgid.x;
    int idx_L = tgid.y;

    if (idx_K >= params.seqlen_kv) return;
    if (idx_L >= params.heads_kv * params.batch_size) return;

    int batch_idx = idx_L / params.heads_kv;
    int head_kv_idx = idx_L % params.heads_kv;
    int gqa_ratio = params.heads_q / params.heads_kv;

    int SQ = params.seqlen_q;
    int SK = params.seqlen_kv;
    int D = params.dim;
    int DV = params.dim_value;
    int H = params.heads_q;
    int HK = params.heads_kv;

    int k_base = batch_idx * SK * HK * D + idx_K * HK * D + head_kv_idx * D;

    constexpr int Q_TILE_SIZE = 2048;
    threadgroup float scores[Q_TILE_SIZE];

    constexpr int MAX_DIM = 1024;
    constexpr int NUM_THREADS = 256;
    constexpr int DIM_PER_THREAD = MAX_DIM / NUM_THREADS;
    float final_acc[DIM_PER_THREAD];

    int num_q_tiles = (SQ + Q_TILE_SIZE - 1) / Q_TILE_SIZE;

    for (int gqa = 0; gqa < gqa_ratio; gqa++) {
        int head_q_idx = head_kv_idx * gqa_ratio + gqa;

        for (int i = 0; i < DIM_PER_THREAD; i++) final_acc[i] = 0.0f;

        for (int tile = 0; tile < num_q_tiles; tile++) {
            int tile_offset = tile * Q_TILE_SIZE;
            int tile_end = min(tile_offset + Q_TILE_SIZE, SQ);
            int tile_len = tile_end - tile_offset;

            for (int idx_Q = tile_offset + (int)tid; idx_Q < tile_end; idx_Q += (int)NUM_THREADS) {
                int q_base = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;

                float acc_qk = 0.0f;
                { int d4 = D / 4;
                  for (int d = 0; d < d4; d++) {
                      acc_qk += dot(float4(*reinterpret_cast<device const half4*>(&Q[q_base + d * 4])),
                                    float4(*reinterpret_cast<device const half4*>(&K[k_base + d * 4])));
                  }
                  for (int d = d4 * 4; d < D; d++) acc_qk += float(Q[q_base + d]) * float(K[k_base + d]);
                }
                acc_qk *= params.attn_scale;

                if (!compute_na_mask(idx_Q, idx_K, params)) {
                    acc_qk = -INFINITY;
                }

                float lse_val = LSE[batch_idx * SQ * H + idx_Q * H + head_q_idx];
                scores[idx_Q - tile_offset] = exp(min(acc_qk - lse_val, 0.0f));
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int i = 0; i < DIM_PER_THREAD; i++) {
                int idx_D = tid + i * NUM_THREADS;
                if (idx_D < DV) {
                    for (int j = 0; j < tile_len; j++) {
                        int idx_Q = j + tile_offset;
                        int do_offset = batch_idx * SQ * H * DV + idx_Q * H * DV + head_q_idx * DV;
                        final_acc[i] += scores[j] * float(dO[do_offset + idx_D]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        int dv_base = batch_idx * SK * HK * DV + idx_K * HK * DV + head_kv_idx * DV;
        for (int i = 0; i < DIM_PER_THREAD; i++) {
            int idx_D = tid + i * NUM_THREADS;
            if (idx_D < DV) {
                if (gqa == 0) {
                    dV[dv_base + idx_D] = half(final_acc[i]);
                } else {
                    dV[dv_base + idx_D] = half(float(dV[dv_base + idx_D]) + final_acc[i]);
                }
            }
        }
    }
}

// ======== BF16 Backward Kernels ========

kernel void na_backward_dQ_bf16(
    device const bfloat* Q       [[buffer(0)]],
    device const bfloat* K       [[buffer(1)]],
    device const bfloat* V       [[buffer(2)]],
    device const bfloat* O       [[buffer(3)]],
    device const bfloat* dO      [[buffer(4)]],
    device const float* LSE      [[buffer(5)]],
    device bfloat* dQ            [[buffer(6)]],
    constant NAParams& params    [[buffer(7)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]
) {
    int idx_Q = tgid.x;
    int idx_L = tgid.y;

    if (idx_Q >= params.seqlen_q) return;
    if (idx_L >= params.heads_q * params.batch_size) return;

    int batch_idx = idx_L / params.heads_q;
    int head_q_idx = idx_L % params.heads_q;
    int head_kv_idx = head_q_idx / (params.heads_q / params.heads_kv);

    int SQ = params.seqlen_q;
    int SK = params.seqlen_kv;
    int D = params.dim;
    int DV = params.dim_value;
    int H = params.heads_q;
    int HK = params.heads_kv;

    int q_base = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;
    int do_base = batch_idx * SQ * H * DV + idx_Q * H * DV + head_q_idx * DV;
    int o_base = do_base;
    int k_batch_offset = batch_idx * SK * HK * D;
    int k_head_offset = head_kv_idx * D;
    int v_batch_offset = batch_idx * SK * HK * DV;
    int v_head_offset = head_kv_idx * DV;

    float lse_val = LSE[batch_idx * SQ * H + idx_Q * H + head_q_idx];

    constexpr int KV_TILE_SIZE = 2048;
    threadgroup float scores[KV_TILE_SIZE];

    constexpr int MAX_DIM = 1024;
    constexpr int NUM_THREADS = 256;
    constexpr int DIM_PER_THREAD = MAX_DIM / NUM_THREADS;
    float final_acc[DIM_PER_THREAD];
    for (int i = 0; i < DIM_PER_THREAD; i++) final_acc[i] = 0.0f;

    int num_kv_tiles = (SK + KV_TILE_SIZE - 1) / KV_TILE_SIZE;

    for (int tile = 0; tile < num_kv_tiles; tile++) {
        int tile_offset = tile * KV_TILE_SIZE;
        int tile_end = min(tile_offset + KV_TILE_SIZE, SK);
        int tile_len = tile_end - tile_offset;

        for (int idx_K = tile_offset + (int)tid; idx_K < tile_end; idx_K += (int)NUM_THREADS) {
            float acc_qk = 0.0f;
            int k_offset = k_batch_offset + idx_K * HK * D + k_head_offset;
            { int d4 = D / 4;
              for (int d = 0; d < d4; d++) {
                  int base = d * 4;
                  float4 q4 = float4(float(Q[q_base + base]), float(Q[q_base + base + 1]),
                                     float(Q[q_base + base + 2]), float(Q[q_base + base + 3]));
                  float4 k4 = float4(float(K[k_offset + base]), float(K[k_offset + base + 1]),
                                     float(K[k_offset + base + 2]), float(K[k_offset + base + 3]));
                  acc_qk += dot(q4, k4);
              }
              for (int d = d4 * 4; d < D; d++) acc_qk += float(Q[q_base + d]) * float(K[k_offset + d]);
            }
            acc_qk *= params.attn_scale;

            float acc_dov = 0.0f;
            int v_offset = v_batch_offset + idx_K * HK * DV + v_head_offset;
            { int d4 = DV / 4;
              for (int d = 0; d < d4; d++) {
                  int base = d * 4;
                  float4 do4 = float4(float(dO[do_base + base]), float(dO[do_base + base + 1]),
                                      float(dO[do_base + base + 2]), float(dO[do_base + base + 3]));
                  float4 v4 = float4(float(V[v_offset + base]), float(V[v_offset + base + 1]),
                                     float(V[v_offset + base + 2]), float(V[v_offset + base + 3]));
                  acc_dov += dot(do4, v4);
              }
              for (int d = d4 * 4; d < DV; d++) acc_dov += float(dO[do_base + d]) * float(V[v_offset + d]);
            }
            acc_dov *= params.attn_scale;

            float acc_doo = 0.0f;
            { int d4 = DV / 4;
              for (int d = 0; d < d4; d++) {
                  int base = d * 4;
                  float4 do4 = float4(float(dO[do_base + base]), float(dO[do_base + base + 1]),
                                      float(dO[do_base + base + 2]), float(dO[do_base + base + 3]));
                  float4 o4 = float4(float(O[o_base + base]), float(O[o_base + base + 1]),
                                     float(O[o_base + base + 2]), float(O[o_base + base + 3]));
                  acc_doo += dot(do4, o4);
              }
              for (int d = d4 * 4; d < DV; d++) acc_doo += float(dO[do_base + d]) * float(O[o_base + d]);
            }
            acc_doo *= params.attn_scale;

            if (!compute_na_mask(idx_Q, idx_K, params)) {
                acc_qk = -INFINITY;
            }

            scores[idx_K - tile_offset] = exp(min(acc_qk - lse_val, 0.0f)) * (acc_dov - acc_doo);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < DIM_PER_THREAD; i++) {
            int idx_D = tid + i * NUM_THREADS;
            if (idx_D < D) {
                for (int j = 0; j < tile_len; j++) {
                    int idx_K = j + tile_offset;
                    int k_offset = k_batch_offset + idx_K * HK * D + k_head_offset;
                    final_acc[i] += scores[j] * float(K[k_offset + idx_D]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    int dq_base = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;
    for (int i = 0; i < DIM_PER_THREAD; i++) {
        int idx_D = tid + i * NUM_THREADS;
        if (idx_D < D) {
            dQ[dq_base + idx_D] = bfloat(final_acc[i]);
        }
    }
}

kernel void na_backward_dK_bf16(
    device const bfloat* Q       [[buffer(0)]],
    device const bfloat* K       [[buffer(1)]],
    device const bfloat* V       [[buffer(2)]],
    device const bfloat* O       [[buffer(3)]],
    device const bfloat* dO      [[buffer(4)]],
    device const float* LSE      [[buffer(5)]],
    device bfloat* dK            [[buffer(6)]],
    constant NAParams& params    [[buffer(7)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]
) {
    int idx_K = tgid.x;
    int idx_L = tgid.y;

    if (idx_K >= params.seqlen_kv) return;
    if (idx_L >= params.heads_kv * params.batch_size) return;

    int batch_idx = idx_L / params.heads_kv;
    int head_kv_idx = idx_L % params.heads_kv;
    int gqa_ratio = params.heads_q / params.heads_kv;

    int SQ = params.seqlen_q;
    int SK = params.seqlen_kv;
    int D = params.dim;
    int DV = params.dim_value;
    int H = params.heads_q;
    int HK = params.heads_kv;

    int k_base = batch_idx * SK * HK * D + idx_K * HK * D + head_kv_idx * D;
    int v_base = batch_idx * SK * HK * DV + idx_K * HK * DV + head_kv_idx * DV;

    constexpr int Q_TILE_SIZE = 2048;
    threadgroup float scores[Q_TILE_SIZE];

    constexpr int MAX_DIM = 1024;
    constexpr int NUM_THREADS = 256;
    constexpr int DIM_PER_THREAD = MAX_DIM / NUM_THREADS;
    float final_acc[DIM_PER_THREAD];

    int num_q_tiles = (SQ + Q_TILE_SIZE - 1) / Q_TILE_SIZE;

    for (int gqa = 0; gqa < gqa_ratio; gqa++) {
        int head_q_idx = head_kv_idx * gqa_ratio + gqa;

        for (int i = 0; i < DIM_PER_THREAD; i++) final_acc[i] = 0.0f;

        for (int tile = 0; tile < num_q_tiles; tile++) {
            int tile_offset = tile * Q_TILE_SIZE;
            int tile_end = min(tile_offset + Q_TILE_SIZE, SQ);
            int tile_len = tile_end - tile_offset;

            for (int idx_Q = tile_offset + (int)tid; idx_Q < tile_end; idx_Q += (int)NUM_THREADS) {
                int q_base = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;
                int do_base = batch_idx * SQ * H * DV + idx_Q * H * DV + head_q_idx * DV;
                int o_base = do_base;

                float acc_qk = 0.0f;
                { int d4 = D / 4;
                  for (int d = 0; d < d4; d++) {
                      int base = d * 4;
                      float4 q4 = float4(float(Q[q_base + base]), float(Q[q_base + base + 1]),
                                         float(Q[q_base + base + 2]), float(Q[q_base + base + 3]));
                      float4 k4 = float4(float(K[k_base + base]), float(K[k_base + base + 1]),
                                         float(K[k_base + base + 2]), float(K[k_base + base + 3]));
                      acc_qk += dot(q4, k4);
                  }
                  for (int d = d4 * 4; d < D; d++) acc_qk += float(Q[q_base + d]) * float(K[k_base + d]);
                }
                acc_qk *= params.attn_scale;

                float acc_dov = 0.0f;
                { int d4 = DV / 4;
                  for (int d = 0; d < d4; d++) {
                      int base = d * 4;
                      float4 do4 = float4(float(dO[do_base + base]), float(dO[do_base + base + 1]),
                                          float(dO[do_base + base + 2]), float(dO[do_base + base + 3]));
                      float4 v4 = float4(float(V[v_base + base]), float(V[v_base + base + 1]),
                                         float(V[v_base + base + 2]), float(V[v_base + base + 3]));
                      acc_dov += dot(do4, v4);
                  }
                  for (int d = d4 * 4; d < DV; d++) acc_dov += float(dO[do_base + d]) * float(V[v_base + d]);
                }
                acc_dov *= params.attn_scale;

                float acc_doo = 0.0f;
                { int d4 = DV / 4;
                  for (int d = 0; d < d4; d++) {
                      int base = d * 4;
                      float4 do4 = float4(float(dO[do_base + base]), float(dO[do_base + base + 1]),
                                          float(dO[do_base + base + 2]), float(dO[do_base + base + 3]));
                      float4 o4 = float4(float(O[o_base + base]), float(O[o_base + base + 1]),
                                         float(O[o_base + base + 2]), float(O[o_base + base + 3]));
                      acc_doo += dot(do4, o4);
                  }
                  for (int d = d4 * 4; d < DV; d++) acc_doo += float(dO[do_base + d]) * float(O[o_base + d]);
                }
                acc_doo *= params.attn_scale;

                if (!compute_na_mask(idx_Q, idx_K, params)) {
                    acc_qk = -INFINITY;
                }

                float lse_val = LSE[batch_idx * SQ * H + idx_Q * H + head_q_idx];
                scores[idx_Q - tile_offset] = exp(min(acc_qk - lse_val, 0.0f)) * (acc_dov - acc_doo);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int i = 0; i < DIM_PER_THREAD; i++) {
                int idx_D = tid + i * NUM_THREADS;
                if (idx_D < D) {
                    for (int j = 0; j < tile_len; j++) {
                        int idx_Q = j + tile_offset;
                        int q_offset = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;
                        final_acc[i] += scores[j] * float(Q[q_offset + idx_D]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (int i = 0; i < DIM_PER_THREAD; i++) {
            int idx_D = tid + i * NUM_THREADS;
            if (idx_D < D) {
                if (gqa == 0) {
                    dK[k_base + idx_D] = bfloat(final_acc[i]);
                } else {
                    dK[k_base + idx_D] = bfloat(float(dK[k_base + idx_D]) + final_acc[i]);
                }
            }
        }
    }
}

kernel void na_backward_dV_bf16(
    device const bfloat* Q       [[buffer(0)]],
    device const bfloat* K       [[buffer(1)]],
    device const bfloat* V       [[buffer(2)]],
    device const bfloat* O       [[buffer(3)]],
    device const bfloat* dO      [[buffer(4)]],
    device const float* LSE      [[buffer(5)]],
    device bfloat* dV            [[buffer(6)]],
    constant NAParams& params    [[buffer(7)]],
    uint2 tgid                   [[threadgroup_position_in_grid]],
    uint tid                     [[thread_index_in_threadgroup]]
) {
    int idx_K = tgid.x;
    int idx_L = tgid.y;

    if (idx_K >= params.seqlen_kv) return;
    if (idx_L >= params.heads_kv * params.batch_size) return;

    int batch_idx = idx_L / params.heads_kv;
    int head_kv_idx = idx_L % params.heads_kv;
    int gqa_ratio = params.heads_q / params.heads_kv;

    int SQ = params.seqlen_q;
    int SK = params.seqlen_kv;
    int D = params.dim;
    int DV = params.dim_value;
    int H = params.heads_q;
    int HK = params.heads_kv;

    int k_base = batch_idx * SK * HK * D + idx_K * HK * D + head_kv_idx * D;

    constexpr int Q_TILE_SIZE = 2048;
    threadgroup float scores[Q_TILE_SIZE];

    constexpr int MAX_DIM = 1024;
    constexpr int NUM_THREADS = 256;
    constexpr int DIM_PER_THREAD = MAX_DIM / NUM_THREADS;
    float final_acc[DIM_PER_THREAD];

    int num_q_tiles = (SQ + Q_TILE_SIZE - 1) / Q_TILE_SIZE;

    for (int gqa = 0; gqa < gqa_ratio; gqa++) {
        int head_q_idx = head_kv_idx * gqa_ratio + gqa;

        for (int i = 0; i < DIM_PER_THREAD; i++) final_acc[i] = 0.0f;

        for (int tile = 0; tile < num_q_tiles; tile++) {
            int tile_offset = tile * Q_TILE_SIZE;
            int tile_end = min(tile_offset + Q_TILE_SIZE, SQ);
            int tile_len = tile_end - tile_offset;

            for (int idx_Q = tile_offset + (int)tid; idx_Q < tile_end; idx_Q += (int)NUM_THREADS) {
                int q_base = batch_idx * SQ * H * D + idx_Q * H * D + head_q_idx * D;

                float acc_qk = 0.0f;
                { int d4 = D / 4;
                  for (int d = 0; d < d4; d++) {
                      int base = d * 4;
                      float4 q4 = float4(float(Q[q_base + base]), float(Q[q_base + base + 1]),
                                         float(Q[q_base + base + 2]), float(Q[q_base + base + 3]));
                      float4 k4 = float4(float(K[k_base + base]), float(K[k_base + base + 1]),
                                         float(K[k_base + base + 2]), float(K[k_base + base + 3]));
                      acc_qk += dot(q4, k4);
                  }
                  for (int d = d4 * 4; d < D; d++) acc_qk += float(Q[q_base + d]) * float(K[k_base + d]);
                }
                acc_qk *= params.attn_scale;

                if (!compute_na_mask(idx_Q, idx_K, params)) {
                    acc_qk = -INFINITY;
                }

                float lse_val = LSE[batch_idx * SQ * H + idx_Q * H + head_q_idx];
                scores[idx_Q - tile_offset] = exp(min(acc_qk - lse_val, 0.0f));
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int i = 0; i < DIM_PER_THREAD; i++) {
                int idx_D = tid + i * NUM_THREADS;
                if (idx_D < DV) {
                    for (int j = 0; j < tile_len; j++) {
                        int idx_Q = j + tile_offset;
                        int do_offset = batch_idx * SQ * H * DV + idx_Q * H * DV + head_q_idx * DV;
                        final_acc[i] += scores[j] * float(dO[do_offset + idx_D]);
                    }
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        int dv_base = batch_idx * SK * HK * DV + idx_K * HK * DV + head_kv_idx * DV;
        for (int i = 0; i < DIM_PER_THREAD; i++) {
            int idx_D = tid + i * NUM_THREADS;
            if (idx_D < DV) {
                if (gqa == 0) {
                    dV[dv_base + idx_D] = bfloat(final_acc[i]);
                } else {
                    dV[dv_base + idx_D] = bfloat(float(dV[dv_base + idx_D]) + final_acc[i]);
                }
            }
        }
    }
}

)";
}

// =============================================================================
// Metal Tiled Backward Kernel Source (Flash-Attention style backward)
// =============================================================================

static NSString* get_metal_tiled_backward_source() {
    return @R"(
#include <metal_stdlib>
using namespace metal;

struct NAParams {
    int batch_size;
    int seqlen_q;
    int seqlen_kv;
    int heads_q;
    int heads_kv;
    int dim;
    int dim_value;
    int num_additional_kv;
    float attn_scale;
    int na_dim;
    int qkv_shape[3];
    int window_size[3];
    int stride[3];
    int dilation[3];
    int is_causal[3];
};

constant constexpr int NUM_THREADS = 256;
constant constexpr int SIMD_SIZE = 32;
constant constexpr int NUM_SIMDGROUPS = NUM_THREADS / SIMD_SIZE;  // 8

// ---- Helper functions ----

static inline int qkv_stride_fn(int na_dim, constant int* shape, int d) {
    int s = 1;
    for (int i = d + 1; i < na_dim; i++) s *= shape[i];
    return s;
}

static inline void idx_to_coord(int idx, int na_dim, constant int* shape, thread int* coord) {
    for (int d = 0; d < na_dim; d++) {
        int s = qkv_stride_fn(na_dim, shape, d);
        coord[d] = idx / s;
        idx = idx % s;
    }
}

static inline int qkv_fix_dilation(int qkv_shape, int dilation, int dilation_group) {
    int padding = 1 - ((dilation_group + (dilation - (qkv_shape % dilation))) / dilation);
    return (qkv_shape / dilation) + padding;
}

static inline int get_win_start_nc(int index, int window_left, int window_right, int stride_val, int length) {
    int stride_group_leader_idx = min((index / stride_val) * stride_val + (stride_val / 2), length - 1);
    return max(stride_group_leader_idx - window_left, 0) +
        ((stride_group_leader_idx + window_right >= length) *
         (length - window_right - stride_group_leader_idx - 1));
}

static inline int get_win_start_causal(int index, int window_left, int window_right, int stride_val, int length) {
    int stride_group_leader_idx = min((index / stride_val) * stride_val + stride_val - 1, length - 1);
    return max(stride_group_leader_idx - window_left - window_right, 0);
}

static inline int get_win_end_nc(int start, int window_size_val) {
    return start + window_size_val;
}

static inline int get_win_end_causal(int index, int length) {
    return min(index + 1, length);
}

static inline bool compute_na_mask(
    int q_idx, int k_idx, int na_dim,
    constant int* qkv_shape, constant int* window_size,
    constant int* stride_arr, constant int* dilation_arr, constant int* is_causal
) {
    int q_cg[3] = {0,0,0}, k_cg[3] = {0,0,0};
    { int rem = q_idx; for (int d = 0; d < na_dim; d++) { int s = 1; for (int i = d+1; i < na_dim; i++) s *= qkv_shape[i]; q_cg[d] = rem / s; rem = rem % s; } }
    { int rem = k_idx; for (int d = 0; d < na_dim; d++) { int s = 1; for (int i = d+1; i < na_dim; i++) s *= qkv_shape[i]; k_cg[d] = rem / s; rem = rem % s; } }
    for (int d = 0; d < na_dim; d++) {
        int q_di = q_cg[d] % dilation_arr[d], k_di = k_cg[d] % dilation_arr[d];
        if (q_di != k_di) return false;
        int q_c = q_cg[d] / dilation_arr[d], k_c = k_cg[d] / dilation_arr[d];
        int eff = qkv_fix_dilation(qkv_shape[d], dilation_arr[d], q_di);
        int wl = window_size[d] / 2, wr = (window_size[d] / 2) + ((window_size[d] % 2) - 1);
        int ws, we;
        if (is_causal[d]) { ws = get_win_start_causal(q_c, wl, wr, stride_arr[d], eff); we = get_win_end_causal(q_c, eff); }
        else { ws = get_win_start_nc(q_c, wl, wr, stride_arr[d], eff); we = get_win_end_nc(ws, window_size[d]); }
        if (k_c < ws || k_c >= we) return false;
    }
    return true;
}

// Precompute per-dim window bounds for a Q position.
// Stores q_di_group[d], win_start_global[d], win_end_global[d] (in global coords)
// so that K neighbor check becomes a simple coord range test.
static inline void precompute_q_window(
    int q_idx, int na_dim,
    constant int* qkv_shape, constant int* window_size,
    constant int* stride_arr, constant int* dilation_arr, constant int* is_causal,
    thread int* q_di_out, thread int* win_start_global, thread int* win_end_global
) {
    int q_cg[3] = {0,0,0};
    { int rem = q_idx; for (int d = 0; d < na_dim; d++) { int s = 1; for (int i = d+1; i < na_dim; i++) s *= qkv_shape[i]; q_cg[d] = rem / s; rem = rem % s; } }
    for (int d = 0; d < na_dim; d++) {
        int q_di = q_cg[d] % dilation_arr[d];
        int q_c = q_cg[d] / dilation_arr[d];
        int eff = qkv_fix_dilation(qkv_shape[d], dilation_arr[d], q_di);
        int wl = window_size[d] / 2, wr = (window_size[d] / 2) + ((window_size[d] % 2) - 1);
        int ws, we;
        if (is_causal[d]) { ws = get_win_start_causal(q_c, wl, wr, stride_arr[d], eff); we = get_win_end_causal(q_c, eff); }
        else { ws = get_win_start_nc(q_c, wl, wr, stride_arr[d], eff); we = get_win_end_nc(ws, window_size[d]); }
        q_di_out[d] = q_di;
        win_start_global[d] = ws * dilation_arr[d] + q_di;
        win_end_global[d] = min((we - 1) * dilation_arr[d] + q_di + 1, qkv_shape[d]);
    }
}

// Fast neighbor check using precomputed Q window bounds.
// Only needs idx_to_coord for K (no window computation).
static inline bool check_na_window(
    int k_idx, int na_dim,
    constant int* qkv_shape, constant int* dilation_arr,
    thread int* q_di, thread int* win_start_global, thread int* win_end_global
) {
    int k_cg[3] = {0,0,0};
    { int rem = k_idx; for (int d = 0; d < na_dim; d++) { int s = 1; for (int i = d+1; i < na_dim; i++) s *= qkv_shape[i]; k_cg[d] = rem / s; rem = rem % s; } }
    for (int d = 0; d < na_dim; d++) {
        if (k_cg[d] % dilation_arr[d] != q_di[d]) return false;
        if (k_cg[d] < win_start_global[d] || k_cg[d] >= win_end_global[d]) return false;
    }
    return true;
}

// Fast neighbor check: precomputed K coords + precomputed Q window bounds from shared mem.
// No coordinate decomposition at all — just dilation modulus + range checks.
static inline bool check_na_window_kq_precomputed(
    int na_dim, constant int* dilation_arr,
    thread int* k_cg, threadgroup int* q_di, threadgroup int* win_start_global, threadgroup int* win_end_global
) {
    for (int d = 0; d < na_dim; d++) {
        if (k_cg[d] % dilation_arr[d] != q_di[d]) return false;
        if (k_cg[d] < win_start_global[d] || k_cg[d] >= win_end_global[d]) return false;
    }
    return true;
}

// Compute KV bounds for a Q position (forward direction)
static inline void compute_kv_bounds(
    int q_idx, int na_dim,
    constant int* qkv_shape, constant int* window_size,
    constant int* stride_arr, constant int* dilation_arr, constant int* is_causal,
    thread int& min_kv, thread int& max_kv
) {
    int q_cg[3] = {0,0,0};
    { int rem = q_idx; for (int d = 0; d < na_dim; d++) { int s = 1; for (int i = d+1; i < na_dim; i++) s *= qkv_shape[i]; q_cg[d] = rem / s; rem = rem % s; } }
    int kv_sc[3] = {0,0,0}, kv_ec[3] = {0,0,0};
    for (int d = 0; d < na_dim; d++) {
        int q_di = q_cg[d] % dilation_arr[d], q_c = q_cg[d] / dilation_arr[d];
        int eff = qkv_fix_dilation(qkv_shape[d], dilation_arr[d], q_di);
        int wl = window_size[d] / 2, wr = (window_size[d] / 2) + ((window_size[d] % 2) - 1);
        int ws, we;
        if (is_causal[d]) { ws = get_win_start_causal(q_c, wl, wr, stride_arr[d], eff); we = get_win_end_causal(q_c, eff); }
        else { ws = get_win_start_nc(q_c, wl, wr, stride_arr[d], eff); we = get_win_end_nc(ws, window_size[d]); }
        kv_sc[d] = ws * dilation_arr[d] + q_di;
        kv_ec[d] = min((we - 1) * dilation_arr[d] + q_di + 1, qkv_shape[d]);
    }
    min_kv = 0; max_kv = 0;
    for (int d = 0; d < na_dim; d++) {
        int s = 1; for (int i = d+1; i < na_dim; i++) s *= qkv_shape[i];
        min_kv += kv_sc[d] * s;
        max_kv += (kv_ec[d] - 1) * s;
    }
    max_kv += 1;
}

// Compute Q bounds for a K position (reverse direction for dKdV)
// Conservative: returns the range of Q positions whose window could include k_idx
static inline void compute_q_bounds_for_k(
    int k_idx, int na_dim,
    constant int* qkv_shape, constant int* window_size,
    constant int* stride_arr, constant int* dilation_arr, constant int* is_causal,
    int seqlen_q,
    thread int& min_q, thread int& max_q
) {
    int k_cg[3] = {0,0,0};
    { int rem = k_idx; for (int d = 0; d < na_dim; d++) { int s = 1; for (int i = d+1; i < na_dim; i++) s *= qkv_shape[i]; k_cg[d] = rem / s; rem = rem % s; } }
    int q_sc[3] = {0,0,0}, q_ec[3] = {0,0,0};
    for (int d = 0; d < na_dim; d++) {
        int k_di = k_cg[d] % dilation_arr[d];
        int k_c = k_cg[d] / dilation_arr[d];
        int wl = window_size[d] / 2;
        // Conservative Q range in dilated coords
        if (is_causal[d]) {
            // Causal: Q at position q can see K at k if k <= q, within window
            // So Q range is [k_c, k_c + wl + wr] clamped
            int wr = (window_size[d] / 2) + ((window_size[d] % 2) - 1);
            int q_min_c = k_c;
            int eff = qkv_fix_dilation(qkv_shape[d], dilation_arr[d], k_di);
            int q_max_c = min(k_c + wl + wr, eff - 1);
            q_sc[d] = q_min_c * dilation_arr[d] + k_di;
            q_ec[d] = min(q_max_c * dilation_arr[d] + k_di + 1, qkv_shape[d]);
        } else {
            // Non-causal: NA boundary correction shifts windows, so Q positions
            // near edges can see K positions further than wl away.
            // Use full (window_size - 1) to be conservative.
            int eff = qkv_fix_dilation(qkv_shape[d], dilation_arr[d], k_di);
            int q_min_c = max(0, k_c - (window_size[d] - 1));
            int q_max_c = min(eff - 1, k_c + (window_size[d] - 1));
            q_sc[d] = q_min_c * dilation_arr[d] + k_di;
            q_ec[d] = min(q_max_c * dilation_arr[d] + k_di + 1, qkv_shape[d]);
        }
    }
    min_q = 0; max_q = 0;
    for (int d = 0; d < na_dim; d++) {
        int s = 1; for (int i = d+1; i < na_dim; i++) s *= qkv_shape[i];
        min_q += q_sc[d] * s;
        max_q += (q_ec[d] - 1) * s;
    }
    max_q += 1;
    min_q = max(0, min_q);
    max_q = min(seqlen_q, max_q);
}

// =============================================================================
// Tiled dQ — Q rows fixed in registers, iterate KV tiles in smem
// =============================================================================

#define TILED_DQ_KERNEL(FNAME, DTYPE, BR, MAX_D) \
kernel void FNAME( \
    device const DTYPE* Q       [[buffer(0)]], \
    device const DTYPE* K       [[buffer(1)]], \
    device const DTYPE* V       [[buffer(2)]], \
    device const DTYPE* dO      [[buffer(3)]], \
    device const float* LSE     [[buffer(4)]], \
    device const DTYPE* O       [[buffer(5)]], \
    device DTYPE* dQ            [[buffer(6)]], \
    constant NAParams& params   [[buffer(7)]], \
    uint2 tgid                  [[threadgroup_position_in_grid]], \
    uint tid                    [[thread_index_in_threadgroup]], \
    uint simd_lane              [[thread_index_in_simdgroup]], \
    uint simdgroup_id           [[simdgroup_index_in_threadgroup]] \
) { \
    constexpr int Br = BR; \
    constexpr int Bc = 32; \
    constexpr int ROWS_PER_SIMD = Br / NUM_SIMDGROUPS; \
    \
    int tile_q_start = (int)tgid.x * Br; \
    int idx_L = (int)tgid.y; \
    if (idx_L >= params.heads_q * params.batch_size) return; \
    \
    int batch_idx = idx_L / params.heads_q; \
    int head_q_idx = idx_L % params.heads_q; \
    int head_kv_idx = head_q_idx / (params.heads_q / params.heads_kv); \
    int SQ = params.seqlen_q, SK = params.seqlen_kv; \
    int D = params.dim, DV = params.dim_value; \
    int H = params.heads_q, HK = params.heads_kv, na_dim = params.na_dim; \
    \
    threadgroup float smem_K[Bc * MAX_D]; \
    threadgroup float smem_V[Bc * MAX_D]; \
    \
    float q_reg[ROWS_PER_SIMD][MAX_D]; \
    float do_reg[ROWS_PER_SIMD][MAX_D]; \
    float dq_acc[ROWS_PER_SIMD][MAX_D]; \
    float row_lse[ROWS_PER_SIMD]; \
    float row_di[ROWS_PER_SIMD]; \
    int q_indices[ROWS_PER_SIMD]; \
    int row_q_di[ROWS_PER_SIMD][3]; \
    int row_win_start[ROWS_PER_SIMD][3]; \
    int row_win_end[ROWS_PER_SIMD][3]; \
    \
    for (int r = 0; r < ROWS_PER_SIMD; r++) { \
        int q_row = tile_q_start + (int)simdgroup_id * ROWS_PER_SIMD + r; \
        q_indices[r] = q_row; \
        for (int dd = 0; dd < MAX_D; dd++) { q_reg[r][dd] = 0.0f; do_reg[r][dd] = 0.0f; dq_acc[r][dd] = 0.0f; } \
        row_lse[r] = 0.0f; row_di[r] = 0.0f; \
        for (int dd = 0; dd < 3; dd++) { row_q_di[r][dd] = 0; row_win_start[r][dd] = 0; row_win_end[r][dd] = 0; } \
        if (q_row < SQ) { \
            int q_base = batch_idx * SQ * H * D + q_row * H * D + head_q_idx * D; \
            int do_base = batch_idx * SQ * H * DV + q_row * H * DV + head_q_idx * DV; \
            for (int dd = 0; dd < D; dd++) q_reg[r][dd] = float(Q[q_base + dd]); \
            for (int dd = 0; dd < DV; dd++) do_reg[r][dd] = float(dO[do_base + dd]); \
            int lse_idx = batch_idx * SQ * H + q_row * H + head_q_idx; \
            row_lse[r] = LSE[lse_idx]; \
            /* Compute Di = dot(dO, O) * attn_scale inline */ \
            { float di_acc = 0.0f; \
              int o_base = batch_idx * SQ * H * DV + q_row * H * DV + head_q_idx * DV; \
              for (int dd = 0; dd < DV; dd++) di_acc += do_reg[r][dd] * float(O[o_base + dd]); \
              row_di[r] = di_acc * params.attn_scale; } \
            precompute_q_window(q_row, na_dim, params.qkv_shape, params.window_size, \
                                params.stride, params.dilation, params.is_causal, \
                                row_q_di[r], row_win_start[r], row_win_end[r]); \
        } \
    } \
    \
    /* KV Bounding Box — as_type bit-cast through smem_K (no extra threadgroup memory) */ \
    int first_tile, last_tile; \
    { \
        int local_min_kv = SK, local_max_kv = 0; \
        for (int r = 0; r < ROWS_PER_SIMD; r++) { \
            if (q_indices[r] < SQ) { \
                int qmin, qmax; \
                compute_kv_bounds(q_indices[r], na_dim, params.qkv_shape, params.window_size, \
                                  params.stride, params.dilation, params.is_causal, qmin, qmax); \
                local_min_kv = min(local_min_kv, qmin); local_max_kv = max(local_max_kv, qmax); \
            } \
        } \
        int sg_mn = simd_min(local_min_kv), sg_mx = simd_max(local_max_kv); \
        if (simd_lane == 0) { \
            smem_K[simdgroup_id * 2 + 0] = as_type<float>(sg_mn); \
            smem_K[simdgroup_id * 2 + 1] = as_type<float>(sg_mx); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        int bbox_min_kv, bbox_max_kv; \
        if (tid == 0) { \
            bbox_min_kv = as_type<int>(smem_K[0]); \
            bbox_max_kv = as_type<int>(smem_K[1]); \
            for (int s = 1; s < NUM_SIMDGROUPS; s++) { \
                bbox_min_kv = min(bbox_min_kv, as_type<int>(smem_K[s * 2])); \
                bbox_max_kv = max(bbox_max_kv, as_type<int>(smem_K[s * 2 + 1])); \
            } \
            smem_K[0] = as_type<float>(bbox_min_kv); \
            smem_K[1] = as_type<float>(bbox_max_kv); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        bbox_min_kv = as_type<int>(smem_K[0]); \
        bbox_max_kv = as_type<int>(smem_K[1]); \
        first_tile = bbox_min_kv / Bc; \
        last_tile = min((bbox_max_kv + Bc - 1) / Bc, (SK + Bc - 1) / Bc); \
    } \
    \
    /* Main KV Tile Loop */ \
    for (int tile_idx = first_tile; tile_idx < last_tile; tile_idx++) { \
        int kv_start = tile_idx * Bc; \
        int kv_end = min(kv_start + Bc, SK); \
        int tile_len = kv_end - kv_start; \
        \
        /* Cooperative load K tile */ \
        int total_k = tile_len * D; \
        for (int i = (int)tid; i < total_k; i += NUM_THREADS) { \
            int kk = i / D, dd = i % D; \
            smem_K[kk*D+dd] = float(K[batch_idx*SK*HK*D + (kv_start+kk)*HK*D + head_kv_idx*D + dd]); \
        } \
        /* Cooperative load V tile */ \
        int total_v = tile_len * DV; \
        for (int i = (int)tid; i < total_v; i += NUM_THREADS) { \
            int kk = i / DV, dd = i % DV; \
            smem_V[kk*DV+dd] = float(V[batch_idx*SK*HK*DV + (kv_start+kk)*HK*DV + head_kv_idx*DV + dd]); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        \
        for (int r = 0; r < ROWS_PER_SIMD; r++) { \
            int q_row = q_indices[r]; \
            if (q_row >= SQ) continue; \
            \
            int k_pos = (int)simd_lane; \
            float score = -INFINITY; \
            float dov_val = 0.0f; \
            if (k_pos < tile_len) { \
                int global_k = kv_start + k_pos; \
                /* Q*K dot product */ \
                float acc = 0.0f; \
                for (int dd = 0; dd < D; dd += 4) { \
                    int rem = min(4, D - dd); \
                    if (rem == 4) { \
                        float4 q4 = float4(q_reg[r][dd], q_reg[r][dd+1], q_reg[r][dd+2], q_reg[r][dd+3]); \
                        float4 k4 = *reinterpret_cast<threadgroup const float4*>(&smem_K[k_pos*D+dd]); \
                        acc += dot(q4, k4); \
                    } else { for (int ddd = 0; ddd < rem; ddd++) acc += q_reg[r][dd+ddd] * smem_K[k_pos*D+dd+ddd]; } \
                } \
                acc *= params.attn_scale; \
                /* NA mask (uses precomputed Q window bounds) */ \
                bool is_nb = check_na_window(global_k, na_dim, params.qkv_shape, params.dilation, row_q_di[r], row_win_start[r], row_win_end[r]); \
                score = is_nb ? acc : -INFINITY; \
                /* dO*V dot product */ \
                for (int dd = 0; dd < DV; dd += 4) { \
                    int rem = min(4, DV - dd); \
                    if (rem == 4) { \
                        float4 do4 = float4(do_reg[r][dd], do_reg[r][dd+1], do_reg[r][dd+2], do_reg[r][dd+3]); \
                        float4 v4 = *reinterpret_cast<threadgroup const float4*>(&smem_V[k_pos*DV+dd]); \
                        dov_val += dot(do4, v4); \
                    } else { for (int ddd = 0; ddd < rem; ddd++) dov_val += do_reg[r][dd+ddd] * smem_V[k_pos*DV+dd+ddd]; } \
                } \
                dov_val *= params.attn_scale; \
            } \
            \
            /* P = exp(score - LSE), dS = P * (dO*V*scale - Di) */ \
            float P = (score != -INFINITY) ? exp(min(score - row_lse[r], 0.0f)) : 0.0f; \
            float dS = P * (dov_val - row_di[r]); \
            \
            /* dQ accumulation: dQ[d] += simd_sum(dS * K[lane][d]) */ \
            if (k_pos < tile_len) { \
                for (int dd = 0; dd < D; dd += 4) { \
                    int rem = min(4, D - dd); \
                    if (rem == 4) { \
                        float4 k4 = *reinterpret_cast<threadgroup const float4*>(&smem_K[k_pos*D+dd]); \
                        dq_acc[r][dd]   += simd_sum(dS * k4.x); \
                        dq_acc[r][dd+1] += simd_sum(dS * k4.y); \
                        dq_acc[r][dd+2] += simd_sum(dS * k4.z); \
                        dq_acc[r][dd+3] += simd_sum(dS * k4.w); \
                    } else { for (int ddd = 0; ddd < rem; ddd++) dq_acc[r][dd+ddd] += simd_sum(dS * smem_K[k_pos*D+dd+ddd]); } \
                } \
            } else { \
                for (int dd = 0; dd < D; dd += 4) { int rem = min(4, D - dd); if (rem == 4) { simd_sum(0.0f); simd_sum(0.0f); simd_sum(0.0f); simd_sum(0.0f); } else { for (int ddd = 0; ddd < rem; ddd++) simd_sum(0.0f); } } \
            } \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } \
    \
    /* Additional KV tile loop */ \
    if (params.num_additional_kv > 0) { \
        int add_first = SQ / Bc; \
        int add_last = (SQ + params.num_additional_kv + Bc - 1) / Bc; \
        add_last = min(add_last, (SK + Bc - 1) / Bc); \
        for (int tile_idx = add_first; tile_idx < add_last; tile_idx++) { \
            if (tile_idx >= first_tile && tile_idx < last_tile) continue; \
            int kv_start = tile_idx * Bc; \
            int kv_end = min(kv_start + Bc, SK); \
            int tile_len = kv_end - kv_start; \
            int total_k = tile_len * D; \
            for (int i = (int)tid; i < total_k; i += NUM_THREADS) { \
                int kk = i / D, dd = i % D; \
                smem_K[kk*D+dd] = float(K[batch_idx*SK*HK*D + (kv_start+kk)*HK*D + head_kv_idx*D + dd]); \
            } \
            int total_v = tile_len * DV; \
            for (int i = (int)tid; i < total_v; i += NUM_THREADS) { \
                int kk = i / DV, dd = i % DV; \
                smem_V[kk*DV+dd] = float(V[batch_idx*SK*HK*DV + (kv_start+kk)*HK*DV + head_kv_idx*DV + dd]); \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            for (int r = 0; r < ROWS_PER_SIMD; r++) { \
                int q_row = q_indices[r]; \
                if (q_row >= SQ) continue; \
                int k_pos = (int)simd_lane; \
                float score = -INFINITY; \
                float dov_val = 0.0f; \
                if (k_pos < tile_len) { \
                    int global_k = kv_start + k_pos; \
                    if (global_k >= SQ && global_k - SQ < params.num_additional_kv) { \
                        float acc = 0.0f; \
                        for (int dd = 0; dd < D; dd += 4) { \
                            int rem = min(4, D - dd); \
                            if (rem == 4) { \
                                float4 q4 = float4(q_reg[r][dd], q_reg[r][dd+1], q_reg[r][dd+2], q_reg[r][dd+3]); \
                                float4 k4 = *reinterpret_cast<threadgroup const float4*>(&smem_K[k_pos*D+dd]); \
                                acc += dot(q4, k4); \
                            } else { for (int ddd = 0; ddd < rem; ddd++) acc += q_reg[r][dd+ddd] * smem_K[k_pos*D+dd+ddd]; } \
                        } \
                        score = acc * params.attn_scale; \
                        for (int dd = 0; dd < DV; dd += 4) { \
                            int rem = min(4, DV - dd); \
                            if (rem == 4) { \
                                float4 do4 = float4(do_reg[r][dd], do_reg[r][dd+1], do_reg[r][dd+2], do_reg[r][dd+3]); \
                                float4 v4 = *reinterpret_cast<threadgroup const float4*>(&smem_V[k_pos*DV+dd]); \
                                dov_val += dot(do4, v4); \
                            } else { for (int ddd = 0; ddd < rem; ddd++) dov_val += do_reg[r][dd+ddd] * smem_V[k_pos*DV+dd+ddd]; } \
                        } \
                        dov_val *= params.attn_scale; \
                    } \
                } \
                float P = (score != -INFINITY) ? exp(min(score - row_lse[r], 0.0f)) : 0.0f; \
                float dS = P * (dov_val - row_di[r]); \
                if (k_pos < tile_len) { \
                    for (int dd = 0; dd < D; dd += 4) { \
                        int rem = min(4, D - dd); \
                        if (rem == 4) { \
                            float4 k4 = *reinterpret_cast<threadgroup const float4*>(&smem_K[k_pos*D+dd]); \
                            dq_acc[r][dd]   += simd_sum(dS * k4.x); \
                            dq_acc[r][dd+1] += simd_sum(dS * k4.y); \
                            dq_acc[r][dd+2] += simd_sum(dS * k4.z); \
                            dq_acc[r][dd+3] += simd_sum(dS * k4.w); \
                        } else { for (int ddd = 0; ddd < rem; ddd++) dq_acc[r][dd+ddd] += simd_sum(dS * smem_K[k_pos*D+dd+ddd]); } \
                    } \
                } else { \
                    for (int dd = 0; dd < D; dd += 4) { int rem = min(4, D - dd); if (rem == 4) { simd_sum(0.0f); simd_sum(0.0f); simd_sum(0.0f); simd_sum(0.0f); } else { for (int ddd = 0; ddd < rem; ddd++) simd_sum(0.0f); } } \
                } \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
    } \
    \
    /* Write dQ */ \
    if (simd_lane == 0) { \
        for (int r = 0; r < ROWS_PER_SIMD; r++) { \
            int q_row = q_indices[r]; \
            if (q_row >= SQ) continue; \
            int dq_base = batch_idx * SQ * H * D + q_row * H * D + head_q_idx * D; \
            for (int dd = 0; dd < D; dd++) dQ[dq_base + dd] = DTYPE(dq_acc[r][dd]); \
        } \
    } \
}

TILED_DQ_KERNEL(na_backward_tiled_dQ_fp32_br32_d32, float, 32, 32)
TILED_DQ_KERNEL(na_backward_tiled_dQ_fp32_br32, float, 32, 64)
TILED_DQ_KERNEL(na_backward_tiled_dQ_fp32_br16, float, 16, 128)
TILED_DQ_KERNEL(na_backward_tiled_dQ_fp16_br32_d32, half, 32, 32)
TILED_DQ_KERNEL(na_backward_tiled_dQ_fp16_br32, half, 32, 64)
TILED_DQ_KERNEL(na_backward_tiled_dQ_fp16_br16, half, 16, 128)
TILED_DQ_KERNEL(na_backward_tiled_dQ_bf16_br32_d32, bfloat, 32, 32)
TILED_DQ_KERNEL(na_backward_tiled_dQ_bf16_br32, bfloat, 32, 64)
TILED_DQ_KERNEL(na_backward_tiled_dQ_bf16_br16, bfloat, 16, 128)

// =============================================================================
// Kernel 3: Fused Tiled dKdV — K/V rows fixed in registers, iterate Q tiles
// =============================================================================

#define TILED_DKDV_KERNEL(FNAME, DTYPE, BR, BC, MAX_D) \
kernel void FNAME( \
    device const DTYPE* Q       [[buffer(0)]], \
    device const DTYPE* K       [[buffer(1)]], \
    device const DTYPE* V       [[buffer(2)]], \
    device const DTYPE* dO      [[buffer(3)]], \
    device const float* LSE     [[buffer(4)]], \
    device const DTYPE* O       [[buffer(5)]], \
    device DTYPE* dK            [[buffer(6)]], \
    device DTYPE* dV            [[buffer(7)]], \
    constant NAParams& params   [[buffer(8)]], \
    uint2 tgid                  [[threadgroup_position_in_grid]], \
    uint tid                    [[thread_index_in_threadgroup]], \
    uint simd_lane              [[thread_index_in_simdgroup]], \
    uint simdgroup_id           [[simdgroup_index_in_threadgroup]] \
) { \
    constexpr int Br = BR; \
    constexpr int Bc = BC; \
    constexpr int ROWS_PER_SIMD = Br / NUM_SIMDGROUPS; \
    \
    int tile_kv_start = (int)tgid.x * Br; \
    int idx_L = (int)tgid.y; \
    if (idx_L >= params.heads_kv * params.batch_size) return; \
    \
    int batch_idx = idx_L / params.heads_kv; \
    int head_kv_idx = idx_L % params.heads_kv; \
    int gqa_ratio = params.heads_q / params.heads_kv; \
    int SQ = params.seqlen_q, SK = params.seqlen_kv; \
    int D = params.dim, DV = params.dim_value; \
    int H = params.heads_q, HK = params.heads_kv, na_dim = params.na_dim; \
    \
    /* Shared memory for Q tile + dO tile + LSE/Di + precomputed Q window bounds */ \
    threadgroup float smem_Q[Bc * MAX_D]; \
    threadgroup float smem_dO[Bc * MAX_D]; \
    threadgroup float smem_LSE[Bc]; \
    threadgroup float smem_Di[Bc]; \
    threadgroup int smem_q_di[Bc * 3]; \
    threadgroup int smem_win_start[Bc * 3]; \
    threadgroup int smem_win_end[Bc * 3]; \
    \
    /* Load K and V rows into registers */ \
    float k_reg[ROWS_PER_SIMD][MAX_D]; \
    float v_reg[ROWS_PER_SIMD][MAX_D]; \
    float dk_acc[ROWS_PER_SIMD][MAX_D]; \
    float dv_acc[ROWS_PER_SIMD][MAX_D]; \
    int kv_indices[ROWS_PER_SIMD]; \
    int kv_coords[ROWS_PER_SIMD][3]; \
    bool kv_is_additional[ROWS_PER_SIMD]; \
    \
    for (int r = 0; r < ROWS_PER_SIMD; r++) { \
        int kv_row = tile_kv_start + (int)simdgroup_id * ROWS_PER_SIMD + r; \
        kv_indices[r] = kv_row; \
        kv_coords[r][0] = 0; kv_coords[r][1] = 0; kv_coords[r][2] = 0; \
        kv_is_additional[r] = (kv_row >= SQ && kv_row < SK); \
        for (int dd = 0; dd < MAX_D; dd++) { k_reg[r][dd] = 0.0f; v_reg[r][dd] = 0.0f; dk_acc[r][dd] = 0.0f; dv_acc[r][dd] = 0.0f; } \
        if (kv_row < SK) { \
            int k_base = batch_idx * SK * HK * D + kv_row * HK * D + head_kv_idx * D; \
            int v_base = batch_idx * SK * HK * DV + kv_row * HK * DV + head_kv_idx * DV; \
            for (int dd = 0; dd < D; dd++) k_reg[r][dd] = float(K[k_base + dd]); \
            for (int dd = 0; dd < DV; dd++) v_reg[r][dd] = float(V[v_base + dd]); \
            /* Precompute K coords for fast NA mask check */ \
            if (!kv_is_additional[r]) { \
                int rem = kv_row; \
                for (int d = 0; d < na_dim; d++) { int s = 1; for (int i = d+1; i < na_dim; i++) s *= params.qkv_shape[i]; kv_coords[r][d] = rem / s; rem = rem % s; } \
            } \
        } \
    } \
    \
    /* Reverse Q bounding box — as_type bit-cast through smem_Q (no extra threadgroup memory) */ \
    int first_tile, last_tile; \
    { \
        int local_min_q = SQ, local_max_q = 0; \
        for (int r = 0; r < ROWS_PER_SIMD; r++) { \
            if (kv_indices[r] < SK) { \
                int kv_row = kv_indices[r]; \
                if (kv_row >= SQ && kv_row - SQ < params.num_additional_kv) { \
                    local_min_q = 0; local_max_q = SQ; \
                } else if (kv_row < SQ) { \
                    int qmin, qmax; \
                    compute_q_bounds_for_k(kv_row, na_dim, params.qkv_shape, params.window_size, \
                                           params.stride, params.dilation, params.is_causal, SQ, qmin, qmax); \
                    local_min_q = min(local_min_q, qmin); local_max_q = max(local_max_q, qmax); \
                } \
            } \
        } \
        int sg_mn = simd_min(local_min_q), sg_mx = simd_max(local_max_q); \
        if (simd_lane == 0) { \
            smem_Q[simdgroup_id * 2 + 0] = as_type<float>(sg_mn); \
            smem_Q[simdgroup_id * 2 + 1] = as_type<float>(sg_mx); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        int bbox_min_q, bbox_max_q; \
        if (tid == 0) { \
            bbox_min_q = as_type<int>(smem_Q[0]); \
            bbox_max_q = as_type<int>(smem_Q[1]); \
            for (int s = 1; s < NUM_SIMDGROUPS; s++) { \
                bbox_min_q = min(bbox_min_q, as_type<int>(smem_Q[s * 2])); \
                bbox_max_q = max(bbox_max_q, as_type<int>(smem_Q[s * 2 + 1])); \
            } \
            smem_Q[0] = as_type<float>(bbox_min_q); \
            smem_Q[1] = as_type<float>(bbox_max_q); \
        } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
        bbox_min_q = as_type<int>(smem_Q[0]); \
        bbox_max_q = as_type<int>(smem_Q[1]); \
        first_tile = bbox_min_q / Bc; \
        last_tile = min((bbox_max_q + Bc - 1) / Bc, (SQ + Bc - 1) / Bc); \
    } \
    \
    /* GQA loop over Q heads mapping to this KV head */ \
    for (int gqa = 0; gqa < gqa_ratio; gqa++) { \
        int head_q_idx = head_kv_idx * gqa_ratio + gqa; \
        \
        /* Q Tile Loop */ \
        for (int tile_idx = first_tile; tile_idx < last_tile; tile_idx++) { \
            int q_start = tile_idx * Bc; \
            int q_end = min(q_start + Bc, SQ); \
            int tile_len = q_end - q_start; \
            \
            /* Cooperative load Q + dO + LSE + Di */ \
            int total_q = tile_len * D; \
            for (int i = (int)tid; i < total_q; i += NUM_THREADS) { \
                int qq = i / D, dd = i % D; \
                smem_Q[qq*D+dd] = float(Q[batch_idx*SQ*H*D + (q_start+qq)*H*D + head_q_idx*D + dd]); \
            } \
            int total_do = tile_len * DV; \
            for (int i = (int)tid; i < total_do; i += NUM_THREADS) { \
                int qq = i / DV, dd = i % DV; \
                smem_dO[qq*DV+dd] = float(dO[batch_idx*SQ*H*DV + (q_start+qq)*H*DV + head_q_idx*DV + dd]); \
            } \
            for (int i = (int)tid; i < tile_len; i += NUM_THREADS) { \
                int lse_idx = batch_idx * SQ * H + (q_start+i) * H + head_q_idx; \
                smem_LSE[i] = LSE[lse_idx]; \
                /* Compute Di = dot(dO, O) * attn_scale inline */ \
                int o_base = batch_idx * SQ * H * DV + (q_start+i) * H * DV + head_q_idx * DV; \
                float di_acc = 0.0f; \
                for (int dd = 0; dd < DV; dd++) di_acc += float(dO[o_base + dd]) * float(O[o_base + dd]); \
                smem_Di[i] = di_acc * params.attn_scale; \
            } \
            /* Cooperatively precompute Q window bounds for this tile */ \
            for (int i = (int)tid; i < tile_len; i += NUM_THREADS) { \
                int global_q = q_start + i; \
                int tmp_di[3], tmp_ws[3], tmp_we[3]; \
                precompute_q_window(global_q, na_dim, params.qkv_shape, params.window_size, \
                                    params.stride, params.dilation, params.is_causal, \
                                    tmp_di, tmp_ws, tmp_we); \
                for (int d = 0; d < 3; d++) { \
                    smem_q_di[i*3+d] = tmp_di[d]; \
                    smem_win_start[i*3+d] = tmp_ws[d]; \
                    smem_win_end[i*3+d] = tmp_we[d]; \
                } \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            \
            for (int r = 0; r < ROWS_PER_SIMD; r++) { \
                int kv_row = kv_indices[r]; \
                if (kv_row >= SK) continue; \
                \
                int q_pos = (int)simd_lane; \
                float score = -INFINITY; \
                float dov_val = 0.0f; \
                float q_lse = 0.0f, q_di = 0.0f; \
                if (q_pos < tile_len) { \
                    int global_q = q_start + q_pos; \
                    /* Q*K dot */ \
                    float acc = 0.0f; \
                    for (int dd = 0; dd < D; dd += 4) { \
                        int rem = min(4, D - dd); \
                        if (rem == 4) { \
                            float4 q4 = *reinterpret_cast<threadgroup const float4*>(&smem_Q[q_pos*D+dd]); \
                            float4 k4 = float4(k_reg[r][dd], k_reg[r][dd+1], k_reg[r][dd+2], k_reg[r][dd+3]); \
                            acc += dot(q4, k4); \
                        } else { for (int ddd = 0; ddd < rem; ddd++) acc += smem_Q[q_pos*D+dd+ddd] * k_reg[r][dd+ddd]; } \
                    } \
                    acc *= params.attn_scale; \
                    /* NA mask: precomputed K coords (registers) + precomputed Q window (shared mem) */ \
                    bool is_nb; \
                    if (kv_is_additional[r]) { is_nb = true; } \
                    else { \
                        is_nb = check_na_window_kq_precomputed(na_dim, params.dilation, \
                            kv_coords[r], &smem_q_di[q_pos*3], &smem_win_start[q_pos*3], &smem_win_end[q_pos*3]); \
                    } \
                    score = is_nb ? acc : -INFINITY; \
                    q_lse = smem_LSE[q_pos]; \
                    q_di = smem_Di[q_pos]; \
                    /* dO*V dot */ \
                    for (int dd = 0; dd < DV; dd += 4) { \
                        int rem = min(4, DV - dd); \
                        if (rem == 4) { \
                            float4 do4 = *reinterpret_cast<threadgroup const float4*>(&smem_dO[q_pos*DV+dd]); \
                            float4 v4 = float4(v_reg[r][dd], v_reg[r][dd+1], v_reg[r][dd+2], v_reg[r][dd+3]); \
                            dov_val += dot(do4, v4); \
                        } else { for (int ddd = 0; ddd < rem; ddd++) dov_val += smem_dO[q_pos*DV+dd+ddd] * v_reg[r][dd+ddd]; } \
                    } \
                    dov_val *= params.attn_scale; \
                } \
                \
                float P = (score != -INFINITY) ? exp(min(score - q_lse, 0.0f)) : 0.0f; \
                float dS = P * (dov_val - q_di); \
                \
                /* dK accumulation */ \
                if (q_pos < tile_len) { \
                    for (int dd = 0; dd < D; dd += 4) { \
                        int rem = min(4, D - dd); \
                        if (rem == 4) { \
                            float4 q4 = *reinterpret_cast<threadgroup const float4*>(&smem_Q[q_pos*D+dd]); \
                            dk_acc[r][dd]   += simd_sum(dS * q4.x); \
                            dk_acc[r][dd+1] += simd_sum(dS * q4.y); \
                            dk_acc[r][dd+2] += simd_sum(dS * q4.z); \
                            dk_acc[r][dd+3] += simd_sum(dS * q4.w); \
                        } else { for (int ddd = 0; ddd < rem; ddd++) dk_acc[r][dd+ddd] += simd_sum(dS * smem_Q[q_pos*D+dd+ddd]); } \
                    } \
                } else { \
                    for (int dd = 0; dd < D; dd += 4) { int rem = min(4, D - dd); if (rem == 4) { simd_sum(0.0f); simd_sum(0.0f); simd_sum(0.0f); simd_sum(0.0f); } else { for (int ddd = 0; ddd < rem; ddd++) simd_sum(0.0f); } } \
                } \
                /* dV accumulation */ \
                if (q_pos < tile_len) { \
                    for (int dd = 0; dd < DV; dd += 4) { \
                        int rem = min(4, DV - dd); \
                        if (rem == 4) { \
                            float4 do4 = *reinterpret_cast<threadgroup const float4*>(&smem_dO[q_pos*DV+dd]); \
                            dv_acc[r][dd]   += simd_sum(P * do4.x); \
                            dv_acc[r][dd+1] += simd_sum(P * do4.y); \
                            dv_acc[r][dd+2] += simd_sum(P * do4.z); \
                            dv_acc[r][dd+3] += simd_sum(P * do4.w); \
                        } else { for (int ddd = 0; ddd < rem; ddd++) dv_acc[r][dd+ddd] += simd_sum(P * smem_dO[q_pos*DV+dd+ddd]); } \
                    } \
                } else { \
                    for (int dd = 0; dd < DV; dd += 4) { int rem = min(4, DV - dd); if (rem == 4) { simd_sum(0.0f); simd_sum(0.0f); simd_sum(0.0f); simd_sum(0.0f); } else { for (int ddd = 0; ddd < rem; ddd++) simd_sum(0.0f); } } \
                } \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
        } \
    } \
    \
    /* Write dK and dV */ \
    if (simd_lane == 0) { \
        for (int r = 0; r < ROWS_PER_SIMD; r++) { \
            int kv_row = kv_indices[r]; \
            if (kv_row >= SK) continue; \
            int dk_base = batch_idx * SK * HK * D + kv_row * HK * D + head_kv_idx * D; \
            int dv_base = batch_idx * SK * HK * DV + kv_row * HK * DV + head_kv_idx * DV; \
            for (int dd = 0; dd < D; dd++) dK[dk_base + dd] = DTYPE(dk_acc[r][dd]); \
            for (int dd = 0; dd < DV; dd++) dV[dv_base + dd] = DTYPE(dv_acc[r][dd]); \
        } \
    } \
}

TILED_DKDV_KERNEL(na_backward_tiled_dKdV_fp32_br32_d32, float, 32, 32, 32)
TILED_DKDV_KERNEL(na_backward_tiled_dKdV_fp32_br32, float, 32, 32, 64)
TILED_DKDV_KERNEL(na_backward_tiled_dKdV_fp32_br16, float, 16, 16, 128)
TILED_DKDV_KERNEL(na_backward_tiled_dKdV_fp16_br32_d32, half, 32, 32, 32)
TILED_DKDV_KERNEL(na_backward_tiled_dKdV_fp16_br32, half, 32, 32, 64)
TILED_DKDV_KERNEL(na_backward_tiled_dKdV_fp16_br16, half, 16, 16, 128)
TILED_DKDV_KERNEL(na_backward_tiled_dKdV_bf16_br32_d32, bfloat, 32, 32, 32)
TILED_DKDV_KERNEL(na_backward_tiled_dKdV_bf16_br32, bfloat, 32, 32, 64)
TILED_DKDV_KERNEL(na_backward_tiled_dKdV_bf16_br16, bfloat, 16, 16, 128)

)";
}

// =============================================================================
// Metal Kernel Cache (Thread-Safe)
// =============================================================================

static std::mutex g_init_mutex;
static std::atomic<bool> g_initialized{false};

static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLLibrary> g_bwd_library = nil;
static id<MTLLibrary> g_tiled_library = nil;
static id<MTLLibrary> g_tiled_bwd_library = nil;
static id<MTLComputePipelineState> g_na_fwd_fp32 = nil;
static id<MTLComputePipelineState> g_na_fwd_fp16 = nil;
static id<MTLComputePipelineState> g_na_fwd_bf16 = nil;
// Tiled forward pipelines (flash-attention style)
static id<MTLComputePipelineState> g_na_fwd_tiled_fp32_br32_d32 = nil;
static id<MTLComputePipelineState> g_na_fwd_tiled_fp32_br32 = nil;
static id<MTLComputePipelineState> g_na_fwd_tiled_fp32_br16 = nil;
static id<MTLComputePipelineState> g_na_fwd_tiled_fp16_br32_d32 = nil;
static id<MTLComputePipelineState> g_na_fwd_tiled_fp16_br32 = nil;
static id<MTLComputePipelineState> g_na_fwd_tiled_fp16_br16 = nil;
static id<MTLComputePipelineState> g_na_fwd_tiled_bf16_br32_d32 = nil;
static id<MTLComputePipelineState> g_na_fwd_tiled_bf16_br32 = nil;
static id<MTLComputePipelineState> g_na_fwd_tiled_bf16_br16 = nil;
static id<MTLComputePipelineState> g_na_bwd_dQ_fp32 = nil;
static id<MTLComputePipelineState> g_na_bwd_dQ_fp16 = nil;
static id<MTLComputePipelineState> g_na_bwd_dQ_bf16 = nil;
static id<MTLComputePipelineState> g_na_bwd_dK_fp32 = nil;
static id<MTLComputePipelineState> g_na_bwd_dK_fp16 = nil;
static id<MTLComputePipelineState> g_na_bwd_dK_bf16 = nil;
static id<MTLComputePipelineState> g_na_bwd_dV_fp32 = nil;
static id<MTLComputePipelineState> g_na_bwd_dV_fp16 = nil;
static id<MTLComputePipelineState> g_na_bwd_dV_bf16 = nil;
// Tiled backward pipelines (Di fused into dQ/dKdV, no separate Di kernel)
static id<MTLComputePipelineState> g_na_bwd_tiled_dQ_fp32_br32_d32 = nil;
static id<MTLComputePipelineState> g_na_bwd_tiled_dQ_fp32_br32 = nil;
static id<MTLComputePipelineState> g_na_bwd_tiled_dQ_fp32_br16 = nil;
static id<MTLComputePipelineState> g_na_bwd_tiled_dQ_fp16_br32_d32 = nil;
static id<MTLComputePipelineState> g_na_bwd_tiled_dQ_fp16_br32 = nil;
static id<MTLComputePipelineState> g_na_bwd_tiled_dQ_fp16_br16 = nil;
static id<MTLComputePipelineState> g_na_bwd_tiled_dQ_bf16_br32_d32 = nil;
static id<MTLComputePipelineState> g_na_bwd_tiled_dQ_bf16_br32 = nil;
static id<MTLComputePipelineState> g_na_bwd_tiled_dQ_bf16_br16 = nil;
static id<MTLComputePipelineState> g_na_bwd_tiled_dKdV_fp32_br32_d32 = nil;
static id<MTLComputePipelineState> g_na_bwd_tiled_dKdV_fp32_br32 = nil;
static id<MTLComputePipelineState> g_na_bwd_tiled_dKdV_fp32_br16 = nil;
static id<MTLComputePipelineState> g_na_bwd_tiled_dKdV_fp16_br32_d32 = nil;
static id<MTLComputePipelineState> g_na_bwd_tiled_dKdV_fp16_br32 = nil;
static id<MTLComputePipelineState> g_na_bwd_tiled_dKdV_fp16_br16 = nil;
static id<MTLComputePipelineState> g_na_bwd_tiled_dKdV_bf16_br32_d32 = nil;
static id<MTLComputePipelineState> g_na_bwd_tiled_dKdV_bf16_br32 = nil;
static id<MTLComputePipelineState> g_na_bwd_tiled_dKdV_bf16_br16 = nil;

static bool init_metal() {
    if (g_initialized.load(std::memory_order_acquire)) {
        return true;
    }

    std::lock_guard<std::mutex> lock(g_init_mutex);

    if (g_initialized.load(std::memory_order_relaxed)) {
        return true;
    }

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        TORCH_CHECK(g_device, "Failed to create Metal device");

        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.mathMode = MTLMathModeFast;

        g_library = [g_device newLibraryWithSource:get_metal_source() options:options error:&error];
        TORCH_CHECK(g_library, "Failed to compile Metal library: ",
                     error ? [[error localizedDescription] UTF8String] : "unknown");

        auto create_pipeline = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [g_library newFunctionWithName:name];
            TORCH_CHECK(fn, "Failed to find Metal function: ", [name UTF8String]);
            id<MTLComputePipelineState> pipeline =
                [g_device newComputePipelineStateWithFunction:fn error:&error];
            TORCH_CHECK(pipeline, "Failed to create pipeline for ", [name UTF8String], ": ",
                         error ? [[error localizedDescription] UTF8String] : "unknown");
            return pipeline;
        };

        g_na_fwd_fp32 = create_pipeline(@"na_forward_fp32");
        g_na_fwd_fp16 = create_pipeline(@"na_forward_fp16");

        @try {
            g_na_fwd_bf16 = create_pipeline(@"na_forward_bf16");
        } @catch (NSException *exception) {
            g_na_fwd_bf16 = nil;
        }

        // Compile tiled forward kernels
        NSError* tiled_error = nil;
        g_tiled_library = [g_device newLibraryWithSource:get_metal_tiled_source() options:options error:&tiled_error];
        TORCH_CHECK(g_tiled_library, "Failed to compile Metal tiled library: ",
                     tiled_error ? [[tiled_error localizedDescription] UTF8String] : "unknown");

        auto create_tiled_pipeline = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [g_tiled_library newFunctionWithName:name];
            TORCH_CHECK(fn, "Failed to find Metal tiled function: ", [name UTF8String]);
            NSError* err = nil;
            id<MTLComputePipelineState> pipeline =
                [g_device newComputePipelineStateWithFunction:fn error:&err];
            TORCH_CHECK(pipeline, "Failed to create tiled pipeline for ", [name UTF8String], ": ",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            return pipeline;
        };

        g_na_fwd_tiled_fp32_br32_d32 = create_tiled_pipeline(@"na_forward_tiled_fp32_br32_d32");
        g_na_fwd_tiled_fp32_br32 = create_tiled_pipeline(@"na_forward_tiled_fp32_br32");
        g_na_fwd_tiled_fp32_br16 = create_tiled_pipeline(@"na_forward_tiled_fp32_br16");
        g_na_fwd_tiled_fp16_br32_d32 = create_tiled_pipeline(@"na_forward_tiled_fp16_br32_d32");
        g_na_fwd_tiled_fp16_br32 = create_tiled_pipeline(@"na_forward_tiled_fp16_br32");
        g_na_fwd_tiled_fp16_br16 = create_tiled_pipeline(@"na_forward_tiled_fp16_br16");

        @try {
            g_na_fwd_tiled_bf16_br32_d32 = create_tiled_pipeline(@"na_forward_tiled_bf16_br32_d32");
            g_na_fwd_tiled_bf16_br32 = create_tiled_pipeline(@"na_forward_tiled_bf16_br32");
            g_na_fwd_tiled_bf16_br16 = create_tiled_pipeline(@"na_forward_tiled_bf16_br16");
        } @catch (NSException *exception) {
            g_na_fwd_tiled_bf16_br32_d32 = nil;
            g_na_fwd_tiled_bf16_br32 = nil;
            g_na_fwd_tiled_bf16_br16 = nil;
        }

        // Compile backward kernels
        NSError* bwd_error = nil;
        g_bwd_library = [g_device newLibraryWithSource:get_metal_backward_source() options:options error:&bwd_error];
        TORCH_CHECK(g_bwd_library, "Failed to compile Metal backward library: ",
                     bwd_error ? [[bwd_error localizedDescription] UTF8String] : "unknown");

        auto create_bwd_pipeline = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [g_bwd_library newFunctionWithName:name];
            TORCH_CHECK(fn, "Failed to find Metal backward function: ", [name UTF8String]);
            NSError* err = nil;
            id<MTLComputePipelineState> pipeline =
                [g_device newComputePipelineStateWithFunction:fn error:&err];
            TORCH_CHECK(pipeline, "Failed to create backward pipeline for ", [name UTF8String], ": ",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            return pipeline;
        };

        g_na_bwd_dQ_fp32 = create_bwd_pipeline(@"na_backward_dQ_fp32");
        g_na_bwd_dK_fp32 = create_bwd_pipeline(@"na_backward_dK_fp32");
        g_na_bwd_dV_fp32 = create_bwd_pipeline(@"na_backward_dV_fp32");
        g_na_bwd_dQ_fp16 = create_bwd_pipeline(@"na_backward_dQ_fp16");
        g_na_bwd_dK_fp16 = create_bwd_pipeline(@"na_backward_dK_fp16");
        g_na_bwd_dV_fp16 = create_bwd_pipeline(@"na_backward_dV_fp16");

        @try {
            g_na_bwd_dQ_bf16 = create_bwd_pipeline(@"na_backward_dQ_bf16");
            g_na_bwd_dK_bf16 = create_bwd_pipeline(@"na_backward_dK_bf16");
            g_na_bwd_dV_bf16 = create_bwd_pipeline(@"na_backward_dV_bf16");
        } @catch (NSException *exception) {
            g_na_bwd_dQ_bf16 = nil;
            g_na_bwd_dK_bf16 = nil;
            g_na_bwd_dV_bf16 = nil;
        }

        // Compile tiled backward kernels
        NSError* tiled_bwd_error = nil;
        g_tiled_bwd_library = [g_device newLibraryWithSource:get_metal_tiled_backward_source() options:options error:&tiled_bwd_error];
        TORCH_CHECK(g_tiled_bwd_library, "Failed to compile Metal tiled backward library: ",
                     tiled_bwd_error ? [[tiled_bwd_error localizedDescription] UTF8String] : "unknown");

        auto create_tiled_bwd_pipeline = [&](NSString* name) -> id<MTLComputePipelineState> {
            id<MTLFunction> fn = [g_tiled_bwd_library newFunctionWithName:name];
            TORCH_CHECK(fn, "Failed to find Metal tiled backward function: ", [name UTF8String]);
            NSError* err = nil;
            id<MTLComputePipelineState> pipeline =
                [g_device newComputePipelineStateWithFunction:fn error:&err];
            TORCH_CHECK(pipeline, "Failed to create tiled backward pipeline for ", [name UTF8String], ": ",
                         err ? [[err localizedDescription] UTF8String] : "unknown");
            return pipeline;
        };

        g_na_bwd_tiled_dQ_fp32_br32_d32 = create_tiled_bwd_pipeline(@"na_backward_tiled_dQ_fp32_br32_d32");
        g_na_bwd_tiled_dQ_fp32_br32 = create_tiled_bwd_pipeline(@"na_backward_tiled_dQ_fp32_br32");
        g_na_bwd_tiled_dQ_fp32_br16 = create_tiled_bwd_pipeline(@"na_backward_tiled_dQ_fp32_br16");
        g_na_bwd_tiled_dQ_fp16_br32_d32 = create_tiled_bwd_pipeline(@"na_backward_tiled_dQ_fp16_br32_d32");
        g_na_bwd_tiled_dQ_fp16_br32 = create_tiled_bwd_pipeline(@"na_backward_tiled_dQ_fp16_br32");
        g_na_bwd_tiled_dQ_fp16_br16 = create_tiled_bwd_pipeline(@"na_backward_tiled_dQ_fp16_br16");
        g_na_bwd_tiled_dKdV_fp32_br32_d32 = create_tiled_bwd_pipeline(@"na_backward_tiled_dKdV_fp32_br32_d32");
        g_na_bwd_tiled_dKdV_fp32_br32 = create_tiled_bwd_pipeline(@"na_backward_tiled_dKdV_fp32_br32");
        g_na_bwd_tiled_dKdV_fp32_br16 = create_tiled_bwd_pipeline(@"na_backward_tiled_dKdV_fp32_br16");
        g_na_bwd_tiled_dKdV_fp16_br32_d32 = create_tiled_bwd_pipeline(@"na_backward_tiled_dKdV_fp16_br32_d32");
        g_na_bwd_tiled_dKdV_fp16_br32 = create_tiled_bwd_pipeline(@"na_backward_tiled_dKdV_fp16_br32");
        g_na_bwd_tiled_dKdV_fp16_br16 = create_tiled_bwd_pipeline(@"na_backward_tiled_dKdV_fp16_br16");

        @try {
            g_na_bwd_tiled_dQ_bf16_br32_d32 = create_tiled_bwd_pipeline(@"na_backward_tiled_dQ_bf16_br32_d32");
            g_na_bwd_tiled_dQ_bf16_br32 = create_tiled_bwd_pipeline(@"na_backward_tiled_dQ_bf16_br32");
            g_na_bwd_tiled_dQ_bf16_br16 = create_tiled_bwd_pipeline(@"na_backward_tiled_dQ_bf16_br16");
            g_na_bwd_tiled_dKdV_bf16_br32_d32 = create_tiled_bwd_pipeline(@"na_backward_tiled_dKdV_bf16_br32_d32");
            g_na_bwd_tiled_dKdV_bf16_br32 = create_tiled_bwd_pipeline(@"na_backward_tiled_dKdV_bf16_br32");
            g_na_bwd_tiled_dKdV_bf16_br16 = create_tiled_bwd_pipeline(@"na_backward_tiled_dKdV_bf16_br16");
        } @catch (NSException *exception) {
            g_na_bwd_tiled_dQ_bf16_br32_d32 = nil;
            g_na_bwd_tiled_dQ_bf16_br32 = nil;
            g_na_bwd_tiled_dQ_bf16_br16 = nil;
            g_na_bwd_tiled_dKdV_bf16_br32_d32 = nil;
            g_na_bwd_tiled_dKdV_bf16_br32 = nil;
            g_na_bwd_tiled_dKdV_bf16_br16 = nil;
        }

        g_initialized.store(true, std::memory_order_release);
    }

    return true;
}

// =============================================================================
// Generic forward implementation
// =============================================================================

template <class StdNADim, class StdCausal>
static void metal_na_forward_impl(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& logsumexp,
    const StdNADim& kernel_size,
    const StdNADim& stride,
    const StdNADim& dilation,
    const StdCausal& is_causal,
    float attn_scale,
    const StdNADim& qkv_shape,
    int num_extra_kv) {

    static constexpr int kNADim = std::tuple_size_v<StdNADim>;
    static_assert(kNADim >= 1 && kNADim <= 3);
    static_assert(std::tuple_size_v<StdCausal> == kNADim);

    TORCH_CHECK(query.is_mps(), "query must be on MPS device");
    TORCH_CHECK(key.is_mps(), "key must be on MPS device");
    TORCH_CHECK(value.is_mps(), "value must be on MPS device");
    TORCH_CHECK(out.is_mps(), "output must be on MPS device");
    TORCH_CHECK(logsumexp.is_mps(), "logsumexp must be on MPS device");

    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(logsumexp.is_contiguous(), "logsumexp must be contiguous");

    TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D (flattened spatial)");

    init_metal();

    int batch_size = query.size(0);
    int seqlen_q = query.size(1);
    int heads_q = query.size(2);
    int dim = query.size(3);
    int heads_kv = key.size(2);
    int seqlen_kv = key.size(1);
    int dim_value = value.size(3);

    TORCH_CHECK(heads_q >= heads_kv, "heads_q must be >= heads_kv");
    TORCH_CHECK(heads_q % heads_kv == 0, "heads_q must be divisible by heads_kv");
    TORCH_CHECK(dim <= 1024, "Metal kernel supports head dim up to 1024");
    TORCH_CHECK(dim_value <= 1024, "Metal kernel supports value head dim up to 1024");

    TORCH_CHECK(query.scalar_type() == at::kFloat ||
                query.scalar_type() == at::kHalf ||
                query.scalar_type() == at::kBFloat16,
                "Only FP32, FP16, and BF16 are supported");

    bool is_bf16 = query.scalar_type() == at::kBFloat16;
    bool bf16_fallback = is_bf16 && g_na_fwd_bf16 == nil;

    auto orig_dtype = query.scalar_type();
    auto query_work = bf16_fallback ? query.to(at::kFloat) : query;
    auto key_work = bf16_fallback ? key.to(at::kFloat) : key;
    auto value_work = bf16_fallback ? value.to(at::kFloat) : value;
    auto out_work = bf16_fallback ? at::empty_like(out, at::kFloat) : out;

    NAParams params;
    params.batch_size = batch_size;
    params.seqlen_q = seqlen_q;
    params.seqlen_kv = seqlen_kv;
    params.heads_q = heads_q;
    params.heads_kv = heads_kv;
    params.dim = dim;
    params.dim_value = dim_value;
    params.num_additional_kv = num_extra_kv;
    params.attn_scale = attn_scale;
    params.na_dim = kNADim;

    for (int i = 0; i < 3; i++) {
        params.qkv_shape[i] = 1;
        params.window_size[i] = 1;
        params.stride[i] = 1;
        params.dilation[i] = 1;
        params.is_causal[i] = 0;
    }

    if constexpr (kNADim >= 1) {
        params.qkv_shape[0] = std::get<0>(qkv_shape);
        params.window_size[0] = std::get<0>(kernel_size);
        params.stride[0] = std::get<0>(stride);
        params.dilation[0] = std::get<0>(dilation);
        params.is_causal[0] = std::get<0>(is_causal) ? 1 : 0;
    }
    if constexpr (kNADim >= 2) {
        params.qkv_shape[1] = std::get<1>(qkv_shape);
        params.window_size[1] = std::get<1>(kernel_size);
        params.stride[1] = std::get<1>(stride);
        params.dilation[1] = std::get<1>(dilation);
        params.is_causal[1] = std::get<1>(is_causal) ? 1 : 0;
    }
    if constexpr (kNADim >= 3) {
        params.qkv_shape[2] = std::get<2>(qkv_shape);
        params.window_size[2] = std::get<2>(kernel_size);
        params.stride[2] = std::get<2>(stride);
        params.dilation[2] = std::get<2>(dilation);
        params.is_causal[2] = std::get<2>(is_causal) ? 1 : 0;
    }

    // Decide whether to use tiled (flash-attention) or reference kernel
    // Tiled kernel: D <= 128, no additional KV tokens (tiled kernel handles them
    // but reference is simpler for edge cases)
    bool use_tiled = (dim <= 128) && (dim_value <= 128);

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();
        id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();

        id<MTLBuffer> q_buf = at::native::mps::getMTLBufferStorage(query_work);
        id<MTLBuffer> k_buf = at::native::mps::getMTLBufferStorage(key_work);
        id<MTLBuffer> v_buf = at::native::mps::getMTLBufferStorage(value_work);
        id<MTLBuffer> o_buf = at::native::mps::getMTLBufferStorage(out_work);
        id<MTLBuffer> lse_buf = at::native::mps::getMTLBufferStorage(logsumexp);

        [encoder setBuffer:q_buf offset:query_work.storage_offset() * query_work.element_size() atIndex:0];
        [encoder setBuffer:k_buf offset:key_work.storage_offset() * key_work.element_size() atIndex:1];
        [encoder setBuffer:v_buf offset:value_work.storage_offset() * value_work.element_size() atIndex:2];
        [encoder setBuffer:o_buf offset:out_work.storage_offset() * out_work.element_size() atIndex:3];
        [encoder setBuffer:lse_buf offset:logsumexp.storage_offset() * logsumexp.element_size() atIndex:4];
        [encoder setBytes:&params length:sizeof(NAParams) atIndex:5];

        if (use_tiled) {
            // Select tiled pipeline based on dtype and head dimension
            id<MTLComputePipelineState> pipeline;
            int Br;
            if (dim <= 32 && dim_value <= 32) {
                Br = 32;
                if (query_work.scalar_type() == at::kHalf) {
                    pipeline = g_na_fwd_tiled_fp16_br32_d32;
                } else if (query_work.scalar_type() == at::kBFloat16) {
                    pipeline = g_na_fwd_tiled_bf16_br32_d32;
                } else {
                    pipeline = g_na_fwd_tiled_fp32_br32_d32;
                }
            } else if (dim <= 64 && dim_value <= 64) {
                Br = 32;
                if (query_work.scalar_type() == at::kHalf) {
                    pipeline = g_na_fwd_tiled_fp16_br32;
                } else if (query_work.scalar_type() == at::kBFloat16) {
                    pipeline = g_na_fwd_tiled_bf16_br32;
                } else {
                    pipeline = g_na_fwd_tiled_fp32_br32;
                }
            } else {
                Br = 16;
                if (query_work.scalar_type() == at::kHalf) {
                    pipeline = g_na_fwd_tiled_fp16_br16;
                } else if (query_work.scalar_type() == at::kBFloat16) {
                    pipeline = g_na_fwd_tiled_bf16_br16;
                } else {
                    pipeline = g_na_fwd_tiled_fp32_br16;
                }
            }

            // Fallback to reference if tiled pipeline unavailable (e.g. BF16 not supported)
            if (pipeline == nil) {
                use_tiled = false;
            } else {
                [encoder setComputePipelineState:pipeline];
                int num_q_tiles = (seqlen_q + Br - 1) / Br;
                MTLSize threadgroupCount = MTLSizeMake(num_q_tiles, heads_q * batch_size, 1);
                MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
                [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
            }
        }

        if (!use_tiled) {
            // Reference kernel: one threadgroup per Q position
            id<MTLComputePipelineState> pipeline;
            if (query_work.scalar_type() == at::kHalf) {
                pipeline = g_na_fwd_fp16;
            } else if (query_work.scalar_type() == at::kBFloat16) {
                pipeline = g_na_fwd_bf16;
            } else {
                pipeline = g_na_fwd_fp32;
            }
            [encoder setComputePipelineState:pipeline];

            MTLSize threadgroupCount = MTLSizeMake(seqlen_q, heads_q * batch_size, 1);
            MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);
            [encoder dispatchThreadgroups:threadgroupCount threadsPerThreadgroup:threadgroupSize];
        }

        // No explicit sync — let PyTorch manage command buffer lifecycle
    }

    if (bf16_fallback) {
        out.copy_(out_work.to(orig_dtype));
    }
}

// =============================================================================
// Public API + Pybind11 Bindings (single .mm file, no Swift needed)
// =============================================================================

static void metal_na1d_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& logsumexp,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& stride,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t>& qkv_shape,
    int num_extra_kv) {
    TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");
    metal_na_forward_impl(
        out, query, key, value, logsumexp,
        kernel_size, stride, dilation, is_causal,
        attn_scale, qkv_shape, num_extra_kv);
}

static void metal_na2d_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& logsumexp,
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t>& dilation,
    const std::tuple<bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t>& qkv_shape,
    int num_extra_kv) {
    TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");
    metal_na_forward_impl(
        out, query, key, value, logsumexp,
        kernel_size, stride, dilation, is_causal,
        attn_scale, qkv_shape, num_extra_kv);
}

static void metal_na3d_forward(
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::Tensor& logsumexp,
    const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t, int32_t>& dilation,
    const std::tuple<bool, bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t, int32_t>& qkv_shape,
    int num_extra_kv) {
    TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");
    metal_na_forward_impl(
        out, query, key, value, logsumexp,
        kernel_size, stride, dilation, is_causal,
        attn_scale, qkv_shape, num_extra_kv);
}

// =============================================================================
// Generic backward implementation
// =============================================================================

template <class StdNADim, class StdCausal>
static void metal_na_backward_impl(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& output,
    const at::Tensor& d_output,
    const at::Tensor& logsumexp,
    at::Tensor& d_query,
    at::Tensor& d_key,
    at::Tensor& d_value,
    const StdNADim& kernel_size,
    const StdNADim& stride,
    const StdNADim& dilation,
    const StdCausal& is_causal,
    float attn_scale,
    const StdNADim& qkv_shape,
    int num_extra_kv) {

    static constexpr int kNADim = std::tuple_size_v<StdNADim>;
    static_assert(kNADim >= 1 && kNADim <= 3);
    static_assert(std::tuple_size_v<StdCausal> == kNADim);

    TORCH_CHECK(query.is_mps(), "query must be on MPS device");
    TORCH_CHECK(key.is_mps(), "key must be on MPS device");
    TORCH_CHECK(value.is_mps(), "value must be on MPS device");
    TORCH_CHECK(output.is_mps(), "output must be on MPS device");
    TORCH_CHECK(d_output.is_mps(), "d_output must be on MPS device");
    TORCH_CHECK(logsumexp.is_mps(), "logsumexp must be on MPS device");

    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.is_contiguous(), "key must be contiguous");
    TORCH_CHECK(value.is_contiguous(), "value must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(d_output.is_contiguous(), "d_output must be contiguous");
    TORCH_CHECK(logsumexp.is_contiguous(), "logsumexp must be contiguous");
    TORCH_CHECK(d_query.is_contiguous(), "d_query must be contiguous");
    TORCH_CHECK(d_key.is_contiguous(), "d_key must be contiguous");
    TORCH_CHECK(d_value.is_contiguous(), "d_value must be contiguous");

    TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D (flattened spatial)");

    init_metal();

    int batch_size = query.size(0);
    int seqlen_q = query.size(1);
    int heads_q = query.size(2);
    int dim = query.size(3);
    int heads_kv = key.size(2);
    int seqlen_kv = key.size(1);
    int dim_value = value.size(3);

    TORCH_CHECK(heads_q >= heads_kv, "heads_q must be >= heads_kv");
    TORCH_CHECK(heads_q % heads_kv == 0, "heads_q must be divisible by heads_kv");
    TORCH_CHECK(dim <= 1024, "Metal kernel supports head dim up to 1024");
    TORCH_CHECK(dim_value <= 1024, "Metal kernel supports value head dim up to 1024");

    TORCH_CHECK(query.scalar_type() == at::kFloat ||
                query.scalar_type() == at::kHalf ||
                query.scalar_type() == at::kBFloat16,
                "Only FP32, FP16, and BF16 are supported");

    bool is_bf16 = query.scalar_type() == at::kBFloat16;
    bool bf16_fallback = is_bf16 && g_na_bwd_dQ_bf16 == nil;

    auto orig_dtype = query.scalar_type();
    auto query_work = bf16_fallback ? query.to(at::kFloat) : query;
    auto key_work = bf16_fallback ? key.to(at::kFloat) : key;
    auto value_work = bf16_fallback ? value.to(at::kFloat) : value;
    auto output_work = bf16_fallback ? output.to(at::kFloat) : output;
    auto d_output_work = bf16_fallback ? d_output.to(at::kFloat) : d_output;
    auto d_query_work = bf16_fallback ? at::zeros_like(d_query, at::kFloat) : d_query;
    auto d_key_work = bf16_fallback ? at::zeros_like(d_key, at::kFloat) : d_key;
    auto d_value_work = bf16_fallback ? at::zeros_like(d_value, at::kFloat) : d_value;

    NAParams params;
    params.batch_size = batch_size;
    params.seqlen_q = seqlen_q;
    params.seqlen_kv = seqlen_kv;
    params.heads_q = heads_q;
    params.heads_kv = heads_kv;
    params.dim = dim;
    params.dim_value = dim_value;
    params.num_additional_kv = num_extra_kv;
    params.attn_scale = attn_scale;
    params.na_dim = kNADim;

    for (int i = 0; i < 3; i++) {
        params.qkv_shape[i] = 1;
        params.window_size[i] = 1;
        params.stride[i] = 1;
        params.dilation[i] = 1;
        params.is_causal[i] = 0;
    }

    if constexpr (kNADim >= 1) {
        params.qkv_shape[0] = std::get<0>(qkv_shape);
        params.window_size[0] = std::get<0>(kernel_size);
        params.stride[0] = std::get<0>(stride);
        params.dilation[0] = std::get<0>(dilation);
        params.is_causal[0] = std::get<0>(is_causal) ? 1 : 0;
    }
    if constexpr (kNADim >= 2) {
        params.qkv_shape[1] = std::get<1>(qkv_shape);
        params.window_size[1] = std::get<1>(kernel_size);
        params.stride[1] = std::get<1>(stride);
        params.dilation[1] = std::get<1>(dilation);
        params.is_causal[1] = std::get<1>(is_causal) ? 1 : 0;
    }
    if constexpr (kNADim >= 3) {
        params.qkv_shape[2] = std::get<2>(qkv_shape);
        params.window_size[2] = std::get<2>(kernel_size);
        params.stride[2] = std::get<2>(stride);
        params.dilation[2] = std::get<2>(dilation);
        params.is_causal[2] = std::get<2>(is_causal) ? 1 : 0;
    }

    @autoreleasepool {
        auto stream = at::mps::getCurrentMPSStream();

        id<MTLBuffer> q_buf = at::native::mps::getMTLBufferStorage(query_work);
        id<MTLBuffer> k_buf = at::native::mps::getMTLBufferStorage(key_work);
        id<MTLBuffer> v_buf = at::native::mps::getMTLBufferStorage(value_work);
        id<MTLBuffer> o_buf = at::native::mps::getMTLBufferStorage(output_work);
        id<MTLBuffer> do_buf = at::native::mps::getMTLBufferStorage(d_output_work);
        id<MTLBuffer> lse_buf = at::native::mps::getMTLBufferStorage(logsumexp);
        id<MTLBuffer> dq_buf = at::native::mps::getMTLBufferStorage(d_query_work);
        id<MTLBuffer> dk_buf = at::native::mps::getMTLBufferStorage(d_key_work);
        id<MTLBuffer> dv_buf = at::native::mps::getMTLBufferStorage(d_value_work);

        size_t q_off = query_work.storage_offset() * query_work.element_size();
        size_t k_off = key_work.storage_offset() * key_work.element_size();
        size_t v_off = value_work.storage_offset() * value_work.element_size();
        size_t o_off = output_work.storage_offset() * output_work.element_size();
        size_t do_off = d_output_work.storage_offset() * d_output_work.element_size();
        size_t lse_off = logsumexp.storage_offset() * logsumexp.element_size();
        size_t dq_off = d_query_work.storage_offset() * d_query_work.element_size();
        size_t dk_off = d_key_work.storage_offset() * d_key_work.element_size();
        size_t dv_off = d_value_work.storage_offset() * d_value_work.element_size();

        bool use_tiled_bwd = (dim <= 128) && (dim_value <= 128);

        MTLSize threadgroupSize = MTLSizeMake(256, 1, 1);

        if (use_tiled_bwd) {
            int Br;
            id<MTLComputePipelineState> tiled_dq_pipeline, tiled_dkdv_pipeline;

            if (dim <= 32 && dim_value <= 32) {
                Br = 32;
                if (query_work.scalar_type() == at::kHalf) {
                    tiled_dq_pipeline = g_na_bwd_tiled_dQ_fp16_br32_d32;
                    tiled_dkdv_pipeline = g_na_bwd_tiled_dKdV_fp16_br32_d32;
                } else if (query_work.scalar_type() == at::kBFloat16) {
                    tiled_dq_pipeline = g_na_bwd_tiled_dQ_bf16_br32_d32;
                    tiled_dkdv_pipeline = g_na_bwd_tiled_dKdV_bf16_br32_d32;
                } else {
                    tiled_dq_pipeline = g_na_bwd_tiled_dQ_fp32_br32_d32;
                    tiled_dkdv_pipeline = g_na_bwd_tiled_dKdV_fp32_br32_d32;
                }
            } else if (dim <= 64 && dim_value <= 64) {
                Br = 32;
                if (query_work.scalar_type() == at::kHalf) {
                    tiled_dq_pipeline = g_na_bwd_tiled_dQ_fp16_br32;
                    tiled_dkdv_pipeline = g_na_bwd_tiled_dKdV_fp16_br32;
                } else if (query_work.scalar_type() == at::kBFloat16) {
                    tiled_dq_pipeline = g_na_bwd_tiled_dQ_bf16_br32;
                    tiled_dkdv_pipeline = g_na_bwd_tiled_dKdV_bf16_br32;
                } else {
                    tiled_dq_pipeline = g_na_bwd_tiled_dQ_fp32_br32;
                    tiled_dkdv_pipeline = g_na_bwd_tiled_dKdV_fp32_br32;
                }
            } else {
                Br = 16;
                if (query_work.scalar_type() == at::kHalf) {
                    tiled_dq_pipeline = g_na_bwd_tiled_dQ_fp16_br16;
                    tiled_dkdv_pipeline = g_na_bwd_tiled_dKdV_fp16_br16;
                } else if (query_work.scalar_type() == at::kBFloat16) {
                    tiled_dq_pipeline = g_na_bwd_tiled_dQ_bf16_br16;
                    tiled_dkdv_pipeline = g_na_bwd_tiled_dKdV_bf16_br16;
                } else {
                    tiled_dq_pipeline = g_na_bwd_tiled_dQ_fp32_br16;
                    tiled_dkdv_pipeline = g_na_bwd_tiled_dKdV_fp32_br16;
                }
            }

            if (!tiled_dq_pipeline || !tiled_dkdv_pipeline) {
                use_tiled_bwd = false;
            } else {
                // ---- Tiled dQ kernel ----
                {
                    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
                    [encoder setComputePipelineState:tiled_dq_pipeline];
                    [encoder setBuffer:q_buf offset:q_off atIndex:0];
                    [encoder setBuffer:k_buf offset:k_off atIndex:1];
                    [encoder setBuffer:v_buf offset:v_off atIndex:2];
                    [encoder setBuffer:do_buf offset:do_off atIndex:3];
                    [encoder setBuffer:lse_buf offset:lse_off atIndex:4];
                    [encoder setBuffer:o_buf offset:o_off atIndex:5];
                    [encoder setBuffer:dq_buf offset:dq_off atIndex:6];
                    [encoder setBytes:&params length:sizeof(NAParams) atIndex:7];

                    int num_q_tiles = (seqlen_q + Br - 1) / Br;
                    MTLSize dq_grid = MTLSizeMake(num_q_tiles, heads_q * batch_size, 1);
                    [encoder dispatchThreadgroups:dq_grid threadsPerThreadgroup:threadgroupSize];
                }

                // ---- Tiled dKdV kernel ----
                {
                    id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
                    [encoder setComputePipelineState:tiled_dkdv_pipeline];
                    [encoder setBuffer:q_buf offset:q_off atIndex:0];
                    [encoder setBuffer:k_buf offset:k_off atIndex:1];
                    [encoder setBuffer:v_buf offset:v_off atIndex:2];
                    [encoder setBuffer:do_buf offset:do_off atIndex:3];
                    [encoder setBuffer:lse_buf offset:lse_off atIndex:4];
                    [encoder setBuffer:o_buf offset:o_off atIndex:5];
                    [encoder setBuffer:dk_buf offset:dk_off atIndex:6];
                    [encoder setBuffer:dv_buf offset:dv_off atIndex:7];
                    [encoder setBytes:&params length:sizeof(NAParams) atIndex:8];

                    int num_kv_tiles = (seqlen_kv + Br - 1) / Br;
                    MTLSize dkdv_grid = MTLSizeMake(num_kv_tiles, heads_kv * batch_size, 1);
                    [encoder dispatchThreadgroups:dkdv_grid threadsPerThreadgroup:threadgroupSize];
                }
            }
        }

        if (!use_tiled_bwd) {
            // Reference backward kernels (fallback for D > 128 or missing pipelines)
            id<MTLComputePipelineState> dq_pipeline, dk_pipeline, dv_pipeline;
            if (query_work.scalar_type() == at::kHalf) {
                dq_pipeline = g_na_bwd_dQ_fp16;
                dk_pipeline = g_na_bwd_dK_fp16;
                dv_pipeline = g_na_bwd_dV_fp16;
            } else if (query_work.scalar_type() == at::kBFloat16) {
                dq_pipeline = g_na_bwd_dQ_bf16;
                dk_pipeline = g_na_bwd_dK_bf16;
                dv_pipeline = g_na_bwd_dV_bf16;
            } else {
                dq_pipeline = g_na_bwd_dQ_fp32;
                dk_pipeline = g_na_bwd_dK_fp32;
                dv_pipeline = g_na_bwd_dV_fp32;
            }

            // ---- dQ kernel: grid = (seqlen_q, heads_q * batch_size) ----
            {
                id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
                [encoder setComputePipelineState:dq_pipeline];
                [encoder setBuffer:q_buf offset:q_off atIndex:0];
                [encoder setBuffer:k_buf offset:k_off atIndex:1];
                [encoder setBuffer:v_buf offset:v_off atIndex:2];
                [encoder setBuffer:o_buf offset:o_off atIndex:3];
                [encoder setBuffer:do_buf offset:do_off atIndex:4];
                [encoder setBuffer:lse_buf offset:lse_off atIndex:5];
                [encoder setBuffer:dq_buf offset:dq_off atIndex:6];
                [encoder setBytes:&params length:sizeof(NAParams) atIndex:7];

                MTLSize dq_grid = MTLSizeMake(seqlen_q, heads_q * batch_size, 1);
                [encoder dispatchThreadgroups:dq_grid threadsPerThreadgroup:threadgroupSize];
            }

            // ---- dK kernel: grid = (seqlen_kv, heads_kv * batch_size) ----
            {
                id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
                [encoder setComputePipelineState:dk_pipeline];
                [encoder setBuffer:q_buf offset:q_off atIndex:0];
                [encoder setBuffer:k_buf offset:k_off atIndex:1];
                [encoder setBuffer:v_buf offset:v_off atIndex:2];
                [encoder setBuffer:o_buf offset:o_off atIndex:3];
                [encoder setBuffer:do_buf offset:do_off atIndex:4];
                [encoder setBuffer:lse_buf offset:lse_off atIndex:5];
                [encoder setBuffer:dk_buf offset:dk_off atIndex:6];
                [encoder setBytes:&params length:sizeof(NAParams) atIndex:7];

                MTLSize dk_grid = MTLSizeMake(seqlen_kv, heads_kv * batch_size, 1);
                [encoder dispatchThreadgroups:dk_grid threadsPerThreadgroup:threadgroupSize];
            }

            // ---- dV kernel: grid = (seqlen_kv, heads_kv * batch_size) ----
            {
                id<MTLComputeCommandEncoder> encoder = stream->commandEncoder();
                [encoder setComputePipelineState:dv_pipeline];
                [encoder setBuffer:q_buf offset:q_off atIndex:0];
                [encoder setBuffer:k_buf offset:k_off atIndex:1];
                [encoder setBuffer:v_buf offset:v_off atIndex:2];
                [encoder setBuffer:o_buf offset:o_off atIndex:3];
                [encoder setBuffer:do_buf offset:do_off atIndex:4];
                [encoder setBuffer:lse_buf offset:lse_off atIndex:5];
                [encoder setBuffer:dv_buf offset:dv_off atIndex:6];
                [encoder setBytes:&params length:sizeof(NAParams) atIndex:7];

                MTLSize dv_grid = MTLSizeMake(seqlen_kv, heads_kv * batch_size, 1);
                [encoder dispatchThreadgroups:dv_grid threadsPerThreadgroup:threadgroupSize];
            }
        }

        // No explicit sync — let PyTorch manage command buffer lifecycle
    }

    if (bf16_fallback) {
        d_query.copy_(d_query_work.to(orig_dtype));
        d_key.copy_(d_key_work.to(orig_dtype));
        d_value.copy_(d_value_work.to(orig_dtype));
    }
}

// =============================================================================
// Backward Public API
// =============================================================================

static void metal_na1d_backward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& output,
    const at::Tensor& d_output,
    const at::Tensor& logsumexp,
    at::Tensor& d_query,
    at::Tensor& d_key,
    at::Tensor& d_value,
    const std::tuple<int32_t>& kernel_size,
    const std::tuple<int32_t>& stride,
    const std::tuple<int32_t>& dilation,
    const std::tuple<bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t>& qkv_shape,
    int num_extra_kv) {
    TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");
    metal_na_backward_impl(
        query, key, value, output, d_output, logsumexp,
        d_query, d_key, d_value,
        kernel_size, stride, dilation, is_causal,
        attn_scale, qkv_shape, num_extra_kv);
}

static void metal_na2d_backward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& output,
    const at::Tensor& d_output,
    const at::Tensor& logsumexp,
    at::Tensor& d_query,
    at::Tensor& d_key,
    at::Tensor& d_value,
    const std::tuple<int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t>& dilation,
    const std::tuple<bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t>& qkv_shape,
    int num_extra_kv) {
    TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");
    metal_na_backward_impl(
        query, key, value, output, d_output, logsumexp,
        d_query, d_key, d_value,
        kernel_size, stride, dilation, is_causal,
        attn_scale, qkv_shape, num_extra_kv);
}

static void metal_na3d_backward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& output,
    const at::Tensor& d_output,
    const at::Tensor& logsumexp,
    at::Tensor& d_query,
    at::Tensor& d_key,
    at::Tensor& d_value,
    const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
    const std::tuple<int32_t, int32_t, int32_t>& stride,
    const std::tuple<int32_t, int32_t, int32_t>& dilation,
    const std::tuple<bool, bool, bool>& is_causal,
    float attn_scale,
    const std::tuple<int32_t, int32_t, int32_t>& qkv_shape,
    int num_extra_kv) {
    TORCH_CHECK(query.dim() == 4, "Tensors must be 4-D.");
    metal_na_backward_impl(
        query, key, value, output, d_output, logsumexp,
        d_query, d_key, d_value,
        kernel_size, stride, dilation, is_causal,
        attn_scale, qkv_shape, num_extra_kv);
}

PYBIND11_MODULE(_metal_natten, m) {
    m.doc() = "NATTEN Metal backend for MPS (Apple Silicon)";

    m.def("metal_na1d_forward", &metal_na1d_forward, "Metal NA1D forward (MPS)");
    m.def("metal_na2d_forward", &metal_na2d_forward, "Metal NA2D forward (MPS)");
    m.def("metal_na3d_forward", &metal_na3d_forward, "Metal NA3D forward (MPS)");

    m.def("metal_na1d_backward", &metal_na1d_backward, "Metal NA1D backward (MPS)");
    m.def("metal_na2d_backward", &metal_na2d_backward, "Metal NA2D backward (MPS)");
    m.def("metal_na3d_backward", &metal_na3d_backward, "Metal NA3D backward (MPS)");
}

