#pragma once

#include <iostream>
#include <ATen/ATen.h>

#include "cute/util/print.hpp"

#include "flash.h"
// #include "natten/cuda/flash_fmha/flash.h"

namespace natten {
namespace cuda {
namespace flash {

struct FlashBwdWorkspaceSize {

  int64_t softmax_lse_log2_size;
  int64_t dsoftmax_sum_size;
  int64_t dQ_accum_size;
  int64_t dQ_semaphore_size;

  int64_t softmax_lse_log2_bytes;
  int64_t dsoftmax_sum_bytes;
  int64_t dQ_accum_bytes;
  int64_t dQ_semaphore_bytes;

  int64_t total_bytes;

};


struct FlashBwdWorkspacePtr {

  void* softmax_lse_log2_ptr;
  void* dsoftmax_sum_ptr;
  void* dQ_accum_ptr;
  void* dQ_semaphore_ptr;
  void* total_ptr;

};


inline int inline_round_up(int spatial_ext, int tile_size) {
  return ((spatial_ext + tile_size - 1) / tile_size) * tile_size;
}


Flash_fwd_params set_flash_fwd_params(
    int cc,
    int num_sm,
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::optional<at::Tensor>& logsumexp,
    int query_tile_size,
    int key_tile_size,
    float attn_scale) {

  Flash_fwd_params params = {};

  int B = query.size(0);
  int Q = query.size(1);
  int Q_rounded = inline_round_up(Q, query_tile_size);
  int K = key.size(1);
  int H = query.size(2);
  int D = query.size(3);
  int D_v = value.size(3);

  params.is_bf16 = query.scalar_type() == torch::kBFloat16;
  params.is_e4m3 = false;

  // Set the pointers and strides.
  params.q_ptr = static_cast<void*>(query.data_ptr());
  params.k_ptr = static_cast<void*>(key.data_ptr());
  params.v_ptr = static_cast<void*>(value.data_ptr());
  params.o_ptr = static_cast<void*>(out.data_ptr());
  // Softmax sum
  params.softmax_lse_ptr = static_cast<void*>(logsumexp.value().data_ptr());

  // All stride are in elements, not bytes.
  params.q_row_stride = query.stride(1);
  params.k_row_stride = key.stride(1);
  params.v_row_stride = value.stride(1);
  params.o_row_stride = out.stride(1);

  params.q_head_stride = query.stride(2);
  params.k_head_stride = key.stride(2);
  params.v_head_stride = value.stride(2);
  params.o_head_stride = out.stride(2);

  params.q_batch_stride = query.stride(0);
  params.k_batch_stride = key.stride(0);
  params.v_batch_stride = value.stride(0);
  params.o_batch_stride = out.stride(0);

  params.v_dim_stride = value.stride(3);

  // Set the dimensions.
  params.b = B;
  params.h = H;
  params.h_k = H;
  params.seqlen_q = Q;
  params.seqlen_k = K;
  params.seqlen_q_rounded = inline_round_up(Q, query_tile_size);
  params.seqlen_k_rounded = inline_round_up(K, key_tile_size);
  params.d = D;
  params.dv = D_v;
  params.d_rounded = inline_round_up(D, 8);

  // Set the different scale values.
  params.scale_softmax = attn_scale;

  // Dropout is not supported
  params.p_dropout = 1.f;
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;

  params.arch = cc;
  params.num_sm = num_sm;

  return params;
}


FlashBwdWorkspaceSize get_flash_bwd_workspace_size(
    int B,
    int Q,
    int K,
    int H,
    int D,
    int Q_tile_size,
    int K_tile_size
    ) {

  static constexpr int64_t kFullBytes = 4;
  static constexpr int64_t kIntBytes = 4;

  int Q_rounded = inline_round_up(Q, Q_tile_size);

  int64_t BHQr = B * H * (int64_t)Q_rounded;

  FlashBwdWorkspaceSize workspace_size;
  workspace_size.softmax_lse_log2_size = BHQr;
  workspace_size.dsoftmax_sum_size = BHQr;
  workspace_size.dQ_accum_size = BHQr * D;
  workspace_size.dQ_semaphore_size = BHQr;

  workspace_size.softmax_lse_log2_bytes = workspace_size.softmax_lse_log2_size * kFullBytes;
  workspace_size.dsoftmax_sum_bytes = workspace_size.dsoftmax_sum_size * kFullBytes;
  workspace_size.dQ_accum_bytes = workspace_size.dQ_accum_size * kFullBytes;
  workspace_size.dQ_semaphore_bytes = workspace_size.dQ_semaphore_size * kIntBytes;
  workspace_size.total_bytes = workspace_size.softmax_lse_log2_bytes + workspace_size.dsoftmax_sum_bytes +
    workspace_size.dQ_accum_bytes + workspace_size.dQ_semaphore_bytes;

  return workspace_size;

}

FlashBwdWorkspacePtr allocate_flash_bwd_workspace(
    void* workspace_ptr, FlashBwdWorkspaceSize workspace_size) {

  int64_t offset = 0;
  FlashBwdWorkspacePtr workspace;

  workspace.total_ptr = workspace_ptr;

  workspace.softmax_lse_log2_ptr = workspace_ptr;
  offset = offset + workspace_size.softmax_lse_log2_size; // increment offset by number of elements
  workspace_ptr = (void*)((float*)workspace_ptr + offset); // move by number of elements
  
  workspace.dsoftmax_sum_ptr = workspace_ptr;
  offset = offset + workspace_size.dsoftmax_sum_size; // increment offset by number of elements
  workspace_ptr = (void*)((float*)workspace_ptr + offset); // move by number of elements

  workspace.dQ_accum_ptr = workspace_ptr;
  offset = offset + workspace_size.dQ_accum_size; // increment offset by number of elements
  workspace_ptr = (void*)((float*)workspace_ptr + offset); // move by number of elements

  workspace.dQ_semaphore_ptr = workspace_ptr;
  offset = offset + workspace_size.dQ_semaphore_size; // increment offset by number of elements

  return workspace;
}


Flash_bwd_params set_flash_bwd_params(
    int cc,
    int num_sm,
    at::Tensor& dq,
    at::Tensor& dk,
    at::Tensor& dv,
    const at::Tensor& dout,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& out,
    const at::optional<at::Tensor>& logsumexp,
    void* softmax_lse_log2_ptr,
    void* dsoftmax_sum_ptr,
    void* dq_semaphore_ptr,
    void* dq_accum_ptr,
    int query_tile_size,
    int key_tile_size,
    float attn_scale) {

  // Allocate all pointers and dimensions.
  Flash_bwd_params params = {};

  int B = query.size(0);
  int Q = query.size(1);
  int Q_rounded = inline_round_up(Q, query_tile_size);
  int K = key.size(1);
  int H = query.size(2);
  int D = query.size(3);
  int D_v = value.size(3);

  params.q_ptr = static_cast<void*>(query.data_ptr());
  params.k_ptr = static_cast<void*>(key.data_ptr());
  params.v_ptr = static_cast<void*>(value.data_ptr());
  params.o_ptr = static_cast<void*>(out.data_ptr());
  params.dq_ptr = static_cast<void*>(dq.data_ptr());
  params.dk_ptr = static_cast<void*>(dk.data_ptr());
  params.dv_ptr = static_cast<void*>(dv.data_ptr());
  params.do_ptr = static_cast<void*>(dout.data_ptr());
  // Softmax sum
  params.softmax_lse_ptr = static_cast<void*>(logsumexp.value().data_ptr());
  params.softmax_lse_log2_ptr = softmax_lse_log2_ptr;
  params.dsoftmax_sum = dsoftmax_sum_ptr;
  params.dq_semaphore = static_cast<int*>(dq_semaphore_ptr);
  params.dq_accum_ptr = dq_accum_ptr;
  params.dk_accum_ptr = nullptr;
  params.dv_accum_ptr = nullptr;


  // All stride are in elements, not bytes.
  params.q_row_stride = query.stride(1);
  params.k_row_stride = key.stride(1);
  params.v_row_stride = value.stride(1);
  params.o_row_stride = out.stride(1);

  params.q_head_stride = query.stride(2);
  params.k_head_stride = key.stride(2);
  params.v_head_stride = value.stride(2);
  params.o_head_stride = out.stride(2);

  params.q_batch_stride = query.stride(0);
  params.k_batch_stride = key.stride(0);
  params.v_batch_stride = value.stride(0);
  params.o_batch_stride = out.stride(0);

  params.dq_row_stride = query.stride(1);
  params.dk_row_stride = key.stride(1);
  params.dv_row_stride = value.stride(1);
  params.do_row_stride = out.stride(1);

  params.dq_head_stride = query.stride(2);
  params.dk_head_stride = key.stride(2);
  params.dv_head_stride = value.stride(2);
  params.do_head_stride = out.stride(2);

  params.dq_batch_stride = query.stride(0);
  params.dk_batch_stride = key.stride(0);
  params.dv_batch_stride = value.stride(0);
  params.do_batch_stride = out.stride(0);

  params.v_dim_stride = value.stride(3);

  // Set the dimensions.
  params.b = B;
  params.h = H;
  params.h_k = H;
  params.seqlen_q = Q;
  params.seqlen_k = K;
  params.seqlen_q_rounded = inline_round_up(Q, query_tile_size);
  params.seqlen_k_rounded = inline_round_up(K, key_tile_size);
  params.d = D;
  params.d_rounded = D;
  params.dv = D;
  params.dv_rounded = D;

  int total_q = params.b * params.seqlen_q; 
  int total_k = params.b * params.seqlen_k; 
  params.total_q = total_q;
  params.total_k = total_k;

  // Set the different scale values.
  params.scale_softmax = attn_scale;
  // params.softcap = 0.f;

  params.p_dropout = 1.f;
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;

  // Hardcoded for A100
  params.arch = cc;
  params.num_sm = num_sm;

  return params;
  // Allocate pointers for dq_accum_ptr, softmax_lse_log2_ptr, dq_semaphore_ptr

}

} // namespace flash
} // namespace cuda
} // namespace natten
