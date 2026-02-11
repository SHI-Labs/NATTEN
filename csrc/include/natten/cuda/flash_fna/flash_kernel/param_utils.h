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
 ******************************************************************************/

#pragma once

#include <iostream>
#include <ATen/ATen.h>

#include "cute/util/print.hpp"

#include "flash.h"
// #include "natten/cuda/flash_fmha/flash_kernel/flash.h"

namespace natten {
namespace cuda {
namespace flash_fna {

struct FlashBwdWorkspaceSize {

  int64_t softmax_lse_log2_size = 0;
  int64_t dsoftmax_sum_size = 0;
  int64_t dQ_accum_size = 0;
  int64_t dQ_semaphore_size = 0;
  int64_t dK_semaphore_size = 0;
  int64_t dV_semaphore_size = 0;

  int64_t softmax_lse_log2_bytes = 0;
  int64_t dsoftmax_sum_bytes = 0;
  int64_t dQ_accum_bytes = 0;
  int64_t dQ_semaphore_bytes = 0;
  int64_t dK_semaphore_bytes = 0;
  int64_t dV_semaphore_bytes = 0;

  int64_t total_bytes = 0;

  bool has_kv_semaphores = false;

};


struct FlashBwdWorkspacePtr {

  void* softmax_lse_log2_ptr = nullptr;
  void* dsoftmax_sum_ptr = nullptr;
  void* dQ_accum_ptr = nullptr;
  int* dQ_semaphore_ptr = nullptr;
  int* dK_semaphore_ptr = nullptr;
  int* dV_semaphore_ptr = nullptr;
  void* total_ptr = nullptr;

};


inline int inline_round_up(int spatial_ext, int tile_size) {
  return ((spatial_ext + tile_size - 1) / tile_size) * tile_size;
}


template<class NADim>
Flash_fna_fwd_params<NADim> set_flash_fna_fwd_params(
    int cc,
    int num_sm,
    at::Tensor& out,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    at::optional<at::Tensor>& logsumexp,
    float attn_scale,
    NADim q_shape,
    NADim kv_shape,
    NADim qkv_shape,
    NADim window_size,
    NADim stride,
    NADim dilation
  ) {

  Flash_fna_fwd_params<NADim> params = {};

  int B = query.size(0);
  int Q = query.size(1);
  int K = key.size(1);
  int H = query.size(2);
  int D = query.size(3);
  int D_v = value.size(3);

  params.q_shape = q_shape;
  params.kv_shape = kv_shape;
  params.qkv_shape = qkv_shape;
  params.window_size = window_size;
  params.stride = stride;
  params.dilation = dilation;
  params.batch_size_actual = B / product(dilation);

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
  params.seqlen_q_rounded = Q; // NOTE (aditya): We know Q, K to be
  params.seqlen_k_rounded = K; // divisible by their resp tile sizes.
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
    int K_tile_size,
    bool deterministic
    ) {

  static constexpr int64_t kFullBytes = sizeof(float);
  static constexpr int64_t kIntBytes = sizeof(int);

  int Q_rounded = inline_round_up(Q, Q_tile_size);
  int K_rounded = inline_round_up(K, K_tile_size);

  int64_t BHQr = B * H * (int64_t)Q_rounded;
  int64_t BHKr = B * H * (int64_t)K_rounded;

  FlashBwdWorkspaceSize workspace_size;

  workspace_size.softmax_lse_log2_size = BHQr;
  workspace_size.dsoftmax_sum_size = BHQr;
  workspace_size.dQ_accum_size = BHQr * D;
  workspace_size.dQ_semaphore_size = BHQr / Q_tile_size;

  workspace_size.softmax_lse_log2_bytes = workspace_size.softmax_lse_log2_size * kFullBytes;
  workspace_size.dsoftmax_sum_bytes = workspace_size.dsoftmax_sum_size * kFullBytes;
  workspace_size.dQ_accum_bytes = workspace_size.dQ_accum_size * kFullBytes;
  workspace_size.dQ_semaphore_bytes = workspace_size.dQ_semaphore_size * kIntBytes;
  workspace_size.total_bytes = (
      workspace_size.softmax_lse_log2_bytes +
      workspace_size.dsoftmax_sum_bytes +
      workspace_size.dQ_accum_bytes +
      workspace_size.dQ_semaphore_bytes
  );

  workspace_size.has_kv_semaphores = false;

  if (deterministic) {
    workspace_size.dK_semaphore_size = BHKr / K_tile_size;
    workspace_size.dV_semaphore_size = BHKr / K_tile_size;

    workspace_size.dK_semaphore_bytes = workspace_size.dK_semaphore_size * kIntBytes;
    workspace_size.dV_semaphore_bytes = workspace_size.dV_semaphore_size * kIntBytes;

    workspace_size.total_bytes += workspace_size.dK_semaphore_bytes;
    workspace_size.total_bytes += workspace_size.dV_semaphore_bytes;

    workspace_size.has_kv_semaphores = true;
  }

  // workspace_size.print_debug();

  return workspace_size;

}

FlashBwdWorkspacePtr allocate_flash_bwd_workspace(
    void* workspace_ptr, FlashBwdWorkspaceSize workspace_size) {

  FlashBwdWorkspacePtr workspace;

  // Marks the start of workspace
  workspace.total_ptr = workspace_ptr;

  // Start pointing and moving workspace pointer after every allocation
  workspace.softmax_lse_log2_ptr = static_cast<void*>(workspace_ptr);
  workspace_ptr = static_cast<char*>(workspace_ptr) + workspace_size.softmax_lse_log2_bytes; // move by number of bytes

  workspace.dsoftmax_sum_ptr = static_cast<void*>(workspace_ptr);
  workspace_ptr = static_cast<char*>(workspace_ptr) + workspace_size.dsoftmax_sum_bytes;

  workspace.dQ_accum_ptr = static_cast<void*>(workspace_ptr);
  workspace_ptr = static_cast<char*>(workspace_ptr) + workspace_size.dQ_accum_bytes;

  workspace.dQ_semaphore_ptr = static_cast<int*>(workspace_ptr);
  workspace_ptr = static_cast<char*>(workspace_ptr) + workspace_size.dQ_semaphore_bytes;
  
  if (workspace_size.has_kv_semaphores) {
    workspace.dK_semaphore_ptr = static_cast<int*>(workspace_ptr);
    workspace_ptr = static_cast<char*>(workspace_ptr) + workspace_size.dK_semaphore_bytes;

    workspace.dV_semaphore_ptr = static_cast<int*>(workspace_ptr);
    workspace_ptr = static_cast<char*>(workspace_ptr) + workspace_size.dV_semaphore_bytes;

  }

  // workspace.print_debug();

  return workspace;
}


template<class NADim>
Flash_fna_bwd_params<NADim> set_flash_fna_bwd_params(
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
    const at::Tensor& logsumexp,
    void* softmax_lse_log2_ptr,
    void* dsoftmax_sum_ptr,
    void* dq_semaphore_ptr,
    void* dk_semaphore_ptr,
    void* dv_semaphore_ptr,
    void* dq_accum_ptr,
    float attn_scale,
    NADim q_shape,
    NADim kv_shape,
    NADim qkv_shape,
    NADim window_size,
    NADim stride,
    NADim dilation,
    bool deterministic) {

  // Allocate all pointers and dimensions.
  Flash_fna_bwd_params<NADim> params = {};

  int B = query.size(0);
  int Q = query.size(1);
  int K = key.size(1);
  int H = query.size(2);
  int D = query.size(3);
  int D_v = value.size(3);

  params.q_shape = q_shape;
  params.kv_shape = kv_shape;
  params.qkv_shape = qkv_shape;
  params.window_size = window_size;
  params.stride = stride;
  params.dilation = dilation;
  params.batch_size_actual = B / product(dilation);

  params.q_ptr = static_cast<void*>(query.data_ptr());
  params.k_ptr = static_cast<void*>(key.data_ptr());
  params.v_ptr = static_cast<void*>(value.data_ptr());
  params.o_ptr = static_cast<void*>(out.data_ptr());
  params.dq_ptr = static_cast<void*>(dq.data_ptr());
  params.dk_ptr = static_cast<void*>(dk.data_ptr());
  params.dv_ptr = static_cast<void*>(dv.data_ptr());
  params.do_ptr = static_cast<void*>(dout.data_ptr());
  // Softmax sum
  params.softmax_lse_ptr = static_cast<void*>(logsumexp.data_ptr());
  params.softmax_lse_log2_ptr = softmax_lse_log2_ptr;
  params.dsoftmax_sum = dsoftmax_sum_ptr;
  params.dq_semaphore = static_cast<int*>(dq_semaphore_ptr);
  params.dq_accum_ptr = dq_accum_ptr;
  params.dk_accum_ptr = nullptr;
  params.dv_accum_ptr = nullptr;

  if (deterministic){
    params.dk_semaphore = static_cast<int*>(dk_semaphore_ptr);
    params.dv_semaphore = static_cast<int*>(dv_semaphore_ptr);
  }
  else {
    params.dk_semaphore = nullptr;
    params.dv_semaphore = nullptr;
  }


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
  params.seqlen_q_rounded = Q;
  params.seqlen_k_rounded = K;
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

  params.arch = cc;
  params.num_sm = num_sm;

  return params;
  // Allocate pointers for dq_accum_ptr, softmax_lse_log2_ptr, dq_semaphore_ptr

}

} // namespace flash_fna
} // namespace cuda
} // namespace natten
