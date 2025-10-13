// This file is added to speed up development iterations. We compile only a subset of kernels and
// test them. This file is NOT auto generated.

// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
// Splitting the different template instantiations to different files to speed up compilation.

#pragma once

#include <iostream>

#include "cute/util/print.hpp"
#include "flash.h"

using namespace natten::cuda::flash;
using namespace cute;

void print_strides(const Flash_bwd_params& params) {
    std::cout << "q_ptr     = " << params.q_ptr << "\n";
    std::cout << "k_ptr     = " << params.k_ptr << "\n";
    std::cout << "v_ptr     = " << params.v_ptr << "\n";
    std::cout << "o_ptr     = " << params.o_ptr << "\n";
    std::cout << "q_row_stride     = " << params.q_row_stride << "\n";
    std::cout << "k_row_stride     = " << params.k_row_stride << "\n";
    std::cout << "v_row_stride     = " << params.v_row_stride << "\n";
    std::cout << "q_head_stride    = " << params.q_head_stride << "\n";
    std::cout << "k_head_stride    = " << params.k_head_stride << "\n";
    std::cout << "v_head_stride    = " << params.v_head_stride << "\n";
    std::cout << "v_dim_stride     = " << params.v_dim_stride << "\n";
    std::cout << "o_row_stride     = " << params.o_row_stride << "\n";
    std::cout << "o_head_stride    = " << params.o_head_stride << "\n";
    std::cout << "q_batch_stride   = " << params.q_batch_stride << "\n";
    std::cout << "o_batch_stride   = " << params.o_batch_stride << "\n";
    std::cout << "k_batch_stride   = " << params.k_batch_stride << "\n";
    std::cout << "v_batch_stride   = " << params.v_batch_stride << "\n";
}

template<
  class StrideQ,
  class StrideK,
  class StrideV,
  class StrideO,
  class StrideLSE,
  class ProblemShapeType
>
Flash_bwd_params set_flash_bwd_params_for_testing(
    const ProblemShapeType& problem_size,
    void* q_ptr,
    void* k_ptr,
    void* v_ptr,
    void* o_ptr,
    void* lse_ptr,
    void* dq_ptr,
    void* dk_ptr,
    void* dv_ptr,
    void* do_ptr,
    void* dsoftmax_sum_ptr,
    void* softmax_lse_log2_ptr,
    void* dq_accum_ptr,
    int* dq_semaphore_ptr,
    StrideQ stride_Q, // same stride for dQ
    StrideK stride_K, // same stride for dK
    StrideV stride_V, // same stride for dV
    StrideO stride_O, // same stride for dO
    StrideLSE stride_LSE,
    float softmax_scale
    ){

  auto [B, H, Q, K, D] = problem_size;

  Flash_bwd_params params = {};

  params.is_bf16 = false;
  params.is_e4m3 = false;

  // Set the pointers and strides.
  params.q_ptr = q_ptr;
  params.k_ptr = k_ptr;
  params.v_ptr = v_ptr;
  params.o_ptr = o_ptr;
  params.dq_ptr = dq_ptr;
  params.dk_ptr = dk_ptr;
  params.dv_ptr = dv_ptr;
  params.do_ptr = do_ptr;
  // Softmax sum
  params.softmax_lse_ptr = lse_ptr;
  params.softmax_lse_log2_ptr = softmax_lse_log2_ptr;
  params.dsoftmax_sum = dsoftmax_sum_ptr;
  params.dq_semaphore = dq_semaphore_ptr;

  // All stride are in elements, not bytes.
  params.q_row_stride = get<2>(stride_Q);
  params.k_row_stride = get<2>(stride_K);
  params.v_row_stride = get<2>(stride_V);
  params.o_row_stride = get<2>(stride_O);

  params.q_head_stride = get<1>(stride_Q);
  params.k_head_stride = get<1>(stride_K);
  params.v_head_stride = get<1>(stride_V);
  params.o_head_stride = get<1>(stride_O);

  params.q_batch_stride = get<0>(stride_Q); 
  params.o_batch_stride = get<0>(stride_O); 
  params.k_batch_stride = get<0>(stride_K); 
  params.v_batch_stride = get<0>(stride_V); 

  // Copy ever same strides for dq, dk, dv, do
  params.dq_row_stride = get<2>(stride_Q);
  params.dk_row_stride = get<2>(stride_K);
  params.dv_row_stride = get<2>(stride_V);
  params.do_row_stride = get<2>(stride_O);

  params.dq_head_stride = get<1>(stride_Q);
  params.dk_head_stride = get<1>(stride_K);
  params.dv_head_stride = get<1>(stride_V);
  params.do_head_stride = get<1>(stride_O);

  params.dq_batch_stride = get<0>(stride_Q); 
  params.do_batch_stride = get<0>(stride_O); 
  params.dk_batch_stride = get<0>(stride_K); 
  params.dv_batch_stride = get<0>(stride_V); 

  params.dq_accum_ptr = dq_accum_ptr;
  params.dk_accum_ptr = nullptr;
  params.dv_accum_ptr = nullptr;

  // params.cu_seqlens_q = nullptr; 
  // params.cu_seqlens_k = nullptr;
  // params.seqused_q = nullptr;
  // params.seqused_k = nullptr;


  // Set the dimensions.
  params.b = B;
  params.h = H;
  params.h_k = H;
  params.seqlen_q = Q;
  params.seqlen_k = K;
  params.seqlen_q_rounded = cutlass::round_up(Q, 64);
  params.seqlen_k_rounded = cutlass::round_up(K, 128);
  params.d = D;
  params.d_rounded = D;
  params.dv = D;
  params.dv_rounded = D;

  int total_q = params.b * params.seqlen_q; 
  int total_k = params.b * params.seqlen_k; 
  params.total_q = total_q;
  params.total_k = total_k;

  // Set the different scale values.
  params.scale_softmax = softmax_scale;
  // params.softcap = 0.f;

  // Set this to probability of keeping an element to simplify things.
  params.p_dropout = 1.f;
  // Convert p from float to int so we don't have to convert the random uint to float to compare.
  // [Minor] We want to round down since when we do the comparison we use <= instead of <
  // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
  // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
  params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
  params.rp_dropout = 1.f / params.p_dropout;

  // Hardcoded for A100
  params.arch = 80;
  params.num_sm = 108;

  return params;
}
