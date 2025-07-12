#pragma once


#include <iostream> 
#include <type_traits> 
#ifdef NATTEN_WITH_CUTLASS
#ifdef NATTEN_WITH_BLACKWELL_FNA
#include <natten/natten.h> 
#include <ATen/ATen.h> 
#include <ATen/cuda/CUDAContext.h> 
#include <c10/cuda/CUDAGuard.h> 
#include <c10/cuda/CUDAStream.h> 
#include <torch/extension.h> 
#include <natten/natten.h> 
#include <natten/helpers.h> 
#include <natten/cuda/fmha_blackwell/fmha_backward.cuh> 
#include <natten_autogen/cuda/blackwell_fmha_bwd/dispatch_head_dim.h> 
namespace natten { 
namespace cuda { 
namespace fmha_blackwell { 
#define DISPATCH_BLACKWELL_FMHA_BACKWARD(dtype, dim, q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (dtype == torch::kFloat16) { \
      DISPATCH_BLACKWELL_FMHA_BACKWARD_float16(dim, q_tile_size, kv_tile_size, __VA_ARGS__); \
    } \
    else if (dtype == torch::kBFloat16) { \
      DISPATCH_BLACKWELL_FMHA_BACKWARD_bfloat16(dim, q_tile_size, kv_tile_size, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "Blackwell FMHA kernel launch failed!" \
                << "Blackwell FMHA does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fmha_blackwell 
#endif 
#endif 

