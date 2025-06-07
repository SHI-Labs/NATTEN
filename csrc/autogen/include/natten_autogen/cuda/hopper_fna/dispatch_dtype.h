#pragma once


#include <iostream> 
#include <type_traits> 
#ifdef NATTEN_WITH_CUTLASS
#ifdef NATTEN_WITH_HOPPER_FNA
#include <natten/natten.h> 
#include <ATen/ATen.h> 
#include <ATen/cuda/CUDAContext.h> 
#include <c10/cuda/CUDAGuard.h> 
#include <c10/cuda/CUDAStream.h> 
#include <torch/extension.h> 
#include <natten/natten.h> 
#include <natten/helpers.h> 
#include <natten/cuda/fna_hopper/fna_forward.cuh> 
#include <natten_autogen/cuda/hopper_fna/dispatch_head_dim.h> 
namespace natten { 
namespace cuda { 
namespace fna_hopper { 
#define DISPATCH_HOPPER_FNA_FORWARD_1D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (dtype == torch::kFloat16) { \
      DISPATCH_HOPPER_FNA_FORWARD_1D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dtype == torch::kBFloat16) { \
      DISPATCH_HOPPER_FNA_FORWARD_1D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA-1D does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (dtype == torch::kFloat16) { \
      DISPATCH_HOPPER_FNA_FORWARD_2D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dtype == torch::kBFloat16) { \
      DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA-2D does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (dtype == torch::kFloat16) { \
      DISPATCH_HOPPER_FNA_FORWARD_3D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else if (dtype == torch::kBFloat16) { \
      DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, kernel_type, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA-3D does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna_hopper 
#endif 
#endif 

