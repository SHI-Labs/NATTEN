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
#include <natten/cuda/fna_blackwell/fna_forward.cuh> 
#include <natten_autogen/cuda/blackwell_fna/dispatch_dtype.h> 
namespace natten { 
namespace cuda { 
namespace fna_blackwell { 
#define DISPATCH_BLACKWELL_FNA_FORWARD(rank, dtype, dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \
  [&] { \
    if constexpr (rank == 1) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_1D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if constexpr (rank == 2) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_2D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else if constexpr (rank == 3) { \
      DISPATCH_BLACKWELL_FNA_FORWARD_3D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "Blackwell FNA kernel launch failed!" \
                << "NATTEN only supports NA1D, 2D, and 3D!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna_blackwell 
#endif 
#endif 

