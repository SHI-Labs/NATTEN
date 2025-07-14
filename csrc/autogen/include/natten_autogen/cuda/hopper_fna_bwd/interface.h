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
#include <natten/cuda/fna_hopper/fna_backward.cuh> 
#include <natten_autogen/cuda/hopper_fna_bwd/dispatch_dtype.h> 
namespace natten { 
namespace cuda { 
namespace fna_hopper { 
#define DISPATCH_HOPPER_FNA_BACKWARD(rank, dtype, dim, is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if constexpr (rank == 1) { \
      DISPATCH_HOPPER_FNA_BACKWARD_1D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if constexpr (rank == 2) { \
      DISPATCH_HOPPER_FNA_BACKWARD_2D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if constexpr (rank == 3) { \
      DISPATCH_HOPPER_FNA_BACKWARD_3D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "Hopper FNA kernel launch failed!" \
                << "NATTEN only supports NA1D, 2D, and 3D!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna_hopper 
#endif 
#endif 

