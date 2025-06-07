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
#include <natten/cuda/fmha_hopper/fmha_forward.cuh> 
#include <natten_autogen/cuda/hopper_fmha/dispatch_tile_size.h> 
namespace natten { 
namespace cuda { 
namespace fmha_hopper { 
#define DISPATCH_HOPPER_FMHA_FORWARD_float16(dim, q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (dim == 32) { \
      DISPATCH_HOPPER_FMHA_FORWARD_float16_headdim32(q_tile_size, kv_tile_size, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 64) { \
      DISPATCH_HOPPER_FMHA_FORWARD_float16_headdim64(q_tile_size, kv_tile_size, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 128) { \
      DISPATCH_HOPPER_FMHA_FORWARD_float16_headdim128(q_tile_size, kv_tile_size, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 256) { \
      DISPATCH_HOPPER_FMHA_FORWARD_float16_headdim256(q_tile_size, kv_tile_size, kernel_type, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "Hopper FMHA kernel launch failed!" \
                << "Hopper FMHA does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_FORWARD_bfloat16(dim, q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (dim == 32) { \
      DISPATCH_HOPPER_FMHA_FORWARD_bfloat16_headdim32(q_tile_size, kv_tile_size, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 64) { \
      DISPATCH_HOPPER_FMHA_FORWARD_bfloat16_headdim64(q_tile_size, kv_tile_size, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 128) { \
      DISPATCH_HOPPER_FMHA_FORWARD_bfloat16_headdim128(q_tile_size, kv_tile_size, kernel_type, __VA_ARGS__); \
    } \
    else if (dim == 256) { \
      DISPATCH_HOPPER_FMHA_FORWARD_bfloat16_headdim256(q_tile_size, kv_tile_size, kernel_type, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "Hopper FMHA kernel launch failed!" \
                << "Hopper FMHA does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fmha_hopper 
#endif 
#endif 

