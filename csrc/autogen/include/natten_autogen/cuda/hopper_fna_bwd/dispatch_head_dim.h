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
#include <natten_autogen/cuda/hopper_fna_bwd/dispatch_cm.h> 
namespace natten { 
namespace cuda { 
namespace fna_hopper { 
#define DISPATCH_HOPPER_FNA_BACKWARD_1D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (dim == 32) { \
      DISPATCH_HOPPER_FNA_BACKWARD_1D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim == 64) { \
      DISPATCH_HOPPER_FNA_BACKWARD_1D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim == 128) { \
      DISPATCH_HOPPER_FNA_BACKWARD_1D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA-1D does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_1D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (dim == 32) { \
      DISPATCH_HOPPER_FNA_BACKWARD_1D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim == 64) { \
      DISPATCH_HOPPER_FNA_BACKWARD_1D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim == 128) { \
      DISPATCH_HOPPER_FNA_BACKWARD_1D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA-1D does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (dim == 32) { \
      DISPATCH_HOPPER_FNA_BACKWARD_2D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim == 64) { \
      DISPATCH_HOPPER_FNA_BACKWARD_2D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim == 128) { \
      DISPATCH_HOPPER_FNA_BACKWARD_2D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA-2D does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_2D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (dim == 32) { \
      DISPATCH_HOPPER_FNA_BACKWARD_2D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim == 64) { \
      DISPATCH_HOPPER_FNA_BACKWARD_2D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim == 128) { \
      DISPATCH_HOPPER_FNA_BACKWARD_2D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA-2D does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_float16(dim, is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (dim == 32) { \
      DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim == 64) { \
      DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim == 128) { \
      DISPATCH_HOPPER_FNA_BACKWARD_3D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA-3D does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16(dim, is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (dim == 32) { \
      DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim == 64) { \
      DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (dim == 128) { \
      DISPATCH_HOPPER_FNA_BACKWARD_3D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \
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

