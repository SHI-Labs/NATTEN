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
#include <natten_autogen/cuda/blackwell_fmha_bwd/kernels.h> 
namespace natten { 
namespace cuda { 
namespace fmha_blackwell { 
#define DISPATCH_BLACKWELL_FMHA_BACKWARD_float16_headdim32(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_blackwell::blackwell_fmha_backward_float16_128x128x32(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Blackwell FMHA kernel launch failed!" \
                << "Blackwell FMHA got invalid Q tile and KV tile combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_BACKWARD_float16_headdim64(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_blackwell::blackwell_fmha_backward_float16_128x128x64(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Blackwell FMHA kernel launch failed!" \
                << "Blackwell FMHA got invalid Q tile and KV tile combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_BACKWARD_float16_headdim128(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_blackwell::blackwell_fmha_backward_float16_128x128x128(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Blackwell FMHA kernel launch failed!" \
                << "Blackwell FMHA got invalid Q tile and KV tile combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_BACKWARD_bfloat16_headdim32(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_blackwell::blackwell_fmha_backward_bfloat16_128x128x32(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Blackwell FMHA kernel launch failed!" \
                << "Blackwell FMHA got invalid Q tile and KV tile combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_BACKWARD_bfloat16_headdim64(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_blackwell::blackwell_fmha_backward_bfloat16_128x128x64(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Blackwell FMHA kernel launch failed!" \
                << "Blackwell FMHA got invalid Q tile and KV tile combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_BACKWARD_bfloat16_headdim128(q_tile_size, kv_tile_size, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 128) { \
  natten::cuda::fmha_blackwell::blackwell_fmha_backward_bfloat16_128x128x128(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Blackwell FMHA kernel launch failed!" \
                << "Blackwell FMHA got invalid Q tile and KV tile combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fmha_blackwell 
#endif 
#endif 

