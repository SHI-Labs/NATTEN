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
#include <natten/cuda/fmha_blackwell/fmha_forward.cuh> 
#include <natten_autogen/cuda/blackwell_fmha/kernels.h> 
namespace natten { 
namespace cuda { 
namespace fmha_blackwell { 
#define DISPATCH_BLACKWELL_FMHA_FORWARD_float16_headdim32(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_float16_256x128x32_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_float16_256x128x32(__VA_ARGS__); \
  } \
} \
    else { \
          std::cerr << "Blackwell FMHA kernel launch failed!" \
                << "Blackwell FMHA got invalid Q tile and KV tile combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_float16_headdim64(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_float16_256x128x64_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_float16_256x128x64(__VA_ARGS__); \
  } \
} \
    else { \
          std::cerr << "Blackwell FMHA kernel launch failed!" \
                << "Blackwell FMHA got invalid Q tile and KV tile combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_float16_headdim128(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_float16_256x128x128_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_float16_256x128x128(__VA_ARGS__); \
  } \
} \
    else { \
          std::cerr << "Blackwell FMHA kernel launch failed!" \
                << "Blackwell FMHA got invalid Q tile and KV tile combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_bfloat16_headdim32(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_bfloat16_256x128x32_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_bfloat16_256x128x32(__VA_ARGS__); \
  } \
} \
    else { \
          std::cerr << "Blackwell FMHA kernel launch failed!" \
                << "Blackwell FMHA got invalid Q tile and KV tile combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_bfloat16_headdim64(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_bfloat16_256x128x64_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_bfloat16_256x128x64(__VA_ARGS__); \
  } \
} \
    else { \
          std::cerr << "Blackwell FMHA kernel launch failed!" \
                << "Blackwell FMHA got invalid Q tile and KV tile combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FMHA_FORWARD_bfloat16_headdim128(q_tile_size, kv_tile_size, persistent, ...) \
  [&] { \
    if (q_tile_size == 256 && \
kv_tile_size == 128) { \
  if (persistent) { \
    natten::cuda::fmha_blackwell::blackwell_fmha_bfloat16_256x128x128_persistent(__VA_ARGS__); \
  } else { \
    natten::cuda::fmha_blackwell::blackwell_fmha_bfloat16_256x128x128(__VA_ARGS__); \
  } \
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

