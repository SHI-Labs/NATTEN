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
#include <natten_autogen/cuda/hopper_fmha/kernels.h> 
namespace natten { 
namespace cuda { 
namespace fmha_hopper { 
#define DISPATCH_HOPPER_FMHA_FORWARD_float16_headdim32(q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (q_tile_size == 64 && \
kv_tile_size == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fmha_hopper::hopper_fmha_float16_64x128x32(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FMHA kernel launch failed!" \
                << "Hopper FMHA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_FORWARD_float16_headdim64(q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (q_tile_size == 64 && \
kv_tile_size == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fmha_hopper::hopper_fmha_float16_64x128x64(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FMHA kernel launch failed!" \
                << "Hopper FMHA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_FORWARD_float16_headdim128(q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fmha_hopper::hopper_fmha_float16_128x128x128_coop(__VA_ARGS__); \
} \
    else if (q_tile_size == 128 && \
kv_tile_size == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fmha_hopper::hopper_fmha_float16_128x128x128_pp(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FMHA kernel launch failed!" \
                << "Hopper FMHA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_FORWARD_float16_headdim256(q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 64 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fmha_hopper::hopper_fmha_float16_128x64x256_coop(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FMHA kernel launch failed!" \
                << "Hopper FMHA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_FORWARD_bfloat16_headdim32(q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (q_tile_size == 64 && \
kv_tile_size == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fmha_hopper::hopper_fmha_bfloat16_64x128x32(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FMHA kernel launch failed!" \
                << "Hopper FMHA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_FORWARD_bfloat16_headdim64(q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (q_tile_size == 64 && \
kv_tile_size == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fmha_hopper::hopper_fmha_bfloat16_64x128x64(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FMHA kernel launch failed!" \
                << "Hopper FMHA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_FORWARD_bfloat16_headdim128(q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fmha_hopper::hopper_fmha_bfloat16_128x128x128_coop(__VA_ARGS__); \
} \
    else if (q_tile_size == 128 && \
kv_tile_size == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fmha_hopper::hopper_fmha_bfloat16_128x128x128_pp(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FMHA kernel launch failed!" \
                << "Hopper FMHA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FMHA_FORWARD_bfloat16_headdim256(q_tile_size, kv_tile_size, kernel_type, ...) \
  [&] { \
    if (q_tile_size == 128 && \
kv_tile_size == 64 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fmha_hopper::hopper_fmha_bfloat16_128x64x256_coop(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FMHA kernel launch failed!" \
                << "Hopper FMHA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fmha_hopper 
#endif 
#endif 

