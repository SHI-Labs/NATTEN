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
#include <natten_autogen/cuda/hopper_fna/kernels.h> 
namespace natten { 
namespace cuda { 
namespace fna_hopper { 
#define DISPATCH_HOPPER_FNA_FORWARD_1D_float16_headdim32_causal0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna1d_float16_64x128x32_Q64_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_1D_float16_headdim32_causal1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna1d_float16_64x128x32_Q64_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_1D_float16_headdim64_causal0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna1d_float16_64x128x64_Q64_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_1D_float16_headdim64_causal1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna1d_float16_64x128x64_Q64_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_1D_float16_headdim128_causal0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna1d_float16_128x128x128_coop_Q128_KV128_causal0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna1d_float16_128x128x128_pp_Q128_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_1D_float16_headdim128_causal1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna1d_float16_128x128x128_coop_Q128_KV128_causal1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna1d_float16_128x128x128_pp_Q128_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_1D_float16_headdim256_causal0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 64 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna1d_float16_128x64x256_coop_Q128_KV64_causal0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_1D_float16_headdim256_causal1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 64 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna1d_float16_128x64x256_coop_Q128_KV64_causal1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_1D_bfloat16_headdim32_causal0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna1d_bfloat16_64x128x32_Q64_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_1D_bfloat16_headdim32_causal1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna1d_bfloat16_64x128x32_Q64_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_1D_bfloat16_headdim64_causal0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna1d_bfloat16_64x128x64_Q64_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_1D_bfloat16_headdim64_causal1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 64 && \
cute::get<0>(kv_tile_shape) == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna1d_bfloat16_64x128x64_Q64_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_1D_bfloat16_headdim128_causal0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna1d_bfloat16_128x128x128_coop_Q128_KV128_causal0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna1d_bfloat16_128x128x128_pp_Q128_KV128_causal0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_1D_bfloat16_headdim128_causal1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna1d_bfloat16_128x128x128_coop_Q128_KV128_causal1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 128 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna1d_bfloat16_128x128x128_pp_Q128_KV128_causal1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_1D_bfloat16_headdim256_causal0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 64 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna1d_bfloat16_128x64x256_coop_Q128_KV64_causal0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_1D_bfloat16_headdim256_causal1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 128 && \
cute::get<0>(kv_tile_shape) == 64 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna1d_bfloat16_128x64x256_coop_Q128_KV64_causal1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim32_causal0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_64x128x32_Q8x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_64x128x32_Q8x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim32_causal0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_64x128x32_Q8x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_64x128x32_Q8x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim32_causal1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_64x128x32_Q8x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_64x128x32_Q8x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim32_causal1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_64x128x32_Q8x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_64x128x32_Q8x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim64_causal0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_64x128x64_Q8x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_64x128x64_Q8x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim64_causal0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_64x128x64_Q8x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_64x128x64_Q8x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim64_causal1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_64x128x64_Q8x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_64x128x64_Q8x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim64_causal1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_64x128x64_Q8x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_64x128x64_Q8x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim128_causal0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_128x128x128_coop_Q16x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_128x128x128_pp_Q16x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim128_causal0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_128x128x128_coop_Q16x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_128x128x128_pp_Q16x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim128_causal1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_128x128x128_coop_Q16x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_128x128x128_pp_Q16x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim128_causal1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_128x128x128_coop_Q16x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_128x128x128_pp_Q16x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim256_causal0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_128x64x256_coop_Q16x8_KV8x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_128x64x256_coop_Q8x16_KV8x8_causal0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim256_causal0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_128x64x256_coop_Q16x8_KV8x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_128x64x256_coop_Q8x16_KV8x8_causal0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim256_causal1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_128x64x256_coop_Q16x8_KV8x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_128x64x256_coop_Q8x16_KV8x8_causal1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_float16_headdim256_causal1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_128x64x256_coop_Q16x8_KV8x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_float16_128x64x256_coop_Q8x16_KV8x8_causal1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim32_causal0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_64x128x32_Q8x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_64x128x32_Q8x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim32_causal0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_64x128x32_Q8x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_64x128x32_Q8x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim32_causal1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_64x128x32_Q8x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_64x128x32_Q8x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim32_causal1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_64x128x32_Q8x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_64x128x32_Q8x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim64_causal0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_64x128x64_Q8x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_64x128x64_Q8x8_KV8x16_causal0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim64_causal0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_64x128x64_Q8x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_64x128x64_Q8x8_KV8x16_causal0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim64_causal1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_64x128x64_Q8x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_64x128x64_Q8x8_KV8x16_causal1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim64_causal1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_64x128x64_Q8x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 16 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_64x128x64_Q8x8_KV8x16_causal1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim128_causal0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_128x128x128_coop_Q16x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_128x128x128_pp_Q16x8_KV16x8_causal0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim128_causal0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_128x128x128_coop_Q16x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_128x128x128_pp_Q16x8_KV16x8_causal0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim128_causal1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_128x128x128_coop_Q16x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_128x128x128_pp_Q16x8_KV16x8_causal1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim128_causal1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_128x128x128_coop_Q16x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 16 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_128x128x128_pp_Q16x8_KV16x8_causal1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim256_causal0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_128x64x256_coop_Q16x8_KV8x8_causal0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_128x64x256_coop_Q8x16_KV8x8_causal0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim256_causal0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_128x64x256_coop_Q16x8_KV8x8_causal0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_128x64x256_coop_Q8x16_KV8x8_causal0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim256_causal1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_128x64x256_coop_Q16x8_KV8x8_causal1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_128x64x256_coop_Q8x16_KV8x8_causal1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_2D_bfloat16_headdim256_causal1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 16 && cute::get<1>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_128x64x256_coop_Q16x8_KV8x8_causal1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 8 && cute::get<1>(q_tile_shape) == 16 && \
cute::get<0>(kv_tile_shape) == 8 && cute::get<1>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna2d_bfloat16_128x64x256_coop_Q8x16_KV8x8_causal1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim32_causal0x0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x32_Q4x4x4_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x32_Q4x4x4_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim32_causal0x0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x32_Q4x4x4_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x32_Q4x4x4_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim32_causal0x1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x32_Q4x4x4_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x32_Q4x4x4_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim32_causal0x1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x32_Q4x4x4_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x32_Q4x4x4_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim32_causal1x0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x32_Q4x4x4_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x32_Q4x4x4_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim32_causal1x0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x32_Q4x4x4_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x32_Q4x4x4_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim32_causal1x1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x32_Q4x4x4_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x32_Q4x4x4_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim32_causal1x1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x32_Q4x4x4_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x32_Q4x4x4_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim64_causal0x0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x64_Q4x4x4_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x64_Q4x4x4_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim64_causal0x0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x64_Q4x4x4_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x64_Q4x4x4_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim64_causal0x1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x64_Q4x4x4_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x64_Q4x4x4_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim64_causal0x1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x64_Q4x4x4_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x64_Q4x4x4_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim64_causal1x0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x64_Q4x4x4_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x64_Q4x4x4_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim64_causal1x0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x64_Q4x4x4_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x64_Q4x4x4_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim64_causal1x1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x64_Q4x4x4_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x64_Q4x4x4_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim64_causal1x1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x64_Q4x4x4_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_64x128x64_Q4x4x4_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim128_causal0x0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_coop_Q4x4x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_pp_Q4x4x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_coop_Q2x8x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_pp_Q2x8x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim128_causal0x0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_coop_Q4x4x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_pp_Q4x4x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_coop_Q2x8x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_pp_Q2x8x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim128_causal0x1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_coop_Q4x4x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_pp_Q4x4x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_coop_Q2x8x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_pp_Q2x8x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim128_causal0x1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_coop_Q4x4x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_pp_Q4x4x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_coop_Q2x8x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_pp_Q2x8x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim128_causal1x0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_coop_Q4x4x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_pp_Q4x4x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_coop_Q2x8x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_pp_Q2x8x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim128_causal1x0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_coop_Q4x4x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_pp_Q4x4x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_coop_Q2x8x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_pp_Q2x8x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim128_causal1x1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_coop_Q4x4x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_pp_Q4x4x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_coop_Q2x8x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_pp_Q2x8x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim128_causal1x1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_coop_Q4x4x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_pp_Q4x4x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_coop_Q2x8x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x128x128_pp_Q2x8x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim256_causal0x0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x64x256_coop_Q4x4x8_KV4x4x4_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x64x256_coop_Q2x8x8_KV4x4x4_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim256_causal0x0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x64x256_coop_Q4x4x8_KV4x4x4_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x64x256_coop_Q2x8x8_KV4x4x4_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim256_causal0x1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x64x256_coop_Q4x4x8_KV4x4x4_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x64x256_coop_Q2x8x8_KV4x4x4_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim256_causal0x1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x64x256_coop_Q4x4x8_KV4x4x4_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x64x256_coop_Q2x8x8_KV4x4x4_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim256_causal1x0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x64x256_coop_Q4x4x8_KV4x4x4_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x64x256_coop_Q2x8x8_KV4x4x4_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim256_causal1x0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x64x256_coop_Q4x4x8_KV4x4x4_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x64x256_coop_Q2x8x8_KV4x4x4_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim256_causal1x1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x64x256_coop_Q4x4x8_KV4x4x4_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x64x256_coop_Q2x8x8_KV4x4x4_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_float16_headdim256_causal1x1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x64x256_coop_Q4x4x8_KV4x4x4_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_float16_128x64x256_coop_Q2x8x8_KV4x4x4_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim32_causal0x0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim32_causal0x0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim32_causal0x1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim32_causal0x1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim32_causal1x0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim32_causal1x0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim32_causal1x1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim32_causal1x1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x32_Q4x4x4_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim64_causal0x0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x64_Q4x4x4_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x64_Q4x4x4_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim64_causal0x0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x64_Q4x4x4_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x64_Q4x4x4_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim64_causal0x1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x64_Q4x4x4_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x64_Q4x4x4_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim64_causal0x1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x64_Q4x4x4_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x64_Q4x4x4_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim64_causal1x0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x64_Q4x4x4_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x64_Q4x4x4_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim64_causal1x0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x64_Q4x4x4_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x64_Q4x4x4_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim64_causal1x1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x64_Q4x4x4_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x64_Q4x4x4_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim64_causal1x1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x64_Q4x4x4_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 4 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::NonPersistent) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_64x128x64_Q4x4x4_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim128_causal0x0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_coop_Q4x4x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_pp_Q4x4x8_KV4x4x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_coop_Q2x8x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_pp_Q2x8x8_KV2x8x8_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim128_causal0x0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_coop_Q4x4x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_pp_Q4x4x8_KV4x4x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_coop_Q2x8x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_pp_Q2x8x8_KV2x8x8_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim128_causal0x1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_coop_Q4x4x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_pp_Q4x4x8_KV4x4x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_coop_Q2x8x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_pp_Q2x8x8_KV2x8x8_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim128_causal0x1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_coop_Q4x4x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_pp_Q4x4x8_KV4x4x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_coop_Q2x8x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_pp_Q2x8x8_KV2x8x8_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim128_causal1x0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_coop_Q4x4x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_pp_Q4x4x8_KV4x4x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_coop_Q2x8x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_pp_Q2x8x8_KV2x8x8_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim128_causal1x0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_coop_Q4x4x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_pp_Q4x4x8_KV4x4x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_coop_Q2x8x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_pp_Q2x8x8_KV2x8x8_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim128_causal1x1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_coop_Q4x4x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_pp_Q4x4x8_KV4x4x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_coop_Q2x8x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_pp_Q2x8x8_KV2x8x8_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim128_causal1x1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_coop_Q4x4x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_pp_Q4x4x8_KV4x4x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_coop_Q2x8x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 2 && cute::get<1>(kv_tile_shape) == 8 && cute::get<2>(kv_tile_shape) == 8 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSPingpong) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x128x128_pp_Q2x8x8_KV2x8x8_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim256_causal0x0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x64x256_coop_Q4x4x8_KV4x4x4_causal0x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x64x256_coop_Q2x8x8_KV4x4x4_causal0x0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim256_causal0x0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x64x256_coop_Q4x4x8_KV4x4x4_causal0x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x64x256_coop_Q2x8x8_KV4x4x4_causal0x0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim256_causal0x1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x64x256_coop_Q4x4x8_KV4x4x4_causal0x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x64x256_coop_Q2x8x8_KV4x4x4_causal0x1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim256_causal0x1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x64x256_coop_Q4x4x8_KV4x4x4_causal0x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x64x256_coop_Q2x8x8_KV4x4x4_causal0x1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim256_causal1x0x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x64x256_coop_Q4x4x8_KV4x4x4_causal1x0x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x64x256_coop_Q2x8x8_KV4x4x4_causal1x0x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim256_causal1x0x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x64x256_coop_Q4x4x8_KV4x4x4_causal1x0x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x64x256_coop_Q2x8x8_KV4x4x4_causal1x0x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim256_causal1x1x0(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x64x256_coop_Q4x4x8_KV4x4x4_causal1x1x0(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x64x256_coop_Q2x8x8_KV4x4x4_causal1x1x0(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_HOPPER_FNA_FORWARD_3D_bfloat16_headdim256_causal1x1x1(q_tile_shape, kv_tile_shape, kernel_type, ...) \
  [&] { \
    if (cute::get<0>(q_tile_shape) == 4 && cute::get<1>(q_tile_shape) == 4 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x64x256_coop_Q4x4x8_KV4x4x4_causal1x1x1(__VA_ARGS__); \
} \
    else if (cute::get<0>(q_tile_shape) == 2 && cute::get<1>(q_tile_shape) == 8 && cute::get<2>(q_tile_shape) == 8 && \
cute::get<0>(kv_tile_shape) == 4 && cute::get<1>(kv_tile_shape) == 4 && cute::get<2>(kv_tile_shape) == 4 && \
kernel_type == natten::cuda::hopper::HopperKernelSchedule::WSCooperative) { \
  natten::cuda::fna_hopper::hopper_fna3d_bfloat16_128x64x256_coop_Q2x8x8_KV4x4x4_causal1x1x1(__VA_ARGS__); \
} \
    else { \
          std::cerr << "Hopper FNA kernel launch failed!" \
                << "Hopper FNA got invalid Q tile, KV tile, and schedule combination." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna_hopper 
#endif 
#endif 

