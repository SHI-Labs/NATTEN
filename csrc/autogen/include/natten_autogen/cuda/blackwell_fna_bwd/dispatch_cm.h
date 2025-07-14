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
#include <natten/cuda/fna_blackwell/fna_backward.cuh> 
#include <natten_autogen/cuda/blackwell_fna_bwd/dispatch_tile_shape.h> 
namespace natten { 
namespace cuda { 
namespace fna_blackwell { 
#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim32_causal0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim32_causal1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim64_causal0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim64_causal1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim128_causal0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_float16_headdim128_causal1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim32_causal0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim32_causal1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim64_causal0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim64_causal1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim128_causal0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_1D_bfloat16_headdim128_causal1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim32_causal0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim32_causal0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim32_causal1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim32_causal1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim64_causal0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim64_causal0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim64_causal1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim64_causal1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim128_causal0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim128_causal0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim128_causal1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_float16_headdim128_causal1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim32_causal0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim32_causal0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim32_causal1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim32_causal1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim64_causal0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim64_causal0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim64_causal1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim64_causal1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim128_causal0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim128_causal0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim128_causal1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_2D_bfloat16_headdim128_causal1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32_causal0x0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32_causal0x0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32_causal0x1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32_causal0x1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32_causal1x0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32_causal1x0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32_causal1x1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim32_causal1x1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64_causal0x0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64_causal0x0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64_causal0x1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64_causal0x1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64_causal1x0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64_causal1x0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64_causal1x1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim64_causal1x1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128_causal0x0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128_causal0x0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128_causal0x1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128_causal0x1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128_causal1x0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128_causal1x0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128_causal1x1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_float16_headdim128_causal1x1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32_causal0x0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32_causal0x0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32_causal0x1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32_causal0x1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32_causal1x0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32_causal1x0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32_causal1x1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim32_causal1x1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64_causal0x0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64_causal0x0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64_causal0x1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64_causal0x1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64_causal1x0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64_causal1x0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64_causal1x1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim64_causal1x1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128(is_causal, q_tile_shape, kv_tile_shape, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128_causal0x0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128_causal0x0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128_causal0x1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128_causal0x1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128_causal1x0x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128_causal1x0x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128_causal1x1x0(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
      DISPATCH_BLACKWELL_FNA_BACKWARD_3D_bfloat16_headdim128_causal1x1x1(q_tile_shape, kv_tile_shape, __VA_ARGS__); \
    } \
    else { \
          std::cerr << "Blackwell FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna_blackwell 
#endif 
#endif 

