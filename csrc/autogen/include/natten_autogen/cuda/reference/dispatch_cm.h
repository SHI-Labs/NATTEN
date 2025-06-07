#pragma once


#include <iostream> 
#include <type_traits> 
#ifdef NATTEN_WITH_CUTLASS
#include <natten/natten.h> 
#include <ATen/ATen.h> 
#include <ATen/cuda/CUDAContext.h> 
#include <c10/cuda/CUDAGuard.h> 
#include <c10/cuda/CUDAStream.h> 
#include <torch/extension.h> 
#include <natten/natten.h> 
#include <natten/helpers.h> 
#include <natten/cuda/reference/fna_reference_forward.hpp> 
#include <natten/cuda/reference/fna_reference_backward.hpp> 
#include <natten_autogen/cuda/reference/kernels.h> 
namespace natten { 
namespace cuda { 
namespace reference { 
#define DISPATCH_REFERENCE_FNA_FORWARD_1D_float32(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
        natten::cuda::reference::reference_fna1d_float32_causal0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
        natten::cuda::reference::reference_fna1d_float32_causal1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_FORWARD_1D_float16(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
        natten::cuda::reference::reference_fna1d_float16_causal0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
        natten::cuda::reference::reference_fna1d_float16_causal1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_FORWARD_1D_bfloat16(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
        natten::cuda::reference::reference_fna1d_bfloat16_causal0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
        natten::cuda::reference::reference_fna1d_bfloat16_causal1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_BACKWARD_1D_float32(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
        natten::cuda::reference::reference_fna1d_backward_float32_causal0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
        natten::cuda::reference::reference_fna1d_backward_float32_causal1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_BACKWARD_1D_float16(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
        natten::cuda::reference::reference_fna1d_backward_float16_causal0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
        natten::cuda::reference::reference_fna1d_backward_float16_causal1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_BACKWARD_1D_bfloat16(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal)) { \
        natten::cuda::reference::reference_fna1d_backward_bfloat16_causal0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal)) { \
        natten::cuda::reference::reference_fna1d_backward_bfloat16_causal1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_FORWARD_2D_float32(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_float32_causal0x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_float32_causal0x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_float32_causal1x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_float32_causal1x1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_FORWARD_2D_float16(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_float16_causal0x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_float16_causal0x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_float16_causal1x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_float16_causal1x1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_FORWARD_2D_bfloat16(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_bfloat16_causal0x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_bfloat16_causal0x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_bfloat16_causal1x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_bfloat16_causal1x1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_BACKWARD_2D_float32(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_backward_float32_causal0x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_backward_float32_causal0x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_backward_float32_causal1x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_backward_float32_causal1x1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_BACKWARD_2D_float16(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_backward_float16_causal0x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_backward_float16_causal0x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_backward_float16_causal1x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_backward_float16_causal1x1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_BACKWARD_2D_bfloat16(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_backward_bfloat16_causal0x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_backward_bfloat16_causal0x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_backward_bfloat16_causal1x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal)) { \
        natten::cuda::reference::reference_fna2d_backward_bfloat16_causal1x1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_FORWARD_3D_float32(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_float32_causal0x0x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_float32_causal0x0x1(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_float32_causal0x1x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_float32_causal0x1x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_float32_causal1x0x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_float32_causal1x0x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_float32_causal1x1x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_float32_causal1x1x1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_FORWARD_3D_float16(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_float16_causal0x0x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_float16_causal0x0x1(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_float16_causal0x1x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_float16_causal0x1x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_float16_causal1x0x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_float16_causal1x0x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_float16_causal1x1x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_float16_causal1x1x1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_FORWARD_3D_bfloat16(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_bfloat16_causal0x0x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_bfloat16_causal0x0x1(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_bfloat16_causal0x1x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_bfloat16_causal0x1x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_bfloat16_causal1x0x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_bfloat16_causal1x0x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_bfloat16_causal1x1x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_bfloat16_causal1x1x1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_BACKWARD_3D_float32(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_float32_causal0x0x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_float32_causal0x0x1(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_float32_causal0x1x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_float32_causal0x1x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_float32_causal1x0x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_float32_causal1x0x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_float32_causal1x1x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_float32_causal1x1x1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_BACKWARD_3D_float16(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_float16_causal0x0x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_float16_causal0x0x1(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_float16_causal0x1x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_float16_causal0x1x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_float16_causal1x0x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_float16_causal1x0x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_float16_causal1x1x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_float16_causal1x1x1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_BACKWARD_3D_bfloat16(is_causal, ...) \
  [&] { \
    if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_bfloat16_causal0x0x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_bfloat16_causal0x0x1(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_bfloat16_causal0x1x0(__VA_ARGS__); \
    } \
    else if (not cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_bfloat16_causal0x1x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_bfloat16_causal1x0x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && not cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_bfloat16_causal1x0x1(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && not cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_bfloat16_causal1x1x0(__VA_ARGS__); \
    } \
    else if (cute::get<0>(is_causal) && cute::get<1>(is_causal) && cute::get<2>(is_causal)) { \
        natten::cuda::reference::reference_fna3d_backward_bfloat16_causal1x1x1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "Reference FNA kernel launch failed!" \
                << "Causal mask dispatcher got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace reference 
#endif 

