#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.cuh> 
#include <natten_autogen/cuda/naive/kernels.h> 
namespace natten { 
namespace cuda { 
namespace naive { 
#define DISPATCH_CM_na1d_pn_cuda_naive_double(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_pn_cuda_naive_double_cm_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal)) { \
      naive::na1d_pn_cuda_naive_double_cm_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_cuda_naive_double got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_pn_cuda_naive_float(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_pn_cuda_naive_float_cm_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal)) { \
      naive::na1d_pn_cuda_naive_float_cm_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_cuda_naive_float got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_pn_cuda_naive_half(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_pn_cuda_naive_half_cm_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal)) { \
      naive::na1d_pn_cuda_naive_half_cm_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_cuda_naive_half got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_pn_cuda_naive_bfloat16(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_pn_cuda_naive_bfloat16_cm_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal)) { \
      naive::na1d_pn_cuda_naive_bfloat16_cm_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_cuda_naive_bfloat16 got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_pn_cuda_naive_double(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_pn_cuda_naive_double_cm_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_pn_cuda_naive_double_cm_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_pn_cuda_naive_double_cm_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_pn_cuda_naive_double_cm_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_naive_double got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_pn_cuda_naive_float(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_pn_cuda_naive_float_cm_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_pn_cuda_naive_float_cm_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_pn_cuda_naive_float_cm_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_pn_cuda_naive_float_cm_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_naive_float got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_pn_cuda_naive_half(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_pn_cuda_naive_half_cm_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_pn_cuda_naive_half_cm_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_pn_cuda_naive_half_cm_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_pn_cuda_naive_half_cm_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_naive_half got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_pn_cuda_naive_bfloat16(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_pn_cuda_naive_bfloat16_cm_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_pn_cuda_naive_bfloat16_cm_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_pn_cuda_naive_bfloat16_cm_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_pn_cuda_naive_bfloat16_cm_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_naive_bfloat16 got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_pn_cuda_naive_double(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_double_cm_0_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_double_cm_0_0_1(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_double_cm_0_1_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_double_cm_0_1_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_double_cm_1_0_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_double_cm_1_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_double_cm_1_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_double_cm_1_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_pn_cuda_naive_double got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_pn_cuda_naive_float(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_float_cm_0_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_float_cm_0_0_1(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_float_cm_0_1_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_float_cm_0_1_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_float_cm_1_0_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_float_cm_1_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_float_cm_1_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_float_cm_1_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_pn_cuda_naive_float got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_pn_cuda_naive_half(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_half_cm_0_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_half_cm_0_0_1(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_half_cm_0_1_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_half_cm_0_1_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_half_cm_1_0_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_half_cm_1_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_half_cm_1_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_half_cm_1_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_pn_cuda_naive_half got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_pn_cuda_naive_bfloat16(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_bfloat16_cm_0_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_bfloat16_cm_0_0_1(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_bfloat16_cm_0_1_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_bfloat16_cm_0_1_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_bfloat16_cm_1_0_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_bfloat16_cm_1_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_bfloat16_cm_1_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_pn_cuda_naive_bfloat16_cm_1_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_pn_cuda_naive_bfloat16 got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_pn_bias_cuda_naive_double(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_pn_bias_cuda_naive_double_cm_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_bias_cuda_naive_double got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_pn_bias_cuda_naive_float(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_pn_bias_cuda_naive_float_cm_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_bias_cuda_naive_float got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_pn_bias_cuda_naive_half(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_pn_bias_cuda_naive_half_cm_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_bias_cuda_naive_half got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_pn_bias_cuda_naive_bfloat16(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_pn_bias_cuda_naive_bfloat16_cm_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_bias_cuda_naive_bfloat16 got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_pn_bias_cuda_naive_double(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_pn_bias_cuda_naive_double_cm_0_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_bias_cuda_naive_double got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_pn_bias_cuda_naive_float(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_pn_bias_cuda_naive_float_cm_0_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_bias_cuda_naive_float got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_pn_bias_cuda_naive_half(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_pn_bias_cuda_naive_half_cm_0_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_bias_cuda_naive_half got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_pn_bias_cuda_naive_bfloat16(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_pn_bias_cuda_naive_bfloat16_cm_0_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_bias_cuda_naive_bfloat16 got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_pn_bias_cuda_naive_double(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_bias_cuda_naive_double_cm_0_0_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_pn_bias_cuda_naive_double got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_pn_bias_cuda_naive_float(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_bias_cuda_naive_float_cm_0_0_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_pn_bias_cuda_naive_float got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_pn_bias_cuda_naive_half(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_bias_cuda_naive_half_cm_0_0_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_pn_bias_cuda_naive_half got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_pn_bias_cuda_naive_bfloat16(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_pn_bias_cuda_naive_bfloat16_cm_0_0_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_pn_bias_cuda_naive_bfloat16 got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_nn_cuda_naive_double(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_nn_cuda_naive_double_cm_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal)) { \
      naive::na1d_nn_cuda_naive_double_cm_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_nn_cuda_naive_double got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_nn_cuda_naive_float(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_nn_cuda_naive_float_cm_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal)) { \
      naive::na1d_nn_cuda_naive_float_cm_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_nn_cuda_naive_float got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_nn_cuda_naive_half(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_nn_cuda_naive_half_cm_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal)) { \
      naive::na1d_nn_cuda_naive_half_cm_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_nn_cuda_naive_half got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_nn_cuda_naive_bfloat16(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_nn_cuda_naive_bfloat16_cm_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal)) { \
      naive::na1d_nn_cuda_naive_bfloat16_cm_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_nn_cuda_naive_bfloat16 got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_nn_cuda_naive_double(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_nn_cuda_naive_double_cm_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_nn_cuda_naive_double_cm_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_nn_cuda_naive_double_cm_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_nn_cuda_naive_double_cm_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_naive_double got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_nn_cuda_naive_float(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_nn_cuda_naive_float_cm_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_nn_cuda_naive_float_cm_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_nn_cuda_naive_float_cm_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_nn_cuda_naive_float_cm_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_naive_float got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_nn_cuda_naive_half(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_nn_cuda_naive_half_cm_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_nn_cuda_naive_half_cm_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_nn_cuda_naive_half_cm_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_nn_cuda_naive_half_cm_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_naive_half got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_nn_cuda_naive_bfloat16(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_nn_cuda_naive_bfloat16_cm_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_nn_cuda_naive_bfloat16_cm_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_nn_cuda_naive_bfloat16_cm_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_nn_cuda_naive_bfloat16_cm_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_naive_bfloat16 got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_nn_cuda_naive_double(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_double_cm_0_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_double_cm_0_0_1(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_double_cm_0_1_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_double_cm_0_1_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_double_cm_1_0_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_double_cm_1_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_double_cm_1_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_double_cm_1_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_nn_cuda_naive_double got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_nn_cuda_naive_float(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_float_cm_0_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_float_cm_0_0_1(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_float_cm_0_1_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_float_cm_0_1_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_float_cm_1_0_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_float_cm_1_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_float_cm_1_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_float_cm_1_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_nn_cuda_naive_float got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_nn_cuda_naive_half(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_half_cm_0_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_half_cm_0_0_1(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_half_cm_0_1_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_half_cm_0_1_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_half_cm_1_0_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_half_cm_1_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_half_cm_1_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_half_cm_1_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_nn_cuda_naive_half got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_nn_cuda_naive_bfloat16(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_bfloat16_cm_0_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_bfloat16_cm_0_0_1(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_bfloat16_cm_0_1_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_bfloat16_cm_0_1_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_bfloat16_cm_1_0_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_bfloat16_cm_1_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_bfloat16_cm_1_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_nn_cuda_naive_bfloat16_cm_1_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_nn_cuda_naive_bfloat16 got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_in_cuda_naive_double(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_in_cuda_naive_double_cm_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal)) { \
      naive::na1d_in_cuda_naive_double_cm_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_in_cuda_naive_double got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_in_cuda_naive_float(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_in_cuda_naive_float_cm_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal)) { \
      naive::na1d_in_cuda_naive_float_cm_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_in_cuda_naive_float got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_in_cuda_naive_half(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_in_cuda_naive_half_cm_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal)) { \
      naive::na1d_in_cuda_naive_half_cm_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_in_cuda_naive_half got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_in_cuda_naive_bfloat16(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_in_cuda_naive_bfloat16_cm_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal)) { \
      naive::na1d_in_cuda_naive_bfloat16_cm_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_in_cuda_naive_bfloat16 got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_in_cuda_naive_double(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_in_cuda_naive_double_cm_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_in_cuda_naive_double_cm_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_in_cuda_naive_double_cm_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_in_cuda_naive_double_cm_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_naive_double got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_in_cuda_naive_float(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_in_cuda_naive_float_cm_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_in_cuda_naive_float_cm_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_in_cuda_naive_float_cm_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_in_cuda_naive_float_cm_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_naive_float got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_in_cuda_naive_half(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_in_cuda_naive_half_cm_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_in_cuda_naive_half_cm_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_in_cuda_naive_half_cm_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_in_cuda_naive_half_cm_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_naive_half got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_in_cuda_naive_bfloat16(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_in_cuda_naive_bfloat16_cm_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_in_cuda_naive_bfloat16_cm_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_in_cuda_naive_bfloat16_cm_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      naive::na2d_in_cuda_naive_bfloat16_cm_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_naive_bfloat16 got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_in_cuda_naive_double(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_double_cm_0_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_double_cm_0_0_1(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_double_cm_0_1_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_double_cm_0_1_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_double_cm_1_0_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_double_cm_1_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_double_cm_1_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_double_cm_1_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_in_cuda_naive_double got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_in_cuda_naive_float(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_float_cm_0_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_float_cm_0_0_1(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_float_cm_0_1_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_float_cm_0_1_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_float_cm_1_0_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_float_cm_1_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_float_cm_1_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_float_cm_1_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_in_cuda_naive_float got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_in_cuda_naive_half(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_half_cm_0_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_half_cm_0_0_1(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_half_cm_0_1_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_half_cm_0_1_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_half_cm_1_0_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_half_cm_1_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_half_cm_1_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_half_cm_1_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_in_cuda_naive_half got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_in_cuda_naive_bfloat16(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_bfloat16_cm_0_0_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_bfloat16_cm_0_0_1(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_bfloat16_cm_0_1_0(__VA_ARGS__); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_bfloat16_cm_0_1_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_bfloat16_cm_1_0_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_bfloat16_cm_1_0_1(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_bfloat16_cm_1_1_0(__VA_ARGS__); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      naive::na3d_in_cuda_naive_bfloat16_cm_1_1_1(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_in_cuda_naive_bfloat16 got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_rpbgrad_cuda_naive_double(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_rpbgrad_cuda_naive_double_cm_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_rpbgrad_cuda_naive_double got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_rpbgrad_cuda_naive_float(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_rpbgrad_cuda_naive_float_cm_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_rpbgrad_cuda_naive_float got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_rpbgrad_cuda_naive_half(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_rpbgrad_cuda_naive_half_cm_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_rpbgrad_cuda_naive_half got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na1d_rpbgrad_cuda_naive_bfloat16(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
      naive::na1d_rpbgrad_cuda_naive_bfloat16_cm_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_rpbgrad_cuda_naive_bfloat16 got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_rpbgrad_cuda_naive_double(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_rpbgrad_cuda_naive_double_cm_0_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_rpbgrad_cuda_naive_double got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_rpbgrad_cuda_naive_float(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_rpbgrad_cuda_naive_float_cm_0_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_rpbgrad_cuda_naive_float got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_rpbgrad_cuda_naive_half(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_rpbgrad_cuda_naive_half_cm_0_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_rpbgrad_cuda_naive_half got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na2d_rpbgrad_cuda_naive_bfloat16(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      naive::na2d_rpbgrad_cuda_naive_bfloat16_cm_0_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_rpbgrad_cuda_naive_bfloat16 got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_rpbgrad_cuda_naive_double(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_rpbgrad_cuda_naive_double_cm_0_0_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_rpbgrad_cuda_naive_double got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_rpbgrad_cuda_naive_float(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_rpbgrad_cuda_naive_float_cm_0_0_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_rpbgrad_cuda_naive_float got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_rpbgrad_cuda_naive_half(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_rpbgrad_cuda_naive_half_cm_0_0_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_rpbgrad_cuda_naive_half got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_CM_na3d_rpbgrad_cuda_naive_bfloat16(is_causal, ...) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      naive::na3d_rpbgrad_cuda_naive_bfloat16_cm_0_0_0(__VA_ARGS__); \
    } \
    else { \
          std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_rpbgrad_cuda_naive_bfloat16 got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace naive 

