#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.cuh> 
#include <natten_autogen/cuda/naive/dispatch_cm.h> 
namespace natten { 
namespace cuda { 
namespace naive { 
#define DISPATCH_DTYPE_na1d_pn_cuda_naive(dtype, is_causal, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_CM_na1d_pn_cuda_naive_double(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_CM_na1d_pn_cuda_naive_float(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_CM_na1d_pn_cuda_naive_half(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_CM_na1d_pn_cuda_naive_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_cuda_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na2d_pn_cuda_naive(dtype, is_causal, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_CM_na2d_pn_cuda_naive_double(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_CM_na2d_pn_cuda_naive_float(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_CM_na2d_pn_cuda_naive_half(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_CM_na2d_pn_cuda_naive_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na3d_pn_cuda_naive(dtype, is_causal, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_CM_na3d_pn_cuda_naive_double(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_CM_na3d_pn_cuda_naive_float(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_CM_na3d_pn_cuda_naive_half(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_CM_na3d_pn_cuda_naive_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_pn_cuda_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na1d_pn_bias_cuda_naive(dtype, is_causal, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_CM_na1d_pn_bias_cuda_naive_double(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_CM_na1d_pn_bias_cuda_naive_float(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_CM_na1d_pn_bias_cuda_naive_half(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_CM_na1d_pn_bias_cuda_naive_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_bias_cuda_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na2d_pn_bias_cuda_naive(dtype, is_causal, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_CM_na2d_pn_bias_cuda_naive_double(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_CM_na2d_pn_bias_cuda_naive_float(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_CM_na2d_pn_bias_cuda_naive_half(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_CM_na2d_pn_bias_cuda_naive_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_bias_cuda_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na3d_pn_bias_cuda_naive(dtype, is_causal, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_CM_na3d_pn_bias_cuda_naive_double(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_CM_na3d_pn_bias_cuda_naive_float(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_CM_na3d_pn_bias_cuda_naive_half(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_CM_na3d_pn_bias_cuda_naive_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_pn_bias_cuda_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na1d_nn_cuda_naive(dtype, is_causal, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_CM_na1d_nn_cuda_naive_double(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_CM_na1d_nn_cuda_naive_float(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_CM_na1d_nn_cuda_naive_half(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_CM_na1d_nn_cuda_naive_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_nn_cuda_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na2d_nn_cuda_naive(dtype, is_causal, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_CM_na2d_nn_cuda_naive_double(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_CM_na2d_nn_cuda_naive_float(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_CM_na2d_nn_cuda_naive_half(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_CM_na2d_nn_cuda_naive_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na3d_nn_cuda_naive(dtype, is_causal, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_CM_na3d_nn_cuda_naive_double(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_CM_na3d_nn_cuda_naive_float(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_CM_na3d_nn_cuda_naive_half(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_CM_na3d_nn_cuda_naive_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_nn_cuda_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na1d_in_cuda_naive(dtype, is_causal, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_CM_na1d_in_cuda_naive_double(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_CM_na1d_in_cuda_naive_float(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_CM_na1d_in_cuda_naive_half(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_CM_na1d_in_cuda_naive_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_in_cuda_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na2d_in_cuda_naive(dtype, is_causal, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_CM_na2d_in_cuda_naive_double(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_CM_na2d_in_cuda_naive_float(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_CM_na2d_in_cuda_naive_half(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_CM_na2d_in_cuda_naive_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na3d_in_cuda_naive(dtype, is_causal, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_CM_na3d_in_cuda_naive_double(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_CM_na3d_in_cuda_naive_float(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_CM_na3d_in_cuda_naive_half(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_CM_na3d_in_cuda_naive_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_in_cuda_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na1d_rpbgrad_cuda_naive(dtype, is_causal, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_CM_na1d_rpbgrad_cuda_naive_double(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_CM_na1d_rpbgrad_cuda_naive_float(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_CM_na1d_rpbgrad_cuda_naive_half(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_CM_na1d_rpbgrad_cuda_naive_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_rpbgrad_cuda_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na2d_rpbgrad_cuda_naive(dtype, is_causal, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_CM_na2d_rpbgrad_cuda_naive_double(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_CM_na2d_rpbgrad_cuda_naive_float(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_CM_na2d_rpbgrad_cuda_naive_half(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_CM_na2d_rpbgrad_cuda_naive_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_rpbgrad_cuda_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na3d_rpbgrad_cuda_naive(dtype, is_causal, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_CM_na3d_rpbgrad_cuda_naive_double(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_CM_na3d_rpbgrad_cuda_naive_float(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_CM_na3d_rpbgrad_cuda_naive_half(is_causal, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_CM_na3d_rpbgrad_cuda_naive_bfloat16(is_causal, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_rpbgrad_cuda_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace naive 

