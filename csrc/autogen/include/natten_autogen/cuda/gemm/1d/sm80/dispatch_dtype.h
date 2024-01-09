#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.cuh> 
#include <natten/gemm_argpack.cuh> 
#include <natten/config.h> 
#include <natten_autogen/cuda/gemm/1d/sm80/dispatch_align.h> 
namespace natten { 
namespace cuda { 
namespace gemm { 
#define DISPATCH_DTYPE_na1d_pn_cuda_gemm_sm80(dtype, dim, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_ALIGNMENT_na1d_pn_cuda_gemm_sm80_double(dim, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_ALIGNMENT_na1d_pn_cuda_gemm_sm80_float(dim, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_ALIGNMENT_na1d_pn_cuda_gemm_sm80_half(dim, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_ALIGNMENT_na1d_pn_cuda_gemm_sm80_bfloat16(dim, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_cuda_gemm does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na1d_nn_cuda_gemm_sm80(dtype, dim, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_ALIGNMENT_na1d_nn_cuda_gemm_sm80_double(dim, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_ALIGNMENT_na1d_nn_cuda_gemm_sm80_float(dim, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_ALIGNMENT_na1d_nn_cuda_gemm_sm80_half(dim, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_ALIGNMENT_na1d_nn_cuda_gemm_sm80_bfloat16(dim, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_nn_cuda_gemm does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na1d_in_cuda_gemm_sm80(dtype, dim, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      DISPATCH_ALIGNMENT_na1d_in_cuda_gemm_sm80_double(dim, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_ALIGNMENT_na1d_in_cuda_gemm_sm80_float(dim, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_ALIGNMENT_na1d_in_cuda_gemm_sm80_half(dim, __VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_ALIGNMENT_na1d_in_cuda_gemm_sm80_bfloat16(dim, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_in_cuda_gemm does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace gemm 

