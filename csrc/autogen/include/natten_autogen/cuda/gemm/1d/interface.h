#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.cuh> 
#include <natten/config.h> 
#include <natten_autogen/cuda/gemm/1d/sm70/dispatch_dtype.h> 
#include <natten_autogen/cuda/gemm/1d/sm75/dispatch_dtype.h> 
#include <natten_autogen/cuda/gemm/1d/sm80/dispatch_dtype.h> 
namespace natten { 
namespace cuda { 
namespace gemm { 
#define LAUNCH_na1d_pn_cuda_gemm(cc, dtype, dim, ...) \
  [&] { \
    if (cc >= 80) { \
      DISPATCH_DTYPE_na1d_pn_cuda_gemm_sm80(dtype, dim, __VA_ARGS__); \
    } \
    else if (cc >= 75) { \
      DISPATCH_DTYPE_na1d_pn_cuda_gemm_sm75(dtype, dim, __VA_ARGS__); \
    } \
    else if (cc >= 70) { \
      DISPATCH_DTYPE_na1d_pn_cuda_gemm_sm70(dtype, dim, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_cuda_gemm does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define LAUNCH_na1d_nn_cuda_gemm(cc, dtype, dim, ...) \
  [&] { \
    if (cc >= 80) { \
      DISPATCH_DTYPE_na1d_nn_cuda_gemm_sm80(dtype, dim, __VA_ARGS__); \
    } \
    else if (cc >= 75) { \
      DISPATCH_DTYPE_na1d_nn_cuda_gemm_sm75(dtype, dim, __VA_ARGS__); \
    } \
    else if (cc >= 70) { \
      DISPATCH_DTYPE_na1d_nn_cuda_gemm_sm70(dtype, dim, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_nn_cuda_gemm does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define LAUNCH_na1d_in_cuda_gemm(cc, dtype, dim, ...) \
  [&] { \
    if (cc >= 80) { \
      DISPATCH_DTYPE_na1d_in_cuda_gemm_sm80(dtype, dim, __VA_ARGS__); \
    } \
    else if (cc >= 75) { \
      DISPATCH_DTYPE_na1d_in_cuda_gemm_sm75(dtype, dim, __VA_ARGS__); \
    } \
    else if (cc >= 70) { \
      DISPATCH_DTYPE_na1d_in_cuda_gemm_sm70(dtype, dim, __VA_ARGS__); \
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

