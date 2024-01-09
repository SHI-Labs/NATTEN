#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.cuh> 
#include <natten/config.h> 
#include <natten_autogen/cuda/gemm/2d/sm70/dispatch_dtype.h> 
#include <natten_autogen/cuda/gemm/2d/sm75/dispatch_dtype.h> 
#include <natten_autogen/cuda/gemm/2d/sm80/dispatch_dtype.h> 
namespace natten { 
namespace cuda { 
namespace gemm { 
#define LAUNCH_na2d_pn_cuda_gemm(cc, dtype, kernel_size, dim, ...) \
  [&] { \
    if (cc >= 80) { \
      DISPATCH_DTYPE_na2d_pn_cuda_gemm_sm80(dtype, kernel_size, dim, __VA_ARGS__); \
    } \
    else if (cc >= 75) { \
      DISPATCH_DTYPE_na2d_pn_cuda_gemm_sm75(dtype, kernel_size, dim, __VA_ARGS__); \
    } \
    else if (cc >= 70) { \
      DISPATCH_DTYPE_na2d_pn_cuda_gemm_sm70(dtype, kernel_size, dim, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define LAUNCH_na2d_nn_cuda_gemm(cc, dtype, kernel_size, dim, ...) \
  [&] { \
    if (cc >= 80) { \
      DISPATCH_DTYPE_na2d_nn_cuda_gemm_sm80(dtype, kernel_size, dim, __VA_ARGS__); \
    } \
    else if (cc >= 75) { \
      DISPATCH_DTYPE_na2d_nn_cuda_gemm_sm75(dtype, kernel_size, dim, __VA_ARGS__); \
    } \
    else if (cc >= 70) { \
      DISPATCH_DTYPE_na2d_nn_cuda_gemm_sm70(dtype, kernel_size, dim, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define LAUNCH_na2d_in_cuda_gemm(cc, dtype, kernel_size, dim, ...) \
  [&] { \
    if (cc >= 80) { \
      DISPATCH_DTYPE_na2d_in_cuda_gemm_sm80(dtype, kernel_size, dim, __VA_ARGS__); \
    } \
    else if (cc >= 75) { \
      DISPATCH_DTYPE_na2d_in_cuda_gemm_sm75(dtype, kernel_size, dim, __VA_ARGS__); \
    } \
    else if (cc >= 70) { \
      DISPATCH_DTYPE_na2d_in_cuda_gemm_sm70(dtype, kernel_size, dim, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace gemm 

