#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.cuh> 
#include <natten/gemm_argpack.cuh> 
#include <natten/config.h> 
#include <natten_autogen/cuda/gemm/2d/sm70/dispatch_kernel_size.h> 
namespace natten { 
namespace cuda { 
namespace gemm { 
#define DISPATCH_DTYPE_na2d_pn_cuda_gemm_sm70(dtype, kernel_size, dim, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_KERNELSIZE_na2d_pn_cuda_gemm_sm70_half(kernel_size, dim, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_sm70 does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na2d_nn_cuda_gemm_sm70(dtype, kernel_size, dim, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_KERNELSIZE_na2d_nn_cuda_gemm_sm70_half(kernel_size, dim, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_sm70 does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na2d_in_cuda_gemm_sm70(dtype, kernel_size, dim, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_KERNELSIZE_na2d_in_cuda_gemm_sm70_half(kernel_size, dim, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_sm70 does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace gemm 

