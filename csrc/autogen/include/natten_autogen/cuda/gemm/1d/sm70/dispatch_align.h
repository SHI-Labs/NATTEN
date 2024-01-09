#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.cuh> 
#include <natten/gemm_argpack.cuh> 
#include <natten/config.h> 
#include <natten_autogen/cuda/gemm/1d/sm70/kernels.h> 
namespace natten { 
namespace cuda { 
namespace gemm { 
#define DISPATCH_ALIGNMENT_na1d_pn_cuda_gemm_sm70_half(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na1d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na1d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na1d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_cuda_gemm_sm70_half requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na1d_nn_cuda_gemm_sm70_half(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na1d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na1d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na1d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_nn_cuda_gemm_sm70_half requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na1d_in_cuda_gemm_sm70_half(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na1d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na1d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na1d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_in_cuda_gemm_sm70_half requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace gemm 

