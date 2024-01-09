#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.cuh> 
#include <natten/gemm_argpack.cuh> 
#include <natten/config.h> 
#include <natten_autogen/cuda/gemm/1d/sm80/kernels.h> 
namespace natten { 
namespace cuda { 
namespace gemm { 
#define DISPATCH_ALIGNMENT_na1d_pn_cuda_gemm_sm80_double(dim, ...) \
  [&] { \
  natten::cuda::gemm::na1d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_sm80_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na1d_pn_cuda_gemm_sm80_float(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na1d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na1d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na1d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_cuda_gemm_sm80_float requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na1d_pn_cuda_gemm_sm80_half(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na1d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na1d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na1d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_cuda_gemm_sm80_half requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na1d_pn_cuda_gemm_sm80_bfloat16(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na1d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_sm80_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na1d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_sm80_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na1d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_sm80_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_cuda_gemm_sm80_bfloat16 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na1d_nn_cuda_gemm_sm80_double(dim, ...) \
  [&] { \
  natten::cuda::gemm::na1d_nn_cuda_gemm_double_64x32x16_32x16x16_8x8x4_3_sm80_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na1d_nn_cuda_gemm_sm80_float(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na1d_nn_cuda_gemm_float_64x32x16_32x16x16_16x8x8_3_sm80_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na1d_nn_cuda_gemm_float_64x32x16_32x16x16_16x8x8_3_sm80_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na1d_nn_cuda_gemm_float_64x32x16_32x16x16_16x8x8_3_sm80_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_nn_cuda_gemm_sm80_float requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na1d_nn_cuda_gemm_sm80_half(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na1d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na1d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na1d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_nn_cuda_gemm_sm80_half requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na1d_nn_cuda_gemm_sm80_bfloat16(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na1d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na1d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na1d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_nn_cuda_gemm_sm80_bfloat16 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na1d_in_cuda_gemm_sm80_double(dim, ...) \
  [&] { \
  natten::cuda::gemm::na1d_in_cuda_gemm_double_64x32x16_32x16x16_8x8x4_3_sm80_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na1d_in_cuda_gemm_sm80_float(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na1d_in_cuda_gemm_float_64x32x16_32x16x16_16x8x8_3_sm80_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na1d_in_cuda_gemm_float_64x32x16_32x16x16_16x8x8_3_sm80_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na1d_in_cuda_gemm_float_64x32x16_32x16x16_16x8x8_3_sm80_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_in_cuda_gemm_sm80_float requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na1d_in_cuda_gemm_sm80_half(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na1d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na1d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na1d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_in_cuda_gemm_sm80_half requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na1d_in_cuda_gemm_sm80_bfloat16(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na1d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na1d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na1d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_in_cuda_gemm_sm80_bfloat16 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace gemm 

