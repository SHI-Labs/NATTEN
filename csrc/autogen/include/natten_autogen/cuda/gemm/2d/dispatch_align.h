#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.cuh> 
#include <natten/gemm_argpack.cuh> 
#include <natten/config.h> 
#include <natten_autogen/cuda/gemm/2d/kernels.h> 
namespace natten { 
namespace cuda { 
namespace gemm { 
#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks3(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks3_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks5(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks5_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks7(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks7_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks9(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks9_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks11(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks11_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks13(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks13_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks15(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks15_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks17(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks17_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks19(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks19_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks21(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks21_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks23(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks23_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks25(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks25_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks27(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks27_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks29(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks29_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks31(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks31_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks33(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks33_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks35(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks35_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks37(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks37_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks39(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks39_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks41(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks41_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks43(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks43_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks45(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks45_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks47(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks47_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks49(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks49_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks51(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks51_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks53(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks53_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks55(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks55_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks57(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks57_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks59(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks59_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks61(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks61_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks63(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks63_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks15(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks15_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks15_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks15_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks15 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks17(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks17_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks17_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks17_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks17 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks19(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks19_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks19_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks19_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks19 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks21(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks21_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks21_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks21_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks21 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks23(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks23_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks23_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks23_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks23 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks25(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks25_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks25_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks25_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks25 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks27(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks27_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks27_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks27_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks27 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks29(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks29_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks29_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks29_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks29 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks31(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks31_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks31_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks31_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks31 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks33(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks33_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks33_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks33_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks33 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks35(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks35_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks35_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks35_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks35 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks37(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks37_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks37_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks37_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks37 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks39(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks39_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks39_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks39_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks39 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks41(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks41_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks41_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks41_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks41 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks43(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks43_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks43_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks43_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks43 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks45(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks45_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks45_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks45_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks45 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks47(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks47_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks47_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks47_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks47 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks49(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks49_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks49_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks49_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks49 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks51(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks51_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks51_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks51_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks51 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks53(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks53_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks53_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks53_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks53 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks55(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks55_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks55_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks55_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks55 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks57(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks57_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks57_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks57_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks57 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks59(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks59_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks59_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks59_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks59 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks61(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks61_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks61_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks61_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks61 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks63(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks63_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks63_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks63_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks63 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks15(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks15_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks15_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks15_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks15 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks17(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks17_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks17_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks17_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks17 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks19(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks19_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks19_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks19_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks19 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks21(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks21_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks21_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks21_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks21 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks23(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks23_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks23_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks23_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks23 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks25(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks25_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks25_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks25_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks25 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks27(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks27_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks27_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks27_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks27 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks29(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks29_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks29_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks29_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks29 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks31(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks31_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks31_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks31_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks31 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks33(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks33_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks33_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks33_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks33 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks35(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks35_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks35_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks35_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks35 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks37(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks37_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks37_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks37_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks37 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks39(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks39_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks39_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks39_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks39 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks41(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks41_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks41_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks41_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks41 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks43(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks43_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks43_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks43_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks43 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks45(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks45_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks45_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks45_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks45 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks47(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks47_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks47_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks47_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks47 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks49(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks49_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks49_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks49_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks49 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks51(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks51_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks51_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks51_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks51 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks53(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks53_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks53_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks53_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks53 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks55(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks55_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks55_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks55_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks55 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks57(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks57_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks57_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks57_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks57 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks59(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks59_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks59_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks59_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks59 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks61(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks61_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks61_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks61_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks61 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks63(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks63_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks63_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks63_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks63 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks15(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks15_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks15_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks15_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks15 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks17(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks17_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks17_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks17_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks17 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks19(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks19_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks19_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks19_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks19 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks21(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks21_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks21_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks21_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks21 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks23(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks23_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks23_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks23_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks23 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks25(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks25_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks25_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks25_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks25 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks27(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks27_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks27_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks27_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks27 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks29(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks29_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks29_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks29_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks29 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks31(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks31_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks31_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks31_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks31 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks33(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks33_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks33_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks33_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks33 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks35(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks35_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks35_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks35_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks35 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks37(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks37_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks37_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks37_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks37 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks39(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks39_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks39_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks39_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks39 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks41(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks41_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks41_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks41_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks41 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks43(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks43_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks43_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks43_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks43 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks45(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks45_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks45_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks45_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks45 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks47(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks47_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks47_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks47_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks47 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks49(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks49_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks49_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks49_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks49 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks51(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks51_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks51_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks51_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks51 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks53(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks53_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks53_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks53_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks53 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks55(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks55_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks55_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks55_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks55 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks57(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks57_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks57_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks57_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks57 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks59(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks59_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks59_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks59_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks59 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks61(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks61_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks61_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks61_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks61 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks63(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks63_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks63_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks63_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks63 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks3(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks3_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks5(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks5_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks7(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks7_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks9(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks9_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks11(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks11_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks13(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks13_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks15(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks15_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks17(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks17_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks19(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks19_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks21(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks21_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks23(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks23_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks25(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks25_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks27(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks27_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks29(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks29_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks31(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks31_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks33(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks33_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks35(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks35_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks37(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks37_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks39(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks39_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks41(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks41_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks43(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks43_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks45(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks45_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks47(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks47_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks49(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks49_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks51(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks51_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks53(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks53_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks55(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks55_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks57(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks57_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks59(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks59_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks61(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks61_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks63(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_nn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_ks63_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks15(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks15_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks15_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks15_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks15 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks17(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks17_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks17_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks17_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks17 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks19(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks19_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks19_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks19_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks19 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks21(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks21_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks21_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks21_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks21 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks23(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks23_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks23_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks23_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks23 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks25(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks25_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks25_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks25_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks25 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks27(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks27_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks27_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks27_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks27 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks29(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks29_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks29_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks29_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks29 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks31(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks31_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks31_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks31_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks31 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks33(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks33_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks33_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks33_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks33 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks35(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks35_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks35_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks35_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks35 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks37(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks37_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks37_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks37_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks37 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks39(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks39_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks39_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks39_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks39 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks41(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks41_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks41_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks41_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks41 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks43(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks43_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks43_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks43_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks43 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks45(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks45_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks45_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks45_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks45 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks47(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks47_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks47_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks47_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks47 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks49(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks49_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks49_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks49_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks49 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks51(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks51_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks51_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks51_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks51 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks53(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks53_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks53_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks53_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks53 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks55(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks55_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks55_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks55_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks55 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks57(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks57_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks57_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks57_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks57 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks59(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks59_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks59_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks59_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks59 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks61(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks61_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks61_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks61_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks61 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks63(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks63_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks63_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks63_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_ks63 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks15(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks15_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks15_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks15_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks15 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks17(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks17_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks17_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks17_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks17 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks19(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks19_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks19_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks19_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks19 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks21(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks21_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks21_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks21_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks21 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks23(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks23_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks23_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks23_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks23 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks25(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks25_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks25_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks25_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks25 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks27(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks27_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks27_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks27_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks27 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks29(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks29_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks29_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks29_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks29 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks31(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks31_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks31_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks31_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks31 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks33(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks33_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks33_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks33_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks33 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks35(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks35_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks35_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks35_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks35 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks37(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks37_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks37_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks37_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks37 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks39(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks39_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks39_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks39_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks39 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks41(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks41_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks41_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks41_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks41 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks43(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks43_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks43_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks43_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks43 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks45(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks45_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks45_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks45_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks45 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks47(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks47_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks47_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks47_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks47 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks49(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks49_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks49_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks49_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks49 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks51(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks51_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks51_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks51_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks51 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks53(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks53_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks53_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks53_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks53 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks55(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks55_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks55_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks55_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks55 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks57(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks57_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks57_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks57_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks57 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks59(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks59_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks59_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks59_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks59 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks61(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks61_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks61_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks61_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks61 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks63(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks63_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks63_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks63_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_ks63 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks15(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks15_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks15_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks15_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks15 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks17(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks17_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks17_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks17_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks17 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks19(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks19_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks19_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks19_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks19 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks21(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks21_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks21_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks21_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks21 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks23(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks23_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks23_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks23_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks23 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks25(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks25_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks25_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks25_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks25 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks27(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks27_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks27_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks27_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks27 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks29(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks29_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks29_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks29_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks29 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks31(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks31_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks31_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks31_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks31 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks33(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks33_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks33_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks33_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks33 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks35(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks35_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks35_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks35_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks35 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks37(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks37_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks37_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks37_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks37 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks39(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks39_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks39_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks39_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks39 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks41(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks41_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks41_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks41_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks41 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks43(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks43_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks43_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks43_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks43 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks45(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks45_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks45_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks45_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks45 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks47(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks47_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks47_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks47_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks47 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks49(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks49_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks49_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks49_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks49 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks51(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks51_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks51_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks51_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks51 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks53(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks53_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks53_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks53_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks53 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks55(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks55_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks55_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks55_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks55 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks57(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks57_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks57_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks57_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks57 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks59(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks59_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks59_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks59_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks59 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks61(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks61_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks61_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks61_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks61 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks63(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks63_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks63_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks63_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_ks63 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks3(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks3_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks5(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks5_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks7(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks7_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks9(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks9_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks11(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks11_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks13(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks13_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks15(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks15_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks17(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks17_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks19(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks19_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks21(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks21_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks23(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks23_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks25(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks25_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks27(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks27_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks29(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks29_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks31(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks31_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks33(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks33_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks35(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks35_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks37(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks37_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks39(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks39_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks41(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks41_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks43(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks43_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks45(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks45_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks47(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks47_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks49(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks49_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks51(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks51_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks53(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks53_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks55(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks55_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks57(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks57_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks59(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks59_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks61(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks61_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks63(dim, ...) \
  [&] { \
  natten::cuda::gemm::na2d_in_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_ks63_align1(__VA_ARGS__); \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks3 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks5 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks7 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks9 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks11 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks13 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks15(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks15_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks15_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks15_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks15 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks17(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks17_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks17_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks17_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks17 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks19(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks19_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks19_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks19_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks19 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks21(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks21_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks21_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks21_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks21 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks23(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks23_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks23_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks23_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks23 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks25(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks25_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks25_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks25_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks25 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks27(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks27_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks27_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks27_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks27 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks29(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks29_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks29_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks29_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks29 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks31(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks31_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks31_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks31_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks31 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks33(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks33_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks33_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks33_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks33 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks35(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks35_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks35_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks35_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks35 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks37(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks37_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks37_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks37_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks37 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks39(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks39_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks39_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks39_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks39 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks41(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks41_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks41_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks41_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks41 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks43(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks43_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks43_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks43_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks43 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks45(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks45_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks45_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks45_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks45 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks47(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks47_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks47_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks47_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks47 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks49(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks49_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks49_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks49_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks49 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks51(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks51_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks51_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks51_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks51 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks53(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks53_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks53_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks53_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks53 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks55(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks55_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks55_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks55_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks55 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks57(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks57_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks57_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks57_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks57 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks59(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks59_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks59_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks59_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks59 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks61(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks61_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks61_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks61_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks61 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks63(dim, ...) \
  [&] { \
    if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks63_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks63_align2(__VA_ARGS__); \
    } \
    else if (dim % 1 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks63_align1(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_ks63 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=float. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks3 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks5 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks7 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks9 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks11 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks13 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks15(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks15_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks15_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks15_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks15 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks17(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks17_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks17_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks17_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks17 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks19(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks19_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks19_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks19_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks19 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks21(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks21_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks21_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks21_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks21 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks23(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks23_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks23_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks23_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks23 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks25(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks25_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks25_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks25_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks25 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks27(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks27_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks27_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks27_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks27 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks29(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks29_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks29_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks29_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks29 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks31(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks31_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks31_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks31_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks31 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks33(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks33_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks33_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks33_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks33 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks35(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks35_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks35_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks35_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks35 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks37(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks37_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks37_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks37_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks37 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks39(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks39_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks39_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks39_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks39 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks41(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks41_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks41_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks41_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks41 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks43(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks43_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks43_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks43_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks43 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks45(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks45_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks45_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks45_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks45 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks47(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks47_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks47_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks47_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks47 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks49(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks49_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks49_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks49_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks49 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks51(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks51_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks51_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks51_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks51 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks53(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks53_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks53_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks53_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks53 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks55(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks55_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks55_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks55_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks55 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks57(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks57_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks57_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks57_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks57 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks59(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks59_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks59_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks59_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks59 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks61(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks61_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks61_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks61_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks61 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks63(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks63_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks63_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks63_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_ks63 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks3 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks5 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks7 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks9 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks11 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks13 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks15(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks15_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks15_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks15_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks15 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks17(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks17_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks17_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks17_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks17 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks19(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks19_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks19_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks19_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks19 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks21(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks21_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks21_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks21_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks21 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks23(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks23_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks23_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks23_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks23 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks25(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks25_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks25_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks25_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks25 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks27(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks27_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks27_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks27_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks27 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks29(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks29_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks29_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks29_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks29 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks31(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks31_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks31_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks31_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks31 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks33(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks33_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks33_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks33_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks33 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks35(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks35_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks35_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks35_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks35 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks37(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks37_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks37_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks37_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks37 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks39(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks39_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks39_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks39_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks39 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks41(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks41_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks41_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks41_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks41 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks43(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks43_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks43_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks43_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks43 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks45(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks45_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks45_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks45_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks45 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks47(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks47_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks47_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks47_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks47 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks49(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks49_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks49_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks49_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks49 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks51(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks51_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks51_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks51_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks51 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks53(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks53_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks53_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks53_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks53 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks55(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks55_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks55_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks55_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks55 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks57(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks57_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks57_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks57_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks57 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks59(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks59_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks59_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks59_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks59 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks61(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks61_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks61_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks61_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks61 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks63(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks63_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks63_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks63_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_ks63 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=bfloat16. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace gemm 

