#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.cuh> 
#include <natten/gemm_argpack.cuh> 
#include <natten/config.h> 
#include <natten_autogen/cuda/gemm/2d/sm70/kernels.h> 
namespace natten { 
namespace cuda { 
namespace gemm { 
#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks15(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks15_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks15_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks15_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks15 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks17(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks17_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks17_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks17_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks17 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks19(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks19_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks19_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks19_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks19 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks21(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks21_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks21_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks21_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks21 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks23(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks23_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks23_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks23_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks23 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks25(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks25_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks25_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks25_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks25 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks27(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks27_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks27_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks27_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks27 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks29(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks29_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks29_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks29_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks29 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks31(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks31_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks31_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks31_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks31 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks33(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks33_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks33_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks33_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks33 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks21(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks21_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks21_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks21_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks21 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks23(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks23_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks23_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks23_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks23 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks25(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks25_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks25_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks25_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks25 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks27(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks27_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks27_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks27_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks27 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks29(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks29_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks29_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks29_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks29 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks31(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks31_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks31_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks31_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks31 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks33(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks33_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks33_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks33_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cuda_gemm_half_128x128x32_64x64x32_8x8x4_2_sm70_ks33 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks3 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks5 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks7 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks9 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks11 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks21(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks21_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks21_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks21_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks21 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks23(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks23_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks23_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks23_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks23 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks25(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks25_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks25_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks25_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks25 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks27(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks27_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks27_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks27_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks27 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks29(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks29_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks29_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks29_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks29 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks31(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks31_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks31_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks31_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks31 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks33(dim, ...) \
  [&] { \
    if (dim % 8 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks33_align8(__VA_ARGS__); \
    } \
    else if (dim % 4 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks33_align4(__VA_ARGS__); \
    } \
    else if (dim % 2 == 0) { \
      natten::cuda::gemm::na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks33_align2(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks33 requires at least 32-bit alignment." \
                << "Got dim=" << dim << ", dtype=half. " \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace gemm 

