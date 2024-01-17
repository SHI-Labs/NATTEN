#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.cuh> 
#include <natten/gemm_argpack.cuh> 
#include <natten/config.h> 
#include <natten_autogen/cuda/gemm/2d/sm75/dispatch_align.h> 
namespace natten { 
namespace cuda { 
namespace gemm { 
#define DISPATCH_KERNELSIZE_na2d_pn_cuda_gemm_sm75_half(kernel_size, dim, ...) \
  [&] { \
    if (kernel_size == 3) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks3(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks5(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks7(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks9(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks11(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks13(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 15) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks15(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 17) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks17(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 19) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks19(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 21) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks21(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 23) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks23(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 25) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks25(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 27) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks27(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 29) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks29(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 31) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks31(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 33) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks33(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 35) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks35(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 37) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks37(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 39) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks39(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 41) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks41(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 43) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks43(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 45) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks45(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 47) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks47(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 49) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks49(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 51) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks51(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 53) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks53(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 55) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks55(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 57) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks57(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 59) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks59(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 61) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks61(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 63) { \
      DISPATCH_ALIGNMENT_na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks63(dim, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed! " \
                << "na2d_pn_cuda_gemm_sm75_half does not support implement " \
                << " kernel size " << kernel_size << ". " \
                << " You may try generating it manually and build from source." \
                << " Refer to NATTEN's github repository for more information." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_KERNELSIZE_na2d_nn_cuda_gemm_sm75_half(kernel_size, dim, ...) \
  [&] { \
    if (kernel_size == 3) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks3(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks5(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks7(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks9(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks11(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks13(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 15) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks15(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 17) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks17(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 19) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks19(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 21) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks21(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 23) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks23(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 25) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks25(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 27) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks27(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 29) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks29(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 31) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks31(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 33) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks33(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 35) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks35(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 37) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks37(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 39) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks39(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 41) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks41(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 43) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks43(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 45) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks45(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 47) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks47(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 49) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks49(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 51) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks51(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 53) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks53(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 55) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks55(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 57) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks57(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 59) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks59(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 61) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks61(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 63) { \
      DISPATCH_ALIGNMENT_na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_ks63(dim, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed! " \
                << "na2d_nn_cuda_gemm_sm75_half does not support implement " \
                << " kernel size " << kernel_size << ". " \
                << " You may try generating it manually and build from source." \
                << " Refer to NATTEN's github repository for more information." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_KERNELSIZE_na2d_in_cuda_gemm_sm75_half(kernel_size, dim, ...) \
  [&] { \
    if (kernel_size == 3) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks3(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks5(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks7(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks9(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks11(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks13(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 15) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks15(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 17) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks17(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 19) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks19(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 21) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks21(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 23) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks23(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 25) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks25(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 27) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks27(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 29) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks29(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 31) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks31(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 33) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks33(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 35) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks35(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 37) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks37(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 39) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks39(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 41) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks41(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 43) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks43(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 45) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks45(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 47) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks47(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 49) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks49(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 51) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks51(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 53) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks53(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 55) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks55(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 57) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks57(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 59) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks59(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 61) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks61(dim, __VA_ARGS__); \
    } \
    else if (kernel_size == 63) { \
      DISPATCH_ALIGNMENT_na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_ks63(dim, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed! " \
                << "na2d_in_cuda_gemm_sm75_half does not support implement " \
                << " kernel size " << kernel_size << ". " \
                << " You may try generating it manually and build from source." \
                << " Refer to NATTEN's github repository for more information." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace gemm 

