#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.cuh> 
#include <natten_autogen/cuda/naive/dispatch_di.h> 
namespace natten { 
namespace cuda { 
namespace naive { 
#define DISPATCH_KERNEL_na1d_pn_cuda_naive_double(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_double_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_double_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_double_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_double_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_double_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_double_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_double_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_pn_cuda_naive_float(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_float_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_float_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_float_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_float_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_float_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_float_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_float_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_pn_cuda_naive_half(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_half_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_half_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_half_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_half_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_half_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_half_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_half_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_pn_cuda_naive_bfloat16(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_bfloat16_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_bfloat16_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_bfloat16_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_bfloat16_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_bfloat16_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_bfloat16_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_pn_cuda_naive_bfloat16_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_pn_cuda_naive_double(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_double_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_double_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_double_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_double_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_double_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_double_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_double_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_pn_cuda_naive_float(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_float_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_float_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_float_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_float_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_float_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_float_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_float_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_pn_cuda_naive_half(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_half_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_half_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_half_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_half_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_half_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_half_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_half_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_pn_cuda_naive_bfloat16(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_bfloat16_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_bfloat16_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_bfloat16_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_bfloat16_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_bfloat16_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_bfloat16_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_pn_cuda_naive_bfloat16_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_pn_cuda_naive_double(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_double_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_double_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_double_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_double_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_double_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_double_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_double_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_pn_cuda_naive_float(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_float_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_float_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_float_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_float_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_float_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_float_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_float_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_pn_cuda_naive_half(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_half_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_half_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_half_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_half_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_half_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_half_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_half_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_pn_cuda_naive_bfloat16(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_bfloat16_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_bfloat16_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_bfloat16_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_bfloat16_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_bfloat16_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_bfloat16_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_pn_cuda_naive_bfloat16_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_pn_bias_cuda_naive_double(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_double_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_double_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_double_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_double_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_double_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_double_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_double_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_pn_bias_cuda_naive_float(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_float_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_float_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_float_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_float_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_float_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_float_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_float_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_pn_bias_cuda_naive_half(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_half_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_half_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_half_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_half_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_half_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_half_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_half_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_pn_bias_cuda_naive_bfloat16(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_bfloat16_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_bfloat16_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_bfloat16_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_bfloat16_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_bfloat16_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_bfloat16_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_pn_bias_cuda_naive_bfloat16_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_pn_bias_cuda_naive_double(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_double_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_double_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_double_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_double_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_double_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_double_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_double_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_pn_bias_cuda_naive_float(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_float_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_float_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_float_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_float_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_float_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_float_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_float_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_pn_bias_cuda_naive_half(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_half_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_half_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_half_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_half_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_half_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_half_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_half_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_pn_bias_cuda_naive_bfloat16(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_bfloat16_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_bfloat16_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_bfloat16_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_bfloat16_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_bfloat16_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_bfloat16_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_pn_bias_cuda_naive_bfloat16_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_pn_bias_cuda_naive_double(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_double_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_double_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_double_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_double_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_double_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_double_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_double_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_pn_bias_cuda_naive_float(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_float_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_float_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_float_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_float_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_float_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_float_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_float_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_pn_bias_cuda_naive_half(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_half_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_half_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_half_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_half_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_half_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_half_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_half_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_pn_bias_cuda_naive_bfloat16(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_bfloat16_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_bfloat16_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_bfloat16_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_bfloat16_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_bfloat16_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_bfloat16_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_pn_bias_cuda_naive_bfloat16_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_nn_cuda_naive_double(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_double_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_double_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_double_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_double_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_double_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_double_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_double_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_nn_cuda_naive_float(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_float_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_float_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_float_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_float_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_float_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_float_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_float_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_nn_cuda_naive_half(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_half_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_half_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_half_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_half_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_half_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_half_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_half_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_nn_cuda_naive_bfloat16(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_bfloat16_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_bfloat16_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_bfloat16_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_bfloat16_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_bfloat16_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_bfloat16_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_nn_cuda_naive_bfloat16_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_nn_cuda_naive_double(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_double_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_double_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_double_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_double_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_double_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_double_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_double_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_nn_cuda_naive_float(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_float_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_float_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_float_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_float_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_float_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_float_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_float_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_nn_cuda_naive_half(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_half_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_half_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_half_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_half_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_half_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_half_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_half_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_nn_cuda_naive_bfloat16(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_bfloat16_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_bfloat16_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_bfloat16_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_bfloat16_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_bfloat16_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_bfloat16_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_nn_cuda_naive_bfloat16_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_nn_cuda_naive_double(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_double_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_double_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_double_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_double_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_double_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_double_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_double_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_nn_cuda_naive_float(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_float_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_float_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_float_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_float_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_float_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_float_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_float_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_nn_cuda_naive_half(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_half_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_half_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_half_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_half_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_half_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_half_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_half_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_nn_cuda_naive_bfloat16(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_bfloat16_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_bfloat16_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_bfloat16_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_bfloat16_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_bfloat16_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_bfloat16_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_nn_cuda_naive_bfloat16_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_in_cuda_naive_double(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_double_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_double_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_double_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_double_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_double_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_double_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_in_cuda_naive_double_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_in_cuda_naive_float(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_float_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_float_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_float_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_float_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_float_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_float_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_in_cuda_naive_float_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_in_cuda_naive_half(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_half_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_half_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_half_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_half_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_half_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_half_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_in_cuda_naive_half_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_in_cuda_naive_bfloat16(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_bfloat16_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_bfloat16_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_bfloat16_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_bfloat16_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_bfloat16_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_in_cuda_naive_bfloat16_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_in_cuda_naive_bfloat16_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_in_cuda_naive_double(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_double_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_double_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_double_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_double_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_double_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_double_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_in_cuda_naive_double_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_in_cuda_naive_float(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_float_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_float_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_float_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_float_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_float_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_float_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_in_cuda_naive_float_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_in_cuda_naive_half(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_half_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_half_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_half_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_half_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_half_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_half_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_in_cuda_naive_half_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_in_cuda_naive_bfloat16(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_bfloat16_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_bfloat16_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_bfloat16_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_bfloat16_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_bfloat16_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_in_cuda_naive_bfloat16_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_in_cuda_naive_bfloat16_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_in_cuda_naive_double(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_double_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_double_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_double_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_double_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_double_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_double_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_in_cuda_naive_double_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_in_cuda_naive_float(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_float_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_float_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_float_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_float_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_float_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_float_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_in_cuda_naive_float_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_in_cuda_naive_half(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_half_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_half_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_half_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_half_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_half_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_half_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_in_cuda_naive_half_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_in_cuda_naive_bfloat16(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_bfloat16_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_bfloat16_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_bfloat16_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_bfloat16_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_bfloat16_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_in_cuda_naive_bfloat16_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_in_cuda_naive_bfloat16_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_rpbgrad_cuda_naive_double(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_double_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_double_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_double_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_double_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_double_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_double_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_double_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_rpbgrad_cuda_naive_float(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_float_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_float_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_float_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_float_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_float_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_float_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_float_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_rpbgrad_cuda_naive_half(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_half_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_half_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_half_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_half_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_half_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_half_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_half_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na1d_rpbgrad_cuda_naive_bfloat16(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_bfloat16_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_bfloat16_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_bfloat16_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_bfloat16_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_bfloat16_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_bfloat16_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_bfloat16_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_rpbgrad_cuda_naive_double(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_double_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_double_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_double_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_double_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_double_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_double_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_double_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_rpbgrad_cuda_naive_float(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_float_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_float_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_float_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_float_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_float_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_float_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_float_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_rpbgrad_cuda_naive_half(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_half_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_half_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_half_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_half_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_half_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_half_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_half_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na2d_rpbgrad_cuda_naive_bfloat16(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_bfloat16_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_bfloat16_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_bfloat16_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_bfloat16_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_bfloat16_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_bfloat16_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_bfloat16_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_rpbgrad_cuda_naive_double(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_double_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_double_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_double_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_double_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_double_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_double_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_double_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_rpbgrad_cuda_naive_float(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_float_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_float_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_float_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_float_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_float_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_float_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_float_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_rpbgrad_cuda_naive_half(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_half_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_half_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_half_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_half_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_half_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_half_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_half_ks_any(dilation, __VA_ARGS__); \
    } \
}();

#define DISPATCH_KERNEL_na3d_rpbgrad_cuda_naive_bfloat16(kernel_size, dilation, ...) \
  [&] { \
        if (kernel_size == 3) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_bfloat16_ks_3(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 5) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_bfloat16_ks_5(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 7) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_bfloat16_ks_7(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 9) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_bfloat16_ks_9(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 11) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_bfloat16_ks_11(dilation, __VA_ARGS__); \
    } \
    else if (kernel_size == 13) { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_bfloat16_ks_13(dilation, __VA_ARGS__); \
    } \
    else { \
      DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_bfloat16_ks_any(dilation, __VA_ARGS__); \
    } \
}();



} // namespace {namespace} 
} // namespace {namespace} 
} // namespace {namespace} 

