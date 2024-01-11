#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.cuh> 
#include <natten_autogen/cuda/naive/kernels.h> 
namespace natten { 
namespace cuda { 
namespace naive { 
#define DISPATCH_DILATION_na1d_pn_cuda_naive_double_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_double_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_double_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_double_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_double_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_double_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_double_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_double_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_double_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_double_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_double_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_double_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_double_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_double_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_double_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_double_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_double_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_double_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_double_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_double_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_double_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_float_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_float_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_float_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_float_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_float_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_float_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_float_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_float_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_float_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_float_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_float_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_float_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_float_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_float_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_float_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_float_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_float_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_float_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_float_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_float_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_float_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_half_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_half_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_half_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_half_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_half_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_half_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_half_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_half_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_half_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_half_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_half_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_half_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_half_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_half_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_half_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_half_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_half_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_half_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_half_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_half_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_half_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_bfloat16_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_bfloat16_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_bfloat16_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_bfloat16_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_bfloat16_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_bfloat16_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_bfloat16_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_bfloat16_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_bfloat16_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_bfloat16_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_bfloat16_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_bfloat16_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_bfloat16_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_bfloat16_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_bfloat16_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_bfloat16_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_bfloat16_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_bfloat16_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_cuda_naive_bfloat16_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_cuda_naive_bfloat16_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_cuda_naive_bfloat16_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_double_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_double_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_double_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_double_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_double_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_double_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_double_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_double_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_double_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_double_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_double_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_double_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_double_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_double_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_double_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_double_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_double_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_double_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_double_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_double_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_double_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_float_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_float_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_float_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_float_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_float_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_float_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_float_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_float_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_float_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_float_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_float_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_float_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_float_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_float_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_float_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_float_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_float_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_float_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_float_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_float_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_float_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_half_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_half_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_half_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_half_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_half_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_half_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_half_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_half_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_half_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_half_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_half_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_half_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_half_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_half_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_half_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_half_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_half_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_half_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_half_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_half_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_half_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_bfloat16_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_bfloat16_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_bfloat16_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_bfloat16_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_bfloat16_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_bfloat16_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_bfloat16_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_bfloat16_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_bfloat16_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_bfloat16_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_bfloat16_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_bfloat16_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_bfloat16_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_bfloat16_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_bfloat16_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_bfloat16_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_bfloat16_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_bfloat16_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_cuda_naive_bfloat16_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_cuda_naive_bfloat16_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_cuda_naive_bfloat16_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_double_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_double_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_double_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_double_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_double_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_double_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_double_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_double_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_double_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_double_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_double_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_double_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_double_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_double_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_double_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_double_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_double_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_double_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_double_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_double_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_double_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_float_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_float_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_float_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_float_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_float_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_float_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_float_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_float_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_float_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_float_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_float_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_float_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_float_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_float_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_float_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_float_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_float_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_float_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_float_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_float_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_float_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_half_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_half_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_half_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_half_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_half_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_half_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_half_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_half_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_half_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_half_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_half_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_half_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_half_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_half_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_half_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_half_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_half_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_half_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_half_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_half_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_half_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_bfloat16_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_bfloat16_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_bfloat16_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_bfloat16_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_bfloat16_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_bfloat16_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_bfloat16_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_bfloat16_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_bfloat16_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_bfloat16_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_bfloat16_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_bfloat16_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_bfloat16_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_bfloat16_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_bfloat16_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_bfloat16_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_bfloat16_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_bfloat16_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_cuda_naive_bfloat16_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_cuda_naive_bfloat16_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_cuda_naive_bfloat16_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_double_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_double_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_double_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_double_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_double_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_double_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_double_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_double_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_double_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_double_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_double_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_double_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_double_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_double_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_double_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_double_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_double_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_double_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_double_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_double_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_double_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_float_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_float_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_float_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_float_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_float_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_float_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_float_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_float_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_float_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_float_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_float_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_float_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_float_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_float_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_float_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_float_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_float_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_float_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_float_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_float_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_float_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_half_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_half_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_half_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_half_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_half_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_half_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_half_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_half_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_half_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_half_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_half_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_half_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_half_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_half_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_half_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_half_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_half_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_half_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_half_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_half_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_half_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_bfloat16_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_bfloat16_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_bfloat16_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_bfloat16_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_bfloat16_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_bfloat16_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_bfloat16_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_bfloat16_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_bfloat16_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_bfloat16_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_bfloat16_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_bfloat16_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_bfloat16_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_bfloat16_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_bfloat16_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_bfloat16_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_bfloat16_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_bfloat16_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_pn_bias_cuda_naive_bfloat16_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_pn_bias_cuda_naive_bfloat16_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_pn_bias_cuda_naive_bfloat16_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_double_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_double_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_double_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_double_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_double_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_double_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_double_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_double_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_double_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_double_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_double_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_double_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_double_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_double_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_double_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_double_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_double_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_double_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_double_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_double_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_double_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_float_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_float_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_float_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_float_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_float_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_float_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_float_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_float_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_float_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_float_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_float_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_float_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_float_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_float_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_float_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_float_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_float_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_float_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_float_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_float_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_float_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_half_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_half_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_half_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_half_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_half_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_half_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_half_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_half_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_half_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_half_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_half_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_half_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_half_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_half_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_half_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_half_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_half_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_half_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_half_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_half_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_half_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_bfloat16_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_bfloat16_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_bfloat16_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_bfloat16_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_bfloat16_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_bfloat16_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_bfloat16_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_bfloat16_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_bfloat16_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_bfloat16_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_bfloat16_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_bfloat16_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_bfloat16_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_bfloat16_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_bfloat16_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_bfloat16_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_bfloat16_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_bfloat16_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_pn_bias_cuda_naive_bfloat16_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_pn_bias_cuda_naive_bfloat16_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_pn_bias_cuda_naive_bfloat16_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_double_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_double_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_double_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_double_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_double_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_double_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_double_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_double_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_double_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_double_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_double_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_double_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_double_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_double_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_double_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_double_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_double_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_double_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_double_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_double_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_double_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_float_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_float_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_float_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_float_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_float_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_float_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_float_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_float_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_float_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_float_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_float_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_float_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_float_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_float_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_float_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_float_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_float_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_float_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_float_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_float_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_float_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_half_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_half_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_half_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_half_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_half_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_half_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_half_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_half_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_half_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_half_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_half_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_half_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_half_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_half_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_half_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_half_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_half_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_half_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_half_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_half_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_half_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_bfloat16_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_bfloat16_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_bfloat16_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_bfloat16_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_bfloat16_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_bfloat16_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_bfloat16_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_bfloat16_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_bfloat16_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_bfloat16_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_bfloat16_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_bfloat16_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_bfloat16_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_bfloat16_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_bfloat16_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_bfloat16_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_bfloat16_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_bfloat16_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_pn_bias_cuda_naive_bfloat16_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_pn_bias_cuda_naive_bfloat16_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_pn_bias_cuda_naive_bfloat16_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_double_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_double_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_double_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_double_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_double_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_double_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_double_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_double_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_double_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_double_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_double_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_double_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_double_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_double_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_double_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_double_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_double_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_double_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_double_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_double_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_double_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_float_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_float_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_float_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_float_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_float_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_float_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_float_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_float_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_float_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_float_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_float_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_float_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_float_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_float_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_float_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_float_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_float_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_float_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_float_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_float_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_float_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_half_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_half_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_half_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_half_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_half_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_half_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_half_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_half_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_half_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_half_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_half_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_half_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_half_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_half_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_half_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_half_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_half_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_half_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_half_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_half_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_half_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_bfloat16_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_bfloat16_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_bfloat16_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_bfloat16_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_bfloat16_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_bfloat16_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_bfloat16_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_bfloat16_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_bfloat16_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_bfloat16_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_bfloat16_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_bfloat16_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_bfloat16_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_bfloat16_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_bfloat16_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_bfloat16_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_bfloat16_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_bfloat16_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_nn_cuda_naive_bfloat16_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_nn_cuda_naive_bfloat16_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_nn_cuda_naive_bfloat16_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_double_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_double_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_double_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_double_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_double_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_double_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_double_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_double_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_double_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_double_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_double_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_double_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_double_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_double_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_double_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_double_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_double_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_double_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_double_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_double_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_double_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_float_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_float_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_float_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_float_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_float_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_float_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_float_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_float_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_float_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_float_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_float_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_float_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_float_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_float_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_float_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_float_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_float_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_float_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_float_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_float_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_float_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_half_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_half_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_half_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_half_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_half_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_half_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_half_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_half_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_half_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_half_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_half_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_half_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_half_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_half_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_half_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_half_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_half_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_half_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_half_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_half_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_half_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_bfloat16_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_bfloat16_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_bfloat16_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_bfloat16_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_bfloat16_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_bfloat16_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_bfloat16_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_bfloat16_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_bfloat16_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_bfloat16_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_bfloat16_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_bfloat16_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_bfloat16_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_bfloat16_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_bfloat16_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_bfloat16_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_bfloat16_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_bfloat16_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_nn_cuda_naive_bfloat16_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_nn_cuda_naive_bfloat16_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_nn_cuda_naive_bfloat16_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_double_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_double_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_double_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_double_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_double_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_double_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_double_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_double_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_double_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_double_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_double_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_double_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_double_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_double_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_double_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_double_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_double_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_double_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_double_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_double_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_double_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_float_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_float_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_float_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_float_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_float_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_float_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_float_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_float_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_float_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_float_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_float_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_float_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_float_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_float_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_float_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_float_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_float_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_float_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_float_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_float_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_float_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_half_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_half_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_half_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_half_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_half_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_half_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_half_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_half_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_half_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_half_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_half_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_half_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_half_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_half_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_half_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_half_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_half_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_half_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_half_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_half_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_half_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_bfloat16_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_bfloat16_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_bfloat16_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_bfloat16_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_bfloat16_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_bfloat16_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_bfloat16_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_bfloat16_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_bfloat16_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_bfloat16_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_bfloat16_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_bfloat16_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_bfloat16_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_bfloat16_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_bfloat16_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_bfloat16_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_bfloat16_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_bfloat16_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_nn_cuda_naive_bfloat16_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_nn_cuda_naive_bfloat16_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_nn_cuda_naive_bfloat16_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_double_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_double_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_double_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_double_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_double_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_double_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_double_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_double_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_double_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_double_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_double_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_double_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_double_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_double_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_double_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_double_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_double_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_double_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_double_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_double_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_double_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_float_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_float_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_float_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_float_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_float_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_float_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_float_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_float_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_float_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_float_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_float_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_float_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_float_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_float_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_float_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_float_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_float_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_float_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_float_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_float_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_float_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_half_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_half_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_half_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_half_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_half_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_half_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_half_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_half_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_half_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_half_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_half_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_half_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_half_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_half_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_half_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_half_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_half_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_half_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_half_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_half_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_half_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_bfloat16_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_bfloat16_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_bfloat16_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_bfloat16_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_bfloat16_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_bfloat16_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_bfloat16_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_bfloat16_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_bfloat16_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_bfloat16_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_bfloat16_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_bfloat16_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_bfloat16_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_bfloat16_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_bfloat16_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_bfloat16_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_bfloat16_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_bfloat16_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_in_cuda_naive_bfloat16_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_in_cuda_naive_bfloat16_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_in_cuda_naive_bfloat16_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_double_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_double_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_double_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_double_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_double_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_double_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_double_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_double_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_double_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_double_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_double_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_double_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_double_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_double_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_double_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_double_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_double_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_double_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_double_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_double_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_double_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_float_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_float_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_float_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_float_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_float_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_float_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_float_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_float_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_float_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_float_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_float_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_float_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_float_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_float_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_float_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_float_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_float_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_float_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_float_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_float_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_float_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_half_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_half_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_half_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_half_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_half_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_half_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_half_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_half_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_half_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_half_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_half_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_half_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_half_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_half_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_half_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_half_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_half_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_half_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_half_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_half_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_half_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_bfloat16_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_bfloat16_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_bfloat16_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_bfloat16_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_bfloat16_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_bfloat16_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_bfloat16_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_bfloat16_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_bfloat16_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_bfloat16_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_bfloat16_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_bfloat16_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_bfloat16_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_bfloat16_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_bfloat16_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_bfloat16_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_bfloat16_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_bfloat16_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_in_cuda_naive_bfloat16_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_in_cuda_naive_bfloat16_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_in_cuda_naive_bfloat16_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_double_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_double_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_double_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_double_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_double_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_double_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_double_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_double_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_double_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_double_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_double_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_double_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_double_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_double_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_double_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_double_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_double_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_double_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_double_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_double_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_double_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_float_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_float_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_float_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_float_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_float_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_float_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_float_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_float_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_float_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_float_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_float_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_float_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_float_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_float_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_float_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_float_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_float_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_float_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_float_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_float_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_float_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_half_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_half_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_half_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_half_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_half_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_half_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_half_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_half_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_half_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_half_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_half_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_half_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_half_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_half_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_half_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_half_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_half_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_half_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_half_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_half_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_half_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_bfloat16_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_bfloat16_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_bfloat16_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_bfloat16_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_bfloat16_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_bfloat16_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_bfloat16_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_bfloat16_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_bfloat16_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_bfloat16_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_bfloat16_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_bfloat16_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_bfloat16_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_bfloat16_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_bfloat16_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_bfloat16_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_bfloat16_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_bfloat16_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_in_cuda_naive_bfloat16_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_in_cuda_naive_bfloat16_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_in_cuda_naive_bfloat16_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_double_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_double_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_double_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_double_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_double_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_double_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_double_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_double_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_double_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_double_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_double_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_double_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_double_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_double_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_double_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_double_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_double_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_double_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_double_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_double_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_double_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_float_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_float_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_float_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_float_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_float_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_float_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_float_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_float_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_float_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_float_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_float_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_float_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_float_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_float_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_float_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_float_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_float_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_float_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_float_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_float_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_float_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_half_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_half_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_half_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_half_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_half_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_half_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_half_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_half_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_half_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_half_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_half_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_half_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_half_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_half_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_half_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_half_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_half_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_half_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_half_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_half_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_half_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_bfloat16_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_bfloat16_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_bfloat16_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_bfloat16_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_bfloat16_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_bfloat16_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_bfloat16_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_bfloat16_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_bfloat16_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_bfloat16_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_bfloat16_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_bfloat16_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_bfloat16_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_bfloat16_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_bfloat16_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_bfloat16_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_bfloat16_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_bfloat16_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na1d_rpbgrad_cuda_naive_bfloat16_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na1d_rpbgrad_cuda_naive_bfloat16_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na1d_rpbgrad_cuda_naive_bfloat16_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_double_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_double_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_double_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_double_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_double_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_double_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_double_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_double_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_double_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_double_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_double_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_double_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_double_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_double_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_double_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_double_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_double_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_double_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_double_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_double_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_double_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_float_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_float_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_float_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_float_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_float_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_float_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_float_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_float_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_float_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_float_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_float_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_float_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_float_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_float_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_float_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_float_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_float_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_float_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_float_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_float_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_float_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_half_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_half_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_half_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_half_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_half_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_half_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_half_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_half_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_half_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_half_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_half_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_half_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_half_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_half_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_half_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_half_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_half_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_half_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_half_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_half_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_half_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_bfloat16_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_bfloat16_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_bfloat16_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_bfloat16_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_bfloat16_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_bfloat16_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_bfloat16_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_bfloat16_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_bfloat16_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_bfloat16_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_bfloat16_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_bfloat16_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_bfloat16_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_bfloat16_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_bfloat16_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_bfloat16_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_bfloat16_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_bfloat16_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na2d_rpbgrad_cuda_naive_bfloat16_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na2d_rpbgrad_cuda_naive_bfloat16_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na2d_rpbgrad_cuda_naive_bfloat16_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_double_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_double_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_double_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_double_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_double_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_double_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_double_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_double_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_double_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_double_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_double_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_double_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_double_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_double_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_double_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_double_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_double_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_double_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_double_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_double_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_double_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_float_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_float_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_float_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_float_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_float_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_float_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_float_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_float_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_float_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_float_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_float_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_float_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_float_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_float_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_float_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_float_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_float_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_float_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_float_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_float_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_float_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_half_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_half_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_half_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_half_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_half_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_half_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_half_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_half_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_half_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_half_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_half_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_half_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_half_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_half_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_half_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_half_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_half_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_half_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_half_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_half_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_half_ks_13_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_bfloat16_ks_any(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_bfloat16_ks_any_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_bfloat16_ks_any_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_bfloat16_ks_3(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_bfloat16_ks_3_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_bfloat16_ks_3_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_bfloat16_ks_5(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_bfloat16_ks_5_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_bfloat16_ks_5_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_bfloat16_ks_7(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_bfloat16_ks_7_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_bfloat16_ks_7_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_bfloat16_ks_9(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_bfloat16_ks_9_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_bfloat16_ks_9_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_bfloat16_ks_11(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_bfloat16_ks_11_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_bfloat16_ks_11_di_any(__VA_ARGS__); \
    } \
}();

#define DISPATCH_DILATION_na3d_rpbgrad_cuda_naive_bfloat16_ks_13(dilation, ...) \
  [&] { \
        if (dilation == 1) { \
      naive::na3d_rpbgrad_cuda_naive_bfloat16_ks_13_di_1(__VA_ARGS__); \
    } \
    else { \
      naive::na3d_rpbgrad_cuda_naive_bfloat16_ks_13_di_any(__VA_ARGS__); \
    } \
}();



} // namespace {namespace} 
} // namespace {namespace} 
} // namespace {namespace} 

