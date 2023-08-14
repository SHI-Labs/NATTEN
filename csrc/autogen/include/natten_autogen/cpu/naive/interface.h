#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/dtypes.h> 
#include <natten_autogen/cpu/naive/kernels.h> 
namespace natten { 
namespace cpu { 
namespace naive { 
#define DISPATCH_DTYPE_na1d_pn_cpu_naive(dtype, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      naive::na1d_pn_cpu_naive_double(__VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      naive::na1d_pn_cpu_naive_float(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_cpu_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na2d_pn_cpu_naive(dtype, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      naive::na2d_pn_cpu_naive_double(__VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      naive::na2d_pn_cpu_naive_float(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_cpu_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na3d_pn_cpu_naive(dtype, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      naive::na3d_pn_cpu_naive_double(__VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      naive::na3d_pn_cpu_naive_float(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_pn_cpu_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na1d_pn_bias_cpu_naive(dtype, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      naive::na1d_pn_bias_cpu_naive_double(__VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      naive::na1d_pn_bias_cpu_naive_float(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_pn_bias_cpu_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na2d_pn_bias_cpu_naive(dtype, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      naive::na2d_pn_bias_cpu_naive_double(__VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      naive::na2d_pn_bias_cpu_naive_float(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_pn_bias_cpu_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na3d_pn_bias_cpu_naive(dtype, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      naive::na3d_pn_bias_cpu_naive_double(__VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      naive::na3d_pn_bias_cpu_naive_float(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_pn_bias_cpu_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na1d_nn_cpu_naive(dtype, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      naive::na1d_nn_cpu_naive_double(__VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      naive::na1d_nn_cpu_naive_float(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_nn_cpu_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na2d_nn_cpu_naive(dtype, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      naive::na2d_nn_cpu_naive_double(__VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      naive::na2d_nn_cpu_naive_float(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_nn_cpu_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na3d_nn_cpu_naive(dtype, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      naive::na3d_nn_cpu_naive_double(__VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      naive::na3d_nn_cpu_naive_float(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_nn_cpu_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na1d_in_cpu_naive(dtype, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      naive::na1d_in_cpu_naive_double(__VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      naive::na1d_in_cpu_naive_float(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_in_cpu_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na2d_in_cpu_naive(dtype, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      naive::na2d_in_cpu_naive_double(__VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      naive::na2d_in_cpu_naive_float(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_in_cpu_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na3d_in_cpu_naive(dtype, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      naive::na3d_in_cpu_naive_double(__VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      naive::na3d_in_cpu_naive_float(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_in_cpu_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na1d_rpbgrad_cpu_naive(dtype, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      naive::na1d_rpbgrad_cpu_naive_double(__VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      naive::na1d_rpbgrad_cpu_naive_float(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na1d_rpbgrad_cpu_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na2d_rpbgrad_cpu_naive(dtype, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      naive::na2d_rpbgrad_cpu_naive_double(__VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      naive::na2d_rpbgrad_cpu_naive_float(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na2d_rpbgrad_cpu_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_DTYPE_na3d_rpbgrad_cpu_naive(dtype, ...) \
  [&] { \
    if (std::is_same<dtype, natten::float64>::value) { \
      naive::na3d_rpbgrad_cpu_naive_double(__VA_ARGS__); \
    } \
    else if (std::is_same<dtype, natten::float32>::value) { \
      naive::na3d_rpbgrad_cpu_naive_float(__VA_ARGS__); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "na3d_rpbgrad_cpu_naive does not support this data type." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cpu 
} // namespace naive 

