#pragma once


#include <iostream> 
#include <type_traits> 
#ifdef NATTEN_WITH_CUTLASS
#include <natten/natten.h> 
#include <ATen/ATen.h> 
#include <ATen/cuda/CUDAContext.h> 
#include <c10/cuda/CUDAGuard.h> 
#include <c10/cuda/CUDAStream.h> 
#include <torch/extension.h> 
#include <natten/natten.h> 
#include <natten/helpers.h> 
#include <natten/cuda/reference/fna_reference_forward.hpp> 
#include <natten/cuda/reference/fna_reference_backward.hpp> 
#include <natten_autogen/cuda/reference/dispatch_dtype.h> 
namespace natten { 
namespace cuda { 
namespace reference { 
#define DISPATCH_REFERENCE_FNA_FORWARD(rank, dtype, is_causal, ...) \
  [&] { \
    if constexpr (rank == 1) { \
      DISPATCH_REFERENCE_FNA_FORWARD_1D(dtype, is_causal, __VA_ARGS__); \
    } \
    else if constexpr (rank == 2) { \
      DISPATCH_REFERENCE_FNA_FORWARD_2D(dtype, is_causal, __VA_ARGS__); \
    } \
    else if constexpr (rank == 3) { \
      DISPATCH_REFERENCE_FNA_FORWARD_3D(dtype, is_causal, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "Reference FNA kernel launch failed!" \
                << "NATTEN only supports NA1D, 2D, and 3D!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_REFERENCE_FNA_BACKWARD(rank, dtype, is_causal, ...) \
  [&] { \
    if constexpr (rank == 1) { \
      DISPATCH_REFERENCE_FNA_BACKWARD_1D(dtype, is_causal, __VA_ARGS__); \
    } \
    else if constexpr (rank == 2) { \
      DISPATCH_REFERENCE_FNA_BACKWARD_2D(dtype, is_causal, __VA_ARGS__); \
    } \
    else if constexpr (rank == 3) { \
      DISPATCH_REFERENCE_FNA_BACKWARD_3D(dtype, is_causal, __VA_ARGS__); \
    } \
    else { \
      std::cerr << "Reference FNA kernel launch failed!" \
                << "NATTEN only supports NA1D, 2D, and 3D!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace reference 
#endif 

