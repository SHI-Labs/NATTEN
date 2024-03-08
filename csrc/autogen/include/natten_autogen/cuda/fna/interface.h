#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/natten.h> 
#include <natten/dtypes.cuh> 
#include <natten/cuda/fna/na_utils.cuh> 
#include <natten/cuda/fna/kernel_forward.h> 
#include <natten_autogen/cuda/fna/dispatch_device.h> 
namespace natten { 
namespace cuda { 
namespace fna { 
#define DISPATCH_FNA_FORWARD_KERNEL(rank, cc, dtype, is_causal, has_rpb, cb) \
  [&] { \
    if constexpr (rank == 1) { \
      DISPATCH_FNA_FORWARD_1D(cc, dtype, is_causal, has_rpb, cb); \
    } \
    else if constexpr (rank == 2) { \
      DISPATCH_FNA_FORWARD_2D(cc, dtype, is_causal, has_rpb, cb); \
    } \
    else if constexpr (rank == 3) { \
      DISPATCH_FNA_FORWARD_3D(cc, dtype, is_causal, has_rpb, cb); \
    } \
    else { \
      std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Invalid spatial extent rank! Only 1, 2, and 3D are supported!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna 

