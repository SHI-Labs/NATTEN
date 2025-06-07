#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/natten.h> 
#include <natten/cuda/fmha/kernel_forward.h> 
#include <natten/cuda/fmha/kernel_backward.h> 
#include <natten_autogen/cuda/fmha/dispatch_dtype.h> 
namespace natten { 
namespace cuda { 
namespace fmha { 
#define DISPATCH_FMHA_FORWARD(cc, dtype, cb) \
  [&] { \
    if (cc < 70 && cc >= 50) { \
      DISPATCH_FMHA_FORWARD_SM50(dtype, cb); \
    } \
    else if (cc < 75 && cc >= 70) { \
      DISPATCH_FMHA_FORWARD_SM70(dtype, cb); \
    } \
    else if (cc < 80 && cc >= 75) { \
      DISPATCH_FMHA_FORWARD_SM75(dtype, cb); \
    } \
    else if (cc >= 80) { \
      DISPATCH_FMHA_FORWARD_SM80(dtype, cb); \
    } \
    else { \
      std::cerr << "NATTEN FMHA kernel launch failed!" \
                << "FMHA is not implemented for this device." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FMHA_BACKWARD(cc, dtype, cb) \
  [&] { \
    if (cc < 70 && cc >= 50) { \
      DISPATCH_FMHA_BACKWARD_SM50(dtype, cb); \
    } \
    else if (cc < 75 && cc >= 70) { \
      DISPATCH_FMHA_BACKWARD_SM70(dtype, cb); \
    } \
    else if (cc < 80 && cc >= 75) { \
      DISPATCH_FMHA_BACKWARD_SM75(dtype, cb); \
    } \
    else if (cc >= 80) { \
      DISPATCH_FMHA_BACKWARD_SM80(dtype, cb); \
    } \
    else { \
      std::cerr << "NATTEN FMHA kernel launch failed!" \
                << "FMHA is not implemented for this device." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fmha 

