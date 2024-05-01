#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/natten.h> 
#include <natten/dtypes.cuh> 
#include <natten/cuda/fna/na_utils.cuh> 
#include <natten/cuda/fna/kernel_forward.h> 
#include <natten/cuda/fna/kernel_backward.h> 
#include <natten_autogen/cuda/fna/dispatch_dtype.h> 
namespace natten { 
namespace cuda { 
namespace fna { 
#define DISPATCH_FNA_FORWARD_1D(cc, dtype, is_causal, has_rpb, computes_lse, cb) \
  [&] { \
    if (cc < 70 && cc >= 50) { \
      DISPATCH_FNA_FORWARD_1D_SM50(dtype, is_causal, has_rpb, computes_lse, cb); \
    } \
    else if (cc < 75 && cc >= 70) { \
      DISPATCH_FNA_FORWARD_1D_SM70(dtype, is_causal, has_rpb, computes_lse, cb); \
    } \
    else if (cc < 80 && cc >= 75) { \
      DISPATCH_FNA_FORWARD_1D_SM75(dtype, is_causal, has_rpb, computes_lse, cb); \
    } \
    else if (cc < 100 && cc >= 80) { \
      DISPATCH_FNA_FORWARD_1D_SM80(dtype, is_causal, has_rpb, computes_lse, cb); \
    } \
    else { \
      std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Fused neighborhood attention is not implemented for this device." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D(cc, dtype, is_causal, has_rpb, computes_lse, cb) \
  [&] { \
    if (cc < 70 && cc >= 50) { \
      DISPATCH_FNA_FORWARD_2D_SM50(dtype, is_causal, has_rpb, computes_lse, cb); \
    } \
    else if (cc < 75 && cc >= 70) { \
      DISPATCH_FNA_FORWARD_2D_SM70(dtype, is_causal, has_rpb, computes_lse, cb); \
    } \
    else if (cc < 80 && cc >= 75) { \
      DISPATCH_FNA_FORWARD_2D_SM75(dtype, is_causal, has_rpb, computes_lse, cb); \
    } \
    else if (cc < 100 && cc >= 80) { \
      DISPATCH_FNA_FORWARD_2D_SM80(dtype, is_causal, has_rpb, computes_lse, cb); \
    } \
    else { \
      std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Fused neighborhood attention is not implemented for this device." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D(cc, dtype, is_causal, has_rpb, computes_lse, cb) \
  [&] { \
    if (cc < 70 && cc >= 50) { \
      DISPATCH_FNA_FORWARD_3D_SM50(dtype, is_causal, has_rpb, computes_lse, cb); \
    } \
    else if (cc < 75 && cc >= 70) { \
      DISPATCH_FNA_FORWARD_3D_SM70(dtype, is_causal, has_rpb, computes_lse, cb); \
    } \
    else if (cc < 80 && cc >= 75) { \
      DISPATCH_FNA_FORWARD_3D_SM75(dtype, is_causal, has_rpb, computes_lse, cb); \
    } \
    else if (cc < 100 && cc >= 80) { \
      DISPATCH_FNA_FORWARD_3D_SM80(dtype, is_causal, has_rpb, computes_lse, cb); \
    } \
    else { \
      std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Fused neighborhood attention is not implemented for this device." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_1D(cc, dtype, is_causal, cb) \
  [&] { \
    if (cc < 70 && cc >= 50) { \
      DISPATCH_FNA_BACKWARD_1D_SM50(dtype, is_causal, cb); \
    } \
    else if (cc < 75 && cc >= 70) { \
      DISPATCH_FNA_BACKWARD_1D_SM70(dtype, is_causal, cb); \
    } \
    else if (cc < 80 && cc >= 75) { \
      DISPATCH_FNA_BACKWARD_1D_SM75(dtype, is_causal, cb); \
    } \
    else if (cc < 100 && cc >= 80) { \
      DISPATCH_FNA_BACKWARD_1D_SM80(dtype, is_causal, cb); \
    } \
    else { \
      std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Fused neighborhood attention is not implemented for this device." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_2D(cc, dtype, is_causal, cb) \
  [&] { \
    if (cc < 70 && cc >= 50) { \
      DISPATCH_FNA_BACKWARD_2D_SM50(dtype, is_causal, cb); \
    } \
    else if (cc < 75 && cc >= 70) { \
      DISPATCH_FNA_BACKWARD_2D_SM70(dtype, is_causal, cb); \
    } \
    else if (cc < 80 && cc >= 75) { \
      DISPATCH_FNA_BACKWARD_2D_SM75(dtype, is_causal, cb); \
    } \
    else if (cc < 100 && cc >= 80) { \
      DISPATCH_FNA_BACKWARD_2D_SM80(dtype, is_causal, cb); \
    } \
    else { \
      std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Fused neighborhood attention is not implemented for this device." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_3D(cc, dtype, is_causal, cb) \
  [&] { \
    if (cc < 70 && cc >= 50) { \
      DISPATCH_FNA_BACKWARD_3D_SM50(dtype, is_causal, cb); \
    } \
    else if (cc < 75 && cc >= 70) { \
      DISPATCH_FNA_BACKWARD_3D_SM70(dtype, is_causal, cb); \
    } \
    else if (cc < 80 && cc >= 75) { \
      DISPATCH_FNA_BACKWARD_3D_SM75(dtype, is_causal, cb); \
    } \
    else if (cc < 100 && cc >= 80) { \
      DISPATCH_FNA_BACKWARD_3D_SM80(dtype, is_causal, cb); \
    } \
    else { \
      std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Fused neighborhood attention is not implemented for this device." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna 

