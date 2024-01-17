#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/natten.h> 
#include <natten/dtypes.cuh> 
#include <natten/cuda/fna/na_utils.cuh> 
#include <natten/cuda/fna/kernel_forward.h> 
#include <natten_autogen/cuda/fna/kernels.h> 
namespace natten { 
namespace cuda { 
namespace fna { 
#define DISPATCH_FNA_FORWARD_1D_SM50_float32(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
 if (has_rpb) {\
  fna1_sm50_float32_cm_0_rpb(cb); \
 } else {\
      fna1_sm50_float32_cm_0(cb); \
 } \
    } \
    else if (std::get<0>(is_causal)) { \
      fna1_sm50_float32_cm_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-1D, SM50, float32) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_1D_SM50_float16(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
 if (has_rpb) {\
  fna1_sm50_float16_cm_0_rpb(cb); \
 } else {\
      fna1_sm50_float16_cm_0(cb); \
 } \
    } \
    else if (std::get<0>(is_causal)) { \
      fna1_sm50_float16_cm_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-1D, SM50, float16) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_1D_SM70_float32(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
 if (has_rpb) {\
  fna1_sm70_float32_cm_0_rpb(cb); \
 } else {\
      fna1_sm70_float32_cm_0(cb); \
 } \
    } \
    else if (std::get<0>(is_causal)) { \
      fna1_sm70_float32_cm_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-1D, SM70, float32) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_1D_SM70_float16(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
 if (has_rpb) {\
  fna1_sm70_float16_cm_0_rpb(cb); \
 } else {\
      fna1_sm70_float16_cm_0(cb); \
 } \
    } \
    else if (std::get<0>(is_causal)) { \
      fna1_sm70_float16_cm_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-1D, SM70, float16) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_1D_SM75_float32(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
 if (has_rpb) {\
  fna1_sm75_float32_cm_0_rpb(cb); \
 } else {\
      fna1_sm75_float32_cm_0(cb); \
 } \
    } \
    else if (std::get<0>(is_causal)) { \
      fna1_sm75_float32_cm_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-1D, SM75, float32) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_1D_SM75_float16(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
 if (has_rpb) {\
  fna1_sm75_float16_cm_0_rpb(cb); \
 } else {\
      fna1_sm75_float16_cm_0(cb); \
 } \
    } \
    else if (std::get<0>(is_causal)) { \
      fna1_sm75_float16_cm_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-1D, SM75, float16) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_1D_SM80_float32(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
 if (has_rpb) {\
  fna1_sm80_float32_cm_0_rpb(cb); \
 } else {\
      fna1_sm80_float32_cm_0(cb); \
 } \
    } \
    else if (std::get<0>(is_causal)) { \
      fna1_sm80_float32_cm_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-1D, SM80, float32) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_1D_SM80_float16(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
 if (has_rpb) {\
  fna1_sm80_float16_cm_0_rpb(cb); \
 } else {\
      fna1_sm80_float16_cm_0(cb); \
 } \
    } \
    else if (std::get<0>(is_causal)) { \
      fna1_sm80_float16_cm_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-1D, SM80, float16) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_1D_SM80_bfloat16(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal)) { \
 if (has_rpb) {\
  fna1_sm80_bfloat16_cm_0_rpb(cb); \
 } else {\
      fna1_sm80_bfloat16_cm_0(cb); \
 } \
    } \
    else if (std::get<0>(is_causal)) { \
      fna1_sm80_bfloat16_cm_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-1D, SM80, bfloat16) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D_SM50_float32(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
 if (has_rpb) {\
  fna2_sm50_float32_cm_0_0_rpb(cb); \
 } else {\
      fna2_sm50_float32_cm_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm50_float32_cm_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      fna2_sm50_float32_cm_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm50_float32_cm_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-2D, SM50, float32) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D_SM50_float16(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
 if (has_rpb) {\
  fna2_sm50_float16_cm_0_0_rpb(cb); \
 } else {\
      fna2_sm50_float16_cm_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm50_float16_cm_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      fna2_sm50_float16_cm_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm50_float16_cm_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-2D, SM50, float16) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D_SM70_float32(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
 if (has_rpb) {\
  fna2_sm70_float32_cm_0_0_rpb(cb); \
 } else {\
      fna2_sm70_float32_cm_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm70_float32_cm_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      fna2_sm70_float32_cm_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm70_float32_cm_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-2D, SM70, float32) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D_SM70_float16(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
 if (has_rpb) {\
  fna2_sm70_float16_cm_0_0_rpb(cb); \
 } else {\
      fna2_sm70_float16_cm_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm70_float16_cm_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      fna2_sm70_float16_cm_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm70_float16_cm_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-2D, SM70, float16) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D_SM75_float32(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
 if (has_rpb) {\
  fna2_sm75_float32_cm_0_0_rpb(cb); \
 } else {\
      fna2_sm75_float32_cm_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm75_float32_cm_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      fna2_sm75_float32_cm_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm75_float32_cm_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-2D, SM75, float32) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D_SM75_float16(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
 if (has_rpb) {\
  fna2_sm75_float16_cm_0_0_rpb(cb); \
 } else {\
      fna2_sm75_float16_cm_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm75_float16_cm_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      fna2_sm75_float16_cm_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm75_float16_cm_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-2D, SM75, float16) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D_SM80_float32(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
 if (has_rpb) {\
  fna2_sm80_float32_cm_0_0_rpb(cb); \
 } else {\
      fna2_sm80_float32_cm_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm80_float32_cm_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      fna2_sm80_float32_cm_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm80_float32_cm_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-2D, SM80, float32) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D_SM80_float16(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
 if (has_rpb) {\
  fna2_sm80_float16_cm_0_0_rpb(cb); \
 } else {\
      fna2_sm80_float16_cm_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm80_float16_cm_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      fna2_sm80_float16_cm_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm80_float16_cm_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-2D, SM80, float16) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D_SM80_bfloat16(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
 if (has_rpb) {\
  fna2_sm80_bfloat16_cm_0_0_rpb(cb); \
 } else {\
      fna2_sm80_bfloat16_cm_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm80_bfloat16_cm_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal)) { \
      fna2_sm80_bfloat16_cm_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal)) { \
      fna2_sm80_bfloat16_cm_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-2D, SM80, bfloat16) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D_SM50_float32(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
 if (has_rpb) {\
  fna3_sm50_float32_cm_0_0_0_rpb(cb); \
 } else {\
      fna3_sm50_float32_cm_0_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm50_float32_cm_0_0_1(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm50_float32_cm_0_1_0(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm50_float32_cm_0_1_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm50_float32_cm_1_0_0(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm50_float32_cm_1_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm50_float32_cm_1_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm50_float32_cm_1_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-3D, SM50, float32) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D_SM50_float16(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
 if (has_rpb) {\
  fna3_sm50_float16_cm_0_0_0_rpb(cb); \
 } else {\
      fna3_sm50_float16_cm_0_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm50_float16_cm_0_0_1(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm50_float16_cm_0_1_0(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm50_float16_cm_0_1_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm50_float16_cm_1_0_0(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm50_float16_cm_1_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm50_float16_cm_1_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm50_float16_cm_1_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-3D, SM50, float16) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D_SM70_float32(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
 if (has_rpb) {\
  fna3_sm70_float32_cm_0_0_0_rpb(cb); \
 } else {\
      fna3_sm70_float32_cm_0_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm70_float32_cm_0_0_1(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm70_float32_cm_0_1_0(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm70_float32_cm_0_1_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm70_float32_cm_1_0_0(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm70_float32_cm_1_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm70_float32_cm_1_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm70_float32_cm_1_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-3D, SM70, float32) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D_SM70_float16(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
 if (has_rpb) {\
  fna3_sm70_float16_cm_0_0_0_rpb(cb); \
 } else {\
      fna3_sm70_float16_cm_0_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm70_float16_cm_0_0_1(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm70_float16_cm_0_1_0(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm70_float16_cm_0_1_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm70_float16_cm_1_0_0(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm70_float16_cm_1_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm70_float16_cm_1_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm70_float16_cm_1_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-3D, SM70, float16) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D_SM75_float32(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
 if (has_rpb) {\
  fna3_sm75_float32_cm_0_0_0_rpb(cb); \
 } else {\
      fna3_sm75_float32_cm_0_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm75_float32_cm_0_0_1(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm75_float32_cm_0_1_0(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm75_float32_cm_0_1_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm75_float32_cm_1_0_0(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm75_float32_cm_1_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm75_float32_cm_1_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm75_float32_cm_1_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-3D, SM75, float32) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D_SM75_float16(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
 if (has_rpb) {\
  fna3_sm75_float16_cm_0_0_0_rpb(cb); \
 } else {\
      fna3_sm75_float16_cm_0_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm75_float16_cm_0_0_1(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm75_float16_cm_0_1_0(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm75_float16_cm_0_1_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm75_float16_cm_1_0_0(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm75_float16_cm_1_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm75_float16_cm_1_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm75_float16_cm_1_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-3D, SM75, float16) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D_SM80_float32(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
 if (has_rpb) {\
  fna3_sm80_float32_cm_0_0_0_rpb(cb); \
 } else {\
      fna3_sm80_float32_cm_0_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm80_float32_cm_0_0_1(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm80_float32_cm_0_1_0(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm80_float32_cm_0_1_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm80_float32_cm_1_0_0(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm80_float32_cm_1_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm80_float32_cm_1_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm80_float32_cm_1_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-3D, SM80, float32) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D_SM80_float16(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
 if (has_rpb) {\
  fna3_sm80_float16_cm_0_0_0_rpb(cb); \
 } else {\
      fna3_sm80_float16_cm_0_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm80_float16_cm_0_0_1(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm80_float16_cm_0_1_0(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm80_float16_cm_0_1_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm80_float16_cm_1_0_0(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm80_float16_cm_1_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm80_float16_cm_1_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm80_float16_cm_1_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-3D, SM80, float16) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D_SM80_bfloat16(is_causal, has_rpb, cb) \
  [&] { \
    if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
 if (has_rpb) {\
  fna3_sm80_bfloat16_cm_0_0_0_rpb(cb); \
 } else {\
      fna3_sm80_bfloat16_cm_0_0_0(cb); \
 } \
    } \
    else if (!std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm80_bfloat16_cm_0_0_1(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm80_bfloat16_cm_0_1_0(cb); \
    } \
    else if (!std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm80_bfloat16_cm_0_1_1(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm80_bfloat16_cm_1_0_0(cb); \
    } \
    else if (std::get<0>(is_causal) && !std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm80_bfloat16_cm_1_0_1(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && !std::get<2>(is_causal)) { \
      fna3_sm80_bfloat16_cm_1_1_0(cb); \
    } \
    else if (std::get<0>(is_causal) && std::get<1>(is_causal) && std::get<2>(is_causal)) { \
      fna3_sm80_bfloat16_cm_1_1_1(cb); \
    } \
    else { \
          std::cerr << "NATTEN FNA kernel launch failed!" \
                << "Causal mask dispatcher (FNA-3D, SM80, bfloat16) got invalid causal mask!" \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna 

