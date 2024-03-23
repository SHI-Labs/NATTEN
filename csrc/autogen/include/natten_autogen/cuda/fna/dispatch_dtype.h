#pragma once


#include <iostream> 
#include <type_traits> 
#include <natten/natten.h> 
#include <natten/dtypes.cuh> 
#include <natten/cuda/fna/na_utils.cuh> 
#include <natten/cuda/fna/kernel_forward.h> 
#include <natten/cuda/fna/kernel_backward.h> 
#include <natten_autogen/cuda/fna/dispatch_cm.h> 
namespace natten { 
namespace cuda { 
namespace fna { 
#define DISPATCH_FNA_FORWARD_1D_SM50(dtype, is_causal, has_rpb, computes_lse, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_FORWARD_1D_SM50_float32(is_causal, has_rpb, computes_lse, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_FORWARD_1D_SM50_float16(is_causal, has_rpb, computes_lse, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-1D does not support this data type on SM50." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_1D_SM70(dtype, is_causal, has_rpb, computes_lse, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_FORWARD_1D_SM70_float32(is_causal, has_rpb, computes_lse, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_FORWARD_1D_SM70_float16(is_causal, has_rpb, computes_lse, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-1D does not support this data type on SM70." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_1D_SM75(dtype, is_causal, has_rpb, computes_lse, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_FORWARD_1D_SM75_float32(is_causal, has_rpb, computes_lse, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_FORWARD_1D_SM75_float16(is_causal, has_rpb, computes_lse, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-1D does not support this data type on SM75." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_1D_SM80(dtype, is_causal, has_rpb, computes_lse, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_FORWARD_1D_SM80_float32(is_causal, has_rpb, computes_lse, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_FORWARD_1D_SM80_float16(is_causal, has_rpb, computes_lse, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_FNA_FORWARD_1D_SM80_bfloat16(is_causal, has_rpb, computes_lse, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-1D does not support this data type on SM80." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D_SM50(dtype, is_causal, has_rpb, computes_lse, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_FORWARD_2D_SM50_float32(is_causal, has_rpb, computes_lse, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_FORWARD_2D_SM50_float16(is_causal, has_rpb, computes_lse, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-2D does not support this data type on SM50." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D_SM70(dtype, is_causal, has_rpb, computes_lse, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_FORWARD_2D_SM70_float32(is_causal, has_rpb, computes_lse, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_FORWARD_2D_SM70_float16(is_causal, has_rpb, computes_lse, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-2D does not support this data type on SM70." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D_SM75(dtype, is_causal, has_rpb, computes_lse, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_FORWARD_2D_SM75_float32(is_causal, has_rpb, computes_lse, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_FORWARD_2D_SM75_float16(is_causal, has_rpb, computes_lse, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-2D does not support this data type on SM75." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_2D_SM80(dtype, is_causal, has_rpb, computes_lse, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_FORWARD_2D_SM80_float32(is_causal, has_rpb, computes_lse, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_FORWARD_2D_SM80_float16(is_causal, has_rpb, computes_lse, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_FNA_FORWARD_2D_SM80_bfloat16(is_causal, has_rpb, computes_lse, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-2D does not support this data type on SM80." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D_SM50(dtype, is_causal, has_rpb, computes_lse, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_FORWARD_3D_SM50_float32(is_causal, has_rpb, computes_lse, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_FORWARD_3D_SM50_float16(is_causal, has_rpb, computes_lse, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-3D does not support this data type on SM50." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D_SM70(dtype, is_causal, has_rpb, computes_lse, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_FORWARD_3D_SM70_float32(is_causal, has_rpb, computes_lse, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_FORWARD_3D_SM70_float16(is_causal, has_rpb, computes_lse, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-3D does not support this data type on SM70." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D_SM75(dtype, is_causal, has_rpb, computes_lse, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_FORWARD_3D_SM75_float32(is_causal, has_rpb, computes_lse, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_FORWARD_3D_SM75_float16(is_causal, has_rpb, computes_lse, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-3D does not support this data type on SM75." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_FORWARD_3D_SM80(dtype, is_causal, has_rpb, computes_lse, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_FORWARD_3D_SM80_float32(is_causal, has_rpb, computes_lse, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_FORWARD_3D_SM80_float16(is_causal, has_rpb, computes_lse, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_FNA_FORWARD_3D_SM80_bfloat16(is_causal, has_rpb, computes_lse, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-3D does not support this data type on SM80." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_1D_SM50(dtype, is_causal, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_BACKWARD_1D_SM50_float32(is_causal, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_BACKWARD_1D_SM50_float16(is_causal, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-1D does not support this data type on SM50." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_1D_SM70(dtype, is_causal, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_BACKWARD_1D_SM70_float32(is_causal, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_BACKWARD_1D_SM70_float16(is_causal, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-1D does not support this data type on SM70." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_1D_SM75(dtype, is_causal, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_BACKWARD_1D_SM75_float32(is_causal, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_BACKWARD_1D_SM75_float16(is_causal, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-1D does not support this data type on SM75." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_1D_SM80(dtype, is_causal, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_BACKWARD_1D_SM80_float32(is_causal, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_BACKWARD_1D_SM80_float16(is_causal, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_FNA_BACKWARD_1D_SM80_bfloat16(is_causal, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-1D does not support this data type on SM80." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_2D_SM50(dtype, is_causal, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_BACKWARD_2D_SM50_float32(is_causal, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_BACKWARD_2D_SM50_float16(is_causal, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-2D does not support this data type on SM50." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_2D_SM70(dtype, is_causal, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_BACKWARD_2D_SM70_float32(is_causal, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_BACKWARD_2D_SM70_float16(is_causal, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-2D does not support this data type on SM70." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_2D_SM75(dtype, is_causal, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_BACKWARD_2D_SM75_float32(is_causal, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_BACKWARD_2D_SM75_float16(is_causal, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-2D does not support this data type on SM75." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_2D_SM80(dtype, is_causal, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_BACKWARD_2D_SM80_float32(is_causal, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_BACKWARD_2D_SM80_float16(is_causal, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_FNA_BACKWARD_2D_SM80_bfloat16(is_causal, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-2D does not support this data type on SM80." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_3D_SM50(dtype, is_causal, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_BACKWARD_3D_SM50_float32(is_causal, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_BACKWARD_3D_SM50_float16(is_causal, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-3D does not support this data type on SM50." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_3D_SM70(dtype, is_causal, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_BACKWARD_3D_SM70_float32(is_causal, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_BACKWARD_3D_SM70_float16(is_causal, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-3D does not support this data type on SM70." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_3D_SM75(dtype, is_causal, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_BACKWARD_3D_SM75_float32(is_causal, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_BACKWARD_3D_SM75_float16(is_causal, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-3D does not support this data type on SM75." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();

#define DISPATCH_FNA_BACKWARD_3D_SM80(dtype, is_causal, cb) \
  [&] { \
    if constexpr (std::is_same<dtype, natten::float32>::value) { \
      DISPATCH_FNA_BACKWARD_3D_SM80_float32(is_causal, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::float16>::value) { \
      DISPATCH_FNA_BACKWARD_3D_SM80_float16(is_causal, cb); \
    } \
    else if constexpr (std::is_same<dtype, natten::bfloat16>::value) { \
      DISPATCH_FNA_BACKWARD_3D_SM80_bfloat16(is_causal, cb); \
    } \
    else { \
      std::cerr << "NATTEN kernel launch failed!" \
                << "FNA-3D does not support this data type on SM80." \
                << std::endl; \
      exit(EXIT_FAILURE); \
    } \
}();



} // namespace natten 
} // namespace cuda 
} // namespace fna 

