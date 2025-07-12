#ifdef NATTEN_WITH_CUTLASS
#ifdef NATTEN_WITH_BLACKWELL_FNA
#include <cuda_runtime.h>
#include <iostream>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <natten/natten.h>
#include <natten/helpers.h>
#include <natten/cuda/fna_blackwell/fna_backward.cuh>
#include <natten_autogen/cuda/blackwell_fna_bwd/kernels.h>
namespace natten { 
namespace cuda { 
namespace fna_blackwell { 

} // namespace fna_blackwell 
} // namespace cuda 
} // namespace natten 
#endif 
#endif 

