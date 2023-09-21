#include <cuda_runtime.h>
#include <natten/dtypes.cuh>
#include <natten/cuda/gemm/na1d.cuh>
#include <natten/gemm_argpack.cuh>
#include <natten/config.h>
namespace natten { 
namespace cuda { 
namespace gemm { 

void na1d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig<64, 64, 32, 32, 32, 32, 16, 8, 8, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<75, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig<64, 64, 32, 32, 32, 32, 16, 8, 8, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<75, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig<64, 64, 32, 32, 32, 32, 16, 8, 8, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<75, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

} 
} 
} 

