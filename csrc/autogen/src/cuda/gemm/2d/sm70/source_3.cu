#include <cuda_runtime.h>
#include <natten/cuda/gemm/na2d.cuh>
#include <natten/dtypes.cuh>
#include <natten/config.h>
#include <natten/gemm_argpack.cuh>
namespace natten { 
namespace cuda { 
namespace gemm { 

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 6>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 6>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks13_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 6>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 7>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 7>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks15_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 7>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 8>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 8>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks17_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 8>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 9>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 9>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks19_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 9>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks21_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 10>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks21_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 10>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks21_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 10>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks23_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 11>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks23_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 11>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks23_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 11>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks25_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 12>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks25_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 12>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks25_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 12>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks27_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 13>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks27_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 13>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks27_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 13>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks29_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 14>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks29_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 14>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks29_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 14>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks31_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 15>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks31_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 15>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks31_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 15>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks33_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 16>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks33_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 16>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_8x8x4_2_sm70_ks33_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 8, 8, 4, 2, 8, 4, 16>;
  using ArchConfig = natten::gemm::detail::ArchArgs<70, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

} 
} 
} 

