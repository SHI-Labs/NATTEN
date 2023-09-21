#include <cuda_runtime.h>
#include <natten/dtypes.cuh>
#include <natten/cuda/gemm/na1d.cuh>
#include <natten/gemm_argpack.cuh>
#include <natten/config.h>
namespace natten { 
namespace cuda { 
namespace gemm { 

void na1d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_sm80_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig<128, 128, 16, 64, 64, 16, 8, 8, 4, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = PointwiseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);
}

void na1d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig<128, 128, 16, 64, 64, 16, 16, 8, 8, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

    }
}

void na1d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig<128, 128, 16, 64, 64, 16, 16, 8, 8, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

    }
}

void na1d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig<128, 128, 16, 64, 64, 16, 16, 8, 8, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

    }
}

void na1d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig<128, 128, 32, 64, 64, 32, 16, 8, 16, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig<128, 128, 32, 64, 64, 32, 16, 8, 16, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig<128, 128, 32, 64, 64, 32, 16, 8, 16, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_sm80_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_BF16
  using GConfig = natten::gemm::detail::GemmConfig<128, 128, 32, 64, 64, 32, 16, 8, 16, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = PointwiseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_sm80_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_BF16
  using GConfig = natten::gemm::detail::GemmConfig<128, 128, 32, 64, 64, 32, 16, 8, 16, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = PointwiseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_pn_cuda_gemm_bfloat16_128x128x32_64x64x32_16x8x16_3_sm80_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_BF16
  using GConfig = natten::gemm::detail::GemmConfig<128, 128, 32, 64, 64, 32, 16, 8, 16, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = PointwiseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_nn_cuda_gemm_double_64x32x16_32x16x16_8x8x4_3_sm80_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig<64, 32, 16, 32, 16, 16, 8, 8, 4, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = NeighborhoodNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);
}

void na1d_nn_cuda_gemm_float_64x32x16_32x16x16_16x8x8_3_sm80_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig<64, 32, 16, 32, 16, 16, 16, 8, 8, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = NeighborhoodNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = NeighborhoodNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

    }
}

void na1d_nn_cuda_gemm_float_64x32x16_32x16x16_16x8x8_3_sm80_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig<64, 32, 16, 32, 16, 16, 16, 8, 8, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = NeighborhoodNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = NeighborhoodNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

    }
}

void na1d_nn_cuda_gemm_float_64x32x16_32x16x16_16x8x8_3_sm80_align1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig<64, 32, 16, 32, 16, 16, 16, 8, 8, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = NeighborhoodNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = NeighborhoodNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

    }
}

void na1d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig<64, 64, 32, 32, 32, 32, 16, 8, 16, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = NeighborhoodNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig<64, 64, 32, 32, 32, 32, 16, 8, 16, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = NeighborhoodNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

} 
} 
} 

