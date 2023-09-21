#include <cuda_runtime.h>
#include <natten/cuda/gemm/na2d.cuh>
#include <natten/dtypes.cuh>
#include <natten/config.h>
#include <natten/gemm_argpack.cuh>
namespace natten { 
namespace cuda { 
namespace gemm { 

void na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_sm80_ks3_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 32, 16, 8, 8, 4, 3, 7, 1, 1>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);
}

void na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_sm80_ks5_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 32, 16, 8, 8, 4, 3, 6, 1, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);
}

void na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_sm80_ks7_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 32, 16, 8, 8, 4, 3, 5, 1, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);
}

void na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_sm80_ks9_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 32, 16, 8, 8, 4, 3, 7, 1, 4>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);
}

void na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_sm80_ks11_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 32, 16, 8, 8, 4, 3, 6, 1, 5>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);
}

void na2d_pn_cuda_gemm_double_64x64x16_32x32x16_8x8x4_3_sm80_ks13_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 32, 16, 8, 8, 4, 3, 10, 1, 6>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);
}

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_sm80_ks15_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 8, 8, 4, 3, 9, 1, 7>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);
}

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_sm80_ks17_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 8, 8, 4, 3, 11, 1, 8>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);
}

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_sm80_ks19_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 8, 8, 4, 3, 13, 1, 9>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);
}

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_sm80_ks21_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 8, 8, 4, 3, 12, 1, 10>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);
}

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_sm80_ks23_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 8, 8, 4, 3, 14, 1, 11>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);
}

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_sm80_ks25_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 8, 8, 4, 3, 13, 1, 12>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);
}

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_sm80_ks27_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 8, 8, 4, 3, 14, 1, 13>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);
}

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_sm80_ks29_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 8, 8, 4, 3, 17, 1, 14>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);
}

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_sm80_ks31_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 8, 8, 4, 3, 17, 1, 15>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);
}

void na2d_pn_cuda_gemm_double_128x128x16_64x64x16_8x8x4_3_sm80_ks33_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 8, 8, 4, 3, 17, 1, 16>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, double>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float64>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks3_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 7, 1, 1>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks3_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 7, 1, 1>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks3_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 7, 1, 1>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks5_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 6, 1, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks5_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 6, 1, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks5_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 6, 1, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks7_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 5, 1, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks7_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 5, 1, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks7_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 5, 1, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks9_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 7, 1, 4>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks9_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 7, 1, 4>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks9_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 7, 1, 4>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks11_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 6, 1, 5>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks11_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 6, 1, 5>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks11_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 6, 1, 5>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks13_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 10, 1, 6>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks13_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 10, 1, 6>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_64x64x16_32x16x16_16x8x8_3_sm80_ks13_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 16, 32, 16, 16, 16, 8, 8, 3, 10, 1, 6>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks15_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 9, 1, 7>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks15_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 9, 1, 7>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks15_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 9, 1, 7>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks17_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 11, 1, 8>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks17_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 11, 1, 8>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks17_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 11, 1, 8>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks19_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 13, 1, 9>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks19_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 13, 1, 9>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks19_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 13, 1, 9>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks21_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 12, 1, 10>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks21_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 12, 1, 10>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks21_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 12, 1, 10>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks23_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 14, 1, 11>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks23_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 14, 1, 11>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks23_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 14, 1, 11>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks25_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 13, 1, 12>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks25_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 13, 1, 12>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks25_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 13, 1, 12>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks27_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 14, 1, 13>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks27_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 14, 1, 13>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks27_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 14, 1, 13>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks29_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 17, 1, 14>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks29_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 17, 1, 14>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks29_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 17, 1, 14>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks31_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 17, 1, 15>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks31_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 17, 1, 15>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks31_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 17, 1, 15>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks33_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 17, 1, 16>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks33_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 17, 1, 16>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_float_128x128x16_64x64x16_16x8x8_3_sm80_ks33_align1(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 16, 64, 64, 16, 16, 8, 8, 3, 17, 1, 16>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, float>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 1, 1>;
    if (natten::kEnableGemmTF32) { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    } else { 
      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

    }
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks3_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 7, 1, 1>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks3_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 7, 1, 1>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks3_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 7, 1, 1>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks5_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 6, 1, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks5_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 6, 1, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks5_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 6, 1, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks7_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 5, 1, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks7_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 5, 1, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks7_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 5, 1, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks9_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 7, 1, 4>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks9_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 7, 1, 4>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks9_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 7, 1, 4>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks11_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 6, 1, 5>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks11_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 6, 1, 5>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks11_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 6, 1, 5>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks13_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 10, 1, 6>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks13_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 10, 1, 6>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks13_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 10, 1, 6>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks15_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 9, 1, 7>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks15_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 9, 1, 7>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks15_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 9, 1, 7>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks17_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 11, 1, 8>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks17_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 11, 1, 8>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks17_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 11, 1, 8>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks19_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 13, 1, 9>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks19_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 13, 1, 9>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks19_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 13, 1, 9>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks21_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 12, 1, 10>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks21_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 12, 1, 10>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks21_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 12, 1, 10>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks23_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 14, 1, 11>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks23_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 14, 1, 11>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks23_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 14, 1, 11>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks25_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 13, 1, 12>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks25_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 13, 1, 12>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks25_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 13, 1, 12>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks27_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 14, 1, 13>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks27_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 14, 1, 13>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks27_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 14, 1, 13>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks29_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 17, 1, 14>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks29_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 17, 1, 14>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks29_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 17, 1, 14>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks31_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 17, 1, 15>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks31_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 17, 1, 15>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks31_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 17, 1, 15>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks33_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 17, 1, 16>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks33_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 17, 1, 16>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks33_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_FP16
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 17, 1, 16>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks3_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_BF16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 7, 1, 1>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks3_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_BF16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 7, 1, 1>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks3_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_BF16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 7, 1, 1>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks5_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_BF16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 6, 1, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks5_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_BF16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 6, 1, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks5_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_BF16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 6, 1, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks7_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_BF16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 5, 1, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks7_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_BF16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 5, 1, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_pn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks7_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  float scale) {

#ifdef NATTEN_ENABLE_BF16
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 5, 1, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = PointwiseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, height, width, dim, kernel_size, dilation, scale);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

} 
} 
} 

