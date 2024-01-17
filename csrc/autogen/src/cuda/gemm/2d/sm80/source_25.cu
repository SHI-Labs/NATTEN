#include <cuda_runtime.h>
#include <iostream>
#include <natten/config.h>
#include <natten/cuda/gemm/na2d.cuh>
#include <natten/dtypes.cuh>
#include <natten/gemm_argpack.cuh>
namespace natten { 
namespace cuda { 
namespace gemm { 

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks7_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks7_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks7_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks9_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 4>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks9_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 4>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks9_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 4>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks11_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 5>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks11_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 5>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks11_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 5>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks13_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 6>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks13_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 6>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks13_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 6>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks15_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 7>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks15_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 7>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks15_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 7>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks17_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 8>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks17_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 8>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks17_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 8>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks19_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 9>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks19_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 9>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks19_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 9>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks21_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 10>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks21_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 10>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks21_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 10>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks23_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 11>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks23_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 11>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks23_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 11>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks25_align8(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 12>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks25_align4(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 12>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_in_cuda_gemm_half_64x64x32_32x32x32_16x8x16_3_sm80_ks25_align2(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 4, 12>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = InverseNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

} 
} 
} 

