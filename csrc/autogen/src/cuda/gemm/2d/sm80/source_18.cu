#include <cuda_runtime.h>
#include <iostream>
#include <natten/dtypes.cuh>
#include <natten/config.h>
#include <natten/gemm_argpack.cuh>
#include <natten/cuda/gemm/na2d.cuh>
namespace natten { 
namespace cuda { 
namespace gemm { 

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks33_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
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
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 11, 2, 16>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = NeighborhoodNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_nn_cuda_gemm_half_128x128x32_64x64x32_16x8x16_3_sm80_ks33_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
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
  using GConfig = natten::gemm::detail::GemmConfig2D<128, 128, 32, 64, 64, 32, 16, 8, 16, 3, 11, 2, 16>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = NeighborhoodNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks3_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
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
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 2, 1>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = NeighborhoodNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks3_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
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
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 2, 1>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = NeighborhoodNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks3_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
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
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 2, 1>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = NeighborhoodNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks5_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
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
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 2, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = NeighborhoodNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks5_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
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
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 2, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = NeighborhoodNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks5_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
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
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 2, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = NeighborhoodNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks7_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
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
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 2, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = NeighborhoodNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks7_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
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
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 2, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = NeighborhoodNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks7_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
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
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 2, 3>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = NeighborhoodNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks9_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
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
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 2, 4>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = NeighborhoodNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks9_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
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
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 2, 4>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = NeighborhoodNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks9_align2(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
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
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 2, 4>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 2, 2>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = NeighborhoodNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

void na2d_nn_cuda_gemm_bfloat16_64x64x32_32x32x32_16x8x16_3_sm80_ks11_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
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
  using GConfig = natten::gemm::detail::GemmConfig2D<64, 64, 32, 32, 32, 32, 16, 8, 16, 3, 8, 2, 5>;
  using ArchConfig = natten::gemm::detail::ArchArgs<80, bfloat16>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::bfloat16>;
  using Kernel = NeighborhoodNeighborhood2D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, scale, stream);
}

} 
} 
} 

