#include <cuda_runtime.h>
#include <iostream>
#include <natten/config.h>
#include <natten/cuda/gemm/na1d.cuh>
#include <natten/dtypes.cuh>
#include <natten/gemm_argpack.cuh>
namespace natten { 
namespace cuda { 
namespace gemm { 

void na1d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_align8(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig<128, 128, 32, 64, 64, 32, 16, 8, 8, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<75, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<8, 8, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation, scale, stream);
}

void na1d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_align4(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig<128, 128, 32, 64, 64, 32, 16, 8, 8, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<75, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<4, 4, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation, scale, stream);
}

void na1d_pn_cuda_gemm_half_128x128x32_64x64x32_16x8x8_2_sm75_align2(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  void * bias_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig<128, 128, 32, 64, 64, 32, 16, 8, 8, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<75, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<2, 2, 1>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = PointwiseNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, bias_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation, scale, stream);
}

void na1d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_align8(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig<64, 64, 32, 32, 32, 32, 16, 8, 8, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<75, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 8, 8>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = NeighborhoodNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation, scale, stream);
}

void na1d_nn_cuda_gemm_half_64x64x32_32x32x32_16x8x8_2_sm75_align4(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation,
  float scale,
  cudaStream_t stream) {
  using GConfig = natten::gemm::detail::GemmConfig<64, 64, 32, 32, 32, 32, 16, 8, 8, 2>;
  using ArchConfig = natten::gemm::detail::ArchArgs<75, half>;
  using AConfig = natten::gemm::detail::AlignmentConfig<1, 4, 4>;
  using DConfig = natten::gemm::detail::DTypeConfig<natten::float16>;
  using Kernel = NeighborhoodNeighborhood1D<GConfig, AConfig, DConfig, ArchConfig>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation, scale, stream);
}

} 
} 
} 

