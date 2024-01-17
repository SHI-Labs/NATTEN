#include <cuda_runtime.h>
#include <iostream>
#include <natten/dtypes.cuh>
#include <natten/naive_argpack.h>
#include <natten/cuda/naive/neighborhood_neighborhood_1d.cuh>
#include <natten/cuda/naive/neighborhood_neighborhood_2d.cuh>
#include <natten/cuda/naive/neighborhood_neighborhood_3d.cuh>
#include <natten/cuda/naive/pointwise_neighborhood_1d.cuh>
#include <natten/cuda/naive/pointwise_neighborhood_2d.cuh>
#include <natten/cuda/naive/pointwise_neighborhood_3d.cuh>
namespace natten { 
namespace cuda { 
namespace naive { 

void na1d_pn_cuda_naive_double_cm_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack1D<natten::float64, 0>;
  using Kernel = PointwiseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_pn_cuda_naive_double_cm_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack1D<natten::float64, 1>;
  using Kernel = PointwiseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_pn_cuda_naive_float_cm_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack1D<natten::float32, 0>;
  using Kernel = PointwiseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_pn_cuda_naive_float_cm_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack1D<natten::float32, 1>;
  using Kernel = PointwiseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_pn_cuda_naive_half_cm_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack1D<natten::float16, 0>;
  using Kernel = PointwiseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na1d_pn_cuda_naive_half_cm_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack1D<natten::float16, 1>;
  using Kernel = PointwiseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na1d_pn_cuda_naive_bfloat16_cm_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack1D<natten::bfloat16, 0>;
  using Kernel = PointwiseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na1d_pn_cuda_naive_bfloat16_cm_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack1D<natten::bfloat16, 1>;
  using Kernel = PointwiseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_pn_cuda_naive_double_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float64, 0, 0>;
  using Kernel = PointwiseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_pn_cuda_naive_double_cm_0_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float64, 0, 1>;
  using Kernel = PointwiseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_pn_cuda_naive_double_cm_1_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float64, 1, 0>;
  using Kernel = PointwiseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_pn_cuda_naive_double_cm_1_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float64, 1, 1>;
  using Kernel = PointwiseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_pn_cuda_naive_float_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float32, 0, 0>;
  using Kernel = PointwiseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_pn_cuda_naive_float_cm_0_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float32, 0, 1>;
  using Kernel = PointwiseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_pn_cuda_naive_float_cm_1_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float32, 1, 0>;
  using Kernel = PointwiseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_pn_cuda_naive_float_cm_1_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float32, 1, 1>;
  using Kernel = PointwiseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_pn_cuda_naive_half_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float16, 0, 0>;
  using Kernel = PointwiseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_pn_cuda_naive_half_cm_0_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float16, 0, 1>;
  using Kernel = PointwiseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_pn_cuda_naive_half_cm_1_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float16, 1, 0>;
  using Kernel = PointwiseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_pn_cuda_naive_half_cm_1_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float16, 1, 1>;
  using Kernel = PointwiseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_pn_cuda_naive_bfloat16_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack2D<natten::bfloat16, 0, 0>;
  using Kernel = PointwiseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_pn_cuda_naive_bfloat16_cm_0_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack2D<natten::bfloat16, 0, 1>;
  using Kernel = PointwiseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_pn_cuda_naive_bfloat16_cm_1_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack2D<natten::bfloat16, 1, 0>;
  using Kernel = PointwiseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_pn_cuda_naive_bfloat16_cm_1_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack2D<natten::bfloat16, 1, 1>;
  using Kernel = PointwiseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_cuda_naive_double_cm_0_0_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float64, 0, 0, 0>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_cuda_naive_double_cm_0_0_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float64, 0, 0, 1>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_cuda_naive_double_cm_0_1_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float64, 0, 1, 0>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_cuda_naive_double_cm_0_1_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float64, 0, 1, 1>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_cuda_naive_double_cm_1_0_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float64, 1, 0, 0>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_cuda_naive_double_cm_1_0_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float64, 1, 0, 1>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_cuda_naive_double_cm_1_1_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float64, 1, 1, 0>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_cuda_naive_double_cm_1_1_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float64, 1, 1, 1>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_cuda_naive_float_cm_0_0_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float32, 0, 0, 0>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_cuda_naive_float_cm_0_0_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float32, 0, 0, 1>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_cuda_naive_float_cm_0_1_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float32, 0, 1, 0>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_cuda_naive_float_cm_0_1_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float32, 0, 1, 1>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_cuda_naive_float_cm_1_0_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float32, 1, 0, 0>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_cuda_naive_float_cm_1_0_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float32, 1, 0, 1>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_cuda_naive_float_cm_1_1_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float32, 1, 1, 0>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_cuda_naive_float_cm_1_1_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float32, 1, 1, 1>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_cuda_naive_half_cm_0_0_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float16, 0, 0, 0>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_cuda_naive_half_cm_0_0_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float16, 0, 0, 1>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_cuda_naive_half_cm_0_1_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float16, 0, 1, 0>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_cuda_naive_half_cm_0_1_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float16, 0, 1, 1>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_cuda_naive_half_cm_1_0_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float16, 1, 0, 0>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_cuda_naive_half_cm_1_0_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float16, 1, 0, 1>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_cuda_naive_half_cm_1_1_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float16, 1, 1, 0>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_cuda_naive_half_cm_1_1_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float16, 1, 1, 1>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_cuda_naive_bfloat16_cm_0_0_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack3D<natten::bfloat16, 0, 0, 0>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_cuda_naive_bfloat16_cm_0_0_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack3D<natten::bfloat16, 0, 0, 1>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_cuda_naive_bfloat16_cm_0_1_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack3D<natten::bfloat16, 0, 1, 0>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_cuda_naive_bfloat16_cm_0_1_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack3D<natten::bfloat16, 0, 1, 1>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_cuda_naive_bfloat16_cm_1_0_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack3D<natten::bfloat16, 1, 0, 0>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_cuda_naive_bfloat16_cm_1_0_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack3D<natten::bfloat16, 1, 0, 1>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_cuda_naive_bfloat16_cm_1_1_0(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack3D<natten::bfloat16, 1, 1, 0>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_cuda_naive_bfloat16_cm_1_1_1(
  int32_t cc,
  cudaStream_t stream,
  bool is_grad,
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack3D<natten::bfloat16, 1, 1, 1>;
  using Kernel = PointwiseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, is_grad, query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na1d_pn_bias_cuda_naive_double_cm_0(
  int32_t cc,
  cudaStream_t stream,
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack1D<natten::float64, 0>;
  using Kernel = PointwiseNeighborhood1DWithBias<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_pn_bias_cuda_naive_float_cm_0(
  int32_t cc,
  cudaStream_t stream,
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack1D<natten::float32, 0>;
  using Kernel = PointwiseNeighborhood1DWithBias<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_pn_bias_cuda_naive_half_cm_0(
  int32_t cc,
  cudaStream_t stream,
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack1D<natten::float16, 0>;
  using Kernel = PointwiseNeighborhood1DWithBias<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na1d_pn_bias_cuda_naive_bfloat16_cm_0(
  int32_t cc,
  cudaStream_t stream,
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack1D<natten::bfloat16, 0>;
  using Kernel = PointwiseNeighborhood1DWithBias<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_pn_bias_cuda_naive_double_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float64, 0, 0>;
  using Kernel = PointwiseNeighborhood2DWithBias<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_pn_bias_cuda_naive_float_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float32, 0, 0>;
  using Kernel = PointwiseNeighborhood2DWithBias<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_pn_bias_cuda_naive_half_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float16, 0, 0>;
  using Kernel = PointwiseNeighborhood2DWithBias<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_pn_bias_cuda_naive_bfloat16_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack2D<natten::bfloat16, 0, 0>;
  using Kernel = PointwiseNeighborhood2DWithBias<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_bias_cuda_naive_double_cm_0_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float64, 0, 0, 0>;
  using Kernel = PointwiseNeighborhood3DWithBias<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_bias_cuda_naive_float_cm_0_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float32, 0, 0, 0>;
  using Kernel = PointwiseNeighborhood3DWithBias<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_pn_bias_cuda_naive_half_cm_0_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float16, 0, 0, 0>;
  using Kernel = PointwiseNeighborhood3DWithBias<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_pn_bias_cuda_naive_bfloat16_cm_0_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack3D<natten::bfloat16, 0, 0, 0>;
  using Kernel = PointwiseNeighborhood3DWithBias<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na1d_nn_cuda_naive_double_cm_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack1D<natten::float64, 0>;
  using Kernel = NeighborhoodNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_nn_cuda_naive_double_cm_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack1D<natten::float64, 1>;
  using Kernel = NeighborhoodNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_nn_cuda_naive_float_cm_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack1D<natten::float32, 0>;
  using Kernel = NeighborhoodNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_nn_cuda_naive_float_cm_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack1D<natten::float32, 1>;
  using Kernel = NeighborhoodNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_nn_cuda_naive_half_cm_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack1D<natten::float16, 0>;
  using Kernel = NeighborhoodNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na1d_nn_cuda_naive_half_cm_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack1D<natten::float16, 1>;
  using Kernel = NeighborhoodNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na1d_nn_cuda_naive_bfloat16_cm_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack1D<natten::bfloat16, 0>;
  using Kernel = NeighborhoodNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na1d_nn_cuda_naive_bfloat16_cm_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t length,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  const std::tuple<int32_t>& kernel_size,
  const std::tuple<int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack1D<natten::bfloat16, 1>;
  using Kernel = NeighborhoodNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_nn_cuda_naive_double_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float64, 0, 0>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_nn_cuda_naive_double_cm_0_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float64, 0, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_nn_cuda_naive_double_cm_1_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float64, 1, 0>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_nn_cuda_naive_double_cm_1_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float64, 1, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_nn_cuda_naive_float_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float32, 0, 0>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_nn_cuda_naive_float_cm_0_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float32, 0, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_nn_cuda_naive_float_cm_1_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float32, 1, 0>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_nn_cuda_naive_float_cm_1_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float32, 1, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_nn_cuda_naive_half_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float16, 0, 0>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_nn_cuda_naive_half_cm_0_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float16, 0, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_nn_cuda_naive_half_cm_1_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float16, 1, 0>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_nn_cuda_naive_half_cm_1_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack2D<natten::float16, 1, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_nn_cuda_naive_bfloat16_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack2D<natten::bfloat16, 0, 0>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_nn_cuda_naive_bfloat16_cm_0_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack2D<natten::bfloat16, 0, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_nn_cuda_naive_bfloat16_cm_1_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack2D<natten::bfloat16, 1, 0>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_nn_cuda_naive_bfloat16_cm_1_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  const std::tuple<int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t>& dilation) {

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack2D<natten::bfloat16, 1, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_nn_cuda_naive_double_cm_0_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float64, 0, 0, 0>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_nn_cuda_naive_double_cm_0_0_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float64, 0, 0, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_nn_cuda_naive_double_cm_0_1_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float64, 0, 1, 0>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_nn_cuda_naive_double_cm_0_1_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int32_t batch_size,
  int32_t heads,
  int32_t depth,
  int32_t height,
  int32_t width,
  int32_t dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  const std::tuple<int32_t, int32_t, int32_t>& kernel_size,
  const std::tuple<int32_t, int32_t, int32_t>& dilation) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float64, 0, 1, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

} 
} 
} 

