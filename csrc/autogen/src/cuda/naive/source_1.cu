#include <cuda_runtime.h>
#include <iostream>
#include <natten/dtypes.cuh>
#include <natten/naive_argpack.h>
#include <natten/cuda/naive/inverse_neighborhood_1d.cuh>
#include <natten/cuda/naive/inverse_neighborhood_2d.cuh>
#include <natten/cuda/naive/inverse_neighborhood_3d.cuh>
#include <natten/cuda/naive/neighborhood_neighborhood_3d.cuh>
#include <natten/cuda/naive/rel_pos_bias_1d.cuh>
#include <natten/cuda/naive/rel_pos_bias_2d.cuh>
#include <natten/cuda/naive/rel_pos_bias_3d.cuh>
namespace natten { 
namespace cuda { 
namespace naive { 

void na3d_nn_cuda_naive_double_cm_1_0_0(
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
  using Arguments = natten::naive::ArgumentPack3D<natten::float64, 1, 0, 0>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_nn_cuda_naive_double_cm_1_0_1(
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
  using Arguments = natten::naive::ArgumentPack3D<natten::float64, 1, 0, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_nn_cuda_naive_double_cm_1_1_0(
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
  using Arguments = natten::naive::ArgumentPack3D<natten::float64, 1, 1, 0>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_nn_cuda_naive_double_cm_1_1_1(
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
  using Arguments = natten::naive::ArgumentPack3D<natten::float64, 1, 1, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_nn_cuda_naive_float_cm_0_0_0(
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
  using Arguments = natten::naive::ArgumentPack3D<natten::float32, 0, 0, 0>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_nn_cuda_naive_float_cm_0_0_1(
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
  using Arguments = natten::naive::ArgumentPack3D<natten::float32, 0, 0, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_nn_cuda_naive_float_cm_0_1_0(
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
  using Arguments = natten::naive::ArgumentPack3D<natten::float32, 0, 1, 0>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_nn_cuda_naive_float_cm_0_1_1(
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
  using Arguments = natten::naive::ArgumentPack3D<natten::float32, 0, 1, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_nn_cuda_naive_float_cm_1_0_0(
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
  using Arguments = natten::naive::ArgumentPack3D<natten::float32, 1, 0, 0>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_nn_cuda_naive_float_cm_1_0_1(
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
  using Arguments = natten::naive::ArgumentPack3D<natten::float32, 1, 0, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_nn_cuda_naive_float_cm_1_1_0(
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
  using Arguments = natten::naive::ArgumentPack3D<natten::float32, 1, 1, 0>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_nn_cuda_naive_float_cm_1_1_1(
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
  using Arguments = natten::naive::ArgumentPack3D<natten::float32, 1, 1, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_nn_cuda_naive_half_cm_0_0_0(
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

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float16, 0, 0, 0>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_nn_cuda_naive_half_cm_0_0_1(
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

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float16, 0, 0, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_nn_cuda_naive_half_cm_0_1_0(
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

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float16, 0, 1, 0>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_nn_cuda_naive_half_cm_0_1_1(
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

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float16, 0, 1, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_nn_cuda_naive_half_cm_1_0_0(
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

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float16, 1, 0, 0>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_nn_cuda_naive_half_cm_1_0_1(
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

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float16, 1, 0, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_nn_cuda_naive_half_cm_1_1_0(
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

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float16, 1, 1, 0>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_nn_cuda_naive_half_cm_1_1_1(
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

if(cc >= 60) {
  using Arguments = natten::naive::ArgumentPack3D<natten::float16, 1, 1, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_nn_cuda_naive_bfloat16_cm_0_0_0(
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

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack3D<natten::bfloat16, 0, 0, 0>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_nn_cuda_naive_bfloat16_cm_0_0_1(
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

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack3D<natten::bfloat16, 0, 0, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_nn_cuda_naive_bfloat16_cm_0_1_0(
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

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack3D<natten::bfloat16, 0, 1, 0>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_nn_cuda_naive_bfloat16_cm_0_1_1(
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

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack3D<natten::bfloat16, 0, 1, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_nn_cuda_naive_bfloat16_cm_1_0_0(
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

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack3D<natten::bfloat16, 1, 0, 0>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_nn_cuda_naive_bfloat16_cm_1_0_1(
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

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack3D<natten::bfloat16, 1, 0, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_nn_cuda_naive_bfloat16_cm_1_1_0(
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

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack3D<natten::bfloat16, 1, 1, 0>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_nn_cuda_naive_bfloat16_cm_1_1_1(
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

if(cc >= 80) {
  using Arguments = natten::naive::ArgumentPack3D<natten::bfloat16, 1, 1, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na1d_in_cuda_naive_double_cm_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_in_cuda_naive_double_cm_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_in_cuda_naive_float_cm_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_in_cuda_naive_float_cm_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_in_cuda_naive_half_cm_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na1d_in_cuda_naive_half_cm_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na1d_in_cuda_naive_bfloat16_cm_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na1d_in_cuda_naive_bfloat16_cm_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_in_cuda_naive_double_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_in_cuda_naive_double_cm_0_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_in_cuda_naive_double_cm_1_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_in_cuda_naive_double_cm_1_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_cm_0_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_cm_1_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_cm_1_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_in_cuda_naive_half_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_in_cuda_naive_half_cm_0_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_in_cuda_naive_half_cm_1_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_in_cuda_naive_half_cm_1_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_in_cuda_naive_bfloat16_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_in_cuda_naive_bfloat16_cm_0_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_in_cuda_naive_bfloat16_cm_1_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_in_cuda_naive_bfloat16_cm_1_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_in_cuda_naive_double_cm_0_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_in_cuda_naive_double_cm_0_0_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_in_cuda_naive_double_cm_0_1_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_in_cuda_naive_double_cm_0_1_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_in_cuda_naive_double_cm_1_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_in_cuda_naive_double_cm_1_0_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_in_cuda_naive_double_cm_1_1_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_in_cuda_naive_double_cm_1_1_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_in_cuda_naive_float_cm_0_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_in_cuda_naive_float_cm_0_0_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_in_cuda_naive_float_cm_0_1_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_in_cuda_naive_float_cm_0_1_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_in_cuda_naive_float_cm_1_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_in_cuda_naive_float_cm_1_0_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_in_cuda_naive_float_cm_1_1_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_in_cuda_naive_float_cm_1_1_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_in_cuda_naive_half_cm_0_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_in_cuda_naive_half_cm_0_0_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_in_cuda_naive_half_cm_0_1_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_in_cuda_naive_half_cm_0_1_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_in_cuda_naive_half_cm_1_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_in_cuda_naive_half_cm_1_0_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_in_cuda_naive_half_cm_1_1_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_in_cuda_naive_half_cm_1_1_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_in_cuda_naive_bfloat16_cm_0_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_in_cuda_naive_bfloat16_cm_0_0_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_in_cuda_naive_bfloat16_cm_0_1_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_in_cuda_naive_bfloat16_cm_0_1_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_in_cuda_naive_bfloat16_cm_1_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_in_cuda_naive_bfloat16_cm_1_0_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_in_cuda_naive_bfloat16_cm_1_1_0(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_in_cuda_naive_bfloat16_cm_1_1_1(
  int32_t cc,
  cudaStream_t stream,
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
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
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na1d_rpbgrad_cuda_naive_double_cm_0(
  int32_t cc,
  cudaStream_t stream,
  void * d_bias_ptr,
  void * d_attn_ptr,
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
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_float_cm_0(
  int32_t cc,
  cudaStream_t stream,
  void * d_bias_ptr,
  void * d_attn_ptr,
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
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_half_cm_0(
  int32_t cc,
  cudaStream_t stream,
  void * d_bias_ptr,
  void * d_attn_ptr,
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
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na1d_rpbgrad_cuda_naive_bfloat16_cm_0(
  int32_t cc,
  cudaStream_t stream,
  void * d_bias_ptr,
  void * d_attn_ptr,
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
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_rpbgrad_cuda_naive_double_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * d_bias_ptr,
  void * d_attn_ptr,
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
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_float_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * d_bias_ptr,
  void * d_attn_ptr,
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
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_half_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * d_bias_ptr,
  void * d_attn_ptr,
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
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na2d_rpbgrad_cuda_naive_bfloat16_cm_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * d_bias_ptr,
  void * d_attn_ptr,
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
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_rpbgrad_cuda_naive_double_cm_0_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * d_bias_ptr,
  void * d_attn_ptr,
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
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_rpbgrad_cuda_naive_float_cm_0_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * d_bias_ptr,
  void * d_attn_ptr,
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
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);
}

void na3d_rpbgrad_cuda_naive_half_cm_0_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * d_bias_ptr,
  void * d_attn_ptr,
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
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

void na3d_rpbgrad_cuda_naive_bfloat16_cm_0_0_0(
  int32_t cc,
  cudaStream_t stream,
  void * d_bias_ptr,
  void * d_attn_ptr,
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
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
cc, stream, d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation);

} else {
std::cerr << "This half type is not supported on the selected device."  << std::endl; 
exit(EXIT_FAILURE); 

}
}

} 
} 
} 

