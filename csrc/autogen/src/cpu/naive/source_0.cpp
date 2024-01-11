#include <natten/dtypes.h>
#include <natten/cpu/naive/pointwise_neighborhood_1d.hpp>
#include <natten/cpu/naive/neighborhood_neighborhood_1d.hpp>
#include <natten/cpu/naive/neighborhood_neighborhood_2d.hpp>
#include <natten/cpu/naive/pointwise_neighborhood_3d.hpp>
#include <natten/cpu/naive/pointwise_neighborhood_2d.hpp>
namespace natten { 
namespace cpu { 
namespace naive { 

void na1d_pn_cpu_naive_double(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation) {
  using Kernel = PointwiseNeighborhood1D<natten::float64>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_pn_cpu_naive_float(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation) {
  using Kernel = PointwiseNeighborhood1D<natten::float32>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na2d_pn_cpu_naive_double(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
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
  int dilation) {
  using Kernel = PointwiseNeighborhood2D<natten::float64>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_pn_cpu_naive_float(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
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
  int dilation) {
  using Kernel = PointwiseNeighborhood2D<natten::float32>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na3d_pn_cpu_naive_double(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d) {
  using Kernel = PointwiseNeighborhood3D<natten::float64>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_pn_cpu_naive_float(
  void * query_ptr,
  void * key_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d) {
  using Kernel = PointwiseNeighborhood3D<natten::float32>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na1d_pn_bias_cpu_naive_double(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation) {
  using Kernel = PointwiseNeighborhood1DWithBias<natten::float64>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_pn_bias_cpu_naive_float(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int kernel_size,
  int dilation) {
  using Kernel = PointwiseNeighborhood1DWithBias<natten::float32>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na2d_pn_bias_cpu_naive_double(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
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
  int dilation) {
  using Kernel = PointwiseNeighborhood2DWithBias<natten::float64>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na2d_pn_bias_cpu_naive_float(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
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
  int dilation) {
  using Kernel = PointwiseNeighborhood2DWithBias<natten::float32>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

void na3d_pn_bias_cpu_naive_double(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d) {
  using Kernel = PointwiseNeighborhood3DWithBias<natten::float64>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_pn_bias_cpu_naive_float(
  void * query_ptr,
  void * key_ptr,
  void * bias_ptr,
  void * attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int64_t attn_stride_0,
  int64_t attn_stride_1,
  int64_t attn_stride_2,
  int64_t attn_stride_3,
  int64_t attn_stride_4,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d) {
  using Kernel = PointwiseNeighborhood3DWithBias<natten::float32>;
  Kernel kernel;
  kernel(
query_ptr, key_ptr, bias_ptr, attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na1d_nn_cpu_naive_double(
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
  int dilation) {
  using Kernel = NeighborhoodNeighborhood1D<natten::float64>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na1d_nn_cpu_naive_float(
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
  int dilation) {
  using Kernel = NeighborhoodNeighborhood1D<natten::float32>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation);
}

void na2d_nn_cpu_naive_double(
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
  int dilation) {
  using Kernel = NeighborhoodNeighborhood2D<natten::float64>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation);
}

} 
} 
} 

