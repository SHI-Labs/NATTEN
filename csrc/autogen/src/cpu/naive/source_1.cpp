#include <natten/dtypes.h>
#include <natten/cpu/naive/inverse_neighborhood_1d.hpp>
#include <natten/cpu/naive/rel_pos_bias_2d.hpp>
#include <natten/cpu/naive/neighborhood_neighborhood_3d.hpp>
#include <natten/cpu/naive/inverse_neighborhood_3d.hpp>
#include <natten/cpu/naive/rel_pos_bias_1d.hpp>
#include <natten/cpu/naive/rel_pos_bias_3d.hpp>
#include <natten/cpu/naive/inverse_neighborhood_2d.hpp>
namespace natten { 
namespace cpu { 
namespace naive { 

void na3d_nn_cpu_naive_double(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d) {
  using Kernel = NeighborhoodNeighborhood3D<natten::float64>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cpu_naive_float(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d) {
  using Kernel = NeighborhoodNeighborhood3D<natten::float32>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na1d_in_cpu_naive_double(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Kernel = InverseNeighborhood1D<natten::float64>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cpu_naive_float(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Kernel = InverseNeighborhood1D<natten::float32>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na2d_in_cpu_naive_double(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Kernel = InverseNeighborhood2D<natten::float64>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cpu_naive_float(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Kernel = InverseNeighborhood2D<natten::float32>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na3d_in_cpu_naive_double(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d) {
  using Kernel = InverseNeighborhood3D<natten::float64>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cpu_naive_float(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d) {
  using Kernel = InverseNeighborhood3D<natten::float32>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na1d_rpbgrad_cpu_naive_double(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Kernel = RelPosBiasGradient1D<natten::float64>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cpu_naive_float(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Kernel = RelPosBiasGradient1D<natten::float32>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cpu_naive_double(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Kernel = RelPosBiasGradient2D<natten::float64>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cpu_naive_float(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Kernel = RelPosBiasGradient2D<natten::float32>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na3d_rpbgrad_cpu_naive_double(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d) {
  using Kernel = RelPosBiasGradient3D<natten::float64>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cpu_naive_float(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int depth,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation,
  int kernel_size_d,
  int dilation_d) {
  using Kernel = RelPosBiasGradient3D<natten::float32>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

} 
} 
} 

