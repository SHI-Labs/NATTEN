#include <natten/dtypes.h>
#include <natten/cpu/naive/inverse_neighborhood_1d.hpp>
#include <natten/cpu/naive/inverse_neighborhood_2d.hpp>
#include <natten/cpu/naive/inverse_neighborhood_3d.hpp>
#include <natten/cpu/naive/neighborhood_neighborhood_2d.hpp>
#include <natten/cpu/naive/neighborhood_neighborhood_3d.hpp>
#include <natten/cpu/naive/rel_pos_bias_1d.hpp>
#include <natten/cpu/naive/rel_pos_bias_2d.hpp>
#include <natten/cpu/naive/rel_pos_bias_3d.hpp>
namespace natten { 
namespace cpu { 
namespace naive { 

void na2d_nn_cpu_naive_float(
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
  const std::tuple<int32_t, int32_t>& dilation,
  const std::tuple<bool, bool>& is_causal) {
  using Kernel = NeighborhoodNeighborhood2D<natten::float32>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, is_causal);
}

void na3d_nn_cpu_naive_double(
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
  const std::tuple<int32_t, int32_t, int32_t>& dilation,
  const std::tuple<bool, bool, bool>& is_causal) {
  using Kernel = NeighborhoodNeighborhood3D<natten::float64>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation, is_causal);
}

void na3d_nn_cpu_naive_float(
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
  const std::tuple<int32_t, int32_t, int32_t>& dilation,
  const std::tuple<bool, bool, bool>& is_causal) {
  using Kernel = NeighborhoodNeighborhood3D<natten::float32>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation, is_causal);
}

void na1d_in_cpu_naive_double(
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
  const std::tuple<int32_t>& dilation,
  const std::tuple<bool>& is_causal) {
  using Kernel = InverseNeighborhood1D<natten::float64>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation, is_causal);
}

void na1d_in_cpu_naive_float(
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
  const std::tuple<int32_t>& dilation,
  const std::tuple<bool>& is_causal) {
  using Kernel = InverseNeighborhood1D<natten::float32>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation, is_causal);
}

void na2d_in_cpu_naive_double(
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
  const std::tuple<int32_t, int32_t>& dilation,
  const std::tuple<bool, bool>& is_causal) {
  using Kernel = InverseNeighborhood2D<natten::float64>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, is_causal);
}

void na2d_in_cpu_naive_float(
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
  const std::tuple<int32_t, int32_t>& dilation,
  const std::tuple<bool, bool>& is_causal) {
  using Kernel = InverseNeighborhood2D<natten::float32>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, is_causal);
}

void na3d_in_cpu_naive_double(
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
  const std::tuple<int32_t, int32_t, int32_t>& dilation,
  const std::tuple<bool, bool, bool>& is_causal) {
  using Kernel = InverseNeighborhood3D<natten::float64>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation, is_causal);
}

void na3d_in_cpu_naive_float(
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
  const std::tuple<int32_t, int32_t, int32_t>& dilation,
  const std::tuple<bool, bool, bool>& is_causal) {
  using Kernel = InverseNeighborhood3D<natten::float32>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation, is_causal);
}

void na1d_rpbgrad_cpu_naive_double(
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
  const std::tuple<int32_t>& dilation,
  const std::tuple<bool>& is_causal) {
  using Kernel = RelPosBiasGradient1D<natten::float64>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation, is_causal);
}

void na1d_rpbgrad_cpu_naive_float(
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
  const std::tuple<int32_t>& dilation,
  const std::tuple<bool>& is_causal) {
  using Kernel = RelPosBiasGradient1D<natten::float32>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, attn_stride_0, attn_stride_1, attn_stride_2, kernel_size, dilation, is_causal);
}

void na2d_rpbgrad_cpu_naive_double(
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
  const std::tuple<int32_t, int32_t>& dilation,
  const std::tuple<bool, bool>& is_causal) {
  using Kernel = RelPosBiasGradient2D<natten::float64>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, is_causal);
}

void na2d_rpbgrad_cpu_naive_float(
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
  const std::tuple<int32_t, int32_t>& dilation,
  const std::tuple<bool, bool>& is_causal) {
  using Kernel = RelPosBiasGradient2D<natten::float32>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, kernel_size, dilation, is_causal);
}

void na3d_rpbgrad_cpu_naive_double(
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
  const std::tuple<int32_t, int32_t, int32_t>& dilation,
  const std::tuple<bool, bool, bool>& is_causal) {
  using Kernel = RelPosBiasGradient3D<natten::float64>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation, is_causal);
}

void na3d_rpbgrad_cpu_naive_float(
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
  const std::tuple<int32_t, int32_t, int32_t>& dilation,
  const std::tuple<bool, bool, bool>& is_causal) {
  using Kernel = RelPosBiasGradient3D<natten::float32>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, attn_stride_0, attn_stride_1, attn_stride_2, attn_stride_3, attn_stride_4, kernel_size, dilation, is_causal);
}

} 
} 
} 

