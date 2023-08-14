#include <cuda_runtime.h>
#include <natten/dtypes.cuh>
#include <natten/naive_argpack.h>
#include <natten/cuda/naive/inverse_neighborhood_2d.cuh>
#include <natten/cuda/naive/rel_pos_bias_2d.cuh>
#include <natten/cuda/naive/rel_pos_bias_3d.cuh>
#include <natten/cuda/naive/inverse_neighborhood_3d.cuh>
#include <natten/cuda/naive/neighborhood_neighborhood_2d.cuh>
#include <natten/cuda/naive/neighborhood_neighborhood_3d.cuh>
#include <natten/cuda/naive/rel_pos_bias_1d.cuh>
#include <natten/cuda/naive/inverse_neighborhood_1d.cuh>
namespace natten { 
namespace cuda { 
namespace naive { 

void na2d_nn_cuda_naive_half_ks_any_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, -1, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_half_ks_3_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 3, -1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_half_ks_3_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 3, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_half_ks_5_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 5, -1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_half_ks_5_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 5, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_half_ks_7_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 7, -1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_half_ks_7_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 7, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_half_ks_9_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 9, -1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_half_ks_9_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 9, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_half_ks_11_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 11, -1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_half_ks_11_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 11, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_half_ks_13_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 13, -1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_half_ks_13_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 13, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_bfloat16_ks_any_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, -1, -1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_bfloat16_ks_any_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, -1, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_bfloat16_ks_3_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 3, -1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_bfloat16_ks_3_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 3, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_bfloat16_ks_5_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 5, -1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_bfloat16_ks_5_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 5, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_bfloat16_ks_7_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 7, -1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_bfloat16_ks_7_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 7, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_bfloat16_ks_9_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 9, -1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_bfloat16_ks_9_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 9, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_bfloat16_ks_11_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 11, -1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_bfloat16_ks_11_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 11, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_bfloat16_ks_13_di_any(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 13, -1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_nn_cuda_naive_bfloat16_ks_13_di_1(
  void * attn_ptr,
  void * value_ptr,
  void * output_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 13, 1>;
  using Kernel = NeighborhoodNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_double_ks_any_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, -1, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_double_ks_any_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, -1, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_double_ks_3_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 3, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_double_ks_3_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 3, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_double_ks_5_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 5, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_double_ks_5_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 5, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_double_ks_7_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 7, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_double_ks_7_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 7, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_double_ks_9_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 9, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_double_ks_9_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 9, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_double_ks_11_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 11, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_double_ks_11_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 11, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_double_ks_13_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 13, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_double_ks_13_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 13, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_float_ks_any_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, -1, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_float_ks_any_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, -1, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_float_ks_3_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 3, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_float_ks_3_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 3, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_float_ks_5_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 5, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_float_ks_5_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 5, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_float_ks_7_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 7, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_float_ks_7_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 7, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_float_ks_9_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 9, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_float_ks_9_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 9, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_float_ks_11_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 11, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_float_ks_11_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 11, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_float_ks_13_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 13, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_float_ks_13_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 13, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_nn_cuda_naive_half_ks_any_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, -1, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_half_ks_any_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, -1, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_half_ks_3_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 3, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_half_ks_3_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 3, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_half_ks_5_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 5, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_half_ks_5_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 5, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_half_ks_7_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 7, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_half_ks_7_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 7, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_half_ks_9_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 9, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_half_ks_9_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 9, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_half_ks_11_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 11, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_half_ks_11_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 11, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_half_ks_13_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 13, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_half_ks_13_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 13, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_bfloat16_ks_any_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, -1, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_bfloat16_ks_any_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, -1, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_bfloat16_ks_3_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 3, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_bfloat16_ks_3_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 3, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_bfloat16_ks_5_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 5, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_bfloat16_ks_5_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 5, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_bfloat16_ks_7_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 7, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_bfloat16_ks_7_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 7, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_bfloat16_ks_9_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 9, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_bfloat16_ks_9_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 9, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_bfloat16_ks_11_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 11, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_bfloat16_ks_11_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 11, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_bfloat16_ks_13_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 13, -1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_nn_cuda_naive_bfloat16_ks_13_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 13, 1>;
  using Kernel = NeighborhoodNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, value_ptr, output_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_double_ks_any_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, -1, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_double_ks_any_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, -1, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_double_ks_3_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 3, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_double_ks_3_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 3, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_double_ks_5_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 5, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_double_ks_5_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 5, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_double_ks_7_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 7, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_double_ks_7_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 7, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_double_ks_9_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 9, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_double_ks_9_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 9, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_double_ks_11_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 11, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_double_ks_11_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 11, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_double_ks_13_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 13, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_double_ks_13_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 13, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_float_ks_any_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, -1, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_float_ks_any_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, -1, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_float_ks_3_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 3, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_float_ks_3_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 3, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_float_ks_5_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 5, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_float_ks_5_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 5, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_float_ks_7_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 7, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_float_ks_7_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 7, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_float_ks_9_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 9, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_float_ks_9_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 9, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_float_ks_11_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 11, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_float_ks_11_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 11, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_float_ks_13_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 13, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_float_ks_13_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 13, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_in_cuda_naive_half_ks_any_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, -1, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_half_ks_any_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, -1, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_half_ks_3_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 3, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_half_ks_3_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 3, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_half_ks_5_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 5, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_half_ks_5_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 5, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_half_ks_7_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 7, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_half_ks_7_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 7, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_half_ks_9_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 9, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_half_ks_9_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 9, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_half_ks_11_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 11, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_half_ks_11_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 11, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_half_ks_13_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 13, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_half_ks_13_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 13, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_bfloat16_ks_any_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, -1, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_bfloat16_ks_any_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, -1, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_bfloat16_ks_3_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 3, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_bfloat16_ks_3_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 3, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_bfloat16_ks_5_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 5, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_bfloat16_ks_5_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 5, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_bfloat16_ks_7_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 7, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_bfloat16_ks_7_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 7, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_bfloat16_ks_9_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 9, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_bfloat16_ks_9_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 9, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_bfloat16_ks_11_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 11, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_bfloat16_ks_11_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 11, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_bfloat16_ks_13_di_any(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 13, -1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_in_cuda_naive_bfloat16_ks_13_di_1(
  void * attn_ptr,
  void * d_output_ptr,
  void * d_value_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 13, 1>;
  using Kernel = InverseNeighborhood1D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_double_ks_any_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, -1, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_double_ks_any_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, -1, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_double_ks_3_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 3, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_double_ks_3_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 3, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_double_ks_5_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 5, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_double_ks_5_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 5, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_double_ks_7_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 7, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_double_ks_7_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 7, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_double_ks_9_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 9, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_double_ks_9_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 9, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_double_ks_11_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 11, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_double_ks_11_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 11, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_double_ks_13_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 13, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_double_ks_13_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 13, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_ks_any_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, -1, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_ks_any_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, -1, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_ks_3_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 3, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_ks_3_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 3, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_ks_5_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 5, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_ks_5_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 5, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_ks_7_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 7, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_ks_7_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 7, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_ks_9_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 9, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_ks_9_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 9, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_ks_11_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 11, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_ks_11_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 11, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_ks_13_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 13, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_float_ks_13_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 13, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_in_cuda_naive_half_ks_any_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, -1, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_half_ks_any_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, -1, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_half_ks_3_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 3, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_half_ks_3_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 3, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_half_ks_5_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 5, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_half_ks_5_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 5, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_half_ks_7_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 7, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_half_ks_7_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 7, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_half_ks_9_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 9, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_half_ks_9_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 9, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_half_ks_11_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 11, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_half_ks_11_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 11, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_half_ks_13_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 13, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_half_ks_13_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 13, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_bfloat16_ks_any_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, -1, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_bfloat16_ks_any_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, -1, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_bfloat16_ks_3_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 3, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_bfloat16_ks_3_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 3, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_bfloat16_ks_5_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 5, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_bfloat16_ks_5_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 5, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_bfloat16_ks_7_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 7, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_bfloat16_ks_7_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 7, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_bfloat16_ks_9_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 9, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_bfloat16_ks_9_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 9, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_bfloat16_ks_11_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 11, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_bfloat16_ks_11_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 11, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_bfloat16_ks_13_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 13, -1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_in_cuda_naive_bfloat16_ks_13_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 13, 1>;
  using Kernel = InverseNeighborhood2D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_double_ks_any_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, -1, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_double_ks_any_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, -1, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_double_ks_3_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 3, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_double_ks_3_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 3, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_double_ks_5_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 5, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_double_ks_5_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 5, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_double_ks_7_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 7, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_double_ks_7_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 7, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_double_ks_9_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 9, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_double_ks_9_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 9, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_double_ks_11_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 11, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_double_ks_11_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 11, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_double_ks_13_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 13, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_double_ks_13_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 13, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_float_ks_any_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, -1, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_float_ks_any_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, -1, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_float_ks_3_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 3, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_float_ks_3_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 3, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_float_ks_5_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 5, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_float_ks_5_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 5, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_float_ks_7_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 7, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_float_ks_7_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 7, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_float_ks_9_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 9, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_float_ks_9_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 9, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_float_ks_11_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 11, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_float_ks_11_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 11, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_float_ks_13_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 13, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_float_ks_13_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 13, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_in_cuda_naive_half_ks_any_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, -1, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_half_ks_any_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, -1, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_half_ks_3_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 3, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_half_ks_3_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 3, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_half_ks_5_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 5, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_half_ks_5_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 5, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_half_ks_7_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 7, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_half_ks_7_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 7, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_half_ks_9_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 9, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_half_ks_9_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 9, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_half_ks_11_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 11, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_half_ks_11_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 11, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_half_ks_13_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 13, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_half_ks_13_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 13, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_bfloat16_ks_any_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, -1, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_bfloat16_ks_any_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, -1, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_bfloat16_ks_3_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 3, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_bfloat16_ks_3_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 3, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_bfloat16_ks_5_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 5, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_bfloat16_ks_5_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 5, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_bfloat16_ks_7_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 7, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_bfloat16_ks_7_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 7, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_bfloat16_ks_9_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 9, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_bfloat16_ks_9_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 9, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_bfloat16_ks_11_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 11, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_bfloat16_ks_11_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 11, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_bfloat16_ks_13_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 13, -1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_in_cuda_naive_bfloat16_ks_13_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 13, 1>;
  using Kernel = InverseNeighborhood3D<Arguments>;
  Kernel kernel;
  kernel(
attn_ptr, d_output_ptr, d_value_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_double_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, -1, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_double_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, -1, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_double_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 3, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_double_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 3, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_double_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 5, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_double_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 5, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_double_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 7, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_double_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 7, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_double_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 9, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_double_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 9, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_double_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 11, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_double_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 11, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_double_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 13, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_double_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 13, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_float_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, -1, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_float_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, -1, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_float_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 3, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_float_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 3, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_float_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 5, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_float_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 5, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_float_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 7, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_float_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 7, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_float_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 9, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_float_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 9, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_float_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 11, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_float_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 11, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_float_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 13, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_float_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 13, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);
}

void na1d_rpbgrad_cuda_naive_half_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, -1, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_half_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, -1, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_half_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 3, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_half_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 3, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_half_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 5, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_half_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 5, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_half_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 7, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_half_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 7, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_half_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 9, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_half_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 9, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_half_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 11, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_half_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 11, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_half_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 13, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_half_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 13, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_bfloat16_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, -1, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_bfloat16_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, -1, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_bfloat16_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 3, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_bfloat16_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 3, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_bfloat16_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 5, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_bfloat16_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 5, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_bfloat16_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 7, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_bfloat16_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 7, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_bfloat16_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 9, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_bfloat16_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 9, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_bfloat16_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 11, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_bfloat16_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 11, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_bfloat16_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 13, -1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na1d_rpbgrad_cuda_naive_bfloat16_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int length,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 13, 1>;
  using Kernel = RelPosBiasGradient1D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, length, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_double_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, -1, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_double_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, -1, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_double_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 3, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_double_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 3, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_double_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 5, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_double_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 5, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_double_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 7, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_double_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 7, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_double_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 9, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_double_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 9, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_double_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 11, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_double_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 11, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_double_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 13, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_double_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float64, 13, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_float_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, -1, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_float_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, -1, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_float_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 3, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_float_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 3, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_float_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 5, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_float_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 5, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_float_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 7, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_float_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 7, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_float_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 9, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_float_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 9, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_float_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 11, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_float_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 11, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_float_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 13, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_float_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {
  using Arguments = natten::naive::ArgumentPack<natten::float32, 13, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);
}

void na2d_rpbgrad_cuda_naive_half_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, -1, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_half_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, -1, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_half_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 3, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_half_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 3, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_half_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 5, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_half_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 5, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_half_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 7, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_half_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 7, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_half_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 9, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_half_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 9, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_half_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 11, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_half_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 11, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_half_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 13, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_half_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 13, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_bfloat16_ks_any_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, -1, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_bfloat16_ks_any_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, -1, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_bfloat16_ks_3_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 3, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_bfloat16_ks_3_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 3, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_bfloat16_ks_5_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 5, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_bfloat16_ks_5_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 5, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_bfloat16_ks_7_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 7, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_bfloat16_ks_7_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 7, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_bfloat16_ks_9_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 9, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_bfloat16_ks_9_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 9, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_bfloat16_ks_11_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 11, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_bfloat16_ks_11_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 11, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_bfloat16_ks_13_di_any(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 13, -1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na2d_rpbgrad_cuda_naive_bfloat16_ks_13_di_1(
  void * d_bias_ptr,
  void * d_attn_ptr,
  int batch_size,
  int heads,
  int height,
  int width,
  int dim,
  int kernel_size,
  int dilation) {

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 13, 1>;
  using Kernel = RelPosBiasGradient2D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, height, width, dim, kernel_size, dilation);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_double_ks_any_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, -1, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_double_ks_any_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, -1, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_double_ks_3_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 3, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_double_ks_3_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 3, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_double_ks_5_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 5, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_double_ks_5_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 5, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_double_ks_7_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 7, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_double_ks_7_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 7, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_double_ks_9_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 9, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_double_ks_9_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 9, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_double_ks_11_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 11, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_double_ks_11_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 11, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_double_ks_13_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 13, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_double_ks_13_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float64, 13, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_float_ks_any_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, -1, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_float_ks_any_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, -1, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_float_ks_3_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 3, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_float_ks_3_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 3, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_float_ks_5_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 5, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_float_ks_5_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 5, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_float_ks_7_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 7, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_float_ks_7_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 7, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_float_ks_9_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 9, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_float_ks_9_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 9, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_float_ks_11_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 11, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_float_ks_11_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 11, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_float_ks_13_di_any(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 13, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_float_ks_13_di_1(
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
  using Arguments = natten::naive::ArgumentPack<natten::float32, 13, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);
}

void na3d_rpbgrad_cuda_naive_half_ks_any_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, -1, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_half_ks_any_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, -1, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_half_ks_3_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 3, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_half_ks_3_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 3, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_half_ks_5_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 5, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_half_ks_5_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 5, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_half_ks_7_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 7, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_half_ks_7_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 7, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_half_ks_9_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 9, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_half_ks_9_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 9, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_half_ks_11_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 11, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_half_ks_11_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 11, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_half_ks_13_di_any(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 13, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_half_ks_13_di_1(
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

#ifdef NATTEN_ENABLE_FP16
  using Arguments = natten::naive::ArgumentPack<natten::float16, 13, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_bfloat16_ks_any_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, -1, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_bfloat16_ks_any_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, -1, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_bfloat16_ks_3_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 3, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_bfloat16_ks_3_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 3, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_bfloat16_ks_5_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 5, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_bfloat16_ks_5_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 5, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_bfloat16_ks_7_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 7, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_bfloat16_ks_7_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 7, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_bfloat16_ks_9_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 9, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_bfloat16_ks_9_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 9, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_bfloat16_ks_11_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 11, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_bfloat16_ks_11_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 11, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_bfloat16_ks_13_di_any(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 13, -1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

void na3d_rpbgrad_cuda_naive_bfloat16_ks_13_di_1(
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

#ifdef NATTEN_ENABLE_BF16
  using Arguments = natten::naive::ArgumentPack<natten::bfloat16, 13, 1>;
  using Kernel = RelPosBiasGradient3D<Arguments>;
  Kernel kernel;
  kernel(
d_bias_ptr, d_attn_ptr, batch_size, heads, depth, height, width, dim, kernel_size, dilation, kernel_size_d, dilation_d);

#else
std::cerr << "NATTEN was not built with support for this half type."  << std::endl; 
exit(EXIT_FAILURE); 

#endif
}

} 
} 
} 

