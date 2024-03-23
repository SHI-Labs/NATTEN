/***************************************************************************************************
 * Copyright (c) 2022-2024 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 **************************************************************************************************/
/*! \file
    \brief compute_delta interface
    mostly used to test the kernel.
*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <natten/natten.h>

#ifdef NATTEN_WITH_CUTLASS
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/half.h>
#include <natten/cuda/compute_delta.cuh>
#include <natten/dtypes.cuh>
#endif

// TODO: merge these two
#define DISPATCH_DOUBLE_ALIGNMENT(dtype_in, dtype_out, fn_name, dim, ...) \
  [&] {                                                                   \
    if (dim % 2 == 0) {                                                   \
      fn_name<dtype_in, dtype_out, 2>(__VA_ARGS__);                       \
    } else {                                                              \
      fn_name<dtype_in, dtype_out, 1>(__VA_ARGS__);                       \
    }                                                                     \
  }()

#define DISPATCH_ALIGNMENT(dtype_in, dtype_out, fn_name, dim, kElements, ...) \
  [&] {                                                                       \
    if (dim % (kElements * 4) == 0) {                                         \
      fn_name<dtype_in, dtype_out, kElements * 4>(__VA_ARGS__);               \
    } else if (dim % (kElements * 2) == 0) {                                  \
      fn_name<dtype_in, dtype_out, kElements * 2>(__VA_ARGS__);               \
    } else if (dim % kElements == 0) {                                        \
      fn_name<dtype_in, dtype_out, kElements>(__VA_ARGS__);                   \
    } else {                                                                  \
      fn_name<dtype_in, dtype_out, 1>(__VA_ARGS__);                           \
    }                                                                         \
  }()

#define DISPATCH_DTYPE_WITH_ALIGNMENT(                                         \
    dtype_out, fn_name, cc, dtype_in, dim, ...)                                \
  [&] {                                                                        \
    if (dtype_in == torch::kFloat) {                                           \
      DISPATCH_ALIGNMENT(float, dtype_out, fn_name, dim, 1, __VA_ARGS__);      \
    } else if (dtype_in == torch::kDouble) {                                   \
      DISPATCH_DOUBLE_ALIGNMENT(double, dtype_out, fn_name, dim, __VA_ARGS__); \
    } else if (dtype_in == torch::kFloat16 && cc >= 50) {                      \
      DISPATCH_ALIGNMENT(                                                      \
          cutlass::half_t, dtype_out, fn_name, dim, 2, __VA_ARGS__);           \
    } else if (dtype_in == torch::kBFloat16 && cc >= 80) {                     \
      DISPATCH_ALIGNMENT(                                                      \
          cutlass::bfloat16_t, dtype_out, fn_name, dim, 2, __VA_ARGS__);       \
    } else {                                                                   \
      std::cerr << "NATTEN (CUDA) does not support data type " << dtype_in     \
                << " on SM " << cc << ")." << std::endl;                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }()

#define DISPATCH_COMPUTE_DELTA(                                              \
    fn_name, device_index, dtype_in, dtype_out, dim, ...)                    \
  [&] {                                                                      \
    cudaDeviceProp* device_props =                                           \
        at::cuda::getDeviceProperties(device_index);                         \
    const int cc = device_props->major * 10 + device_props->minor;           \
    if (dtype_out == torch::kFloat) {                                        \
      DISPATCH_DTYPE_WITH_ALIGNMENT(                                         \
          float, fn_name, cc, dtype_in, dim, __VA_ARGS__);                   \
    } else if (dtype_out == torch::kDouble) {                                \
      DISPATCH_DTYPE_WITH_ALIGNMENT(                                         \
          double, fn_name, cc, dtype_in, dim, __VA_ARGS__);                  \
    } else if (dtype_out == torch::kFloat16 && cc >= 50) {                   \
      DISPATCH_DTYPE_WITH_ALIGNMENT(                                         \
          cutlass::half_t, fn_name, cc, dtype_in, dim, __VA_ARGS__);         \
    } else if (dtype_out == torch::kBFloat16 && cc >= 80) {                  \
      DISPATCH_DTYPE_WITH_ALIGNMENT(                                         \
          cutlass::bfloat16_t, fn_name, cc, dtype_in, dim, __VA_ARGS__);     \
    } else {                                                                 \
      std::cerr << "NATTEN (CUDA) does not support data type " << dtype_out  \
                << " on device with index " << device_index << " (SM " << cc \
                << ")." << std::endl;                                        \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  }()

namespace natten {
namespace pytorch {
namespace cuda {

void compute_delta(
    const at::Tensor& out,
    const at::Tensor& d_out,
    at::Tensor& delta,
    int32_t num_rows,
    int32_t dim) {
#ifdef NATTEN_WITH_CUTLASS

  NATTEN_CHECK(
      out.scalar_type() == d_out.scalar_type(),
      "`compute_delta` expects `out` and `d_out` to have the same dtype.");

  // TODO: bfloat16 tests fail, and it is not deterministic.
  // It most likely isn't alignment mismatch, but disabling half type outputs
  // for this, since we always keep delta in full precision.
  NATTEN_CHECK(
      delta.scalar_type() != torch::kFloat16 &&
          delta.scalar_type() != torch::kBFloat16,
      "`compute_delta` does not support half type outputs.");

  // Fetch workspace size
  int64_t workspace_size_bytes = 0;
  DISPATCH_COMPUTE_DELTA(
      natten::cuda::compute_delta_get_workspace_size,
      out.device().index(),
      out.scalar_type(),
      delta.scalar_type(),
      dim,
      workspace_size_bytes,
      num_rows,
      dim);

  // Allocate memory for workspace (if needed)
  torch::Tensor workspace;
  if (workspace_size_bytes) {
    workspace = torch::empty(
        {workspace_size_bytes}, out.options().dtype(at::ScalarType::Byte));
  }

  // Run the reduction
  DISPATCH_COMPUTE_DELTA(
      natten::cuda::compute_delta,
      out.device().index(),
      out.scalar_type(),
      delta.scalar_type(),
      dim,
      at::cuda::getCurrentCUDAStream(out.device().index()),
      static_cast<void*>(out.data_ptr()),
      static_cast<void*>(d_out.data_ptr()),
      static_cast<void*>(delta.data_ptr()),
      workspace_size_bytes ? static_cast<void*>(workspace.data_ptr()) : nullptr,
      workspace_size_bytes,
      num_rows,
      dim);
#else
  NATTEN_FAILURE(
      "`compute_delta` is only available when NATTEN is built with CUTLASS.");
#endif
}

} // namespace cuda
} // namespace pytorch
} // namespace natten
