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

#pragma once

#include "natten/dtypes.cuh"

#define DISPATCH_DTYPE(device_index, stream, c10_dtype, fn_name, ...)        \
  [&] {                                                                      \
    cudaDeviceProp* device_props =                                           \
        at::cuda::getDeviceProperties(device_index);                         \
    const int cc = device_props->major * 10 + device_props->minor;           \
    const size_t max_smem = device_props->sharedMemPerBlockOptin;            \
    if (c10_dtype == torch::kFloat) {                                        \
      fn_name<natten::float32>(cc, max_smem, stream, __VA_ARGS__);           \
    } else if (c10_dtype == torch::kDouble) {                                \
      fn_name<natten::float64>(cc, max_smem, stream, __VA_ARGS__);           \
    } else if (c10_dtype == torch::kFloat16 && cc >= 50) {                   \
      fn_name<natten::float16>(cc, max_smem, stream, __VA_ARGS__);           \
    } else if (c10_dtype == torch::kBFloat16 && cc >= 80) {                  \
      fn_name<natten::bfloat16>(cc, max_smem, stream, __VA_ARGS__);          \
    } else {                                                                 \
      std::cerr << "NATTEN (CUDA) does not support data type " << c10_dtype  \
                << " on device with index " << device_index << " (SM " << cc \
                << ")." << std::endl;                                        \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  }()
