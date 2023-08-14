/***************************************************************************************************
 * Copyright (c) 2023 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#include <ATen/ATen.h>

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


#ifdef NATTEN_WITH_CUDA
#define DISPATCH_DEVICE(c10_type, kernel_name, ...)                                                \
  [&] {                                                                                            \
    if (c10_type.is_cpu()) {                                                                       \
      natten::pytorch::cpu::kernel_name(__VA_ARGS__);                                              \
    }                                                                                              \
    else if (c10_type.is_cuda()) {                                                                 \
      natten::pytorch::cuda::kernel_name(__VA_ARGS__);                                             \
    }                                                                                              \
    else {                                                                                         \
      std::cerr << "NATTEN does not support "                                                      \
                << c10_type << " devices yet."                                                     \
                << std::endl;                                                                      \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  }()
#else
#define DISPATCH_DEVICE(c10_type, kernel_name, ...)                                                \
  [&] {                                                                                            \
    if (c10_type.is_cpu()) {                                                                       \
      natten::pytorch::cpu::kernel_name(__VA_ARGS__);                                              \
    }                                                                                              \
    else if (c10_type.is_cuda()) {                                                                 \
      std::cerr << "NATTEN was not built with "                                                    \
                << c10_type << " support."                                                         \
                << std::endl;                                                                      \
    }                                                                                              \
    else {                                                                                         \
      std::cerr << "NATTEN does not support "                                                      \
                << c10_type << " devices yet."                                                     \
                << std::endl;                                                                      \
      exit(EXIT_FAILURE);                                                                          \
    }                                                                                              \
  }()
#endif
