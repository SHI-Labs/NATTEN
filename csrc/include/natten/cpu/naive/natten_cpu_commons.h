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
    \brief Holds common functions shared between ops.
*/

#pragma once

namespace natten {
namespace cpu {
namespace naive {

template <typename T>
struct AttnMask {
  static_assert(std::numeric_limits<T>::is_iec559);

  static inline auto value(bool is_grad) {
    return is_grad ? T(0.0) : -std::numeric_limits<T>::infinity();
  }
};

inline int32_t get_backward_window_start(
    int32_t index,
    int32_t kernel_size,
    int32_t neighborhood_size,
    int32_t dilation,
    bool is_causal) {
  if (!is_causal) {
    return (index < kernel_size * dilation)
        ? (index % dilation)
        : index - neighborhood_size * dilation;
  }
  return index;
}

inline int32_t get_backward_window_end(
    int32_t index,
    int32_t length,
    int32_t kernel_size,
    int32_t neighborhood_size,
    int32_t dilation,
    bool is_causal) {
  if (!is_causal) {
    return (index >= length - kernel_size * dilation)
        ? (length)
        : (index + (neighborhood_size + 1) * dilation);
  }
  return std::min(index + kernel_size * dilation, length);
}

inline int32_t get_window_start(
    int32_t index,
    int32_t length,
    int32_t kernel_size,
    int32_t neighborhood_size,
    int32_t dilation,
    bool is_causal) {
  auto dilation_idx = index % dilation;
  auto index_pdp = index / dilation;
  auto length_pdp = (length + dilation - 1) / dilation;
  auto num_padded = (length_pdp * dilation) - length;
  length_pdp -= (dilation_idx >= dilation - num_padded) ? 1 : 0;
  int32_t start_idx;
  if (!is_causal) {
    start_idx = std::max(index_pdp - neighborhood_size, 0) +
        (index_pdp + neighborhood_size >= length_pdp) *
            (length_pdp - index_pdp - neighborhood_size - 1);
  } else {
    start_idx = std::max(index_pdp - kernel_size + 1, 0);
  }
  return start_idx * dilation + dilation_idx;
}

inline int32_t get_window_end(
    int32_t index,
    int32_t start_index,
    int32_t length,
    int32_t kernel_size,
    int32_t neighborhood_size,
    int32_t dilation,
    bool is_causal) {
  if (!is_causal) {
    return std::min(length, start_index + kernel_size * dilation);
  }
  return std::min(length, index + dilation);
}

inline int32_t get_pb_start(
    int32_t index,
    int32_t length,
    int32_t kernel_size,
    int32_t neighborhood_size,
    int32_t dilation) {
  if (dilation <= 1)
    return neighborhood_size +
        (index < neighborhood_size) * (neighborhood_size - index) +
        (index + neighborhood_size >= length) *
        (length - index - 1 - neighborhood_size);
  if (index - neighborhood_size * dilation < 0)
    return kernel_size - 1 - (index / dilation);
  if (index + neighborhood_size * dilation >= length)
    return (length - index - 1) / dilation;
  return neighborhood_size;
}

} // namespace naive
} // namespace cpu
} // namespace natten
