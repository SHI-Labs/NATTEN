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
    \brief Holds dispatchers, and common functions shared between ops.
*/

#pragma once

namespace natten {
namespace cpu {
namespace naive {

inline int get_backward_window_start(
    const int index,
    const int KERNEL_SIZE,
    const int NEIGHBORHOOD_SIZE,
    const int dilation) {
  return (index < KERNEL_SIZE * dilation)
      ? (index % dilation)
      : index - NEIGHBORHOOD_SIZE * dilation;
}

inline int get_backward_window_end(
    const int index,
    const int length,
    const int KERNEL_SIZE,
    const int NEIGHBORHOOD_SIZE,
    const int dilation) {
  return (index >= length - KERNEL_SIZE * dilation)
      ? (length)
      : (index + (NEIGHBORHOOD_SIZE + 1) * dilation);
}

inline int get_window_start(
    const int index,
    const int length,
    const int KERNEL_SIZE,
    const int NEIGHBORHOOD_SIZE,
    const int dilation) {
  if (dilation <= 1)
    return std::max(index - NEIGHBORHOOD_SIZE, 0) +
        (index + NEIGHBORHOOD_SIZE >= length) *
        (length - index - NEIGHBORHOOD_SIZE - 1);
  int ni = index - NEIGHBORHOOD_SIZE * dilation;
  if (ni < 0)
    return index % dilation;
  if (index + NEIGHBORHOOD_SIZE * dilation >= length) {
    const int imodd = index % dilation;
    const int a = int(length / dilation) * dilation;
    const int b = length - a;
    if (imodd < b)
      return length - b + imodd - 2 * NEIGHBORHOOD_SIZE * dilation;
    return a + imodd - KERNEL_SIZE * dilation;
  }
  return ni;
}

inline int get_pb_start(
    const int index,
    const int length,
    const int KERNEL_SIZE,
    const int NEIGHBORHOOD_SIZE,
    const int dilation) {
  if (dilation <= 1)
    return NEIGHBORHOOD_SIZE +
        (index < NEIGHBORHOOD_SIZE) * (NEIGHBORHOOD_SIZE - index) +
        (index + NEIGHBORHOOD_SIZE >= length) *
        (length - index - 1 - NEIGHBORHOOD_SIZE);
  if (index - NEIGHBORHOOD_SIZE * dilation < 0)
    return KERNEL_SIZE - 1 - (index / dilation);
  if (index + NEIGHBORHOOD_SIZE * dilation >= length)
    return (length - index - 1) / dilation;
  return NEIGHBORHOOD_SIZE;
}

} // namespace naive
} // namespace cpu
} // namespace natten
