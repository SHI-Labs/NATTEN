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

#include <iostream>

namespace natten {

#define NATTEN_FAILURE(msg)       \
  std::cerr << #msg << std::endl; \
  exit(EXIT_FAILURE);

#define NATTEN_CHECK(cond, msg)     \
  if (!(cond)) {                    \
    std::cerr << #msg << std::endl; \
    exit(EXIT_FAILURE);             \
  }

// Causal mask helpers
template <typename TupleType>
bool any_true(TupleType v);

template <>
inline bool any_true(std::tuple<bool> v) {
  return std::get<0>(v);
}

template <>
inline bool any_true(std::tuple<bool, bool> v) {
  return std::get<0>(v) || std::get<1>(v);
}

template <>
inline bool any_true(std::tuple<bool, bool, bool> v) {
  return std::get<0>(v) || std::get<1>(v) || std::get<2>(v);
}

// Kernel size helpers
template <typename TupleType>
int32_t flatten(TupleType v);

template <>
inline int32_t flatten(std::tuple<int32_t> v) {
  return std::get<0>(v);
}

template <>
inline int32_t flatten(std::tuple<int32_t, int32_t> v) {
  return std::get<0>(v) * std::get<1>(v);
}

template <>
inline int32_t flatten(std::tuple<int32_t, int32_t, int32_t> v) {
  return std::get<0>(v) * std::get<1>(v) * std::get<2>(v);
}

template <typename TupleType>
bool all_dims_match(TupleType v);

template <>
inline bool all_dims_match(std::tuple<int32_t> v) {
  return true;
}

template <>
inline bool all_dims_match(std::tuple<int32_t, int32_t> v) {
  return std::get<0>(v) == std::get<1>(v);
}

template <>
inline bool all_dims_match(std::tuple<int32_t, int32_t, int32_t> v) {
  return std::get<0>(v) == std::get<1>(v) && std::get<1>(v) == std::get<2>(v);
}

// Runtime accessors

template <typename TupleType>
int32_t get_from_tuple(TupleType v, size_t index);

template <>
inline int32_t get_from_tuple(std::tuple<int32_t> v, size_t index) {
  NATTEN_CHECK(index == 0, "Got invalid index for tuple of size 1.");
  return std::get<0>(v);
}

template <>
inline int32_t get_from_tuple(std::tuple<int32_t, int32_t> v, size_t index) {
  NATTEN_CHECK(
      index == 0 || index == 1, "Got invalid index for tuple of size 2.");
  if (index == 0) {
    return std::get<0>(v);
  }
  return std::get<1>(v);
}

template <>
inline int32_t get_from_tuple(
    std::tuple<int32_t, int32_t, int32_t> v,
    size_t index) {
  NATTEN_CHECK(
      index == 0 || index == 1 || index == 2,
      "Got invalid index for tuple of size 3.");
  if (index == 0) {
    return std::get<0>(v);
  } else if (index == 1) {
    return std::get<1>(v);
  }
  return std::get<2>(v);
}

} // namespace natten
