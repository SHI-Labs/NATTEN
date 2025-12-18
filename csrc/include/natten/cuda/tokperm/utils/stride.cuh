/***************************************************************************************************
 * Copyright (c) 2022-2025 Ali Hassani.
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

#include <cute/layout.hpp>
#include <cute/tensor.hpp>

namespace natten::tokperm::utils {

using namespace cute;

namespace detail {

template <class Shape>
struct StrideHelper {
  static_assert(cute::rank(Shape{}) >= 1);
  static_assert(depth(Shape{}) == 1);
  static constexpr int Rank = cute::rank(Shape{});

  using StrideType = decltype(replace<Rank - 1>(Shape{}, _1{}));

  CUTE_HOST_DEVICE static constexpr StrideType make_stride(Shape const& shape) {
    if constexpr (Rank == 1) {
      return cute::make_stride(_1{});
    } else {
      StrideType stride;

      get<Rank - 1>(stride) = _1{};
      get<Rank - 2>(stride) = get<Rank - 1>(shape);
      cute::for_each(cute::make_range<0, Rank - 1>{}, [&](auto i) {
        static_assert(i < Rank - 1);
        get<Rank - i - 2>(stride) =
            get<Rank - i - 1>(shape) * get<Rank - i - 1>(stride);
      });

      return stride;
    }
  }
};

} // namespace detail

template <class Shape>
CUTE_HOST_DEVICE static constexpr auto make_torch_contiguous_stride(
    Shape const& shape) {
  auto flattened_shape = cute::flatten(shape);
  auto flattened_stride =
      detail::StrideHelper<decltype(flattened_shape)>::make_stride(
          flattened_shape);
  return cute::unflatten(flattened_stride, shape);
}

} // namespace natten::tokperm::utils
