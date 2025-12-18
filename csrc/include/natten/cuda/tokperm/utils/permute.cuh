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
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>

#include <natten/cuda/utils/cutlass.cuh>

#include <natten/cuda/tokperm/utils/stride.cuh>

namespace natten::tokperm::utils {

using namespace cute;

// `permute_tokens`: utility for applying permutations to arbitrary-sized token
// views. Expects rank-4 inputs (batch, (*tokens), heads, head_dim)
//
// NOTE: does not check Permutation to ensure it's a valid permutation
// I.e. permute_tokens<1,1> repeats token dimension 1 without raising an error.
template <int... Permutation, class ProblemShape>
CUTE_HOST_DEVICE constexpr ProblemShape permute_tokens(
    ProblemShape problem_shape) {
  static_assert(rank(ProblemShape{}) == 4);

  return cute::make_tuple(
      get<0>(problem_shape),
      select<Permutation...>(get<1>(problem_shape)),
      get<2>(problem_shape),
      get<3>(problem_shape));
}

// `permute_tokens_varlen`
// Expects rank-2 inputs ((*tokens), heads * head_dim)
//
// NOTE: does not check Permutation to ensure it's a valid permutation
// I.e. permute_tokens<1,1> repeats token dimension 1 without raising an error.
template <int... Permutation, class ProblemShape>
CUTE_HOST_DEVICE constexpr ProblemShape permute_tokens_varlen(
    ProblemShape problem_shape) {
  static_assert(rank(ProblemShape{}) == 2);

  return cute::make_tuple(
      select<Permutation...>(get<0>(problem_shape)), get<1>(problem_shape));
}

} // namespace natten::tokperm::utils
