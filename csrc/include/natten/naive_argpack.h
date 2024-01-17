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

namespace natten {
namespace naive {

// TODO:

template <typename scalar_t, bool CausalMask0_>
struct ArgumentPack1D {
  using Dtype = scalar_t;
  struct CausalMask {
    static constexpr bool Dim0 = CausalMask0_;
  };
};

template <typename scalar_t, bool CausalMask0_, bool CausalMask1_>
struct ArgumentPack2D {
  using Dtype = scalar_t;
  struct CausalMask {
    static constexpr bool Dim0 = CausalMask0_;
    static constexpr bool Dim1 = CausalMask1_;
  };
};

template <
    typename scalar_t,
    bool CausalMask0_,
    bool CausalMask1_,
    bool CausalMask2_>
struct ArgumentPack3D {
  using Dtype = scalar_t;
  struct CausalMask {
    static constexpr bool Dim0 = CausalMask0_;
    static constexpr bool Dim1 = CausalMask1_;
    static constexpr bool Dim2 = CausalMask2_;
  };
};

} // namespace naive
} // namespace natten
