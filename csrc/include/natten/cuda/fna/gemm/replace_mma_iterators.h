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

#include <cutlass/gemm/threadblock/mma_multistage.h>
#include <cutlass/gemm/threadblock/mma_pipelined.h>

namespace natten {
namespace cuda {
namespace fna {

// Technically we can easily replace iterators with a derived
// class, but `MakeCustomMma` dispatches based on Mma classes,
// which makes it really difficult to add arbitrary derived classes,
// so to avoid that, we just replace the Mma iterators by creating
// new Mma instantiations instead of using inheritance.
template <typename Mma, typename NewIteratorA, typename NewIteratorB>
struct ReplaceMmaIterators;

template <
    typename Shape,
    typename IteratorA,
    typename SmemIteratorA,
    cutlass::arch::CacheOperation::Kind CacheOpA,
    typename IteratorB,
    typename SmemIteratorB,
    cutlass::arch::CacheOperation::Kind CacheOpB,
    typename ElementC,
    typename LayoutC,
    typename Policy,
    int kStages,
    cutlass::gemm::SharedMemoryClearOption SharedMemoryClear,
    typename NewIteratorA,
    typename NewIteratorB>
struct ReplaceMmaIterators<
    cutlass::gemm::threadblock::MmaMultistage<
        Shape,
        IteratorA,
        SmemIteratorA,
        CacheOpA,
        IteratorB,
        SmemIteratorB,
        CacheOpB,
        ElementC,
        LayoutC,
        Policy,
        kStages,
        SharedMemoryClear>,
    NewIteratorA,
    NewIteratorB> {
  using Mma = cutlass::gemm::threadblock::MmaMultistage<
      Shape,
      NewIteratorA,
      SmemIteratorA,
      CacheOpA,
      NewIteratorB,
      SmemIteratorB,
      CacheOpB,
      ElementC,
      LayoutC,
      Policy,
      kStages,
      SharedMemoryClear>;
};

template <
    typename Shape,
    typename IteratorA,
    typename SmemIteratorA,
    typename IteratorB,
    typename SmemIteratorB,
    typename ElementC,
    typename LayoutC,
    typename Policy,
    typename NewIteratorA,
    typename NewIteratorB>
struct ReplaceMmaIterators<
    cutlass::gemm::threadblock::MmaPipelined<
        Shape,
        IteratorA,
        SmemIteratorA,
        IteratorB,
        SmemIteratorB,
        ElementC,
        LayoutC,
        Policy>,
    NewIteratorA,
    NewIteratorB> {
  using Mma = cutlass::gemm::threadblock::MmaPipelined<
      Shape,
      NewIteratorA,
      SmemIteratorA,
      NewIteratorB,
      SmemIteratorB,
      ElementC,
      LayoutC,
      Policy>;
};

} // namespace fna
} // namespace cuda
} // namespace natten
