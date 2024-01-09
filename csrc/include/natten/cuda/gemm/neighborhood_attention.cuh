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
    \brief
This file contains definitions and utility functions for describing neighborhood
attention problem sizes in terms of inputs (BNHWD), attention (BNHW[RS]),
dilation (dilation_h, dilation_w). Furthermore, it defines helper functions to
map cutlass' implicit gemm tensor extents, sizes, data types to that of
neighborhood attention extents, sizes, and data types.
                        * Mapping neighborhood attention to Gemm computation *
We employ ImplicitGemm to implement neighborhood attention, the same algorithm
used in cutlass to implement convolutions. ImplicitGemm algorithm runs gemm
operation on input tensors and attention tensors. The underlying gemm operation
follows the standard gemm definition: C = A * B + C A and B are input matrices
                            C is source and output matrix
NATTEN breaks down neighborhood attention to two operations: QK(optionally
+RPB), and AV. The former takes in two input tensors (meaning they are two batch
x heads x height x width x dim tensors), also referred to here as pointwise
tensors, and extracts neighborhoods in the second tensor (keys), and multiplies
the first tensor (queries) by it. We label this operation pointwise-neighborhood
(PN). This operation produces attention weights, which is also a 5D tensor
(batch x heads x height x width x neighborhood size). The second operation, AV,
takes attention weights and applies them to the other pointwise tensor (values),
by basically treating the latter as convolution activations, and the former as
dynamic filters. We label this operation neighborhood-neighborhood. Now moving
on to gradients, the gradient of attention weights can be computed with a simple
PN of the output gradient and the values tensor. So PN serves as both QK-forward
and A-backward. Additionally, the gradient of the queries tensor can also be
computed with a simple NN of the attention gradient and the original keys
tensor. As a result, NN also serves as both AV-forward and Q-backward. However,
K-backward and V-backward are a more of a challenge. Both reduce to each other,
meaning they only need a single operation/kernel as well. The difference is that
they need an inverse-neighborhood-neighborhood (IN) operation. Unlike
convolutions, neighborhoods in na do not maintain a symmetric relationship;
meaning that if a key-value pixel, y, is in the neighborhood of some query
pixel, x, then x's corresponding pixel in the key-value pair is not necessarily
in the neighborhood of y's corresponding pixel in query pixels. In short, for
every key-value gradient, there's as few as half and as many as 1.5 times the
number of gradient neighbors. These two gradients therefore require an inverse
neighborhood operation. K-backward is the multiplication of query's
inverse-neighborhood activation and that of attention gradient. Similarly,
V-backward is the multiplication of output gradient's inverse neighborhood
activation, and that of attention weights.

For the three operators (PN, NN, IN), ImplicitGemm matrices A, B, and C are
mapped onto NA tensors Input (query, key, value, output), Attention (attention
weights) as per the below table:
        ___________________________________________________________________________
               NAOperator      |        A        |      B         |       C
        ___________________________________________________________________________
        |                      |                 |                | | | PN |
Input (PW)   |   Input (N)    |   Attention   | |        NN            |
Attention    |   Input (N)    |   Input (PW)  | |        IN            |
Attention (IN)  |   Input (IN)   |   Input (PW)  |
        ___________________________________________________________________________
*/

#pragma once

namespace natten {
namespace cuda {
namespace gemm {

////////////////////////////////////////////////////////////////////////////////////////////////////

// NA1dShape does not exist, because the "multi-axis tiling" is unnecessary in
// 1D.

template <int T = 1, int T_ext = 1, int T_ext_mul = 1>
struct NA2dShape {
  static int const kTile = T;
  static int const kStride = T + T_ext * T_ext_mul;
};

/// Neighborhood Attention operators
enum class Operator {
  kPN, ///< pointwise-neighborhood: QK-forward and A-backward
  kNN, ///< neighborhood-neighborhood: AV-forward and Q-backward
  kIN ///< inverse neighborhood: K-backward and V-backward
};

// Kernel launcher
// Duplicate of CUTLASS's
template <typename ArchTag, typename Operator>
__global__ void Kernel(typename Operator::Params params) {
#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 700)
#if (__CUDA_ARCH__ >= 800)
  if constexpr (std::is_same<ArchTag, cutlass::arch::Sm80>::value) {
#elif (__CUDA_ARCH__ < 800) && (__CUDA_ARCH__ >= 750)
  if constexpr (std::is_same<ArchTag, cutlass::arch::Sm75>::value) {
#elif (__CUDA_ARCH__ < 750) && (__CUDA_ARCH__ >= 700)
  if constexpr (std::is_same<ArchTag, cutlass::arch::Sm70>::value) {
#endif
    // Dynamic shared memory base pointer
    extern __shared__ int SharedStorageBase[];

    // Declare pointer to dynamic shared memory.
    typename Operator::SharedStorage* shared_storage =
        reinterpret_cast<typename Operator::SharedStorage*>(SharedStorageBase);

    Operator op;

    op(params, *shared_storage);
  }
#else
  printf("Kernel not supported on this device / CUDA version.\n");
  asm volatile("brkpt;\n");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace gemm
} // namespace cuda
} // namespace natten
