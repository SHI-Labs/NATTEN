/***************************************************************************************************
 * Copyright (c) 2023 Ali Hassani.
 **************************************************************************************************/
/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/layout/matrix.h>

#include "natten/cuda/gemm/neighborhood_attention.cuh"
#include "natten/cuda/gemm/na2d_problem_size.cuh"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace natten {
namespace cuda {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  typename Layout_ = cutlass::layout::TensorNDHWC
>
struct NA2dAnalyticParams {

  using Layout = Layout_;

  Layout layout;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  NA2dAnalyticParams() { }

  CUTLASS_HOST_DEVICE
  NA2dAnalyticParams(
    NA2dProblemSize const &,  // unused; placeholder to match other Params interfaces.
    Layout const &layout
  ): layout(layout) { }
};

template<
  typename NAShape_,
  typename Layout_ = cutlass::layout::TensorNDHWC
>
struct NA2dOptimizedParams {

  using NAShape = NAShape_;
  using Layout = Layout_;

  Layout layout;

  cutlass::FastDivmod stride_divmod;
  cutlass::FastDivmod tile_divmod;

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  NA2dOptimizedParams():
     stride_divmod(NAShape::kStride),
     tile_divmod(NAShape::kTile) { }

  CUTLASS_HOST_DEVICE
  NA2dOptimizedParams(
    NA2dProblemSize const &,  // unused; placeholder to match other Params interfaces.
    Layout const &layout
  ): layout(layout),
     stride_divmod(NAShape::kStride),
     tile_divmod(NAShape::kTile) { }
};

} // namespace threadblock
} // namespace gemm
} // namespace cuda
} // namespace natten

/////////////////////////////////////////////////////////////////////////////////////////////////
