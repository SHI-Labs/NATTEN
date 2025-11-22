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

#include "cutlass/cutlass.h"

#include "natten/cuda/flash_fna/flash_kernel/flash.h"
#include "natten/cuda/flash_fna/flash_kernel/flash_fwd_launch_template.h"

namespace natten {
namespace cuda {
namespace flash_fna {

template <int Arch, typename Element, int HeadDim, int kBlockM, int kBlockN,
          class NADim, class QTileShape, class KVTileShape, class Causal>
struct FlashFnaForwardKernel {

  void run(Flash_fna_fwd_params params, cudaStream_t stream){

    auto flash_fwd = run_flash_fwd<
        /* Arch= */ Arch,
        /* kHeadDim= */ HeadDim,
        /* kHeadDimV= */ HeadDim,
        /* kBlockM= */ kBlockM,
        /* kBlockN= */ kBlockN,
        /* Element= */ Element,
        /* ElementOut= */ Element,
        /* PackGQA= */ false,
        /* V_colmajor= */ false,
        /* NADim= */ NADim,
        /* QTileShape= */ QTileShape,
        /* KVTileShape= */ KVTileShape,
        /* Causal= */ Causal 
    >;

    flash_fwd(params, stream);
  }
};

} // namespace flash_fna
} // namespace cuda
} // namespace natten
