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

#include "natten/cuda/flash_fna/flash_kernel/flash_bwd_launch_template.h"

namespace natten {
namespace cuda {
namespace flash_fna {

struct Config {
    int  Stages_dO;
    int  Stages_dS_or_QSm80;
    bool SdP_swapAB;
    bool dKV_swapAB;
    bool dQ_swapAB;
    int  NumMmaWarpGroups;
    int  AtomLayoutMSdP;
    int  AtomLayoutNdKV;
    int  AtomLayoutMdQ;
    bool V_in_regs;
};

template <int HeadDim, int Arch>
constexpr Config get_config() {
    if constexpr (HeadDim == 32) {
        if constexpr (Arch == 86 || Arch == 89) {
            return Config{2, 2, false, false, false, 2, 2, 4, 2, true};
        } else if constexpr (Arch == 80) {
            return Config{2, 2, false, false, false, 2, 4, 4, 4, false};
        }
    } else if constexpr (HeadDim == 64) {
        if constexpr (Arch == 86 || Arch == 89) {
            return Config{2, 2, false, false, false, 2, 2, 4, 2, true};
        } else if constexpr (Arch == 80) {
            return Config{2, 2, false, false, false, 2, 4, 4, 4, false};
        }
    } else if constexpr (HeadDim == 96) {
        if constexpr (Arch == 86 || Arch == 89) {
            return Config{1, 2, false, false, false, 2, 2, 4, 2, true};
        } else if constexpr (Arch == 80) {
            return Config{2, 2, false, false, false, 2, 2, 4, 2, false};
        }
    } else if constexpr (HeadDim == 128) {
        if constexpr (Arch == 86 || Arch == 89) {
            return Config{1, 2, false, false, false, 2, 2, 2, 2, true};
        } else if constexpr (Arch == 80) {
            return Config{2, 2, false, false, false, 2, 2, 2, 2, false};
        }
    } else if constexpr (HeadDim == 192) {
        if constexpr (Arch == 86 || Arch == 89) {
            return Config{1, 1, false, false, false, 2, 2, 2, 2, true};
        } else if constexpr (Arch == 80) {
            return Config{1, 2, false, true, false, 2, 4, 2, 2, false};
        }
    } else if constexpr (HeadDim == 256) {
        if constexpr (Arch == 86 || Arch == 89) {
            return Config{1, 1, false, false, false, 2, 2, 2, 1, true};
        } else if constexpr (Arch == 80) {
            return Config{1, 1, false, false, false, 2, 4, 2, 2, false};
        }
    } else {
        static_assert(HeadDim == -1, "Unsupported HeadDim/Arch combination");
    }
}


template <int Arch, typename Element, int HeadDim, int kBlockM, int kBlockN, bool Deterministic>
struct FlashFnaBackwardKernel {

  void run(Flash_fna_bwd_params params, cudaStream_t stream) {

    static constexpr Config config = get_config<HeadDim, Arch>();

    auto flash_bwd = run_flash_bwd<
      /* Arch= */               Arch,
      /* kHeadDim= */           HeadDim,
      /* kBlockM= */            kBlockM,
      /* kBlockN= */            kBlockN,
      /* Element= */            Element,
      /* Deterministic= */      Deterministic,
      /* GQA= */                false,
      /* Stages_dO= */          config.Stages_dO,
      /* Stages_dS_or_QSm80= */ config.Stages_dS_or_QSm80,
      /* SdP_swapAB= */         config.SdP_swapAB,
      /* dKV_swapAB= */         config.dKV_swapAB,
      /* dQ_swapAB= */          config.dQ_swapAB,
      /* NumMmaWarpGroups= */   config.NumMmaWarpGroups,
      /* AtomLayoutMSdP= */     config.AtomLayoutMSdP,
      /* AtomLayoutNdKV= */     config.AtomLayoutNdKV,
      /* AtomLayoutMdQ= */      config.AtomLayoutMdQ,
      /* V_in_regs= */          config.V_in_regs
    >;

    flash_bwd(params, stream);
  }
};

} // namespace flash_fna
} // namespace cuda
} // namespace natten
