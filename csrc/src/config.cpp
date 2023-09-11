/***************************************************************************************************
 * Copyright (c) 2023 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include "natten/config.h"

namespace natten {

#ifdef NATTEN_WITH_CUDA
bool kHasCUDA = true;
#ifdef NATTEN_WITH_CUTLASS
bool kHasGEMM = true;
#else
bool kHasGEMM = false;
#endif
#else
bool kHasCUDA = false;
bool kHasGEMM = false;
#endif
#ifdef NATTEN_ENABLE_FP16
bool kHasHalf = true;
#else
bool kHasHalf = false;
#endif
#ifdef NATTEN_ENABLE_BF16
bool kHasBFloat = true;
#else
bool kHasBFloat = false;
#endif

bool has_cuda() {
  return kHasCUDA;
}

bool has_half() {
  return kHasHalf;
}

bool has_bfloat() {
  return kHasBFloat;
}

bool has_gemm() {
  return kHasGEMM;
}

bool kEnableTiledNA  = true;
bool kEnableGemmNA   = true;

bool get_tiled_na() {
  return kEnableTiledNA;
}

bool get_gemm_na() {
  return kEnableGemmNA;
}

void set_tiled_na(bool v) {
  kEnableTiledNA = v;
}

void set_gemm_na(bool v) {
  kEnableGemmNA = v;
}

#if (NATTEN_CUTLASS_TARGET_SM >= 80)
bool kEnableGemmTF32 = true;

bool get_gemm_tf32() {
  return kEnableGemmTF32;
}

void set_gemm_tf32(bool v) {
  kEnableGemmTF32 = v;
}
#else
bool kEnableGemmTF32 = false;

bool get_gemm_tf32() {
  return false;
}

void set_gemm_tf32(bool v) {
}
#endif

} // namespace natten

