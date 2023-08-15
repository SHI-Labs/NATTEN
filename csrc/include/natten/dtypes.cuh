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

#pragma once
#ifdef NATTEN_ENABLE_FP16
#include <cuda_fp16.h>
#endif
#ifdef NATTEN_ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace natten {

namespace dummy {

  struct sym_bf16 {};
  struct sym_f16 {};
  struct sym_tf32 {};

}

using float64 = double;
using float32 = float;
#ifdef NATTEN_ENABLE_FP16
using float16  = __half;
#else
using float16 = dummy::sym_f16;
#endif
#ifdef NATTEN_ENABLE_BF16
using bfloat16 = __nv_bfloat16;
#else
using bfloat16 = dummy::sym_bf16;
#endif

using tf32 = dummy::sym_tf32;

} // namespace natten

