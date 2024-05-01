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

#include <torch/extension.h>
#include <vector>

#include "natten/config.h"
#include "natten/pytorch/compute_delta.h"
#include "natten/pytorch/na1d.h"
#include "natten/pytorch/na2d.h"
#include "natten/pytorch/na3d.h"

namespace natten {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "na1d_forward", &natten::pytorch::na1d_forward, "NA1D forward (fused)");

  m.def(
      "na2d_forward", &natten::pytorch::na2d_forward, "NA2D forward (fused)");

  m.def(
      "na3d_forward", &natten::pytorch::na3d_forward, "NA3D forward (fused)");

  m.def(
      "na1d_backward", &natten::pytorch::na1d_backward, "NA1D backward (fused)");

  m.def(
      "na2d_backward", &natten::pytorch::na2d_backward, "NA2D backward (fused)");

  m.def(
      "na3d_backward", &natten::pytorch::na3d_backward, "NA3D backward (fused)");

  m.def(
      "na1d_qk_forward", &natten::pytorch::na1d_qk_forward, "NA1D QK forward");
  m.def(
      "na1d_qk_backward",
      &natten::pytorch::na1d_qk_backward,
      "NA1D QK backward");
  m.def(
      "na1d_av_forward", &natten::pytorch::na1d_av_forward, "NA1D AV forward");
  m.def(
      "na1d_av_backward",
      &natten::pytorch::na1d_av_backward,
      "NA1D AV backward");

  m.def(
      "na2d_qk_forward", &natten::pytorch::na2d_qk_forward, "NA2D QK forward");
  m.def(
      "na2d_qk_backward",
      &natten::pytorch::na2d_qk_backward,
      "NA2D QK backward");
  m.def(
      "na2d_av_forward", &natten::pytorch::na2d_av_forward, "NA2D AV forward");
  m.def(
      "na2d_av_backward",
      &natten::pytorch::na2d_av_backward,
      "NA2D AV backward");

  m.def(
      "na3d_qk_forward", &natten::pytorch::na3d_qk_forward, "NA3D QK forward");
  m.def(
      "na3d_qk_backward",
      &natten::pytorch::na3d_qk_backward,
      "NA3D QK backward");
  m.def(
      "na3d_av_forward", &natten::pytorch::na3d_av_forward, "NA3D AV forward");
  m.def(
      "na3d_av_backward",
      &natten::pytorch::na3d_av_backward,
      "NA3D AV backward");

  m.def(
      "has_cuda", &natten::has_cuda, "Whether NATTEN was compiled with CUDA.");
  m.def(
      "has_gemm",
      &natten::has_gemm,
      "Whether NATTEN was compiled with GEMM kernels.");

  // Only implemented for 2D NA's PN operator when dim_per_head == 32.
  m.def(
      "get_tiled_na",
      &natten::get_tiled_na,
      "Use tiled NA implementations when available.");
  m.def(
      "set_tiled_na",
      &natten::set_tiled_na,
      "Use tiled NA implementations when available.");

  // Only supports NA1D and NA2D, requires SM80 and above.
  m.def(
      "get_gemm_na",
      &natten::get_gemm_na,
      "Use GEMM-based NA implementations when available.");
  m.def(
      "set_gemm_na",
      &natten::set_gemm_na,
      "Use GEMM-based NA implementations when available.");

  // Only applies to Gemm NA kernels.
  m.def(
      "get_gemm_tf32",
      &natten::get_gemm_tf32,
      "Use tiled NA implementations when available.");
  m.def(
      "set_gemm_tf32",
      &natten::set_gemm_tf32,
      "Use tiled NA implementations when available.");

  // Bindings to test misc kernels
  m.def(
      "compute_delta", &natten::pytorch::compute_delta, "Compute delta");
}

} // namespace natten
