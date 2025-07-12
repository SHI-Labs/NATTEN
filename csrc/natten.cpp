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

#include <torch/extension.h>
#include <vector>

#include "natten/compute_delta.h"
#include "natten/reference.h"
#include "natten/fna.h"
#include "natten/blackwell_fna.h"
#include "natten/hopper_fna.h"
#include "natten/fmha.h"
#include "natten/blackwell_fmha.h"
#include "natten/hopper_fmha.h"

namespace natten {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // CUTLASS 3.X kernels
  //// SM100 - Blackwell FNA
  m.def(
      "blackwell_na1d_forward", &natten::blackwell_na1d_forward, "NA1D forward (fused, SM100)");

  m.def(
      "blackwell_na2d_forward", &natten::blackwell_na2d_forward, "NA2D forward (fused, SM100)");
                                                                 
  m.def(                                                         
      "blackwell_na3d_forward", &natten::blackwell_na3d_forward, "NA3D forward (fused, SM100)");

  m.def(
      "blackwell_na1d_backward", &natten::blackwell_na1d_backward, "NA1D backward (fused, SM100)");

  m.def(
      "blackwell_na2d_backward", &natten::blackwell_na2d_backward, "NA2D backward (fused, SM100)");
                                                                 
  m.def(                                                         
      "blackwell_na3d_backward", &natten::blackwell_na3d_backward, "NA3D backward (fused, SM100)");

  ////// SM100 - FMHA
  m.def(
      "blackwell_fmha_forward", &natten::blackwell_fmha_forward, "FMHA forward (fused, SM100)");
  m.def(
      "blackwell_fmha_backward", &natten::blackwell_fmha_backward, "FMHA backward (fused, SM100)");

  //// SM90 - Hopper FNA
  m.def(
      "hopper_na1d_forward", &natten::hopper_na1d_forward, "NA1D forward (fused, SM90)");

  m.def(
      "hopper_na2d_forward", &natten::hopper_na2d_forward, "NA2D forward (fused, SM90)");
                                                                 
  m.def(                                                         
      "hopper_na3d_forward", &natten::hopper_na3d_forward, "NA3D forward (fused, SM90)");

  m.def(
      "hopper_na1d_backward", &natten::hopper_na1d_backward, "NA1D backward (fused, SM90)");

  m.def(
      "hopper_na2d_backward", &natten::hopper_na2d_backward, "NA2D backward (fused, SM90)");
                                                                 
  m.def(                                                         
      "hopper_na3d_backward", &natten::hopper_na3d_backward, "NA3D backward (fused, SM90)");

  ////// SM90 - FMHA
  m.def(
      "hopper_fmha_forward", &natten::hopper_fmha_forward, "FMHA forward (fused, SM90)");

  m.def(
      "hopper_fmha_backward", &natten::hopper_fmha_backward, "FMHA backward (fused, SM90)");

  // CUTLASS 2.X kernels
  //// SM50/SM70/SM75/SM80 - Original FNA
  m.def(
      "na1d_forward", &natten::na1d_forward, "NA1D forward (fused)");

  m.def(
      "na2d_forward", &natten::na2d_forward, "NA2D forward (fused)");

  m.def(
      "na3d_forward", &natten::na3d_forward, "NA3D forward (fused)");

  m.def(
      "na1d_backward", &natten::na1d_backward, "NA1D backward (fused)");

  m.def(
      "na2d_backward", &natten::na2d_backward, "NA2D backward (fused)");

  m.def(
      "na3d_backward", &natten::na3d_backward, "NA3D backward (fused)");

  ////// SM50/SM70/SM75/SM80 - FMHA
  m.def(
      "fmha_forward", &natten::fmha_forward, "FMHA forward (fused)");

  m.def(
      "fmha_backward", &natten::fmha_backward, "FMHA backward (fused)");

  // Reference kernels
  m.def(
      "reference_na1d_forward", &natten::reference_na1d_forward, "Reference NA1D forward");

  m.def(
      "reference_na2d_forward", &natten::reference_na2d_forward, "Reference NA2D forward");

  m.def(
      "reference_na3d_forward", &natten::reference_na3d_forward, "Reference NA3D forward");

  m.def(
      "reference_na1d_backward", &natten::reference_na1d_backward, "Reference NA1D backward");

  m.def(
      "reference_na2d_backward", &natten::reference_na2d_backward, "Reference NA2D backward");

  m.def(
      "reference_na3d_backward", &natten::reference_na3d_backward, "Reference NA3D backward");

  // Misc kernels
  m.def(
      "compute_delta", &natten::compute_delta, "Compute delta");
}

} // namespace natten
