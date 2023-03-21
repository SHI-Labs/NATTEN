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

#include <torch/extension.h>
#include <vector>

#include "context.h"
#include "natten1dav.h"
#include "natten1dqkrpb.h"
#include "natten2dav.h"
#include "natten2dqkrpb.h"

namespace natten {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("natten1dqkrpb_forward", &natten1dqkrpb_forward, "NATTEN1DQK+RPB forward");
  m.def("natten1dqkrpb_backward", &natten1dqkrpb_backward, "NATTEN1DQK+RPB backward");
  m.def("natten1dav_forward", &natten1dav_forward, "NATTEN1DAV forward");
  m.def("natten1dav_backward", &natten1dav_backward, "NATTEN1DAV backward");

  m.def("natten2dqkrpb_forward", &natten2dqkrpb_forward, "NATTEN2DQK+RPB forward");
  m.def("natten2dqkrpb_backward", &natten2dqkrpb_backward, "NATTEN2DQK+RPB backward");
  m.def("natten2dav_forward", &natten2dav_forward, "NATTEN2DAV forward");
  m.def("natten2dav_backward", &natten2dav_backward, "NATTEN2DAV backward");
}

} // namespace natten
