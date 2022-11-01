/*
NATTEN TORCH EXTENSION

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/
#include <torch/extension.h>
#include <vector>
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
