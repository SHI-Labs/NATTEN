/*
NATTEN TORCH EXTENSION (CUDA)

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/
#include <torch/extension.h>
#include <vector>
#include "cu/natten1dav.h"
#include "cu/natten1dqkrpb.h"
#include "cu/nattenav.h"
#include "cu/nattenqkrpb.h"

namespace natten {
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("natten1dqkrpb_forward", &natten1dqkrpb_forward, "NATTEN1DQK+RPB forward (CUDA)");
      m.def("natten1dqkrpb_backward", &natten1dqkrpb_backward, "NATTEN1DQK+RPB backward (CUDA)");
      m.def("natten1dav_forward", &natten1dav_forward, "NATTEN1DAV forward (CUDA)");
      m.def("natten1dav_backward", &natten1dav_backward, "NATTEN1DAV backward (CUDA)");

      m.def("natten2dqkrpb_forward", &nattenqkrpb_forward, "NATTENQK+RPB forward (CUDA)");
      m.def("natten2dqkrpb_backward", &nattenqkrpb_backward, "NATTENQK+RPB backward (CUDA)");
      m.def("natten2dav_forward", &nattenav_forward, "NATTENAV forward (CUDA)");
      m.def("natten2dav_backward", &nattenav_backward, "NATTENAV backward (CUDA)");
    }
} // namespace natten
