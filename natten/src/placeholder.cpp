/*
NATTEN TORCH EXTENSION (CPU Placeholder)

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/
#include <torch/extension.h>
#include <vector>

namespace natten {

torch::Tensor natten1dqkrpb_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb,
    const int dilation) {
    TORCH_CHECK(false, "NATTEN does not have a CPU build yet. Please refer to our website for CUDA builds.");
}

std::vector<torch::Tensor> natten1dqkrpb_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const int dilation) {
    TORCH_CHECK(false, "NATTEN does not have a CPU build yet. Please refer to our website for CUDA builds.");
}

torch::Tensor natten1dav_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int dilation) {
    TORCH_CHECK(false, "NATTEN does not have a CPU build yet. Please refer to our website for CUDA builds.");
}

std::vector<torch::Tensor> natten1dav_backward(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int dilation) {
    TORCH_CHECK(false, "NATTEN does not have a CPU build yet. Please refer to our website for CUDA builds.");
}

torch::Tensor nattenqkrpb_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb,
    const int dilation) {
    TORCH_CHECK(false, "NATTEN does not have a CPU build yet. Please refer to our website for CUDA builds.");
}

std::vector<torch::Tensor> nattenqkrpb_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const int dilation) {
    TORCH_CHECK(false, "NATTEN does not have a CPU build yet. Please refer to our website for CUDA builds.");
}

torch::Tensor nattenav_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int dilation) {
    TORCH_CHECK(false, "NATTEN does not have a CPU build yet. Please refer to our website for CUDA builds.");
}

std::vector<torch::Tensor> nattenav_backward(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int dilation) {
    TORCH_CHECK(false, "NATTEN does not have a CPU build yet. Please refer to our website for CUDA builds.");
}


    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("natten1dqkrpb_forward", &natten1dqkrpb_forward, "NATTEN1DQK+RPB forward (CPU)");
      m.def("natten1dqkrpb_backward", &natten1dqkrpb_backward, "NATTEN1DQK+RPB backward (CPU)");
      m.def("natten1dav_forward", &natten1dav_forward, "NATTEN1DAV forward (CPU)");
      m.def("natten1dav_backward", &natten1dav_backward, "NATTEN1DAV backward (CPU)");

      m.def("natten2dqkrpb_forward", &nattenqkrpb_forward, "NATTENQK+RPB forward (CPU)");
      m.def("natten2dqkrpb_backward", &nattenqkrpb_backward, "NATTENQK+RPB backward (CPU)");
      m.def("natten2dav_forward", &nattenav_forward, "NATTENAV forward (CPU)");
      m.def("natten2dav_backward", &nattenav_backward, "NATTENAV backward (CPU)");
    }
} // namespace natten
