/*
NATTEN1D-QKRPB TORCH EXTENSION

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
*/
#include <torch/extension.h>
#include <vector>

namespace natten {

// CPU forward declarations
torch::Tensor natten1dqkrpb_cpu_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb,
    const int dilation);

// CPU backward declarations
std::vector<torch::Tensor> natten1dqkrpb_cpu_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const int dilation);

#if defined(WITH_CUDA)
// CUDA forward declarations
torch::Tensor natten1dqkrpb_cuda_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb,
    const int dilation);

torch::Tensor natten1dqkrpb_cuda_forward_fp16(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb,
    const int dilation);

// CUDA backward declarations
std::vector<torch::Tensor> natten1dqkrpb_cuda_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const int dilation);

std::vector<torch::Tensor> natten1dqkrpb_cuda_backward_fp16(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const int dilation);

#endif

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor natten1dqkrpb_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb,
    const int dilation) {
    CHECK_CONTIGUOUS(query);
    CHECK_CONTIGUOUS(key);
    CHECK_CONTIGUOUS(rpb);
    assert(query.device().is_cuda() == key.device().is_cuda() && rpb.device().is_cuda() == key.device().is_cuda());
    if (query.device().is_cuda()) {
#if defined(WITH_CUDA)
        bool half = ::detail::scalar_type(query.scalar_type()) == at::ScalarType::Half;
        if (half)
            return natten1dqkrpb_cuda_forward_fp16(query, key, rpb, dilation);
        return natten1dqkrpb_cuda_forward(query, key, rpb, dilation);
#else
    AT_ERROR("NATTEN is not compiled with CUDA! Please make sure you installed correctly by referring to shi-labs.com/natten.");
#endif
    }
    return natten1dqkrpb_cpu_forward(query, key, rpb, dilation);
}

std::vector<torch::Tensor> natten1dqkrpb_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const int dilation) {
    CHECK_CONTIGUOUS(d_attn);
    CHECK_CONTIGUOUS(query);
    CHECK_CONTIGUOUS(key);
    assert(query.device().is_cuda() == key.device().is_cuda() && d_attn.device().is_cuda() == key.device().is_cuda());
    if (query.device().is_cuda()) {
#if defined(WITH_CUDA)
        bool half = ::detail::scalar_type(query.scalar_type()) == at::ScalarType::Half;
        if (half)
            return natten1dqkrpb_cuda_backward_fp16(d_attn, query, key, dilation);
        return natten1dqkrpb_cuda_backward(d_attn, query, key, dilation);
#else
    AT_ERROR("NATTEN is not compiled with CUDA! Please make sure you installed correctly by referring to shi-labs.com/natten.");
#endif
    }
    return natten1dqkrpb_cpu_backward(d_attn, query, key, dilation);
}
} // namespace natten
