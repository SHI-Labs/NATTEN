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
    const bool biasEnabled,
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
    const bool biasEnabled,
    const int dilation);

std::vector<torch::Tensor> natten1dqkrpb_cuda_backward_fp16(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled,
    const int dilation);

#endif

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor natten1dqkrpb_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const at::optional<at::Tensor> &rpb_opt,
    const int kernel_size,
    const int dilation) {
    CHECK_CONTIGUOUS(query);
    CHECK_CONTIGUOUS(key);
    const int heads = query.size(1);
    bool biasEnabled = rpb_opt.has_value();
    auto rpb = biasEnabled ? rpb_opt.value() : torch::zeros({heads, 2 * kernel_size - 1}, query.options());
    assert(int((rpb.size(1) + 1) / 2) == kernel_size);
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
    const bool biasEnabled,
    const int kernel_size,
    const int dilation) {
    CHECK_CONTIGUOUS(d_attn);
    CHECK_CONTIGUOUS(query);
    CHECK_CONTIGUOUS(key);
    assert(query.device().is_cuda() == key.device().is_cuda() && d_attn.device().is_cuda() == key.device().is_cuda());
    if (query.device().is_cuda()) {
#if defined(WITH_CUDA)
        bool half = ::detail::scalar_type(query.scalar_type()) == at::ScalarType::Half;
        if (half)
            return natten1dqkrpb_cuda_backward_fp16(d_attn, query, key, biasEnabled, dilation);
        return natten1dqkrpb_cuda_backward(d_attn, query, key, biasEnabled, dilation);
#else
    AT_ERROR("NATTEN is not compiled with CUDA! Please make sure you installed correctly by referring to shi-labs.com/natten.");
#endif
    }
    return natten1dqkrpb_cpu_backward(d_attn, query, key, biasEnabled, dilation);
}
} // namespace natten
