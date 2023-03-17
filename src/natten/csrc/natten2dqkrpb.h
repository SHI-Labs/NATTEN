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
torch::Tensor natten2dqkrpb_cpu_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb,
    const int dilation);

// CPU backward declarations
std::vector<torch::Tensor> natten2dqkrpb_cpu_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled,
    const int dilation);

#if defined(WITH_CUDA)
// CUDA forward declarations
torch::Tensor natten2dqkrpb_cuda_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb,
    const int dilation);

torch::Tensor natten2dqkrpb_cuda_forward_fp16(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb,
    const int dilation);

torch::Tensor natten2dqkrpb_cuda_forward_tiled_32(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb,
    const int dilation);

torch::Tensor natten2dqkrpb_cuda_forward_fp16_tiled_32(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &rpb,
    const int dilation);

// CUDA backward declarations
std::vector<torch::Tensor> natten2dqkrpb_cuda_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled,
    const int dilation);

std::vector<torch::Tensor> natten2dqkrpb_cuda_backward_fp16(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled,
    const int dilation);

#endif

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

torch::Tensor natten2dqkrpb_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const at::optional<at::Tensor> &rpb_opt,
    const int kernel_size,
    const int dilation) {
    CHECK_CONTIGUOUS(query);
    CHECK_CONTIGUOUS(key);
    const int heads = query.size(1);
    auto rpb = rpb_opt.has_value() ? rpb_opt.value() : torch::zeros({heads, 2 * kernel_size - 1, 2 * kernel_size - 1}, query.options());
    int exp_kernel_size = (rpb.size(1) + 1) / 2;
    assert(exp_kernel_size == kernel_size);
    CHECK_CONTIGUOUS(rpb);
    assert(query.device().is_cuda() == key.device().is_cuda() && rpb.device().is_cuda() == key.device().is_cuda());
    if (query.device().is_cuda()) {
#if defined(WITH_CUDA)
        int dim = query.size(4);
        bool half = ::detail::scalar_type(query.scalar_type()) == at::ScalarType::Half;
        if ((
            kernel_size == 7 || kernel_size == 3 || kernel_size == 5 ||
            kernel_size == 9 || kernel_size == 11 || kernel_size == 13
            ) && dim == 32){
            if (half)
                return natten2dqkrpb_cuda_forward_fp16_tiled_32(query, key, rpb, dilation);
            return natten2dqkrpb_cuda_forward_tiled_32(query, key, rpb, dilation);
        }
        if (half)
            return natten2dqkrpb_cuda_forward_fp16(query, key, rpb, dilation);
        return natten2dqkrpb_cuda_forward(query, key, rpb, dilation);
#else
    AT_ERROR("NATTEN is not compiled with CUDA! Please make sure you installed correctly by referring to shi-labs.com/natten.");
#endif
    }
    return natten2dqkrpb_cpu_forward(query, key, rpb, dilation);
}

std::vector<torch::Tensor> natten2dqkrpb_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled,
    const int kernel_size,
    const int dilation) {
    CHECK_CONTIGUOUS(d_attn);
    CHECK_CONTIGUOUS(query);
    CHECK_CONTIGUOUS(key);
    assert(int(sqrt(d_attn.size(4))) == kernel_size);
    assert(query.device().is_cuda() == key.device().is_cuda() && d_attn.device().is_cuda() == key.device().is_cuda());
    if (query.device().is_cuda()) {
#if defined(WITH_CUDA)
        bool half = ::detail::scalar_type(query.scalar_type()) == at::ScalarType::Half;
        if (half)
            return natten2dqkrpb_cuda_backward_fp16(d_attn, query, key, biasEnabled, dilation);
        return natten2dqkrpb_cuda_backward(d_attn, query, key, biasEnabled, dilation);
#else
    AT_ERROR("NATTEN is not compiled with CUDA! Please make sure you installed correctly by referring to shi-labs.com/natten.");
#endif
    }
    return natten2dqkrpb_cpu_backward(d_attn, query, key, biasEnabled, dilation);
}
} // namespace natten
