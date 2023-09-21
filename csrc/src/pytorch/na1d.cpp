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
/*! \file
    \brief Neighborhood Attention 1D Torch interface
*/

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include "natten/pytorch/cpu/na1d.h"
#ifdef NATTEN_WITH_CUDA
#include "natten/pytorch/cuda/na1d.cuh"
#endif

#include "natten/pytorch/helpers.h"

namespace natten {
namespace pytorch {

at::Tensor na1d_qk_forward(
    const at::Tensor &query,
    const at::Tensor &key,
    const at::optional<at::Tensor> &bias,
    const int kernel_size,
    const int dilation) {
    TORCH_CHECK(kernel_size > 1 && kernel_size % 2 == 1, "Kernel size must be an odd number greater than 1.");
    TORCH_CHECK(dilation >= 1, "Dilation must be a nonnegative integer.");
    CHECK_CONTIGUOUS(query);
    CHECK_CONTIGUOUS(key);
    TORCH_CHECK(query.scalar_type() == key.scalar_type(), "Query and key tensors must match in dtype.");
    TORCH_CHECK(query.dim() == key.dim() && query.dim() == 4, "Expected query and key to be two 4-D tensors.");
    TORCH_CHECK(
            query.size(0) == key.size(0) && 
            query.size(1) == key.size(1) && 
            query.size(2) == key.size(2) && 
            query.size(3) == key.size(3), "Expected query and key to be of the same shape.");
    TORCH_CHECK(query.device().is_cuda() == key.device().is_cuda(), "Expected both query and key to be on the same device.");
    if (bias.has_value()) {
        TORCH_CHECK(query.scalar_type() == bias.value().scalar_type(), "Query, key, and bias tensors must match in dtype.");
        TORCH_CHECK(bias.value().device().is_cuda() == key.device().is_cuda(), 
                "Expected positional bias to be on the same device as the query and key tensors.");
        CHECK_CONTIGUOUS(bias.value());
        TORCH_CHECK(bias.value().size(0) == query.size(1), "Expected bias.shape[0] == query.shape[1] == heads.");
        TORCH_CHECK(int((bias.value().size(1) + 1) / 2) == kernel_size, "Invalid bias shape.");
    }
    int batch_size = query.size(0);
    int heads      = query.size(1);
    int length     = query.size(2);
    int dim        = query.size(3);
    TORCH_CHECK(kernel_size * dilation <= length, "Kernel size * dilation must be less than or equal to sequence length.");
    auto attn = torch::empty({batch_size, heads, length, kernel_size}, query.options());
    DISPATCH_DEVICE(query.device(), na1d_qk_forward,
            query,
            key,
            bias,
            attn,
            batch_size, heads, length, dim,
            kernel_size, dilation);
    return attn;
}


std::vector<at::Tensor> na1d_qk_backward(
    const at::Tensor &d_attn,
    const at::Tensor &query,
    const at::Tensor &key,
    const bool has_bias,
    const int kernel_size,
    const int dilation) {
    TORCH_CHECK(kernel_size > 1 && kernel_size % 2 == 1, "Kernel size must be an odd number greater than 1.");
    TORCH_CHECK(dilation >= 1, "Dilation must be a nonnegative integer.");
    CHECK_CONTIGUOUS(query);
    CHECK_CONTIGUOUS(key);
    CHECK_CONTIGUOUS(d_attn);
    TORCH_CHECK(query.scalar_type() == key.scalar_type(), "Query and key tensors must match in dtype.");
    TORCH_CHECK(query.dim() == key.dim() && query.dim() == d_attn.dim() && query.dim() == 4, 
            "Expected query, key, and d_attn to be 4-D tensors.");
    TORCH_CHECK(
            query.size(0) == key.size(0) && 
            query.size(1) == key.size(1) && 
            query.size(2) == key.size(2) && 
            query.size(3) == key.size(3), "Expected query and key to be of the same shape.");
    TORCH_CHECK(
            query.size(0) == d_attn.size(0) && 
            query.size(1) == d_attn.size(1) && 
            query.size(2) == d_attn.size(2) && 
            kernel_size == d_attn.size(3), "Wrong shape for d_attn.");
    TORCH_CHECK(query.device().is_cuda() == key.device().is_cuda() && query.device().is_cuda() == d_attn.device().is_cuda(), 
            "Expected query, key, and d_attn to be on the same device.");
    int batch_size = query.size(0);
    int heads      = query.size(1);
    int length     = query.size(2);
    int dim        = query.size(3);
    TORCH_CHECK(kernel_size * dilation <= length, "Kernel size * dilation must be less than or equal to sequence length.");
    auto d_query = torch::empty_like(query);
    auto d_key   = torch::empty_like(key);
    at::Tensor d_bias;
    if (has_bias) {
        auto rpb_dtype = query.scalar_type() == torch::kFloat64 ? query.scalar_type() : torch::kFloat32;
        auto options = query.options().dtype(rpb_dtype);
        d_bias = torch::zeros({heads, 2 * kernel_size - 1}, options);
    }
    DISPATCH_DEVICE(d_attn.device(), na1d_qk_backward,
            d_attn,
            query,
            key,
            d_query,
            d_key,
            d_bias,
            batch_size, heads, length, dim,
            kernel_size, dilation);
    return {d_query, d_key, d_bias};
}

at::Tensor na1d_av_forward(
    const at::Tensor &attn,
    const at::Tensor &value,
    const int kernel_size,
    const int dilation) {
    TORCH_CHECK(kernel_size > 1 && kernel_size % 2 == 1, "Kernel size must be an odd number greater than 1.");
    TORCH_CHECK(dilation >= 1, "Dilation must be a nonnegative integer.");
    CHECK_CONTIGUOUS(attn);
    CHECK_CONTIGUOUS(value);
    TORCH_CHECK(attn.scalar_type() == value.scalar_type(), "Attention and value tensors must match in dtype.");
    TORCH_CHECK(attn.dim() == value.dim() && attn.dim() == 4, "Expected attention and value to be two 4-D tensors.");
    TORCH_CHECK(
            attn.size(0) == value.size(0) && 
            attn.size(1) == value.size(1) && 
            attn.size(2) == value.size(2), "Expected attention and value to match in batch size, heads, and length.");
    TORCH_CHECK(kernel_size == attn.size(3), "Attention weights per token do not match kernel size.");
    TORCH_CHECK(attn.device().is_cuda() == value.device().is_cuda(), "Expected both attention and value to be on the same device.");
    int batch_size = value.size(0);
    int heads      = value.size(1);
    int length     = value.size(2);
    int dim        = value.size(3);
    TORCH_CHECK(kernel_size * dilation <= length, "Kernel size * dilation must be less than or equal to sequence length.");
    auto output = torch::empty_like(value);
    DISPATCH_DEVICE(attn.device(), na1d_av_forward,
            attn,
            value,
            output,
            batch_size, heads, length, dim,
            kernel_size, dilation);
    return output;
}

std::vector<at::Tensor> na1d_av_backward(
    const at::Tensor &d_out,
    const at::Tensor &attn,
    const at::Tensor &value,
    const int kernel_size,
    const int dilation) {
    TORCH_CHECK(kernel_size > 1 && kernel_size % 2 == 1, "Kernel size must be an odd number greater than 1.");
    TORCH_CHECK(dilation >= 1, "Dilation must be a nonnegative integer.");
    CHECK_CONTIGUOUS(attn);
    CHECK_CONTIGUOUS(value);
    CHECK_CONTIGUOUS(d_out);
    TORCH_CHECK(d_out.scalar_type() == value.scalar_type(), "d_out and value tensors must match in dtype.");
    TORCH_CHECK(d_out.dim() == value.dim() && d_out.dim() == attn.dim() && d_out.dim() == 4, 
            "Expected d_out, value, and attn to be 4-D tensors.");
    TORCH_CHECK(
            d_out.size(0) == value.size(0) && 
            d_out.size(1) == value.size(1) && 
            d_out.size(2) == value.size(2) && 
            d_out.size(3) == value.size(3), "Expected d_out and value to be of the same shape.");
    TORCH_CHECK(
            d_out.size(0) == attn.size(0) && 
            d_out.size(1) == attn.size(1) && 
            d_out.size(2) == attn.size(2) && 
            kernel_size == attn.size(3), "Wrong shape for attn.");
    TORCH_CHECK(d_out.device().is_cuda() == value.device().is_cuda() && d_out.device().is_cuda() == attn.device().is_cuda(), 
            "Expected d_out, value, and attn to be on the same device.");
    int batch_size = d_out.size(0);
    int heads      = d_out.size(1);
    int length     = d_out.size(2);
    int dim        = d_out.size(3);
    TORCH_CHECK(kernel_size * dilation <= length, "Kernel size * dilation must be less than or equal to sequence length.");
    auto d_attn  = torch::empty_like(attn);
    auto d_value = torch::empty_like(value);
    DISPATCH_DEVICE(attn.device(), na1d_av_backward,
            d_out,
            attn,
            value,
            d_attn,
            d_value,
            batch_size, heads, length, dim,
            kernel_size, dilation);
    return {d_attn, d_value};
}

} // namespace pytorch
} // namespace natten
