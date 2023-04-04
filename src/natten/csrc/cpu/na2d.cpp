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
    \brief Neighborhood Attention 2D - CPU interface
*/

#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include "natten_cpu_commons.h"
#include "pointwise_neighborhood_2d.cpp"
#include "neighborhood_neighborhood_2d.cpp"
#include "inverse_neighborhood_2d.cpp"
#include "rel_pos_bias_2d.cpp"

namespace natten {

torch::Tensor natten2dqkrpb_cpu_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const at::optional<at::Tensor> &rpb,
    const int kernel_size,
    const int dilation) {
    int batch_size = query.size(0);
    int heads = query.size(1);
    int height = query.size(2);
    int width = query.size(3);
    int dim = query.size(4);
    CHECK_FEATMAP(height, width, kernel_size, dilation);

    auto attn = torch::zeros(
            {batch_size, heads, height, width, kernel_size*kernel_size}, query.options());

    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "natten2dqkrpb_cpu_forward", ([&] {
        if (rpb.has_value()) {
            LAUNCH_DNA_KNS(kernel_size, dilation, pointwise_neighborhood_2d_bias, 
                    query.accessor<scalar_t, 5>(), key.accessor<scalar_t, 5>(), 
                    rpb.value().accessor<scalar_t, 3>(), attn.accessor<scalar_t, 5>(), 
                    height, width, heads, kernel_size, dilation, dim, batch_size);
        } else {
            LAUNCH_DNA_KNS(kernel_size, dilation, pointwise_neighborhood_2d, 
                    query.accessor<scalar_t, 5>(), key.accessor<scalar_t, 5>(), 
                    attn.accessor<scalar_t, 5>(), 
                    height, width, heads, kernel_size, dilation, dim, batch_size);
        }
    }));
    return attn;
}

std::vector<torch::Tensor> natten2dqkrpb_cpu_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled,
    const int kernel_size,
    const int dilation) {
    int batch_size = query.size(0);
    int heads = query.size(1);
    int height = query.size(2);
    int width = query.size(3);
    int dim = query.size(4);
    CHECK_FEATMAP(height, width, kernel_size, dilation);
    int RPB_MAX = kernel_size * 2 - 1;
   
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    at::Tensor d_rpb;
    if (biasEnabled)
        d_rpb = torch::zeros({heads, RPB_MAX, RPB_MAX}, d_attn.options());

    AT_DISPATCH_FLOATING_TYPES(d_query.scalar_type(), "natten2dqkrpb_backward_cpu", ([&] {
        const auto d_attn_a = d_attn.accessor<scalar_t, 5>();
        const auto query_a = query.accessor<scalar_t, 5>();
        const auto key_a = key.accessor<scalar_t, 5>();
        auto d_query_a = d_query.accessor<scalar_t, 5>();
        auto d_key_a = d_key.accessor<scalar_t, 5>();
        if (biasEnabled) {
            auto d_rpb_a = d_rpb.accessor<scalar_t, 3>();
            LAUNCH_DNA_KNS(kernel_size, dilation, rel_pos_bias_gradient_2d, 
                    d_rpb_a, d_attn_a, height, width, heads, kernel_size, dilation, batch_size);
        }
        LAUNCH_DNA_KNS(kernel_size, dilation, neighborhood_neighborhood_2d, 
                d_attn_a, key_a, d_query_a, height, width, heads, kernel_size, dilation, dim, batch_size);
        LAUNCH_DNA_KNS(kernel_size, dilation, inverse_neighborhood_2d, 
                d_attn_a, query_a, d_key_a, height, width, heads, kernel_size, dilation, dim, batch_size);
    }));
    return {d_query, d_key, d_rpb};
}

torch::Tensor natten2dav_cpu_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size,
    const int dilation) {
    AT_ASSERTM(attn.device().is_cpu(), "attn must be a CPU tensor");
    AT_ASSERTM(value.device().is_cpu(), "value must be a CPU tensor");
    int batch_size = value.size(0);
    int heads = value.size(1);
    int height = value.size(2);
    int width = value.size(3);
    int dim = value.size(4);
    CHECK_FEATMAP(height, width, kernel_size, dilation);

    auto out = torch::zeros_like(value);

    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "natten2dav_forward_cpu", ([&] {
        LAUNCH_DNA_KNS(kernel_size, dilation, neighborhood_neighborhood_2d, 
                attn.accessor<scalar_t, 5>(), value.accessor<scalar_t, 5>(), out.accessor<scalar_t, 5>(), 
                height, width, heads, kernel_size, dilation, dim, batch_size);
    }));
    return out;
}

std::vector<torch::Tensor> natten2dav_cpu_backward(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size,
    const int dilation) {
    AT_ASSERTM(d_out.device().is_cpu(), "d_out must be a CPU tensor");
    AT_ASSERTM(attn.device().is_cpu(), "attn must be a CPU tensor");
    AT_ASSERTM(value.device().is_cpu(), "value must be a CPU tensor");
    int batch_size = value.size(0);
    int heads = value.size(1);
    int height = value.size(2);
    int width = value.size(3);
    int dim = value.size(4);
    CHECK_FEATMAP(height, width, kernel_size, dilation);

    auto d_attn = torch::zeros_like(attn);
    auto d_value = torch::zeros_like(value);

    AT_DISPATCH_FLOATING_TYPES(d_attn.scalar_type(), "natten2dav_backward_cpu", ([&] {
        const auto d_out_a = d_out.accessor<scalar_t, 5>();
        const auto attn_a = attn.accessor<scalar_t, 5>();
        const auto value_a = value.accessor<scalar_t, 5>();
        auto d_attn_a = d_attn.accessor<scalar_t, 5>();
        auto d_value_a = d_value.accessor<scalar_t, 5>();
        LAUNCH_DNA_KNS(kernel_size, dilation, pointwise_neighborhood_2d, 
                d_out_a, value_a, d_attn_a, height, width, heads, kernel_size, dilation, dim, batch_size);
        LAUNCH_DNA_KNS(kernel_size, dilation, inverse_neighborhood_2d, 
                attn_a, d_out_a, d_value_a, height, width, heads, kernel_size, dilation, dim, batch_size);
    }));
    return {d_attn, d_value};
}

} // namespace natten
