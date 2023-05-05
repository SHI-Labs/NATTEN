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
    \brief Neighborhood Attention 3D - CPU interface
*/

#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>

#include "natten_cpu_commons.h"
#include "pointwise_neighborhood_3d.cpp"
#include "neighborhood_neighborhood_3d.cpp"
#include "inverse_neighborhood_3d.cpp"
#include "rel_pos_bias_3d.cpp"

namespace natten {

torch::Tensor natten3dqkrpb_cpu_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const at::optional<at::Tensor> &rpb,
    const int kernel_size_d,
    const int kernel_size,
    const int dilation_d,
    const int dilation) {
    int batch_size = query.size(0);
    int heads = query.size(1);
    int depth = query.size(2);
    int height = query.size(3);
    int width = query.size(4);
    int dim = query.size(5);
    CHECK_3DFEATMAP(depth, height, width, kernel_size, kernel_size_d, dilation, dilation_d);

    auto attn = torch::zeros(
            {batch_size, heads, depth, height, width, kernel_size_d*kernel_size*kernel_size}, query.options());

    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "natten3dqkrpb_cpu_forward", ([&] {
        if (rpb.has_value()) {
            LAUNCH_NA_KDNDS(kernel_size, kernel_size_d,   pointwise_neighborhood_3d_bias, 
                    query.accessor<scalar_t, 6>(), key.accessor<scalar_t, 6>(), 
                    rpb.value().accessor<scalar_t, 4>(), attn.accessor<scalar_t, 6>(), 
                    depth, height, width, heads, kernel_size, kernel_size_d, dilation, dilation_d, dim, batch_size);
        } else {
            LAUNCH_NA_KDNDS(kernel_size, kernel_size_d,   pointwise_neighborhood_3d, 
                    query.accessor<scalar_t, 6>(), key.accessor<scalar_t, 6>(), 
                    attn.accessor<scalar_t, 6>(), 
                    depth, height, width, heads, kernel_size, kernel_size_d, dilation, dilation_d, dim, batch_size);
        }
    }));
    return attn;
}

std::vector<torch::Tensor> natten3dqkrpb_cpu_backward(
    const torch::Tensor &d_attn,
    const torch::Tensor &query,
    const torch::Tensor &key,
    const bool biasEnabled,
    const int kernel_size_d,
    const int kernel_size,
    const int dilation_d,
    const int dilation) {
    int batch_size = query.size(0);
    int heads = query.size(1);
    int depth = query.size(2);
    int height = query.size(3);
    int width = query.size(4);
    int dim = query.size(5);
    CHECK_3DFEATMAP(depth, height, width, kernel_size, kernel_size_d, dilation, dilation_d);
    int RPB_MAX = kernel_size * 2 - 1;
   
    auto d_query = torch::zeros_like(query);
    auto d_key = torch::zeros_like(key);
    at::Tensor d_rpb;
    if (biasEnabled)
        d_rpb = torch::zeros({heads, RPB_MAX, RPB_MAX, RPB_MAX}, d_attn.options());

    AT_DISPATCH_FLOATING_TYPES(d_query.scalar_type(), "natten3dqkrpb_backward_cpu", ([&] {
        const auto d_attn_a = d_attn.accessor<scalar_t, 6>();
        const auto query_a = query.accessor<scalar_t, 6>();
        const auto key_a = key.accessor<scalar_t, 6>();
        auto d_query_a = d_query.accessor<scalar_t, 6>();
        auto d_key_a = d_key.accessor<scalar_t, 6>();
        if (biasEnabled) {
            auto d_rpb_a = d_rpb.accessor<scalar_t, 4>();
            LAUNCH_NA_KDNDS(kernel_size, kernel_size_d,   rel_pos_bias_gradient_3d, 
                    d_rpb_a, d_attn_a, depth, height, width, heads, kernel_size, kernel_size_d, dilation, dilation_d, batch_size);
        }
        LAUNCH_NA_KDNDS(kernel_size, kernel_size_d,   neighborhood_neighborhood_3d, 
                d_attn_a, key_a, d_query_a, depth, height, width, heads, kernel_size, kernel_size_d, dilation, dilation_d, dim, batch_size);
        LAUNCH_NA_KDNDS(kernel_size, kernel_size_d,   inverse_neighborhood_3d, 
                d_attn_a, query_a, d_key_a, depth, height, width, heads, kernel_size, kernel_size_d, dilation, dilation_d, dim, batch_size);
    }));
    return {d_query, d_key, d_rpb};
}

torch::Tensor natten3dav_cpu_forward(
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size_d,
    const int kernel_size,
    const int dilation_d,
    const int dilation) {
    AT_ASSERTM(attn.device().is_cpu(), "attn must be a CPU tensor");
    AT_ASSERTM(value.device().is_cpu(), "value must be a CPU tensor");
    int batch_size = value.size(0);
    int heads = value.size(1);
    int depth = value.size(2);
    int height = value.size(3);
    int width = value.size(4);
    int dim = value.size(5);
    CHECK_3DFEATMAP(depth, height, width, kernel_size, kernel_size_d, dilation, dilation_d);

    auto out = torch::zeros_like(value);

    AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), "natten3dav_forward_cpu", ([&] {
        LAUNCH_NA_KDNDS(kernel_size, kernel_size_d,   neighborhood_neighborhood_3d, 
                attn.accessor<scalar_t, 6>(), value.accessor<scalar_t, 6>(), out.accessor<scalar_t, 6>(), 
                depth, height, width, heads, kernel_size, kernel_size_d, dilation, dilation_d, dim, batch_size);
    }));
    return out;
}

std::vector<torch::Tensor> natten3dav_cpu_backward(
    const torch::Tensor &d_out,
    const torch::Tensor &attn,
    const torch::Tensor &value,
    const int kernel_size_d,
    const int kernel_size,
    const int dilation_d,
    const int dilation) {
    AT_ASSERTM(d_out.device().is_cpu(), "d_out must be a CPU tensor");
    AT_ASSERTM(attn.device().is_cpu(), "attn must be a CPU tensor");
    AT_ASSERTM(value.device().is_cpu(), "value must be a CPU tensor");
    int batch_size = value.size(0);
    int heads = value.size(1);
    int depth = value.size(2);
    int height = value.size(3);
    int width = value.size(4);
    int dim = value.size(5);
    CHECK_3DFEATMAP(depth, height, width, kernel_size, kernel_size_d, dilation, dilation_d);

    auto d_attn = torch::zeros_like(attn);
    auto d_value = torch::zeros_like(value);

    AT_DISPATCH_FLOATING_TYPES(d_attn.scalar_type(), "natten3dav_backward_cpu", ([&] {
        const auto d_out_a = d_out.accessor<scalar_t, 6>();
        const auto attn_a = attn.accessor<scalar_t, 6>();
        const auto value_a = value.accessor<scalar_t, 6>();
        auto d_attn_a = d_attn.accessor<scalar_t, 6>();
        auto d_value_a = d_value.accessor<scalar_t, 6>();
        LAUNCH_NA_KDNDS(kernel_size, kernel_size_d,   pointwise_neighborhood_3d, 
                d_out_a, value_a, d_attn_a, depth, height, width, heads, kernel_size, kernel_size_d, dilation, dilation_d, dim, batch_size);
        LAUNCH_NA_KDNDS(kernel_size, kernel_size_d,   inverse_neighborhood_3d, 
                attn_a, d_out_a, d_value_a, depth, height, width, heads, kernel_size, kernel_size_d, dilation, dilation_d, dim, batch_size);
    }));
    return {d_attn, d_value};
}

} // namespace natten
