/***************************************************************************************************
 * Copyright (c) 2022-2025 Ali Hassani.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
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
    \brief Token Permute / Unpermute
*/

#pragma once
#include <ATen/ATen.h>

#include <natten/natten.h>

namespace natten {

void token_permute_1d(
    at::Tensor& out,
    const at::Tensor& in,
    const std::tuple<int32_t>& tile_shape,
    const std::tuple<int32_t>& dilation,
    bool flip_tiled_dims);

void token_permute_2d(
    at::Tensor& out,
    const at::Tensor& in,
    const std::tuple<int32_t, int32_t>& tile_shape,
    const std::tuple<int32_t, int32_t>& dilation,
    bool flip_tiled_dims);

void token_permute_3d(
    at::Tensor& out,
    const at::Tensor& in,
    const std::tuple<int32_t, int32_t, int32_t>& tile_shape,
    const std::tuple<int32_t, int32_t, int32_t>& dilation,
    bool flip_tiled_dims);

void token_unpermute_1d(
    at::Tensor& out,
    const at::Tensor& in,
    const std::tuple<int32_t>& tile_shape,
    const std::tuple<int32_t>& dilation,
    bool flip_tiled_dims);

void token_unpermute_2d(
    at::Tensor& out,
    const at::Tensor& in,
    const std::tuple<int32_t, int32_t>& tile_shape,
    const std::tuple<int32_t, int32_t>& dilation,
    bool flip_tiled_dims);

void token_unpermute_3d(
    at::Tensor& out,
    const at::Tensor& in,
    const std::tuple<int32_t, int32_t, int32_t>& tile_shape,
    const std::tuple<int32_t, int32_t, int32_t>& dilation,
    bool flip_tiled_dims);

void token_permute_varlen_1d(
    at::Tensor& out,
    const at::Tensor& in,
    const at::Tensor& offsets_original,
    const at::Tensor& offsets_tokperm,
    const at::Tensor& token_layouts,
    const at::optional<at::Tensor>&
        dilations, // per-batch dilations, if desired
    int32_t seqlen_max,
    const std::tuple<int32_t>& tile_shape,
    const std::tuple<int32_t>& dilation,
    bool flip_tiled_dims);

void token_permute_varlen_2d(
    at::Tensor& out,
    const at::Tensor& in,
    const at::Tensor& offsets_original,
    const at::Tensor& offsets_tokperm,
    const at::Tensor& token_layouts,
    const at::optional<at::Tensor>&
        dilations, // per-batch dilations, if desired
    int32_t seqlen_max,
    const std::tuple<int32_t, int32_t>& tile_shape,
    const std::tuple<int32_t, int32_t>& dilation,
    bool flip_tiled_dims);

void token_permute_varlen_3d(
    at::Tensor& out,
    const at::Tensor& in,
    const at::Tensor& offsets_original,
    const at::Tensor& offsets_tokperm,
    const at::Tensor& token_layouts,
    const at::optional<at::Tensor>&
        dilations, // per-batch dilations, if desired
    int32_t seqlen_max,
    const std::tuple<int32_t, int32_t, int32_t>& tile_shape,
    const std::tuple<int32_t, int32_t, int32_t>& dilation,
    bool flip_tiled_dims);

void token_unpermute_varlen_1d(
    at::Tensor& out,
    const at::Tensor& in,
    const at::Tensor& offsets_original,
    const at::Tensor& offsets_tokperm,
    const at::Tensor& token_layouts,
    const at::optional<at::Tensor>&
        dilations, // per-batch dilations, if desired
    int32_t seqlen_max,
    const std::tuple<int32_t>& tile_shape,
    const std::tuple<int32_t>& dilation,
    bool flip_tiled_dims);

void token_unpermute_varlen_2d(
    at::Tensor& out,
    const at::Tensor& in,
    const at::Tensor& offsets_original,
    const at::Tensor& offsets_tokperm,
    const at::Tensor& token_layouts,
    const at::optional<at::Tensor>&
        dilations, // per-batch dilations, if desired
    int32_t seqlen_max,
    const std::tuple<int32_t, int32_t>& tile_shape,
    const std::tuple<int32_t, int32_t>& dilation,
    bool flip_tiled_dims);

void token_unpermute_varlen_3d(
    at::Tensor& out,
    const at::Tensor& in,
    const at::Tensor& offsets_original,
    const at::Tensor& offsets_tokperm,
    const at::Tensor& token_layouts,
    const at::optional<at::Tensor>&
        dilations, // per-batch dilations, if desired
    int32_t seqlen_max,
    const std::tuple<int32_t, int32_t, int32_t>& tile_shape,
    const std::tuple<int32_t, int32_t, int32_t>& dilation,
    bool flip_tiled_dims);

} // namespace natten
