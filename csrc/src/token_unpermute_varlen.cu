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
    \brief Varlen Token Unpermute
*/

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <natten/helpers.h>
#include <natten/natten.h>

#ifdef NATTEN_WITH_CUTLASS
#include <natten/cuda/tokperm/tokperm_varlen.hpp>

template <typename StdTuple>
auto std_tuple_to_cute_tuple(StdTuple a) {
  static_assert(
      std::tuple_size_v<StdTuple> > 0 && std::tuple_size_v<StdTuple> < 4);

  if constexpr (std::tuple_size_v<StdTuple> == 1) {
    return cute::make_tuple(static_cast<int>(std::get<0>(a)));
  } else if constexpr (std::tuple_size_v<StdTuple> == 2) {
    return cute::make_tuple(
        static_cast<int>(std::get<0>(a)), static_cast<int>(std::get<1>(a)));
  } else {
    return cute::make_tuple(
        static_cast<int>(std::get<0>(a)),
        static_cast<int>(std::get<1>(a)),
        static_cast<int>(std::get<2>(a)));
  }
}

#endif

namespace natten {

template <class StdNADim>
void token_unpermute_varlen_generic(
    at::Tensor& out,
    const at::Tensor& in,
    const at::Tensor& offsets_pre_permute,
    const at::Tensor& offsets_post_permute,
    const at::Tensor& token_layouts,
    int32_t seqlen_max,
    const StdNADim& tile_shape,
    const StdNADim& dilation,
    bool flip_tiled_dims) {
  static_assert(
      std::tuple_size_v<StdNADim> > 0 && std::tuple_size_v<StdNADim> < 4);
  static constexpr int kNADim = std::tuple_size_v<StdNADim>;

#ifdef NATTEN_WITH_CUTLASS
  CHECK_CONTIGUOUS(out);
  CHECK_CONTIGUOUS(in);
  CHECK_CUDA(out);
  CHECK_CUDA(in);

  CheckIfPropertiesMatch(in, out);

  at::cuda::OptionalCUDAGuard device_guard(in.device());

  TORCH_CHECK(
      in.dim() == 4,
      "Varlen Token Unpermute expects rank-4 inputs and outputs, got ",
      "input.dim()=",
      in.dim());

  TORCH_CHECK(
      out.dim() == 4,
      "Varlen Token Unpermute expects rank-4 inputs and outputs, got ",
      "output.dim()=",
      out.dim());

  TORCH_CHECK(
      in.size(0) == out.size(0) && in.size(0) == 1,
      "Varlen Token Unpermute input and output must have batch=1, got ",
      "input.shape[0]=",
      in.size(0),
      ", output.shape[0]=",
      out.size(0));

  TORCH_CHECK(
      out.size(2) == in.size(2),
      "Varlen Token Unpermute input and output heads must match, got ",
      "output.shape[-2]=",
      out.size(2),
      ", input.shape[-2]=",
      in.size(2));

  TORCH_CHECK(
      out.size(3) == in.size(3),
      "Varlen Token Unpermute input and output dims must match, got ",
      "output.shape[-1]=",
      out.size(3),
      ", input.shape[-1]=",
      in.size(3));

  TORCH_CHECK(
      offsets_pre_permute.dim() == 1 && offsets_post_permute.dim() == 1,
      "Varlen Token Permute offsets must be rank-1 tensors, got "
      "offsets_pre_permute.dim()=",
      offsets_pre_permute.dim(),
      ", offsets_post_permute.dim()=",
      offsets_post_permute.dim());

  TORCH_CHECK(
      offsets_pre_permute.size(0) > 1,
      "Varlen Token Permute pre-permute offsets must contain at least 2 elements, got "
      "offsets_pre_permute.shape[0]=",
      offsets_pre_permute.size(0));

  int batch = offsets_pre_permute.size(0) - 1;
  int seqlen = out.size(1);
  int heads = out.size(2);
  int dim = out.size(3);

  auto tile_shape_ = std_tuple_to_cute_tuple(tile_shape);
  auto dilation_ = std_tuple_to_cute_tuple(dilation);
  auto num_dilation_groups = cute::size(dilation_);

  TORCH_CHECK(
      offsets_post_permute.size(0) == num_dilation_groups * batch + 1,
      "Varlen Token Permute post-permute offsets must contain exactly ",
      "batch (=",
      batch,
      ") * math.prod(dilation) (=",
      num_dilation_groups,
      ") + 1"
      "elements, which is = ",
      num_dilation_groups * batch + 1,
      ", got ",
      "offsets_post_permute.shape[0]=",
      offsets_post_permute.size(0));

  int device_id = in.device().index();
  auto cuda_stream = at::cuda::getCurrentCUDAStream(device_id);

  cudaDeviceProp* device_props = at::cuda::getDeviceProperties(device_id);
  const int cc = device_props->major * 10 + device_props->minor;

  bool is_fp8_allowed = cc == 100 || cc == 103;

  TORCH_CHECK(
      in.scalar_type() == torch::kFloat16 ||
          in.scalar_type() == torch::kBFloat16 ||
          in.scalar_type() == torch::kFloat32 ||
          (is_fp8_allowed &&
           (in.scalar_type() == c10::ScalarType::Float8_e4m3fn ||
            in.scalar_type() == c10::ScalarType::Float8_e5m2)),
      "Varlen Token Unpermute only supports FP32, FP16, BF16, and FP8 operands (Blackwell DC-class only) for now.");

  TORCH_CHECK(
      offsets_pre_permute.scalar_type() == torch::kInt &&
          offsets_post_permute.scalar_type() == torch::kInt,
      "Varlen Token Permute only supports Int32 offsets.");

  TORCH_CHECK(
      token_layouts.scalar_type() == torch::kInt,
      "Varlen Token Permute expects token_layouts to be an int32 tensor.");

  TORCH_CHECK(
      token_layouts.dim() == 2,
      "Varlen Token Permute expects token_layouts to be a rank-2 tensor.");

  TORCH_CHECK(
      token_layouts.size(0) == batch,
      "Varlen Token Permute expects token_layouts.shape[0] == batch (=",
      batch,
      "), got ",
      token_layouts.size(0));

  TORCH_CHECK(
      token_layouts.size(1) == kNADim,
      "Varlen Token Permute ",
      kNADim,
      "-D expects token_layouts.shape[1] == ",
      kNADim,
      ", got ",
      token_layouts.size(1));

  using CuteTuple = decltype(tile_shape_);

  bool success = false;
  if (in.scalar_type() == torch::kFloat32) {
    success = natten::tokperm::token_unpermute_varlen_op(
        reinterpret_cast<float*>(in.data_ptr()),
        reinterpret_cast<float*>(out.data_ptr()),
        batch,
        seqlen_max,
        heads,
        dim,
        reinterpret_cast<int32_t*>(offsets_pre_permute.data_ptr()),
        reinterpret_cast<int32_t*>(offsets_post_permute.data_ptr()),
        reinterpret_cast<CuteTuple*>(token_layouts.data_ptr()),
        tile_shape_,
        dilation_,
        flip_tiled_dims,
        cuda_stream);
  } else if (in.scalar_type() == torch::kFloat16) {
    success = natten::tokperm::token_unpermute_varlen_op(
        reinterpret_cast<cutlass::half_t*>(in.data_ptr()),
        reinterpret_cast<cutlass::half_t*>(out.data_ptr()),
        batch,
        seqlen_max,
        heads,
        dim,
        reinterpret_cast<int32_t*>(offsets_pre_permute.data_ptr()),
        reinterpret_cast<int32_t*>(offsets_post_permute.data_ptr()),
        reinterpret_cast<CuteTuple*>(token_layouts.data_ptr()),
        tile_shape_,
        dilation_,
        flip_tiled_dims,
        cuda_stream);
  } else if (in.scalar_type() == torch::kBFloat16) {
    success = natten::tokperm::token_unpermute_varlen_op(
        reinterpret_cast<cutlass::bfloat16_t*>(in.data_ptr()),
        reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr()),
        batch,
        seqlen_max,
        heads,
        dim,
        reinterpret_cast<int32_t*>(offsets_pre_permute.data_ptr()),
        reinterpret_cast<int32_t*>(offsets_post_permute.data_ptr()),
        reinterpret_cast<CuteTuple*>(token_layouts.data_ptr()),
        tile_shape_,
        dilation_,
        flip_tiled_dims,
        cuda_stream);
  }
#if defined(NATTEN_WITH_BLACKWELL_FNA)
  else if (
      is_fp8_allowed && in.scalar_type() == c10::ScalarType::Float8_e4m3fn) {
    success = natten::tokperm::token_unpermute_varlen_op(
        reinterpret_cast<cutlass::float_e4m3_t*>(in.data_ptr()),
        reinterpret_cast<cutlass::float_e4m3_t*>(out.data_ptr()),
        batch,
        seqlen_max,
        heads,
        dim,
        reinterpret_cast<int32_t*>(offsets_pre_permute.data_ptr()),
        reinterpret_cast<int32_t*>(offsets_post_permute.data_ptr()),
        reinterpret_cast<CuteTuple*>(token_layouts.data_ptr()),
        tile_shape_,
        dilation_,
        flip_tiled_dims,
        cuda_stream);
  } else if (
      is_fp8_allowed && in.scalar_type() == c10::ScalarType::Float8_e5m2) {
    success = natten::tokperm::token_unpermute_varlen_op(
        reinterpret_cast<cutlass::float_e5m2_t*>(in.data_ptr()),
        reinterpret_cast<cutlass::float_e5m2_t*>(out.data_ptr()),
        batch,
        seqlen_max,
        heads,
        dim,
        reinterpret_cast<int32_t*>(offsets_pre_permute.data_ptr()),
        reinterpret_cast<int32_t*>(offsets_post_permute.data_ptr()),
        reinterpret_cast<CuteTuple*>(token_layouts.data_ptr()),
        tile_shape_,
        dilation_,
        flip_tiled_dims,
        cuda_stream);
  }
#endif

  TORCH_CHECK(success, "Token Unpermute kernel launch failed.");

#else
  TORCH_CHECK(false, "libnatten not compiled with CUTLASS.");
#endif
}

void token_unpermute_varlen_1d(
    at::Tensor& out,
    const at::Tensor& in,
    const at::Tensor& offsets_pre_permute,
    const at::Tensor& offsets_post_permute,
    const at::Tensor& token_layouts,
    int32_t seqlen_max,
    const std::tuple<int32_t>& tile_shape,
    const std::tuple<int32_t>& dilation,
    bool flip_tiled_dims) {
  token_unpermute_varlen_generic(
      out,
      in,
      offsets_pre_permute,
      offsets_post_permute,
      token_layouts,
      seqlen_max,
      tile_shape,
      dilation,
      flip_tiled_dims);
}

void token_unpermute_varlen_2d(
    at::Tensor& out,
    const at::Tensor& in,
    const at::Tensor& offsets_pre_permute,
    const at::Tensor& offsets_post_permute,
    const at::Tensor& token_layouts,
    int32_t seqlen_max,
    const std::tuple<int32_t, int32_t>& tile_shape,
    const std::tuple<int32_t, int32_t>& dilation,
    bool flip_tiled_dims) {
  token_unpermute_varlen_generic(
      out,
      in,
      offsets_pre_permute,
      offsets_post_permute,
      token_layouts,
      seqlen_max,
      tile_shape,
      dilation,
      flip_tiled_dims);
}

void token_unpermute_varlen_3d(
    at::Tensor& out,
    const at::Tensor& in,
    const at::Tensor& offsets_pre_permute,
    const at::Tensor& offsets_post_permute,
    const at::Tensor& token_layouts,
    int32_t seqlen_max,
    const std::tuple<int32_t, int32_t, int32_t>& tile_shape,
    const std::tuple<int32_t, int32_t, int32_t>& dilation,
    bool flip_tiled_dims) {
  token_unpermute_varlen_generic(
      out,
      in,
      offsets_pre_permute,
      offsets_post_permute,
      token_layouts,
      seqlen_max,
      tile_shape,
      dilation,
      flip_tiled_dims);
}

} // namespace natten
