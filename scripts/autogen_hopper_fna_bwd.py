# Copyright (c) 2022 - 2026 Ali Hassani.
#
# This script is intended to emit fused kernel instantiations into
# a variable number of source files, generate appropriate headers
# and a single dispatcher interface, which will be used by the
# NATTEN API to call the kernels.
#
# NOTE: these scripts are heavily under-documented, and
# overly-repetitive, and will be replaced in future PRs.
# Please use it with caution.

import argparse
import math
import os
from typing import List, Tuple


DEFAULT_OUTPUT_DIR = "csrc/"

SUPPORTED_CONFIGS_BACKWARD = {
    16: {
        32: [
            (64, 128, 32),
            (128, 128, 32),
        ],
        64: [
            (64, 128, 64),
            (128, 128, 64),
        ],
        128: [
            (64, 128, 128),
        ],
    },
}


KERNEL_DECL_TEMPLATE = """
void {kernel_name}(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      void* ptr_dQ,
      void* ptr_dK,
      void* ptr_dV,
      void* ptr_dO,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads,
      int dim,
      {DimType} q_shape,
      {DimType} kv_shape,
      {DimType} qkv_shape,
      {DimType} window_size,
      {DimType} stride,
      {DimType} dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options);
"""


KERNEL_IMPL_TEMPLATE = """
void {kernel_name}(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      void* ptr_dQ,
      void* ptr_dK,
      void* ptr_dV,
      void* ptr_dO,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads,
      int dim,
      {DimType} q_shape,
      {DimType} kv_shape,
      {DimType} qkv_shape,
      {DimType} window_size,
      {DimType} stride,
      {DimType} dilation,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {{

  using Causal = {Causal};
  using QTileShape = {QTileShape};
  using KVTileShape = {KVTileShape};
  using GemmShape = {GEMMShape};
  using Kernel = natten::cuda::fna_hopper::KernelBackward<
    {dtype}, Causal, QTileShape, KVTileShape, GemmShape>;

  Kernel kernel;
  auto args = kernel.initialize(
      ptr_Q,
      ptr_K,
      ptr_V,
      ptr_O,
      ptr_LSE,
      ptr_dQ,
      ptr_dK,
      ptr_dV,
      ptr_dO,
      batch_size,
      seqlen_q,
      seqlen_k,
      heads,
      dim,
      q_shape,
      kv_shape,
      qkv_shape,
      window_size,
      stride,
      dilation,
      device_id,
      attn_scale);

  auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
  auto workspace = at::empty({{bytes}}, tensor_options.dtype(at::ScalarType::Byte));
  auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
  kernel.run(args, workspace_ptr, stream);
}}
"""


class DataType:
    def __init__(self, name, short_name, torch_name, bits):
        self.name = name
        self.bits = bits
        self.short_name = short_name
        self.torch_name = torch_name


Half = DataType("cutlass::half_t", "float16", "torch::kFloat16", 16)
BFloat = DataType("cutlass::bfloat16_t", "bfloat16", "torch::kBFloat16", 16)


def iterable_to_static_cute_tuple(shape_in) -> str:
    shape = ", ".join([f"cute::Int<{x}>" for x in shape_in])
    return f"cute::tuple<{shape}>"


def get_dim_type(na_dim: int) -> str:
    shape = ", ".join(["int" for _ in range(na_dim)])
    return f"cute::tuple<{shape}>"


class HopperFnaInstance:
    def __init__(
        self,
        na_dim: int,
        dtype: DataType,
        gemm_shape: Tuple[int, int, int],
        q_tile_shape: tuple,
        kv_tile_shape: tuple,
        causal: tuple,
    ):
        assert 0 < na_dim <= 3
        assert len(gemm_shape) == 3
        assert na_dim == len(q_tile_shape)
        assert na_dim == len(kv_tile_shape)
        assert na_dim == len(causal)
        self.na_dim = na_dim
        self.gemm_shape = gemm_shape
        self.q_tile_shape = q_tile_shape
        self.kv_tile_shape = kv_tile_shape
        self.causal = causal
        self.dtype = dtype
        self.max_head_dim = gemm_shape[2]

    def get_q_tile_shape_cute(self) -> str:
        return iterable_to_static_cute_tuple(self.q_tile_shape)

    def get_kv_tile_shape_cute(self) -> str:
        return iterable_to_static_cute_tuple(self.kv_tile_shape)

    def get_causal_cute(self) -> str:
        consts = ", ".join(
            ["cute::true_type" if c else "cute::false_type" for c in self.causal]
        )
        return f"cute::tuple<{consts}>"

    def get_gemm_shape_cute(self) -> str:
        return iterable_to_static_cute_tuple(self.gemm_shape)

    def get_name(self) -> str:
        name = f"hopper_fna{self.na_dim}d_backward"
        name += f"_{self.dtype.short_name}"
        name += "_" + "x".join([str(x) for x in self.gemm_shape])
        name += "_Q" + "x".join([str(x) for x in self.q_tile_shape])
        name += "_KV" + "x".join([str(x) for x in self.kv_tile_shape])
        name += "_causal" + "x".join(["1" if c else "0" for c in self.causal])
        return name

    def get_decl(self) -> str:
        return KERNEL_DECL_TEMPLATE.format(
            kernel_name=self.get_name(), DimType=get_dim_type(self.na_dim)
        )

    def get_impl(self) -> str:
        return KERNEL_IMPL_TEMPLATE.format(
            kernel_name=self.get_name(),
            DimType=get_dim_type(self.na_dim),
            Causal=self.get_causal_cute(),
            QTileShape=self.get_q_tile_shape_cute(),
            KVTileShape=self.get_kv_tile_shape_cute(),
            GEMMShape=self.get_gemm_shape_cute(),
            dtype=self.dtype.name,
        )


def write_combined_source_file(path, filename, headers, kernels):
    source_head = []
    source_head += ["#ifdef NATTEN_WITH_CUTLASS\n"]
    source_head += ["#ifdef NATTEN_WITH_HOPPER_FNA\n"]

    source_head += ["#include <cuda_runtime.h>\n"]
    source_head += ["#include <iostream>\n"]

    source_head += ["#include <ATen/ATen.h>\n"]
    source_head += ["#include <ATen/cuda/CUDAContext.h>\n"]
    source_head += ["#include <c10/cuda/CUDAGuard.h>\n"]
    source_head += ["#include <c10/cuda/CUDAStream.h>\n"]
    source_head += ["#include <torch/extension.h>\n"]

    source_head += ["#include <natten/natten.h>\n"]
    source_head += ["#include <natten/helpers.h>\n"]

    source_head += ["#include <natten/cuda/fna_hopper/fna_backward.cuh>\n"]

    for header in headers:
        source_head += [f"#include <{header}>\n"]

    source_head += ["namespace natten { \n"]
    source_head += ["namespace cuda { \n"]
    source_head += ["namespace fna_hopper { \n\n"]

    source_head = "".join(source_head)

    source_body = []
    for kernel in kernels:
        source_body += "\n\n" + kernel.get_impl() + "\n\n"
    source_body = "".join(source_body)

    source_foot = "".join(
        [
            "} // namespace fna_hopper \n",
            "} // namespace cuda \n",
            "} // namespace natten \n",
            "#endif \n",
            "#endif \n",
            "\n",
        ]
    )
    filename = f"{path}/{filename}"
    with open(filename, "w") as f:
        f.write(source_head)
        f.write(source_body)
        f.write(source_foot)


class NaDimDispatcher:
    def __init__(
        self,
    ):
        self.name = "DISPATCH_HOPPER_FNA_BACKWARD"
        self.dims = []

    def append(self, na_dim: int):
        self.dims.append(na_dim)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name}(rank, dtype, dim, is_causal, q_tile_shape, kv_tile_shape, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, na_dim in enumerate(self.dims):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if constexpr (rank == {na_dim})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name}_{na_dim}D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "Hopper FNA kernel launch failed!" \\\n'
        dispatcher_str += (
            '                << "' + "NATTEN only supports NA1D, 2D, and 3D!" + '" \\\n'
        )
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();"
        dispatcher_str += "\n\n"
        return dispatcher_str


class DTypeDispatcher:
    def __init__(self, na_dim: int):
        self.dtypes: List[DataType] = []
        self.na_dim = na_dim
        self.name = f"DISPATCH_HOPPER_FNA_BACKWARD_{self.na_dim}D"

    def append(self, dtype: DataType):
        self.dtypes.append(dtype)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name}(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, dtype in enumerate(self.dtypes):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (dtype == {dtype.torch_name})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name}_{dtype.short_name}(dim, is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "Hopper FNA kernel launch failed!" \\\n'
        dispatcher_str += (
            '                << "'
            + f"Hopper FNA-{self.na_dim}D does not support this data type."
            + '" \\\n'
        )
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();"
        dispatcher_str += "\n\n"
        return dispatcher_str


class HeadDimDispatcher:
    def __init__(self, na_dim: int, dtype: DataType):
        self.na_dim = na_dim
        self.dtype = dtype
        self.name = (
            f"DISPATCH_HOPPER_FNA_BACKWARD_{self.na_dim}D_{self.dtype.short_name}"
        )

        self.dims: List[int] = []

    def append(self, dim: int):
        self.dims.append(dim)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name}(dim, is_causal, q_tile_shape, kv_tile_shape, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, dim in enumerate(self.dims):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (dim == {dim})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name}_headdim{dim}(is_causal, q_tile_shape, kv_tile_shape, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "Hopper FNA kernel launch failed!" \\\n'
        dispatcher_str += (
            '                << "'
            + f"Hopper FNA-{self.na_dim}D does not support this data type."
            + '" \\\n'
        )
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();"
        dispatcher_str += "\n\n"
        return dispatcher_str


class CausalMaskDispatcher:
    def __init__(
        self,
        na_dim: int,
        dtype: DataType,
        head_dim: int,
    ):
        self.na_dim = na_dim
        self.dtype = dtype
        self.head_dim = head_dim

        self.name = f"DISPATCH_HOPPER_FNA_BACKWARD_{self.na_dim}D_{self.dtype.short_name}_headdim{head_dim}"

        self.cms: List = []

    def append(self, cm):
        assert len(cm) == self.na_dim, f"{cm} incompatible for {self.na_dim}D NA."
        self.cms.append(cm)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += (
            f"#define {self.name}(is_causal, q_tile_shape, kv_tile_shape, ...) \\\n"
        )
        dispatcher_str += "  [&] { \\\n"
        i = 0
        for cm in self.cms:
            cm_str = "causal" + "x".join(["1" if c else "0" for c in cm])
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            i += 1
            dispatcher_str += "if ("
            for dim in range(self.na_dim):
                dispatcher_str += (
                    f"cute::get<{dim}>(is_causal)"
                    if cm[dim]
                    else f"not cute::get<{dim}>(is_causal)"
                )
                if dim != self.na_dim - 1:
                    dispatcher_str += " && "
            dispatcher_str += ")"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name}_{cm_str}(q_tile_shape, kv_tile_shape, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += "    "
        dispatcher_str += '      std::cerr << "Hopper FNA kernel launch failed!" \\\n'
        dispatcher_str += (
            '                << "'
            + "Causal mask dispatcher got invalid causal mask!"
            + '" \\\n'
        )
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();"
        dispatcher_str += "\n\n"
        return dispatcher_str


class ConfigDispatcher:
    def __init__(
        self,
        na_dim: int,
        dtype: DataType,
        head_dim: int,
        causal_mask: tuple,
    ):
        self.na_dim = na_dim
        self.dtype = dtype
        self.head_dim = head_dim
        self.causal_mask = causal_mask
        cm_str = "_causal" + "x".join(["1" if c else "0" for c in self.causal_mask])

        self.name = f"DISPATCH_HOPPER_FNA_BACKWARD_{self.na_dim}D_{self.dtype.short_name}_headdim{head_dim}"
        self.name += cm_str

        self.configs: List = []

    def append(self, config):
        assert len(config) == 2
        assert isinstance(config[0], tuple) and len(config[0]) == self.na_dim
        assert isinstance(config[1], tuple) and len(config[1]) == self.na_dim
        self.configs.append(config)

    def get_kernel_instance(self, q_tile_shape, kv_tile_shape):
        gemm_M = math.prod(q_tile_shape)
        gemm_N = math.prod(kv_tile_shape)
        gemm_K = self.head_dim
        gemm_shape = (gemm_M, gemm_N, gemm_K)

        supported_configs = SUPPORTED_CONFIGS_BACKWARD[self.dtype.bits][self.head_dim]
        assert (
            gemm_shape in supported_configs
        ), f"{gemm_shape=} not in supported configs {supported_configs=}"

        kernel = HopperFnaInstance(
            na_dim=self.na_dim,
            dtype=self.dtype,
            gemm_shape=gemm_shape,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            causal=self.causal_mask,
        )

        return kernel

    def get_target_name(self, q_tile_shape, kv_tile_shape):
        kernel = self.get_kernel_instance(q_tile_shape, kv_tile_shape)
        return kernel.get_name()

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name}(q_tile_shape, kv_tile_shape, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        i = 0
        for q_tile_shape, kv_tile_shape in self.configs:
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            i += 1
            dispatcher_str += "if ("
            for dim in range(self.na_dim):
                dispatcher_str += (
                    f"cute::get<{dim}>(q_tile_shape) == {q_tile_shape[dim]}"
                )
                dispatcher_str += " && "
            dispatcher_str += "\\\n"
            for dim in range(self.na_dim):
                dispatcher_str += (
                    f"cute::get<{dim}>(kv_tile_shape) == {kv_tile_shape[dim]}"
                )
                if dim < self.na_dim - 1:
                    dispatcher_str += " && "
            dispatcher_str += ")"
            dispatcher_str += " { \\\n"

            dispatcher_str += f"  natten::cuda::fna_hopper::{self.get_target_name(q_tile_shape, kv_tile_shape)}(__VA_ARGS__); \\\n"

            dispatcher_str += "} \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += "    "
        dispatcher_str += '      std::cerr << "Hopper FNA kernel launch failed!" \\\n'
        dispatcher_str += (
            '                << "'
            + "Hopper FNA Backward got invalid Q tile and KV tile combination."
            + '" \\\n'
        )
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();"
        dispatcher_str += "\n\n"
        return dispatcher_str


def write_header_file(content, path, namespaces, extra_includes=None):
    extra_includes = extra_includes or []
    header_head = [
        "#pragma once\n",
        "\n\n",
    ]
    header_head += ["#include <iostream> \n"]
    header_head += ["#include <type_traits> \n"]
    header_head += ["#ifdef NATTEN_WITH_CUTLASS\n"]
    header_head += ["#ifdef NATTEN_WITH_HOPPER_FNA\n"]
    for incl in extra_includes:
        header_head += [f"#include <{incl}> \n"]

    for namespace in namespaces:
        header_head += [f"namespace {namespace}", " { \n"]

    header_foot = [
        "\n\n",
    ]
    for namespace in namespaces:
        header_foot += ["} ", f"// namespace {namespace}", " \n"]
    header_foot += [
        "#endif \n",
        "#endif \n",
        "\n",
    ]
    with open(path, "w") as f:
        f.write("".join(header_head))
        f.write(content)
        f.write("".join(header_foot))


def generate_hopper_fna_kernels(path, num_splits=2):

    NA_DIMS = [1, 2, 3]

    SUPPORTED_DTYPES = [
        Half,
        BFloat,
    ]

    HEAD_DIMS = [32, 64, 128]

    CAUSAL_MASKS = {
        1: [(False,), (True,)],
        2: [(False, False), (False, True), (True, False), (True, True)],
        3: [
            (False, False, False),
            (False, False, True),
            (False, True, False),
            (False, True, True),
            (True, False, False),
            (True, False, True),
            (True, True, False),
            (True, True, True),
        ],
    }

    CONFIGS = {
        1: {
            16: {
                32: [
                    ((64,), (128,)),
                    ((128,), (128,)),
                ],
                64: [
                    ((64,), (128,)),
                    ((128,), (128,)),
                ],
                128: [
                    ((64,), (128,)),
                ],
            },
        },
        2: {
            16: {
                32: [
                    ((8, 8), (16, 8)),
                    ((8, 8), (8, 16)),
                    ((16, 8), (16, 8)),
                    ((16, 8), (8, 16)),
                ],
                64: [
                    ((8, 8), (16, 8)),
                    ((8, 8), (8, 16)),
                    ((16, 8), (16, 8)),
                    ((16, 8), (8, 16)),
                ],
                128: [
                    ((8, 8), (16, 8)),
                    ((8, 8), (8, 16)),
                ],
            },
        },
        3: {
            16: {
                32: [
                    ((4, 4, 4), (4, 4, 8)),
                    ((4, 4, 4), (2, 8, 8)),
                    ((4, 4, 8), (4, 4, 8)),
                    ((4, 4, 8), (2, 8, 8)),
                ],
                64: [
                    ((4, 4, 4), (4, 4, 8)),
                    ((4, 4, 4), (2, 8, 8)),
                    ((4, 4, 8), (4, 4, 8)),
                    ((4, 4, 8), (2, 8, 8)),
                ],
                128: [
                    ((4, 4, 4), (4, 4, 8)),
                    ((4, 4, 4), (2, 8, 8)),
                    ((2, 4, 8), (2, 8, 8)),
                    ((1, 8, 8), (2, 8, 8)),
                ],
            },
        },
    }

    dtype_dispatchers = []
    head_dim_dispatchers = []
    cm_dispatchers = []
    config_dispatchers = []
    kernels = []

    rank_dispatcher = NaDimDispatcher()
    for na_dim in NA_DIMS:
        rank_dispatcher.append(na_dim)

        dtype_dispatcher = DTypeDispatcher(na_dim=na_dim)
        for dtype in SUPPORTED_DTYPES:
            dtype_dispatcher.append(dtype)

            head_dim_dispatcher = HeadDimDispatcher(dtype=dtype, na_dim=na_dim)

            for head_dim in HEAD_DIMS:
                head_dim_dispatcher.append(head_dim)

                cm_dispatcher = CausalMaskDispatcher(
                    na_dim=na_dim, dtype=dtype, head_dim=head_dim
                )
                for cm in CAUSAL_MASKS[na_dim]:
                    cm_dispatcher.append(cm)

                    config_dispatcher = ConfigDispatcher(
                        dtype=dtype,
                        head_dim=head_dim,
                        na_dim=na_dim,
                        causal_mask=cm,
                    )

                    for q_tile_shape, kv_tile_shape in CONFIGS[na_dim][dtype.bits][
                        head_dim
                    ]:
                        config_dispatcher.append((q_tile_shape, kv_tile_shape))
                        kernels.append(
                            config_dispatcher.get_kernel_instance(
                                q_tile_shape, kv_tile_shape
                            )
                        )
                    config_dispatchers.append(config_dispatcher)
                cm_dispatchers.append(cm_dispatcher)
            head_dim_dispatchers.append(head_dim_dispatcher)
        dtype_dispatchers.append(dtype_dispatcher)

    #

    path_to_sources = f"{path}/autogen/src/cuda/hopper_fna_bwd/"
    rel_header = "natten_autogen/cuda/hopper_fna_bwd/"
    path_to_header_dir = f"{path}/autogen/include/{rel_header}"

    os.makedirs(path_to_sources, exist_ok=False)
    os.makedirs(path_to_header_dir, exist_ok=False)

    path_headers = f"{path_to_header_dir}kernels.h"
    path_rank = f"{path_to_header_dir}interface.h"
    path_dtype = f"{path_to_header_dir}dispatch_dtype.h"
    path_head_dim = f"{path_to_header_dir}dispatch_head_dim.h"
    path_cm = f"{path_to_header_dir}dispatch_cm.h"
    path_tile_shape = f"{path_to_header_dir}dispatch_tile_shape.h"

    rel_path_headers = f"{rel_header}kernels.h"
    rel_path_dtype = f"{rel_header}dispatch_dtype.h"
    rel_path_head_dim = f"{rel_header}dispatch_head_dim.h"
    rel_path_cm = f"{rel_header}dispatch_cm.h"
    rel_path_tile_shape = f"{rel_header}dispatch_tile_shape.h"

    rank_disp = rank_dispatcher.get_dispatcher()

    dtype_disp = ""
    for dispatcher in dtype_dispatchers:
        dtype_disp += dispatcher.get_dispatcher()

    head_dim_disp = ""
    for dispatcher in head_dim_dispatchers:
        head_dim_disp += dispatcher.get_dispatcher()

    cm_disp = ""
    for dispatcher in cm_dispatchers:
        cm_disp += dispatcher.get_dispatcher()

    config_disp = ""
    for dispatcher in config_dispatchers:
        config_disp += dispatcher.get_dispatcher()

    headers = ""
    for kernel in kernels:
        headers += kernel.get_decl()

    assert (
        len(kernels) >= num_splits
    ), f"Generated {len(kernels)} kernels, but got {num_splits=}."
    split_size = len(kernels) // num_splits
    num_splits_with_res = len(kernels) % num_splits
    kernels_emitted = []
    kernels_split = []
    for split_idx in range(num_splits):
        kernel_start_idx = split_size * split_idx + min(num_splits_with_res, split_idx)
        num_kernels_in_split = split_size + (
            1 if split_idx < num_splits_with_res else 0
        )
        kernel_end_idx = kernel_start_idx + num_kernels_in_split
        assert kernel_end_idx <= len(kernels)
        pth_set = set()
        source_list = []
        for kernel_idx in range(kernel_start_idx, kernel_end_idx):
            kernel = kernels[kernel_idx]
            # pth_set.add(kernel.path_to_header)
            source_list.append(kernel)
            kernels_emitted.append(kernel_idx)
        pth_set.add(rel_path_headers)
        write_combined_source_file(
            path_to_sources, f"source_{split_idx}.cu", sorted(pth_set), source_list
        )
        kernels_split.append(source_list)
        # print(f"{split_idx=}, {kernel_start_idx=}, {kernel_end_idx=}, {len(kernels_emitted)=}")
    assert split_idx == num_splits - 1, f"Expected {split_idx=} == {num_splits=} - 1"
    assert len(kernels_emitted) == len(kernels)
    assert sorted(kernels_emitted) == [
        x for x in range(len(kernels))
    ], f"{sorted(kernels_emitted)=}"
    assert all(len(x) > 0 for x in kernels_split)

    namespaces = ["natten", "cuda", "fna_hopper"]
    cuda_headers = [
        "natten/natten.h",
        "ATen/ATen.h",
        "ATen/cuda/CUDAContext.h",
        "c10/cuda/CUDAGuard.h",
        "c10/cuda/CUDAStream.h",
        "torch/extension.h",
        "natten/natten.h",
        "natten/helpers.h",
        "natten/cuda/fna_hopper/fna_backward.cuh",
    ]
    write_header_file(rank_disp, path_rank, namespaces, cuda_headers + [rel_path_dtype])
    write_header_file(
        dtype_disp, path_dtype, namespaces, cuda_headers + [rel_path_head_dim]
    )
    write_header_file(
        head_dim_disp, path_head_dim, namespaces, cuda_headers + [rel_path_cm]
    )
    write_header_file(
        cm_disp, path_cm, namespaces, cuda_headers + [rel_path_tile_shape]
    )
    write_header_file(
        config_disp, path_tile_shape, namespaces, cuda_headers + [rel_path_headers]
    )
    write_header_file(headers, path_headers, namespaces, cuda_headers)


def generate_hopper_fna(output_directory: str, num_splits: int):
    generate_hopper_fna_kernels(output_directory, num_splits=num_splits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--output-directory",
        default=DEFAULT_OUTPUT_DIR,
        help="Path to the directory where the auto-generated "
        "kernel instantiations are dumped. "
        f"Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=8,
        help="Number of source files into which the kernels are split. Default: 8.",
    )
    args = parser.parse_args()
    generate_hopper_fna(args.output_directory, args.num_splits)
