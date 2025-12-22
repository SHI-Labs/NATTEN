# Copyright (c) 2022-2025 Ali Hassani.
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


SUPPORTED_GEMM_SHAPES_FORWARD = [
    (256, 128, 32),
    (256, 128, 64),
    (256, 128, 128),
]


KERNEL_DECL_TEMPLATE = """
void {kernel_name}(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      float attn_scale,
      // fna / fusion parameters
      {DimType} q_shape,
      {DimType} kv_shape,
      {DimType} qkv_shape,
      {DimType} window_size,
      {DimType} stride,
      {DimType} dilation,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      void* ptr_token_layouts,
      // init/launch params
      int device_id,
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
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads_q,
      int heads_kv,
      int dim,
      float attn_scale,
      // fna / fusion parameters
      {DimType} q_shape,
      {DimType} kv_shape,
      {DimType} qkv_shape,
      {DimType} window_size,
      {DimType} stride,
      {DimType} dilation,
      // varlen parameters
      bool is_varlen,
      int max_seqlen_Q,
      int max_seqlen_KV,
      void* ptr_cumulative_seqlen_Q,
      void* ptr_cumulative_seqlen_KV,
      void* ptr_token_layouts,
      // init/launch params
      int device_id,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {{

  using Causal = {Causal};
  using QTileShape = {QTileShape};
  using KVTileShape = {KVTileShape};
  using Config = {GEMMShape};
  using Kernel = natten::cuda::fna_blackwell::KernelForward<
    {dtype}, Causal, QTileShape, KVTileShape, Config, {is_persistent}, false>;
  using VarlenKernel = natten::cuda::fna_blackwell::KernelForward<
    {dtype}, Causal, QTileShape, KVTileShape, Config, {is_persistent}, true>;

  auto launch_kernel = [&](auto& kernel) {{
    auto args = kernel.initialize(
        ptr_Q,
        ptr_K,
        ptr_V,
        ptr_O,
        ptr_LSE,
        batch_size,
        seqlen_q,
        seqlen_k,
        heads_q,
        heads_kv,
        dim,
        attn_scale,
        q_shape,
        kv_shape,
        qkv_shape,
        window_size,
        stride,
        dilation,
        // varlen
        max_seqlen_Q,
        max_seqlen_KV,
        ptr_cumulative_seqlen_Q,
        ptr_cumulative_seqlen_KV,
        ptr_token_layouts,
        device_id);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({{bytes}}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  }};

  if (is_varlen) {{
    VarlenKernel kernel;
    launch_kernel(kernel);
  }}
  else {{
    Kernel kernel;
    launch_kernel(kernel);
  }}
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
E4M3 = DataType("cutlass::float_e4m3_t", "e4m3", "c10::ScalarType::Float8_e4m3fn", 8)
E5M2 = DataType("cutlass::float_e5m2_t", "e5m2", "c10::ScalarType::Float8_e5m2", 8)


def iterable_to_static_cute_tuple(shape_in) -> str:
    shape = ", ".join([f"cute::Int<{x}>" for x in shape_in])
    return f"cute::tuple<{shape}>"


def get_dim_type(na_dim: int) -> str:
    shape = ", ".join(["int" for _ in range(na_dim)])
    return f"cute::tuple<{shape}>"


class BlackwellFnaInstance:
    def __init__(
        self,
        na_dim: int,
        dtype: DataType,
        gemm_shape: Tuple[int, int, int],
        q_tile_shape: tuple,
        kv_tile_shape: tuple,
        causal: tuple,
        is_persistent: bool,
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
        self.is_persistent = is_persistent

    def get_q_tile_shape_cute(self) -> str:
        return iterable_to_static_cute_tuple(self.q_tile_shape)

    def get_kv_tile_shape_cute(self) -> str:
        return iterable_to_static_cute_tuple(self.kv_tile_shape)

    def get_gemm_shape_cute(self) -> str:
        return iterable_to_static_cute_tuple(self.gemm_shape)

    def get_causal_cute(self) -> str:
        consts = ", ".join(
            ["cute::true_type" if c else "cute::false_type" for c in self.causal]
        )
        return f"cute::tuple<{consts}>"

    def get_name(self) -> str:
        name = f"blackwell_fna{self.na_dim}d"
        name += f"_{self.dtype.short_name}"
        name += "_" + "x".join([str(x) for x in self.gemm_shape])
        name += "_Q" + "x".join([str(x) for x in self.q_tile_shape])
        name += "_KV" + "x".join([str(x) for x in self.kv_tile_shape])
        name += "_causal" + "x".join(["1" if c else "0" for c in self.causal])
        if self.is_persistent:
            name += "_persistent"
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
            is_persistent="true" if self.is_persistent else "false",
            dtype=self.dtype.name,
        )


def write_combined_source_file(path, filename, headers, kernels):
    source_head = []
    source_head += ["#ifdef NATTEN_WITH_CUTLASS\n"]
    source_head += ["#ifdef NATTEN_WITH_BLACKWELL_FNA\n"]

    source_head += ["#include <cuda_runtime.h>\n"]
    source_head += ["#include <iostream>\n"]

    source_head += ["#include <ATen/ATen.h>\n"]
    source_head += ["#include <ATen/cuda/CUDAContext.h>\n"]
    source_head += ["#include <c10/cuda/CUDAGuard.h>\n"]
    source_head += ["#include <c10/cuda/CUDAStream.h>\n"]
    source_head += ["#include <torch/extension.h>\n"]

    source_head += ["#include <natten/natten.h>\n"]
    source_head += ["#include <natten/helpers.h>\n"]

    source_head += ["#include <natten/cuda/fna_blackwell/fna_forward.cuh>\n"]

    for header in headers:
        source_head += [f"#include <{header}>\n"]

    source_head += ["namespace natten { \n"]
    source_head += ["namespace cuda { \n"]
    source_head += ["namespace fna_blackwell { \n\n"]

    source_head = "".join(source_head)

    source_body = []
    for kernel in kernels:
        source_body += "\n\n" + kernel.get_impl() + "\n\n"
    source_body = "".join(source_body)

    source_foot = "".join(
        [
            "} // namespace fna_blackwell \n",
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
    def __init__(self):
        self.name = "DISPATCH_BLACKWELL_FNA_FORWARD"
        self.dims = []

    def append(self, na_dim: int):
        self.dims.append(na_dim)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name}(rank, dtype, dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, na_dim in enumerate(self.dims):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if constexpr (rank == {na_dim})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name}_{na_dim}D(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += (
            '      std::cerr << "Blackwell FNA kernel launch failed!" \\\n'
        )
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
        self.name = f"DISPATCH_BLACKWELL_FNA_FORWARD_{self.na_dim}D"

    def append(self, dtype: DataType):
        self.dtypes.append(dtype)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name}(dtype, dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, dtype in enumerate(self.dtypes):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (dtype == {dtype.torch_name})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name}_{dtype.short_name}(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += (
            '      std::cerr << "Blackwell FNA kernel launch failed!" \\\n'
        )
        dispatcher_str += (
            '                << "'
            + f"Blackwell FNA-{self.na_dim}D does not support this data type."
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
            f"DISPATCH_BLACKWELL_FNA_FORWARD_{self.na_dim}D_{self.dtype.short_name}"
        )

        self.dims: List[int] = []

    def append(self, dim: int):
        self.dims.append(dim)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name}(dim, is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, dim in enumerate(self.dims):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (dim == {dim})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name}_headdim{dim}(is_causal, q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += (
            '      std::cerr << "Blackwell FNA kernel launch failed!" \\\n'
        )
        dispatcher_str += (
            '                << "'
            + f"Blackwell FNA-{self.na_dim}D does not support this data type."
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

        self.name = f"DISPATCH_BLACKWELL_FNA_FORWARD_{self.na_dim}D_{self.dtype.short_name}_headdim{head_dim}"

        self.cms: List = []

    def append(self, cm):
        assert len(cm) == self.na_dim, f"{cm} incompatible for {self.na_dim}D NA."
        self.cms.append(cm)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name}(is_causal, q_tile_shape, kv_tile_shape, persistent, ...) \\\n"
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
            dispatcher_str += f"  {self.name}_{cm_str}(q_tile_shape, kv_tile_shape, persistent, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += "    "
        dispatcher_str += (
            '      std::cerr << "Blackwell FNA kernel launch failed!" \\\n'
        )
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


class TileShapeDispatcher:
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

        self.name = f"DISPATCH_BLACKWELL_FNA_FORWARD_{self.na_dim}D_{self.dtype.short_name}_headdim{head_dim}"
        self.name += cm_str

        self.tile_shapes: List = []

    def append(self, tile_shape):
        assert len(tile_shape) == 2
        assert len(tile_shape[0]) == self.na_dim
        assert len(tile_shape[1]) == self.na_dim
        self.tile_shapes.append(tile_shape)

    def get_kernel_instance(self, q_tile_shape, kv_tile_shape, persistent):
        gemm_M = math.prod(q_tile_shape)
        gemm_N = math.prod(kv_tile_shape)
        gemm_K = self.head_dim
        gemm_shape = (gemm_M, gemm_N, gemm_K)

        assert gemm_shape in SUPPORTED_GEMM_SHAPES_FORWARD

        kernel = BlackwellFnaInstance(
            na_dim=self.na_dim,
            dtype=self.dtype,
            gemm_shape=gemm_shape,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            causal=self.causal_mask,
            is_persistent=persistent,
        )

        return kernel

    def get_target_name(self, q_tile_shape, kv_tile_shape, persistent):
        kernel = self.get_kernel_instance(q_tile_shape, kv_tile_shape, persistent)
        return kernel.get_name()

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += (
            f"#define {self.name}(q_tile_shape, kv_tile_shape, persistent, ...) \\\n"
        )
        dispatcher_str += "  [&] { \\\n"
        i = 0
        for q_tile_shape, kv_tile_shape in self.tile_shapes:
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
                if dim != self.na_dim - 1:
                    dispatcher_str += " && "
            dispatcher_str += ")"
            dispatcher_str += " { \\\n"

            dispatcher_str += "  if (persistent) { \\\n"
            dispatcher_str += f"    natten::cuda::fna_blackwell::{self.get_target_name(q_tile_shape, kv_tile_shape, True)}(__VA_ARGS__); \\\n"
            dispatcher_str += "  } else { \\\n"
            dispatcher_str += f"    natten::cuda::fna_blackwell::{self.get_target_name(q_tile_shape, kv_tile_shape, False)}(__VA_ARGS__); \\\n"
            dispatcher_str += "  } \\\n"

            dispatcher_str += "} \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += "    "
        dispatcher_str += (
            '      std::cerr << "Blackwell FNA kernel launch failed!" \\\n'
        )
        dispatcher_str += (
            '                << "'
            + "Blackwell FNA got invalid Q tile and KV tile combination."
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
    header_head += ["#ifdef NATTEN_WITH_BLACKWELL_FNA\n"]
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


def generate_blackwell_fna_kernels(path, num_splits=2):

    NA_DIMS = [1, 2, 3]

    SUPPORTED_DTYPES = [
        Half,
        BFloat,
        E4M3,
        E5M2,
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

    TILE_SHAPES = {
        1: [
            ((256,), (128,)),
        ],
        2: [
            ((16, 16), (16, 8)),
            ((16, 16), (8, 16)),
            ((8, 32), (8, 16)),
            ((8, 32), (4, 32)),
        ],
        3: [
            ((8, 4, 8), (4, 4, 8)),
            ((8, 4, 8), (2, 8, 8)),
            ((2, 8, 16), (4, 4, 8)),
            ((2, 8, 16), (2, 8, 8)),
            ((4, 4, 16), (2, 4, 16)),
            ((2, 16, 8), (2, 8, 8)),
            ((4, 8, 8), (2, 8, 8)),
        ],
    }

    dtype_dispatchers = []
    head_dim_dispatchers = []
    cm_dispatchers = []
    tile_shape_dispatchers = []
    kernels = []

    rank_dispatcher = NaDimDispatcher()
    for na_dim in NA_DIMS:
        rank_dispatcher.append(na_dim)

        dtype_dispatcher = DTypeDispatcher(na_dim=na_dim)
        for dtype in SUPPORTED_DTYPES:
            dtype_dispatcher.append(dtype)

            head_dim_dispatcher = HeadDimDispatcher(na_dim=na_dim, dtype=dtype)

            for head_dim in HEAD_DIMS:
                head_dim_dispatcher.append(head_dim)

                cm_dispatcher = CausalMaskDispatcher(
                    na_dim=na_dim, dtype=dtype, head_dim=head_dim
                )
                for cm in CAUSAL_MASKS[na_dim]:
                    cm_dispatcher.append(cm)

                    tile_shape_dispatcher = TileShapeDispatcher(
                        na_dim=na_dim,
                        dtype=dtype,
                        head_dim=head_dim,
                        causal_mask=cm,
                    )

                    for tile_shape in TILE_SHAPES[na_dim]:
                        tile_shape_dispatcher.append(tile_shape)
                        for persistent in [False, True]:
                            kernels.append(
                                tile_shape_dispatcher.get_kernel_instance(
                                    tile_shape[0], tile_shape[1], persistent
                                )
                            )
                    tile_shape_dispatchers.append(tile_shape_dispatcher)
                cm_dispatchers.append(cm_dispatcher)
            head_dim_dispatchers.append(head_dim_dispatcher)
        dtype_dispatchers.append(dtype_dispatcher)

    #

    path_to_sources = f"{path}/autogen/src/cuda/blackwell_fna/"
    rel_header = "natten_autogen/cuda/blackwell_fna/"
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

    tile_shape_disp = ""
    for dispatcher in tile_shape_dispatchers:
        tile_shape_disp += dispatcher.get_dispatcher()

    headers = ""
    for kernel in kernels:
        headers += kernel.get_decl()

    assert (
        len(kernels) >= num_splits
    ), f"Generated {len(kernels)} kernels, but got {num_splits=}."
    # print(f"Generated {len(kernels)} kernels, got {num_splits=}.")
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

    namespaces = ["natten", "cuda", "fna_blackwell"]
    cuda_headers = [
        "natten/natten.h",
        "ATen/ATen.h",
        "ATen/cuda/CUDAContext.h",
        "c10/cuda/CUDAGuard.h",
        "c10/cuda/CUDAStream.h",
        "torch/extension.h",
        "natten/natten.h",
        "natten/helpers.h",
        "natten/cuda/fna_blackwell/fna_forward.cuh",
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
        tile_shape_disp, path_tile_shape, namespaces, cuda_headers + [rel_path_headers]
    )
    write_header_file(headers, path_headers, namespaces, cuda_headers)


def generate_blackwell_fna(output_directory: str, num_splits: int):
    generate_blackwell_fna_kernels(output_directory, num_splits=num_splits)


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
        default=56,
        help="Number of source files into which the kernels are split. Default: 56.",
    )
    args = parser.parse_args()
    generate_blackwell_fna(args.output_directory, args.num_splits)
