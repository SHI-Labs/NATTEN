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

import os
from typing import List, Tuple

import click


DEFAULT_OUTPUT_DIR = "csrc/"


SUPPORTED_GEMM_SHAPES_FORWARD = [
    (256, 128, 32),
    (256, 128, 64),
    (256, 128, 128),
]

SUPPORTED_GEMM_SHAPES_BACKWARD = [
    (128, 128, 32),
    (128, 128, 64),
    (128, 128, 128),
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
      int heads,
      int dim,
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
      int batch_size,
      int seqlen_q,
      int seqlen_k,
      int heads,
      int dim,
      int device_id,
      float attn_scale,
      cudaStream_t stream,
      at::TensorOptions tensor_options) {{

  using GemmShape = {GEMMShape};
  using Kernel = natten::cuda::fmha_blackwell::KernelForward<
    {dtype}, GemmShape, {is_persistent}, false>;
  using KernelWithResidualMask = natten::cuda::fmha_blackwell::KernelForward<
    {dtype}, GemmShape, {is_persistent}, true>;

  bool no_mask_required = seqlen_k % get<1>(GemmShape{{}}) == 0;
  if (no_mask_required) {{
    Kernel kernel;
    auto args = kernel.initialize(
        ptr_Q,
        ptr_K,
        ptr_V,
        ptr_O,
        ptr_LSE,
        batch_size,
        seqlen_q,
        seqlen_k,
        heads,
        dim,
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({{bytes}}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
  }} else {{
    KernelWithResidualMask kernel;
    auto args = kernel.initialize(
        ptr_Q,
        ptr_K,
        ptr_V,
        ptr_O,
        ptr_LSE,
        batch_size,
        seqlen_q,
        seqlen_k,
        heads,
        dim,
        device_id,
        attn_scale);

    auto bytes = static_cast<int64_t>(kernel.get_workspace_size(args));
    auto workspace = at::empty({{bytes}}, tensor_options.dtype(at::ScalarType::Byte));
    auto workspace_ptr = static_cast<void*>(workspace.data_ptr());
    kernel.run(args, workspace_ptr, stream);
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


def iterable_to_static_cute_tuple(shape_in) -> str:
    shape = ", ".join([f"cute::Int<{x}>" for x in shape_in])
    return f"cute::tuple<{shape}>"


class BlackwellFmhaInstance:
    def __init__(
        self,
        dtype: DataType,
        gemm_shape: Tuple[int, int, int],
        is_persistent: bool,
        is_backward: bool,
    ):
        assert len(gemm_shape) == 3
        self.gemm_shape = gemm_shape
        self.dtype = dtype
        self.max_head_dim = gemm_shape[2]
        self.is_persistent = is_persistent
        self.is_backward = is_backward

    def get_gemm_shape_cute(self) -> str:
        return iterable_to_static_cute_tuple(self.gemm_shape)

    def get_name(self) -> str:
        backward_str = "" if not self.is_backward else "_backward"
        name = f"blackwell_fmha{backward_str}"
        name += f"_{self.dtype.short_name}"
        name += "_" + "x".join([str(x) for x in self.gemm_shape])
        if self.is_persistent:
            name += "_persistent"
        return name

    def get_decl(self) -> str:
        return KERNEL_DECL_TEMPLATE.format(
            kernel_name=self.get_name(),
        )

    def get_impl(self) -> str:
        assert not self.is_backward
        return KERNEL_IMPL_TEMPLATE.format(
            kernel_name=self.get_name(),
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

    source_head += ["#include <natten/cuda/fmha_blackwell/fmha_forward.cuh>\n"]

    for header in headers:
        source_head += [f"#include <{header}>\n"]

    source_head += ["namespace natten { \n"]
    source_head += ["namespace cuda { \n"]
    source_head += ["namespace fmha_blackwell { \n\n"]

    source_head = "".join(source_head)

    source_body = []
    for kernel in kernels:
        source_body += "\n\n" + kernel.get_impl() + "\n\n"
    source_body = "".join(source_body)

    source_foot = "".join(
        [
            "} // namespace fmha_blackwell \n",
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


class DTypeDispatcher:
    def __init__(self, is_backward: bool):
        self.dtypes: List[DataType] = []
        fwd_bwd_str = "BACKWARD" if is_backward else "FORWARD"
        self.name = f"DISPATCH_BLACKWELL_FMHA_{fwd_bwd_str}"
        self.is_backward = is_backward

    def append(self, dtype: DataType):
        self.dtypes.append(dtype)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name}(dtype, dim, q_tile_size, kv_tile_size, persistent, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, dtype in enumerate(self.dtypes):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (dtype == {dtype.torch_name})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name}_{dtype.short_name}(dim, q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += (
            '      std::cerr << "Blackwell FMHA kernel launch failed!" \\\n'
        )
        dispatcher_str += (
            '                << "'
            + "Blackwell FMHA does not support this data type."
            + '" \\\n'
        )
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();"
        dispatcher_str += "\n\n"
        return dispatcher_str


class HeadDimDispatcher:
    def __init__(self, is_backward: bool, dtype: DataType):
        self.dtype = dtype
        fwd_bwd_str = "BACKWARD" if is_backward else "FORWARD"
        self.name = f"DISPATCH_BLACKWELL_FMHA_{fwd_bwd_str}_{self.dtype.short_name}"
        self.is_backward = is_backward

        self.dims: List[int] = []

    def append(self, dim: int):
        self.dims.append(dim)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += (
            f"#define {self.name}(dim, q_tile_size, kv_tile_size, persistent, ...) \\\n"
        )
        dispatcher_str += "  [&] { \\\n"
        for i, dim in enumerate(self.dims):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (dim == {dim})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name}_headdim{dim}(q_tile_size, kv_tile_size, persistent, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += (
            '      std::cerr << "Blackwell FMHA kernel launch failed!" \\\n'
        )
        dispatcher_str += (
            '                << "'
            + "Blackwell FMHA does not support this data type."
            + '" \\\n'
        )
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();"
        dispatcher_str += "\n\n"
        return dispatcher_str


class TileSizeDispatcher:
    def __init__(
        self,
        is_backward: bool,
        dtype: DataType,
        head_dim: int,
    ):
        self.dtype = dtype
        self.head_dim = head_dim

        fwd_bwd_str = "BACKWARD" if is_backward else "FORWARD"
        self.name = f"DISPATCH_BLACKWELL_FMHA_{fwd_bwd_str}_{self.dtype.short_name}_headdim{head_dim}"
        self.is_backward = is_backward

        self.tile_sizes: List = []

    def append(self, tile_size):
        assert len(tile_size) == 2
        assert isinstance(tile_size[0], int)
        assert isinstance(tile_size[1], int)
        self.tile_sizes.append(tile_size)

    def get_kernel_instance(self, q_tile_size, kv_tile_size, persistent):
        gemm_M = q_tile_size
        gemm_N = kv_tile_size
        gemm_K = self.head_dim
        gemm_shape = (gemm_M, gemm_N, gemm_K)

        supported_gemm_shapes = (
            SUPPORTED_GEMM_SHAPES_BACKWARD
            if self.is_backward
            else SUPPORTED_GEMM_SHAPES_FORWARD
        )
        assert gemm_shape in supported_gemm_shapes

        kernel = BlackwellFmhaInstance(
            dtype=self.dtype,
            gemm_shape=gemm_shape,
            is_persistent=persistent,
            is_backward=self.is_backward,
        )

        return kernel

    def get_target_name(self, q_tile_size, kv_tile_size, persistent):
        kernel = self.get_kernel_instance(q_tile_size, kv_tile_size, persistent)
        return kernel.get_name()

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += (
            f"#define {self.name}(q_tile_size, kv_tile_size, persistent, ...) \\\n"
        )
        dispatcher_str += "  [&] { \\\n"
        i = 0
        for q_tile_size, kv_tile_size in self.tile_sizes:
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            i += 1
            dispatcher_str += "if ("
            dispatcher_str += f"q_tile_size == {q_tile_size}"
            dispatcher_str += " && "
            dispatcher_str += "\\\n"
            dispatcher_str += f"kv_tile_size == {kv_tile_size}"
            dispatcher_str += ")"
            dispatcher_str += " { \\\n"

            dispatcher_str += "  if (persistent) { \\\n"
            dispatcher_str += f"    natten::cuda::fmha_blackwell::{self.get_target_name(q_tile_size, kv_tile_size, True)}(__VA_ARGS__); \\\n"
            dispatcher_str += "  } else { \\\n"
            dispatcher_str += f"    natten::cuda::fmha_blackwell::{self.get_target_name(q_tile_size, kv_tile_size, False)}(__VA_ARGS__); \\\n"
            dispatcher_str += "  } \\\n"

            dispatcher_str += "} \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += "    "
        dispatcher_str += (
            '      std::cerr << "Blackwell FMHA kernel launch failed!" \\\n'
        )
        dispatcher_str += (
            '                << "'
            + "Blackwell FMHA got invalid Q tile and KV tile combination."
            + '" \\\n'
        )
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();"
        dispatcher_str += "\n\n"
        return dispatcher_str


def write_header_file(content, path, namespaces, extra_includes=[]):
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


def generate_blackwell_fmha_kernels(path, num_splits=2):

    SUPPORTED_DTYPES = [
        Half,
        BFloat,
    ]

    HEAD_DIMS = [32, 64, 128]

    TILE_SIZES = [
        (256, 128),
    ]

    head_dim_dispatchers = []
    tile_size_dispatchers = []
    kernels = []

    dtype_dispatcher = DTypeDispatcher(is_backward=False)
    for dtype in SUPPORTED_DTYPES:
        dtype_dispatcher.append(dtype)

        head_dim_dispatcher = HeadDimDispatcher(is_backward=False, dtype=dtype)

        for head_dim in HEAD_DIMS:
            head_dim_dispatcher.append(head_dim)

            tile_size_dispatcher = TileSizeDispatcher(
                is_backward=False,
                dtype=dtype,
                head_dim=head_dim,
            )

            for tile_size in TILE_SIZES:
                tile_size_dispatcher.append(tile_size)
                for persistent in [False, True]:
                    kernels.append(
                        tile_size_dispatcher.get_kernel_instance(
                            tile_size[0], tile_size[1], persistent
                        )
                    )
            tile_size_dispatchers.append(tile_size_dispatcher)
        head_dim_dispatchers.append(head_dim_dispatcher)

    #

    path_to_sources = f"{path}/autogen/src/cuda/blackwell_fmha/"
    rel_header = "natten_autogen/cuda/blackwell_fmha/"
    path_to_header_dir = f"{path}/autogen/include/{rel_header}"

    os.makedirs(path_to_sources, exist_ok=False)
    os.makedirs(path_to_header_dir, exist_ok=False)

    path_headers = f"{path_to_header_dir}kernels.h"
    path_dtype = f"{path_to_header_dir}interface.h"
    path_head_dim = f"{path_to_header_dir}dispatch_head_dim.h"
    path_tile_size = f"{path_to_header_dir}dispatch_tile_size.h"

    rel_path_headers = f"{rel_header}kernels.h"
    rel_path_head_dim = f"{rel_header}dispatch_head_dim.h"
    rel_path_tile_size = f"{rel_header}dispatch_tile_size.h"

    dtype_disp = dtype_dispatcher.get_dispatcher()

    head_dim_disp = ""
    for dispatcher in head_dim_dispatchers:
        head_dim_disp += dispatcher.get_dispatcher()

    tile_size_disp = ""
    for dispatcher in tile_size_dispatchers:
        tile_size_disp += dispatcher.get_dispatcher()

    headers = ""
    for kernel in kernels:
        headers += kernel.get_decl()

    split_size = (len(kernels) + num_splits - 1) // num_splits
    kernels_emitted = []
    for split_idx in range(num_splits):
        kernel_start_idx = split_size * split_idx
        kernel_end_idx = min(kernel_start_idx + split_size, len(kernels))
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
    assert split_idx == num_splits - 1, f"Expected {split_idx=} == {num_splits=} - 1"
    assert len(kernels_emitted) == len(kernels) and sorted(kernels_emitted) == [
        x for x in range(len(kernels))
    ]

    namespaces = ["natten", "cuda", "fmha_blackwell"]
    cuda_headers = [
        "natten/natten.h",
        "ATen/ATen.h",
        "ATen/cuda/CUDAContext.h",
        "c10/cuda/CUDAGuard.h",
        "c10/cuda/CUDAStream.h",
        "torch/extension.h",
        "natten/natten.h",
        "natten/helpers.h",
        "natten/cuda/fmha_blackwell/fmha_forward.cuh",
    ]
    write_header_file(
        dtype_disp, path_dtype, namespaces, cuda_headers + [rel_path_head_dim]
    )
    write_header_file(
        head_dim_disp, path_head_dim, namespaces, cuda_headers + [rel_path_tile_size]
    )
    write_header_file(
        tile_size_disp, path_tile_size, namespaces, cuda_headers + [rel_path_headers]
    )
    write_header_file(headers, path_headers, namespaces, cuda_headers)


@click.command()
@click.option(
    "-o",
    "--output-directory",
    default=DEFAULT_OUTPUT_DIR,
    help="Path to the directory where the auto-generated "
    "kernel instantiations are dumped. "
    f"Default: {DEFAULT_OUTPUT_DIR}",
)
@click.option(
    "--num-splits",
    default=2,
    help="Number of source files into which the kernels are split. Default: 2.",
)
def generate_blackwell_fmha(output_directory: str, num_splits: int):
    generate_blackwell_fmha_kernels(output_directory, num_splits=num_splits)


if __name__ == "__main__":
    generate_blackwell_fmha()
