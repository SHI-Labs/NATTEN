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
import os
from enum import Enum
from typing import List, Tuple


DEFAULT_OUTPUT_DIR = "csrc/"


SUPPORTED_CONFIGS_BACKWARD = {
     80: {32: [(128, 128, 32)],
          64: [(128, 128, 64)],
          96: [(64, 128, 96)],
          128: [(64, 128, 128)],
          192: [(64, 80, 192)],
          256: [(64, 64, 256)]},
     86: {32: [(64, 128, 32)],
          64: [(64, 128, 64)],
          96: [(64, 128, 96)],
          128: [(64, 96, 128)],
          192: [(64, 64, 192)],
          256: [(32, 64, 256)]},
     89: {32: [(64, 128, 32)],
          64: [(64, 128, 64)],
          96: [(64, 128, 96)],
          128: [(64, 96, 128)],
          192: [(64, 64, 192)],
          256: [(32, 64, 256)]}
}

KERNEL_DECL_TEMPLATE = """
void {kernel_name}(
    natten::cuda::flash::Flash_bwd_params params,
    cudaStream_t stream);
"""


KERNEL_IMPL_TEMPLATE = """
void {kernel_name}(
    natten::cuda::flash::Flash_bwd_params params,
    cudaStream_t stream) {{

    using Kernel = natten::cuda::flash::FlashBackwardKernel<
      /* Arch= */ {cc},
      /* Element= */ {dtype},
      /* HeadDim= */ {GEMMShape[2]},
      /* kBlockM= */ {GEMMShape[0]},
      /* kBlockN= */ {GEMMShape[1]},
      /* Deterministic= */ {deterministic_str}
    >;

    Kernel flash_bwd_kernel;

    flash_bwd_kernel.run(params, stream);
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


class FlashFmhaInstance:
    def __init__(
        self,
        dtype: DataType,
        gemm_shape: Tuple[int, int, int],
        cc: int,
        deterministic_str: str
    ):
        assert len(gemm_shape) == 3
        self.gemm_shape = gemm_shape
        self.dtype = dtype
        self.max_head_dim = gemm_shape[2]
        self.cc = cc
        self.deterministic_str = deterministic_str

    def get_gemm_shape_cute(self) -> str:
        return self.gemm_shape

    def get_name(self) -> str:
        name = "flash_fmha_bwd"
        name += f"_sm{self.cc}"
        name += f"_{self.dtype.short_name}"
        name += "_" + "x".join([str(x) for x in self.gemm_shape])
        name += f"_deterministic" if self.deterministic_str == "true" else "_nondeterministic"
        return name

    def get_decl(self) -> str:
        return KERNEL_DECL_TEMPLATE.format(
            kernel_name=self.get_name(),
        )

    def get_impl(self) -> str:
        return KERNEL_IMPL_TEMPLATE.format(
            cc=self.cc,
            kernel_name=self.get_name(),
            GEMMShape=self.get_gemm_shape_cute(),
            dtype=self.dtype.name,
            deterministic_str=self.deterministic_str
        )


def write_combined_source_file(path, filename, headers, kernels):
    source_head = []
    source_head += ["#ifdef NATTEN_WITH_CUTLASS\n"]

    source_head += ["#include <cuda_runtime.h>\n"]
    source_head += ["#include <iostream>\n"]

    source_head += ["#include <ATen/ATen.h>\n"]
    source_head += ["#include <ATen/cuda/CUDAContext.h>\n"]
    source_head += ["#include <c10/cuda/CUDAGuard.h>\n"]
    source_head += ["#include <c10/cuda/CUDAStream.h>\n"]
    source_head += ["#include <torch/extension.h>\n"]

    source_head += ["#include <natten/natten.h>\n"]
    source_head += ["#include <natten/helpers.h>\n"]

    source_head += ["#include <natten/cuda/flash_fmha/flash_fmha_backward.cuh>\n"]

    for header in headers:
        source_head += [f"#include <{header}>\n"]

    source_head += ["namespace natten { \n"]
    source_head += ["namespace cuda { \n"]
    source_head += ["namespace flash { \n\n"]

    source_head = "".join(source_head)

    source_body = []
    for kernel in kernels:
        source_body += "\n\n" + kernel.get_impl() + "\n\n"
    source_body = "".join(source_body)

    source_foot = "".join(
        [
            "} // namespace flash \n",
            "} // namespace cuda \n",
            "} // namespace natten \n",
            "#endif \n",
            "\n",
        ]
    )
    filename = f"{path}/{filename}"
    with open(filename, "w") as f:
        f.write(source_head)
        f.write(source_body)
        f.write(source_foot)

# We need four dispatchers:
#   1. Datatype: f16 or bf16
#   2. Headdim: 32, 64, 128, 256
#   3. Arch: 80, 86, 89
#   4. Deterministic: T/F


class DTypeDispatcher:
    def __init__(self):
        self.dtypes = []
        self.name = "DISPATCH_FLASH_FMHA_BACKWARD"

    def append(self, dtype: DataType):
        self.dtypes.append(dtype)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name}(dtype, dim, cc, deterministic, q_tile_size, kv_tile_size, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, dtype in enumerate(self.dtypes):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (dtype == {dtype.torch_name})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name}_{dtype.short_name}(dim, cc, deterministic, q_tile_size, kv_tile_size, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "Flash FMHA backward kernel launch failed!" \\\n'
        dispatcher_str += (
            '                << "'
            + "Flash FMHA does not support this data type."
            + '" \\\n'
        )
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();"
        dispatcher_str += "\n\n"
        return dispatcher_str


class HeadDimDispatcher:
    def __init__(self, dtype: DataType):
        self.dtype = dtype
        self.name = f"DISPATCH_FLASH_FMHA_BACKWARD_{self.dtype.short_name}"

        self.dims: List[int] = []

    def append(self, dim: int):
        self.dims.append(dim)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name}(dim, cc, deterministic, q_tile_size, kv_tile_size, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, dim in enumerate(self.dims):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (dim == {dim})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name}_headdim{dim}(cc, deterministic, q_tile_size, kv_tile_size, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "Flash FMHA backward kernel launch failed!" \\\n'
        dispatcher_str += (
            '                << "'
            + "Flash FMHA does not support this headdim."
            + '" \\\n'
        )
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();"
        dispatcher_str += "\n\n"
        return dispatcher_str


class ArchDispatcher:
    def __init__(self, dtype: DataType, headdim: int):
        self.dtype = dtype
        self.headdim = headdim
        self.name = f"DISPATCH_FLASH_FMHA_BACKWARD_{self.dtype.short_name}_headdim{self.headdim}"

        self.ccs: List[int] = []

    def append(self, cc: int):
        self.ccs.append(cc)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name}(cc, deterministic, q_tile_size, kv_tile_size, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, cc in enumerate(self.ccs):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (cc == {cc})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name}_sm{cc}(deterministic, q_tile_size, kv_tile_size, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "Flash FMHA backward kernel launch failed!" \\\n'
        dispatcher_str += (
            '                << "'
            + "Flash FMHA does not support this architecture."
            + '" \\\n'
        )
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();"
        dispatcher_str += "\n\n"
        return dispatcher_str


class DeterministicDispatcher:
    def __init__(self, dtype: DataType, headdim: int, cc: int):
        self.dtype = dtype
        self.headdim = headdim
        self.cc = cc
        self.name = f"DISPATCH_FLASH_FMHA_BACKWARD_{self.dtype.short_name}_headdim{self.headdim}_sm{cc}"

        self.deterministics: List[int] = []

    def append(self, d: int):
        self.deterministics.append(d)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name}(deterministic, q_tile_size, kv_tile_size, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, d in enumerate(self.deterministics):
            deterministic_str = "_deterministic" if d == "true" else "_nondeterministic"
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (deterministic == {d})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name}{deterministic_str}(q_tile_size, kv_tile_size, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "Flash FMHA backward kernel launch failed!" \\\n'
        dispatcher_str += (
            '                << "'
            + "Flash FMHA does not support this architecture."
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
        dtype: DataType,
        head_dim: int,
        cc: int,
        deterministic_str: str
    ):
        self.dtype = dtype
        self.head_dim = head_dim
        self.cc = cc
        self.deterministic = deterministic_str
        self.deterministic_str = "deterministic" if deterministic_str == "true"\
            else "nondeterministic"

        self.name = (
            f"DISPATCH_FLASH_FMHA_BACKWARD_{self.dtype.short_name}_headdim{head_dim}_sm{cc}_{self.deterministic_str}"
        )

        self.configs: List = []

    def append(self, config):
        assert len(config) == 2
        assert isinstance(config[0], int)
        assert isinstance(config[1], int)
        self.configs.append(config)

    def get_kernel_instance(self, q_tile_size, kv_tile_size):
        gemm_M = q_tile_size
        gemm_N = kv_tile_size
        gemm_K = self.head_dim
        gemm_shape = (gemm_M, gemm_N, gemm_K)
        config = gemm_shape

        supported_configs = SUPPORTED_CONFIGS_BACKWARD[self.cc][self.head_dim]
        assert (
            config in supported_configs
        ), f"{config=} not in supported configs {supported_configs=}"
        assert self.cc in (80, 86, 89), f"{cc=} not supported for flash attention."

        kernel = FlashFmhaInstance(
            dtype=self.dtype,
            gemm_shape=gemm_shape,
            cc=self.cc,
            deterministic_str=self.deterministic
        )

        return kernel

    def get_target_name(self, q_tile_size, kv_tile_size, cc):
        kernel = self.get_kernel_instance(q_tile_size, kv_tile_size)
        return kernel.get_name()

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += (
            f"#define {self.name}(q_tile_size, kv_tile_size, ...) \\\n"
        )
        dispatcher_str += "  [&] { \\\n"
        i = 0
        for q_tile_size, kv_tile_size in self.configs:
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

            dispatcher_str += f"  natten::cuda::flash::{self.get_target_name(q_tile_size, kv_tile_size, self.cc)}(__VA_ARGS__); \\\n"

            dispatcher_str += "} \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += "    "
        dispatcher_str += '      std::cerr << "Flash FMHA kernel launch failed!" \\\n'
        dispatcher_str += (
            '                << "'
            + "Flash FMHA got invalid Q tile, KV tile combination."
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
        "\n",
    ]
    with open(path, "w") as f:
        f.write("".join(header_head))
        f.write(content)
        f.write("".join(header_foot))


def generate_flash_fmha_kernels(path, num_splits=2):

    SUPPORTED_DTYPES = [
        Half,
        BFloat,
    ]

    SUPPORTED_ARCHS = [80, 86, 89]

    HEAD_DIMS = [32, 64, 96, 128, 192, 256]

    head_dim_dispatchers = []
    config_dispatchers = []
    arch_dispatchers = []
    deterministic_dispatchers = []
    kernels = []

    dtype_dispatcher = DTypeDispatcher()
    for dtype in SUPPORTED_DTYPES:
        dtype_dispatcher.append(dtype)

        head_dim_dispatcher = HeadDimDispatcher(dtype=dtype)

        for head_dim in HEAD_DIMS:
            head_dim_dispatcher.append(head_dim)

            arch_dispatcher = ArchDispatcher(dtype, head_dim)
            for cc in SUPPORTED_ARCHS:
                arch_dispatcher.append(cc)

                deterministic_dispatcher = DeterministicDispatcher(dtype, head_dim, cc)

                for deterministic_str in ["true", "false"]:
                    deterministic_dispatcher.append(deterministic_str)
                    config_dispatcher = ConfigDispatcher(
                        dtype=dtype,
                        head_dim=head_dim,
                        cc=cc,
                        deterministic_str=deterministic_str
                    )

                    for q_tile_size, kv_tile_size, _ in SUPPORTED_CONFIGS_BACKWARD[cc][head_dim]:
                        config_dispatcher.append((q_tile_size, kv_tile_size))
                        kernels.append(
                            config_dispatcher.get_kernel_instance(
                                q_tile_size, kv_tile_size
                            )
                        )
                    config_dispatchers.append(config_dispatcher)
                deterministic_dispatchers.append(deterministic_dispatcher)
            arch_dispatchers.append(arch_dispatcher)
        head_dim_dispatchers.append(head_dim_dispatcher)

    path_to_sources = f"{path}/autogen/src/cuda/flash_fmha_bwd/"
    rel_header = "natten_autogen/cuda/flash_fmha_bwd/"
    path_to_header_dir = f"{path}/autogen/include/{rel_header}"

    os.makedirs(path_to_sources, exist_ok=False)
    os.makedirs(path_to_header_dir, exist_ok=False)

    path_headers = f"{path_to_header_dir}kernels.h"
    path_dtype = f"{path_to_header_dir}interface.h"
    path_head_dim = f"{path_to_header_dir}dispatch_head_dim.h"
    path_tile_size = f"{path_to_header_dir}dispatch_tile_size.h"
    path_arch = f"{path_to_header_dir}dispatch_arch.h"
    path_deterministic = f"{path_to_header_dir}dispatch_deterministic.h"

    rel_path_headers = f"{rel_header}kernels.h"
    rel_path_head_dim = f"{rel_header}dispatch_head_dim.h"
    rel_path_tile_size = f"{rel_header}dispatch_tile_size.h"
    rel_path_arch = f"{rel_header}dispatch_arch.h"
    rel_path_deterministic = f"{rel_header}dispatch_deterministic.h"

    dtype_disp = dtype_dispatcher.get_dispatcher()

    head_dim_disp = ""
    for dispatcher in head_dim_dispatchers:
        head_dim_disp += dispatcher.get_dispatcher()

    config_disp = ""
    for dispatcher in config_dispatchers:
        config_disp += dispatcher.get_dispatcher()

    arch_disp = ""
    for dispatcher in arch_dispatchers:
        arch_disp += dispatcher.get_dispatcher()

    deterministic_disp = ""
    for dispatcher in deterministic_dispatchers:
        deterministic_disp += dispatcher.get_dispatcher()

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

    namespaces = ["natten", "cuda", "flash"]
    cuda_headers = [
        "natten/natten.h",
        "ATen/ATen.h",
        "ATen/cuda/CUDAContext.h",
        "c10/cuda/CUDAGuard.h",
        "c10/cuda/CUDAStream.h",
        "torch/extension.h",
        "natten/natten.h",
        "natten/helpers.h",
        "natten/cuda/flash_fmha/flash_fmha_backward.cuh",
    ]
    write_header_file(
        dtype_disp, path_dtype, namespaces, cuda_headers + [rel_path_head_dim]
    )
    write_header_file(
        head_dim_disp, path_head_dim, namespaces, cuda_headers + [rel_path_arch]
    )
    write_header_file(
        arch_disp, path_arch, namespaces, cuda_headers + [rel_path_deterministic]
    )
    write_header_file(
        deterministic_disp, path_deterministic, namespaces, cuda_headers + [rel_path_tile_size]
    )
    write_header_file(
        config_disp, path_tile_size, namespaces, cuda_headers + [rel_path_headers]
    )
    write_header_file(headers, path_headers, namespaces, cuda_headers)


def generate_flash_fmha(output_directory: str, num_splits: int):
    generate_flash_fmha_kernels(output_directory, num_splits=num_splits)


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
        default=1,
        help="Number of source files into which the kernels are split. Default: 1.",
    )
    args = parser.parse_args()
    generate_flash_fmha(args.output_directory, args.num_splits)
