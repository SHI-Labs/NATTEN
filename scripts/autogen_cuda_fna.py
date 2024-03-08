# Copyright (c) 2022-2024 Ali Hassani.
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
from typing import List, Optional, Tuple

import click


DEFAULT_OUTPUT_DIR = "csrc/"

_SM_TO_UPPER_BOUND = {
    50: 70,
    70: 75,
    75: 80,
    80: 100,
}

_SM_TO_LOWER_BOUND = {
    50: 50,
    70: 70,
    75: 75,
    80: 80,
}


class KernelConfig:
    def __init__(self, na_dim: int, sm: int, gemm_shape: Tuple[int, int, int]):
        assert 0 < na_dim <= 3
        assert 50 <= sm <= 80
        assert len(gemm_shape) == 3
        self.na_dim = na_dim
        self.sm = sm
        self.gemm_shape = gemm_shape

    def get_name(self) -> str:
        return f"fna{self.na_dim}_{self.gemm_shape[0]}x{self.gemm_shape[1]}x{self.gemm_shape[2]}_sm{self.sm}"


class KernelConfigList:
    def __init__(self, na_dim: int, sm: int, gemm_shapes: List[Tuple[int, int, int]]):
        assert 0 < na_dim <= 3
        assert 50 <= sm <= 80
        self.na_dim = na_dim
        self.sm = sm
        self.gemm_shapes = gemm_shapes

    @property
    def configs(self) -> List[KernelConfig]:
        return [
            KernelConfig(na_dim=self.na_dim, sm=self.sm, gemm_shape=gemm_shape)
            for gemm_shape in self.gemm_shapes
        ]

    def get_name(self) -> str:
        return f"fna{self.na_dim}_sm{self.sm}"


class DataType:
    def __init__(self, name, natten_name, short_name, bits, min_sm):
        self.name = name
        self.natten_name = natten_name
        self.bits = bits
        self.short_name = short_name
        self.min_sm = min_sm


NATTEN_Float = DataType("float", "natten::float32", "float32", 32, 0)
NATTEN_Half = DataType("cutlass::half_t", "natten::float16", "float16", 16, 50)
NATTEN_BFloat = DataType("cutlass::bfloat16_t", "natten::bfloat16", "bfloat16", 16, 80)


KERNEL_DECL_TEMPLATE = """__global__ void __launch_bounds__(
    {CPP_CLASS}::kNumThreads,
    {CPP_CLASS}::kMinBlocksPerSm)
{NAME}(typename {CPP_CLASS}::Params p);
"""


KERNEL_IMPL_TEMPLATE = """__global__ void __launch_bounds__(
    {CPP_CLASS}::kNumThreads,
    {CPP_CLASS}::kMinBlocksPerSm)
{NAME}(typename {CPP_CLASS}::Params p) {{
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= {SM}0
#if __CUDA_ARCH__ < {SM_MAX}0
  if (!p.advance_to_block()) {{
    return;
  }}
  {CPP_CLASS}::attention_kernel(p);
  return;
#endif
#endif
    printf(
        "FATAL: FNA kernel `{NAME}` was built for SM{SM}, but attempted to launch from SM%d\\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}}
"""

SM_TO_SM_MAX = {
    50: 70,
    70: 75,
    75: 80,
    80: 100,
}


class FusedNAKernel:
    def __init__(
        self,
        dtype: DataType,
        config: KernelConfig,
        causal_mask: List,
        has_rpb: Optional[bool] = False,
    ):
        self.dtype = dtype
        self.config = config
        self.causal_mask = causal_mask
        self.name_cc = f"{self.config.get_name()}_{dtype.short_name}"
        assert len(self.causal_mask) == self.config.na_dim
        cm_str = "_".join([str(int(x)) for x in self.causal_mask])
        self.name_cc += f"_cm_{cm_str}"
        self.path_to_header = "natten/cuda/fna/kernel_forward.h"
        self.aligned = True
        self.has_rpb = has_rpb
        if self.has_rpb:
            self.name_cc += "_rpb"

    @property
    def causal_mask_inst(self) -> str:
        return (
            "CausalMask<"
            + ", ".join(["true" if x else "false" for x in self.causal_mask])
            + ">"
        )

    @property
    def cpp_class(self) -> str:
        template_args = ", ".join(
            [
                str(self.config.na_dim),
                self.causal_mask_inst,
                self.dtype.name,
                f"cutlass::arch::Sm{self.config.sm}",
                "true" if self.aligned else "false",
                str(self.config.gemm_shape[0]),
                str(self.config.gemm_shape[1]),
                str(self.config.gemm_shape[2]),
                "true" if self.has_rpb else "false",
            ]
        )
        return f"FusedNeighborhoodAttentionKernel<{template_args}>"

    def header(self):
        return KERNEL_DECL_TEMPLATE.format(
            CPP_CLASS=self.cpp_class,
            NAME=self.name_cc,
        )

    def source(self):
        return KERNEL_IMPL_TEMPLATE.format(
            CPP_CLASS=self.cpp_class,
            NAME=self.name_cc,
            SM=self.config.sm,
            SM_MAX=SM_TO_SM_MAX[self.config.sm],
        )


class FusedNAKernelBundle:
    def __init__(
        self,
        na_dim: int,
        sm: int,
        dtype: DataType,
        config_list: KernelConfigList,
        causal_mask: List,
        has_rpb: Optional[bool] = False,
    ):
        self.na_dim = na_dim
        self.sm = sm
        self.dtype = dtype
        self.config_list = config_list
        self.causal_mask = causal_mask
        self.name_cc = f"{self.config_list.get_name()}_{dtype.short_name}"
        assert len(self.causal_mask) == self.config_list.na_dim
        cm_str = "_".join([str(int(x)) for x in self.causal_mask])
        self.name_cc += f"_cm_{cm_str}"
        self.path_to_header = "natten/cuda/fna/kernel_forward.h"
        self.aligned = True
        self.has_rpb = has_rpb
        if self.has_rpb:
            self.name_cc += "_rpb"

    @property
    def kernels(self):
        return [
            FusedNAKernel(
                dtype=self.dtype,
                causal_mask=self.causal_mask,
                config=config,
                has_rpb=self.has_rpb,
            )
            for config in self.config_list.configs
        ]

    def header(self):
        decl = "\n\n"
        decl += "///////////////////////////////////////////////////////////////////\n"
        decl += f"// FNA-{self.na_dim}D / {self.dtype.short_name} / SM{self.sm}\n"
        decl += "///////////////////////////////////////////////////////////////////"

        for kernel in self.kernels:
            decl += "\n\n"
            decl += kernel.header()

        decl += f"\n\ntemplate <typename T>\nvoid {self.name_cc}"
        decl += "(T cb) {\n"
        for kernel in self.kernels:
            decl += f"  cb({kernel.cpp_class}(), {kernel.name_cc});\n"
        decl += "}"
        return decl

    def source(self):
        impl = "\n\n"
        impl += "///////////////////////////////////////////////////////////////////\n"
        impl += f"// FNA-{self.na_dim}D / {self.dtype.short_name} / SM{self.sm}\n"
        impl += "///////////////////////////////////////////////////////////////////"

        for kernel in self.kernels:
            impl += "\n\n"
            impl += kernel.source()
        return impl


def write_combined_source_file(path, filename, headers, sources):
    source_head = []
    source_head += ["#include <cuda_runtime.h>\n"]
    source_head += ["#include <iostream>\n"]
    source_head += ["#include <natten/dtypes.cuh>\n"]

    for header in headers:
        source_head += [f"#include <{header}>\n"]

    source_head += ["namespace natten { \n"]
    source_head += ["namespace cuda { \n"]
    source_head += ["namespace fna { \n\n"]

    source_head = "".join(source_head)

    source_body = []
    for source in sources:
        source_body += source.source()
    source_body = "".join(source_body)

    source_foot = "".join(
        [
            "} \n",
            "} \n",
            "} \n",
            "\n",
        ]
    )
    filename = f"{path}/{filename}"
    with open(filename, "w") as f:
        f.write(source_head)
        f.write(source_body)
        f.write(source_foot)


class RankDispatcher:
    def __init__(self):
        self.name_cc = "DISPATCH_FNA_FORWARD_KERNEL"
        self.name_target = "DISPATCH_FNA_FORWARD_"
        self.dims = []

    def append(self, na_dim: int):
        self.dims.append(na_dim)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += (
            f"#define {self.name_cc}(rank, cc, dtype, is_causal, has_rpb, cb) \\\n"
        )
        dispatcher_str += "  [&] { \\\n"
        for i, na_dim in enumerate(self.dims):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if constexpr (rank == {na_dim})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name_target}{na_dim}D(cc, dtype, is_causal, has_rpb, cb); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "NATTEN FNA kernel launch failed!" \\\n'
        dispatcher_str += (
            '                << "'
            + "Invalid spatial extent rank! Only 1, 2, and 3D are supported!"
            + '" \\\n'
        )
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();"
        dispatcher_str += "\n\n"
        return dispatcher_str


class DeviceDispatcher:
    def __init__(self, na_dim: int):
        self.na_dim = na_dim
        self.name_cc = f"DISPATCH_FNA_FORWARD_{self.na_dim}D"
        self.name_target = self.name_cc + "_SM"
        self.devices: List[int] = []

    def append(self, sm: int):
        self.devices.append(sm)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += (
            f"#define {self.name_cc}(cc, dtype, is_causal, has_rpb, cb) \\\n"
        )
        dispatcher_str += "  [&] { \\\n"
        for i, sm in enumerate(self.devices):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += (
                f"if (cc < {_SM_TO_UPPER_BOUND[sm]} && cc >= {_SM_TO_LOWER_BOUND[sm]})"
            )
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += (
                f"  {self.name_target}{sm}(dtype, is_causal, has_rpb, cb); \\\n"
            )
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "NATTEN FNA kernel launch failed!" \\\n'
        dispatcher_str += (
            '                << "'
            + "Fused neighborhood attention is not implemented for this device."
            + '" \\\n'
        )
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();"
        dispatcher_str += "\n\n"
        return dispatcher_str


class DataTypeDispatcher:
    def __init__(self, na_dim: int, sm: int):
        self.dtypes: List[DataType] = []
        self.na_dim = na_dim
        self.sm = sm
        self.name_cc = f"DISPATCH_FNA_FORWARD_{self.na_dim}D_SM{self.sm}"
        self.name_target = self.name_cc

    def append(self, dtype: DataType):
        self.dtypes.append(dtype)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(dtype, is_causal, has_rpb, cb) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, dtype in enumerate(self.dtypes):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += (
                f"if constexpr (std::is_same<dtype, {dtype.natten_name}>::value)"
            )
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += (
                f"  {self.name_target}_{dtype.short_name}(is_causal, has_rpb, cb); \\\n"
            )
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "NATTEN kernel launch failed!" \\\n'
        dispatcher_str += (
            '                << "'
            + f"FNA-{self.na_dim}D does not support this data type on SM{self.sm}."
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
        sm: int,
        dtype: DataType,
    ):
        self.na_dim = na_dim
        self.sm = sm
        self.dtype = dtype
        self.name_cc = (
            f"DISPATCH_FNA_FORWARD_{self.na_dim}D_SM{self.sm}_{self.dtype.short_name}"
        )
        self.name_target = f"fna{self.na_dim}_sm{self.sm}_{self.dtype.short_name}"
        self.cms: List = []

    def append(self, cm):
        assert len(cm) == self.na_dim
        self.cms.append(cm)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(is_causal, has_rpb, cb) \\\n"
        dispatcher_str += "  [&] { \\\n"
        i = 0
        for cm in self.cms:
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            i += 1
            cm_str = "_".join([str(int(x)) for x in cm])
            dispatcher_str += "if ("
            for dim in range(self.na_dim):
                dispatcher_str += (
                    f"std::get<{dim}>(is_causal)"
                    if cm[dim]
                    else f"!std::get<{dim}>(is_causal)"
                )
                if dim != self.na_dim - 1:
                    dispatcher_str += " && "
            dispatcher_str += ")"
            dispatcher_str += " { \\\n"
            if not any(cm):
                dispatcher_str += " if (has_rpb) {\\\n"
                dispatcher_str += f"  {self.name_target}_cm_{cm_str}_rpb(cb); \\\n"
                dispatcher_str += " } else {\\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name_target}_cm_{cm_str}(cb); \\\n"
            if not any(cm):
                dispatcher_str += " } \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += "    "
        dispatcher_str += '      std::cerr << "NATTEN FNA kernel launch failed!" \\\n'
        dispatcher_str += (
            '                << "'
            + f"Causal mask dispatcher (FNA-{self.na_dim}D, SM{self.sm}, {self.dtype.short_name}) got invalid causal mask!"
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
        "\n",
    ]
    with open(path, "w") as f:
        f.write("".join(header_head))
        f.write(content)
        f.write("".join(header_foot))


def generate_cuda_kernels(path, num_splits=2):

    CUDA_DTYPES = [
        NATTEN_Float,
        NATTEN_Half,
        NATTEN_BFloat,
    ]

    CAUSAL_MASKS = {
        1: [[0], [1]],
        2: [[0, 0], [0, 1], [1, 0], [1, 1]],
        3: [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
    }

    SUPPORTED_ARCHS = [50, 70, 75, 80]

    GEMM_SHAPES = [
        # K = 32
        (32, 128, 32),
        (64, 64, 32),
        (64, 128, 32),
        # K = 64
        (32, 128, 64),
        (64, 64, 64),
        (64, 128, 64),
        # K = 128
        (32, 128, 128),
        (64, 64, 128),
        (64, 128, 128),
        # K = 65536
        (32, 128, 2**16),
        (64, 128, 2**16),
        (64, 64, 2**16),
    ]
    device_dispatchers = []
    dtype_dispatchers = []
    cm_dispatchers = []
    kernels = []

    rank_dispatcher = RankDispatcher()
    for na_dim in [1, 2, 3]:
        rank_dispatcher.append(na_dim)
        device_dispatcher = DeviceDispatcher(na_dim=na_dim)
        for sm in SUPPORTED_ARCHS:
            device_dispatcher.append(sm)
            dtype_dispatcher = DataTypeDispatcher(na_dim=na_dim, sm=sm)
            for dtype in CUDA_DTYPES:
                if dtype.min_sm > sm:
                    continue
                dtype_dispatcher.append(dtype)
                cm_dispatcher = CausalMaskDispatcher(na_dim=na_dim, sm=sm, dtype=dtype)
                for cm in CAUSAL_MASKS[na_dim]:
                    cm_dispatcher.append(cm)
                    new_kernel = FusedNAKernelBundle(
                        na_dim=na_dim,
                        sm=sm,
                        dtype=dtype,
                        config_list=KernelConfigList(
                            na_dim=na_dim,
                            sm=sm,
                            gemm_shapes=GEMM_SHAPES,  # GEMM_SHAPES[na_dim][sm]
                        ),
                        causal_mask=cm,
                    )
                    kernels.append(new_kernel)
                    if not any(cm):
                        new_kernel_with_rpb_support = FusedNAKernelBundle(
                            na_dim=na_dim,
                            sm=sm,
                            dtype=dtype,
                            config_list=KernelConfigList(
                                na_dim=na_dim,
                                sm=sm,
                                gemm_shapes=GEMM_SHAPES,  # GEMM_SHAPES[na_dim][sm]
                            ),
                            causal_mask=cm,
                            has_rpb=True,
                        )
                        kernels.append(new_kernel_with_rpb_support)
                cm_dispatchers.append(cm_dispatcher)
            dtype_dispatchers.append(dtype_dispatcher)
        device_dispatchers.append(device_dispatcher)

    #

    path_to_sources = f"{path}/autogen/src/cuda/fna/"
    rel_header = "natten_autogen/cuda/fna/"
    path_to_header_dir = f"{path}/autogen/include/{rel_header}"

    os.makedirs(path_to_sources, exist_ok=False)
    os.makedirs(path_to_header_dir, exist_ok=False)

    path_headers = f"{path_to_header_dir}kernels.h"
    path_rank = f"{path_to_header_dir}interface.h"
    path_device = f"{path_to_header_dir}dispatch_device.h"
    path_dtype = f"{path_to_header_dir}dispatch_dtype.h"
    path_cm = f"{path_to_header_dir}dispatch_cm.h"

    rel_path_headers = f"{rel_header}kernels.h"
    rel_path_device = f"{rel_header}dispatch_device.h"
    rel_path_dtype = f"{rel_header}dispatch_dtype.h"
    rel_path_cm = f"{rel_header}dispatch_cm.h"

    rank_disp = rank_dispatcher.get_dispatcher()

    device_disp = ""
    for dispatcher in device_dispatchers:
        device_disp += dispatcher.get_dispatcher()

    dtype_disp = ""
    for dispatcher in dtype_dispatchers:
        dtype_disp += dispatcher.get_dispatcher()

    cm_disp = ""
    for dispatcher in cm_dispatchers:
        cm_disp += dispatcher.get_dispatcher()

    headers = ""
    for kernel in kernels:
        headers += kernel.header()

    split_size = (len(kernels) + num_splits - 1) // num_splits
    kernels_emitted = []
    for split_idx in range(num_splits):
        kernel_start_idx = split_size * split_idx
        kernel_end_idx = min(kernel_start_idx + split_size, len(kernels))
        pth_set = set()
        source_list = []
        for kernel_idx in range(kernel_start_idx, kernel_end_idx):
            kernel = kernels[kernel_idx]
            pth_set.add(kernel.path_to_header)
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

    namespaces = ["natten", "cuda", "fna"]
    cuda_headers = [
        "natten/natten.h",
        "natten/dtypes.cuh",
        "natten/cuda/fna/na_utils.cuh",
        "natten/cuda/fna/kernel_forward.h",
    ]
    write_header_file(
        rank_disp, path_rank, namespaces, cuda_headers + [rel_path_device]
    )
    write_header_file(
        device_disp, path_device, namespaces, cuda_headers + [rel_path_dtype]
    )
    write_header_file(dtype_disp, path_dtype, namespaces, cuda_headers + [rel_path_cm])
    write_header_file(cm_disp, path_cm, namespaces, cuda_headers + [rel_path_headers])
    # write_header_file(ks_disp, path_ks, namespaces, cuda_headers + [rel_path_di])
    # write_header_file(di_disp, path_di, namespaces, cuda_headers + [rel_path_headers])
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
    default=16,
    help="Number of source files into which the kernels are split. Default: 16.",
)
def generate_cuda_fused(output_directory: str, num_splits: int):
    generate_cuda_kernels(output_directory, num_splits=num_splits)


if __name__ == "__main__":
    generate_cuda_fused()
