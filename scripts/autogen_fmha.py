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
import os
from typing import List, Tuple


DEFAULT_OUTPUT_DIR = "csrc/"

_SM_TO_UPPER_BOUND = {
    50: 70,
    70: 75,
    75: 80,
    80: None,
}

_SM_TO_LOWER_BOUND = {
    50: 50,
    70: 70,
    75: 75,
    80: 80,
}


class KernelConfig:
    def __init__(self, sm: int, gemm_shape: Tuple[int, int, int]):
        assert 50 <= sm <= 80
        assert len(gemm_shape) == 3
        self.sm = sm
        self.gemm_shape = gemm_shape

    def get_name(self, is_backward: bool) -> str:
        backward_str = "" if not is_backward else "_backward"
        return f"fmha{backward_str}_{self.gemm_shape[0]}x{self.gemm_shape[1]}x{self.gemm_shape[2]}_sm{self.sm}"


class KernelConfigList:
    def __init__(self, sm: int, gemm_shapes: List[Tuple[int, int, int]]):
        assert 50 <= sm <= 80
        self.sm = sm
        self.gemm_shapes = gemm_shapes

    @property
    def configs(self) -> List[KernelConfig]:
        return [
            KernelConfig(sm=self.sm, gemm_shape=gemm_shape)
            for gemm_shape in self.gemm_shapes
        ]

    def get_name(self, is_backward: bool) -> str:
        backward_str = "" if not is_backward else "_backward"
        return f"fmha{backward_str}_sm{self.sm}"


class DataType:
    def __init__(self, name, torch_name, short_name, bits, min_sm):
        self.name = name
        self.torch_name = torch_name
        self.bits = bits
        self.short_name = short_name
        self.min_sm = min_sm


NATTEN_Float = DataType("float", "torch::kFloat32", "float32", 32, 0)
NATTEN_Half = DataType("cutlass::half_t", "torch::kFloat16", "float16", 16, 50)
NATTEN_BFloat = DataType("cutlass::bfloat16_t", "torch::kBFloat16", "bfloat16", 16, 80)


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
        "FATAL: fmha kernel `{NAME}` was built for SM{SM}, but attempted to launch from SM%d\\n",
        int(__CUDA_ARCH__ + 0) / 10);
#endif
}}
"""

SM_TO_SM_MAX = {
    50: 70,
    70: 75,
    75: 80,
    80: 130,
}


class FusedNAKernel:
    def __init__(
        self,
        is_backward: bool,
        dtype: DataType,
        config: KernelConfig,
    ):
        self.dtype = dtype
        self.config = config
        self.name_cc = f"{self.config.get_name(is_backward)}_{dtype.short_name}"
        self.path_to_header = (
            "natten/cuda/fmha/kernel_forward.h"
            if not is_backward
            else "natten/cuda/fmha/kernel_backward.h"
        )
        self.aligned = True
        self.is_backward = is_backward

    @property
    def cpp_class(self) -> str:
        if self.is_backward:
            template_args = ", ".join(
                [
                    self.dtype.name,
                    f"cutlass::arch::Sm{self.config.sm}",
                    "true" if self.aligned else "false",
                    # "false", # preload
                    str(self.config.gemm_shape[0]),
                    str(self.config.gemm_shape[1]),
                    str(self.config.gemm_shape[2]),
                    "false",  # QKV divides tile sizes
                ]
            )
            return f"AttentionBackwardKernel<{template_args}>"
        else:
            template_args = ", ".join(
                [
                    self.dtype.name,
                    f"cutlass::arch::Sm{self.config.sm}",
                    "true" if self.aligned else "false",
                    str(self.config.gemm_shape[0]),
                    str(self.config.gemm_shape[1]),
                    str(self.config.gemm_shape[2]),
                ]
            )
            return f"AttentionKernel<{template_args}>"

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
        is_backward: bool,
        sm: int,
        dtype: DataType,
        config_list: KernelConfigList,
    ):
        self.sm = sm
        self.dtype = dtype
        self.config_list = config_list
        self.name_cc = f"{self.config_list.get_name(is_backward)}_{dtype.short_name}"
        self.path_to_header = (
            "natten/cuda/fmha/kernel_forward.h"
            if not is_backward
            else "natten/cuda/fmha/kernel_backward.h"
        )
        self.aligned = True
        self.is_backward = is_backward

    @property
    def kernels(self):
        return [
            FusedNAKernel(
                is_backward=self.is_backward,
                dtype=self.dtype,
                config=config,
            )
            for config in self.config_list.configs
        ]

    def header(self):
        decl = "\n\n"
        decl += "///////////////////////////////////////////////////////////////////\n"
        decl += f"// FMHA / {self.dtype.short_name} / SM{self.sm}"
        decl += "Backward Kernel" if self.is_backward else ""
        decl += "\n"
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
        impl += f"// FMHA / {self.dtype.short_name} / SM{self.sm}\n"
        impl += "///////////////////////////////////////////////////////////////////"

        for kernel in self.kernels:
            impl += "\n\n"
            impl += kernel.source()
        return impl


def write_combined_source_file(path, filename, headers, sources):
    source_head = []
    source_head += ["#include <cuda_runtime.h>\n"]
    source_head += ["#include <iostream>\n"]

    for header in headers:
        source_head += [f"#include <{header}>\n"]

    source_head += ["namespace natten { \n"]
    source_head += ["namespace cuda { \n"]
    source_head += ["namespace fmha { \n\n"]

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


class DeviceDispatcher:
    def __init__(self, is_backward: bool):
        self.name_cc = (
            "DISPATCH_FMHA_FORWARD" if not is_backward else "DISPATCH_FMHA_BACKWARD"
        )
        self.name_target = self.name_cc + "_SM"
        self.devices: List[int] = []
        self.is_backward = is_backward

    def append(self, sm: int):
        self.devices.append(sm)

    def get_dispatcher(self):
        dispatcher_str = ""
        if self.is_backward:
            dispatcher_str += f"#define {self.name_cc}(cc, dtype, cb) \\\n"
        else:
            dispatcher_str += f"#define {self.name_cc}(cc, dtype, cb) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, sm in enumerate(self.devices):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += "if ("
            if _SM_TO_UPPER_BOUND[sm] is not None:
                dispatcher_str += f"cc < {_SM_TO_UPPER_BOUND[sm]} && "
            dispatcher_str += f"cc >= {_SM_TO_LOWER_BOUND[sm]})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            if self.is_backward:
                dispatcher_str += f"  {self.name_target}{sm}(dtype, cb); \\\n"
            else:
                dispatcher_str += f"  {self.name_target}{sm}(dtype, cb); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "NATTEN FMHA kernel launch failed!" \\\n'
        dispatcher_str += (
            '                << "'
            + "FMHA is not implemented for this device."
            + '" \\\n'
        )
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();"
        dispatcher_str += "\n\n"
        return dispatcher_str


class DataTypeDispatcher:
    def __init__(self, is_backward: bool, sm: int):
        self.dtypes: List[DataType] = []
        self.sm = sm
        self.name_cc = (
            f"DISPATCH_FMHA_FORWARD_SM{self.sm}"
            if not is_backward
            else f"DISPATCH_FMHA_BACKWARD_SM{self.sm}"
        )
        backward_str = "" if not is_backward else "_backward"
        self.name_target = f"fmha{backward_str}_sm{self.sm}"
        self.is_backward = is_backward

    def append(self, dtype: DataType):
        self.dtypes.append(dtype)

    def get_dispatcher(self):
        dispatcher_str = ""
        if self.is_backward:
            dispatcher_str += f"#define {self.name_cc}(dtype, cb) \\\n"
        else:
            dispatcher_str += f"#define {self.name_cc}(dtype, cb) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, dtype in enumerate(self.dtypes):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (dtype == {dtype.torch_name})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            if self.is_backward:
                dispatcher_str += f"  {self.name_target}_{dtype.short_name}(cb); \\\n"
            else:
                dispatcher_str += f"  {self.name_target}_{dtype.short_name}(cb); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "NATTEN kernel launch failed!" \\\n'
        dispatcher_str += (
            '                << "'
            + f"FMHA does not support this data type on SM{self.sm}."
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

    SUPPORTED_ARCHS = [50, 70, 75, 80]

    CUDA_DTYPES = [
        NATTEN_Float,
        NATTEN_Half,
        NATTEN_BFloat,
    ]

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
    dtype_dispatchers = []
    kernels = []

    device_dispatcher = DeviceDispatcher(is_backward=False)
    for sm in SUPPORTED_ARCHS:
        device_dispatcher.append(sm)
        dtype_dispatcher = DataTypeDispatcher(is_backward=False, sm=sm)
        for dtype in CUDA_DTYPES:
            if dtype.min_sm > sm:
                continue
            dtype_dispatcher.append(dtype)
            kernels.append(
                FusedNAKernelBundle(
                    is_backward=False,
                    sm=sm,
                    dtype=dtype,
                    config_list=KernelConfigList(
                        sm=sm,
                        gemm_shapes=GEMM_SHAPES,
                    ),
                )
            )
        dtype_dispatchers.append(dtype_dispatcher)

    #

    BACKWARD_GEMM_SHAPES = {
        80: {
            NATTEN_Half: [
                # K = 32
                (64, 64, 32),
                # K = 64
                (64, 64, 64),
                # K = 128
                (64, 64, 128),
                (128, 128, 128),  # SM80/SM90 only
                # K = 65536
                (64, 64, 2**16),
                (128, 64, 2**16),  # SM80/SM90 only
            ],
            NATTEN_BFloat: [
                # K = 32
                (64, 64, 32),
                # K = 64
                (64, 64, 64),
                # K = 128
                (64, 64, 128),
                (128, 128, 128),  # SM80/SM90 only
                # K = 65536
                (64, 64, 2**16),
                (128, 64, 2**16),  # SM80/SM90 only
            ],
            NATTEN_Float: [
                # K = 32
                (64, 64, 32),
                # K = 64
                (64, 64, 64),
                # K = 128
                (64, 64, 128),
                (128, 64, 128),  # SM80/SM90 only
                # K = 65536
                (64, 64, 2**16),
                (128, 64, 2**16),  # SM80/SM90 only
            ],
        },
        75: {
            NATTEN_Half: [
                # K = 32
                (64, 64, 32),
                # K = 64
                (64, 64, 64),
                # K = 128
                (64, 64, 128),
                # K = 65536
                (64, 64, 2**16),
            ],
            NATTEN_Float: [
                # K = 32
                (64, 64, 32),
                # K = 64
                (64, 64, 64),
                # K = 128
                (64, 64, 128),
                # K = 65536
                (64, 64, 2**16),
            ],
        },
        70: {
            NATTEN_Half: [
                # K = 32
                (64, 64, 32),
                # K = 64
                (64, 64, 64),
                # K = 128
                (64, 64, 128),
                (128, 64, 128),  # SM70 only?
                # K = 65536
                (64, 64, 2**16),
                (128, 64, 2**16),  # SM70 only?
            ],
            NATTEN_Float: [
                # K = 32
                (64, 64, 32),
                # K = 64
                (64, 64, 64),
                # K = 128
                (64, 64, 128),
                # K = 65536
                (64, 64, 2**16),
            ],
        },
        50: {
            NATTEN_Half: [
                # K = 32
                (64, 64, 32),
                # K = 64
                (64, 64, 64),
                # K = 128
                (64, 64, 128),
                # K = 65536
                (64, 64, 2**16),
            ],
            NATTEN_Float: [
                # K = 32
                (64, 64, 32),
                # K = 64
                (64, 64, 64),
                # K = 128
                (64, 64, 128),
                # K = 65536
                (64, 64, 2**16),
            ],
        },
    }

    # Backward kernels
    device_dispatcher_bwd = DeviceDispatcher(is_backward=True)
    for sm in SUPPORTED_ARCHS:
        device_dispatcher_bwd.append(sm)
        dtype_dispatcher = DataTypeDispatcher(is_backward=True, sm=sm)
        for dtype in CUDA_DTYPES:
            if dtype.min_sm > sm:
                continue
            dtype_dispatcher.append(dtype)
            kernels.append(
                FusedNAKernelBundle(
                    is_backward=True,
                    sm=sm,
                    dtype=dtype,
                    config_list=KernelConfigList(
                        sm=sm,
                        gemm_shapes=BACKWARD_GEMM_SHAPES[sm][dtype],
                    ),
                )
            )
        dtype_dispatchers.append(dtype_dispatcher)

    #

    path_to_sources = f"{path}/autogen/src/cuda/fmha/"
    rel_header = "natten_autogen/cuda/fmha/"
    path_to_header_dir = f"{path}/autogen/include/{rel_header}"

    os.makedirs(path_to_sources, exist_ok=False)
    os.makedirs(path_to_header_dir, exist_ok=False)

    path_headers = f"{path_to_header_dir}kernels.h"
    path_device = f"{path_to_header_dir}interface.h"
    path_dtype = f"{path_to_header_dir}dispatch_dtype.h"

    rel_path_headers = f"{rel_header}kernels.h"
    rel_path_dtype = f"{rel_header}dispatch_dtype.h"

    device_disp = (
        device_dispatcher.get_dispatcher() + device_dispatcher_bwd.get_dispatcher()
    )

    dtype_disp = ""
    for dispatcher in dtype_dispatchers:
        dtype_disp += dispatcher.get_dispatcher()

    headers = ""
    for kernel in kernels:
        headers += kernel.header()

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
            pth_set.add(kernel.path_to_header)
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

    namespaces = ["natten", "cuda", "fmha"]
    cuda_headers = [
        "natten/natten.h",
        "natten/cuda/fmha/kernel_forward.h",
        "natten/cuda/fmha/kernel_backward.h",
    ]
    write_header_file(
        device_disp, path_device, namespaces, cuda_headers + [rel_path_dtype]
    )
    write_header_file(
        dtype_disp, path_dtype, namespaces, cuda_headers + [rel_path_headers]
    )
    write_header_file(headers, path_headers, namespaces, cuda_headers)


def generate_cuda_fused(output_directory: str, num_splits: int):
    generate_cuda_kernels(output_directory, num_splits=num_splits)


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
        default=12,
        help="Number of source files into which the kernels are split. Default: 12.",
    )
    args = parser.parse_args()
    generate_cuda_fused(args.output_directory, args.num_splits)
