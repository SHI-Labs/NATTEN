# Copyright (c) 2022-2024 Ali Hassani.
#
# This script is intended to emit naive kernel instantiations into
# a variable number of source files, generate appropriate headers
# and a single dispatcher interface, which will be used by the
# NATTEN API to call the kernels.
#
# NOTE: these scripts are heavily under-documented, and
# overly-repetitive, and will be replaced in future PRs.
# Please use it with caution.

import os
from enum import Enum
from itertools import product
from typing import List

import click


DEFAULT_OUTPUT_DIR = "csrc/"


class Operation(Enum):
    PN = 0
    PN_BIAS = 1
    NN = 2
    IN = 3
    RPBGRAD = 4


ALL_OPS = [
    Operation.PN,
    Operation.PN_BIAS,
    Operation.NN,
    Operation.IN,
    Operation.RPBGRAD,
]


class Problem(Enum):
    NA1D = 0
    NA2D = 1
    NA3D = 2


def naive_op_to_filename(operation: Operation, dim: Problem):
    out = ""
    if operation in [Operation.PN, Operation.PN_BIAS]:
        out += "pointwise_neighborhood"
    elif operation in [Operation.NN]:
        out += "neighborhood_neighborhood"
    elif operation in [Operation.IN]:
        out += "inverse_neighborhood"
    elif operation in [Operation.RPBGRAD]:
        out += "rel_pos_bias"
    else:
        raise ValueError()

    if dim == Problem.NA1D:
        out += "_1d"
    elif dim == Problem.NA2D:
        out += "_2d"
    elif dim == Problem.NA3D:
        out += "_3d"
    else:
        raise ValueError()

    return out


def naive_op_to_template_name(operation: Operation, dim: Problem):
    out = ""
    if operation in [Operation.PN, Operation.PN_BIAS]:
        out += "PointwiseNeighborhood"
    elif operation in [Operation.NN]:
        out += "NeighborhoodNeighborhood"
    elif operation in [Operation.IN]:
        out += "InverseNeighborhood"
    elif operation in [Operation.RPBGRAD]:
        out += "RelPosBiasGradient"
    else:
        raise ValueError()

    if dim == Problem.NA1D:
        out += "1D"
    elif dim == Problem.NA2D:
        out += "2D"
    elif dim == Problem.NA3D:
        out += "3D"
    else:
        raise ValueError()

    if operation == Operation.PN_BIAS:
        out += "WithBias"

    return out


ALL_SIZES = [
    Problem.NA1D,
    Problem.NA2D,
    Problem.NA3D,
]


dim_to_cc = {
    Problem.NA1D: "1d",
    Problem.NA2D: "2d",
    Problem.NA3D: "3d",
}


class DataType:
    def __init__(self, name, natten_dtype, bits):
        self.name = name
        self.natten_dtype = natten_dtype
        self.bits = bits


NATTEN_Double = DataType("double", "natten::float64", 64)
NATTEN_Float = DataType("float", "natten::float32", 32)
NATTEN_Half = DataType("half", "natten::float16", 16)
NATTEN_BFloat = DataType("bfloat16", "natten::bfloat16", 16)

#####################################################################################


class CArg:
    def __init__(self, dtype: str, name: str):
        self.dtype = dtype
        self.name = name

    def __str__(self):
        return f"{self.dtype} {self.name}"


CC_ARG = [
    CArg("const int", "cc"),
    CArg("cudaStream_t", "stream"),
]

COMMON_ARGS = [
    CArg("int", "kernel_size"),
    CArg("int", "dilation"),
]

EXTRA_3D_ARGS = [
    CArg("int", "kernel_size_d"),
    CArg("int", "dilation_d"),
]

PN_COMMON_ARGS = [
    CArg("void *", "query_ptr"),
    CArg("void *", "key_ptr"),
    CArg("void *", "attn_ptr"),
]

PN_BIAS_COMMON_ARGS = [
    CArg("void *", "query_ptr"),
    CArg("void *", "key_ptr"),
    CArg("void *", "bias_ptr"),
    CArg("void *", "attn_ptr"),
]

NN_COMMON_ARGS = [
    CArg("void *", "attn_ptr"),
    CArg("void *", "value_ptr"),
    CArg("void *", "output_ptr"),
]

IN_COMMON_ARGS = [
    CArg("void *", "attn_ptr"),
    CArg("void *", "d_output_ptr"),
    CArg("void *", "d_value_ptr"),
]

RPBGRAD_COMMON_ARGS = [
    CArg("void *", "d_bias_ptr"),
    CArg("void *", "d_attn_ptr"),
]

NA1D_PROBLEM_SIZE_ARGS = [
    CArg("int", "batch_size"),
    CArg("int", "heads"),
    CArg("int", "length"),
    CArg("int", "dim"),
    CArg("int64_t", "attn_stride_0"),
    CArg("int64_t", "attn_stride_1"),
    CArg("int64_t", "attn_stride_2"),
]

NA2D_PROBLEM_SIZE_ARGS = [
    CArg("int", "batch_size"),
    CArg("int", "heads"),
    CArg("int", "height"),
    CArg("int", "width"),
    CArg("int", "dim"),
    CArg("int64_t", "attn_stride_0"),
    CArg("int64_t", "attn_stride_1"),
    CArg("int64_t", "attn_stride_2"),
    CArg("int64_t", "attn_stride_3"),
]

NA3D_PROBLEM_SIZE_ARGS = [
    CArg("int", "batch_size"),
    CArg("int", "heads"),
    CArg("int", "depth"),
    CArg("int", "height"),
    CArg("int", "width"),
    CArg("int", "dim"),
    CArg("int64_t", "attn_stride_0"),
    CArg("int64_t", "attn_stride_1"),
    CArg("int64_t", "attn_stride_2"),
    CArg("int64_t", "attn_stride_3"),
    CArg("int64_t", "attn_stride_4"),
]


carg_map = {
    Problem.NA1D: {
        Operation.PN: CC_ARG + PN_COMMON_ARGS + NA1D_PROBLEM_SIZE_ARGS + COMMON_ARGS,
        Operation.PN_BIAS: (
            CC_ARG + PN_BIAS_COMMON_ARGS + NA1D_PROBLEM_SIZE_ARGS + COMMON_ARGS
        ),
        Operation.NN: CC_ARG + NN_COMMON_ARGS + NA1D_PROBLEM_SIZE_ARGS + COMMON_ARGS,
        Operation.IN: CC_ARG + IN_COMMON_ARGS + NA1D_PROBLEM_SIZE_ARGS + COMMON_ARGS,
        Operation.RPBGRAD: (
            CC_ARG + RPBGRAD_COMMON_ARGS + NA1D_PROBLEM_SIZE_ARGS + COMMON_ARGS
        ),
    },
    Problem.NA2D: {
        Operation.PN: CC_ARG + PN_COMMON_ARGS + NA2D_PROBLEM_SIZE_ARGS + COMMON_ARGS,
        Operation.PN_BIAS: (
            CC_ARG + PN_BIAS_COMMON_ARGS + NA2D_PROBLEM_SIZE_ARGS + COMMON_ARGS
        ),
        Operation.NN: CC_ARG + NN_COMMON_ARGS + NA2D_PROBLEM_SIZE_ARGS + COMMON_ARGS,
        Operation.IN: CC_ARG + IN_COMMON_ARGS + NA2D_PROBLEM_SIZE_ARGS + COMMON_ARGS,
        Operation.RPBGRAD: (
            CC_ARG + RPBGRAD_COMMON_ARGS + NA2D_PROBLEM_SIZE_ARGS + COMMON_ARGS
        ),
    },
    Problem.NA3D: {
        Operation.PN: (
            CC_ARG
            + PN_COMMON_ARGS
            + NA3D_PROBLEM_SIZE_ARGS
            + COMMON_ARGS
            + EXTRA_3D_ARGS
        ),
        Operation.PN_BIAS: (
            CC_ARG
            + PN_BIAS_COMMON_ARGS
            + NA3D_PROBLEM_SIZE_ARGS
            + COMMON_ARGS
            + EXTRA_3D_ARGS
        ),
        Operation.NN: (
            CC_ARG
            + NN_COMMON_ARGS
            + NA3D_PROBLEM_SIZE_ARGS
            + COMMON_ARGS
            + EXTRA_3D_ARGS
        ),
        Operation.IN: (
            CC_ARG
            + IN_COMMON_ARGS
            + NA3D_PROBLEM_SIZE_ARGS
            + COMMON_ARGS
            + EXTRA_3D_ARGS
        ),
        Operation.RPBGRAD: (
            CC_ARG
            + RPBGRAD_COMMON_ARGS
            + NA3D_PROBLEM_SIZE_ARGS
            + COMMON_ARGS
            + EXTRA_3D_ARGS
        ),
    },
}

#####################################################################################


class TemplateIntParameter:
    def __init__(self, value: int):
        self.value = value

    def is_default(self):
        return self.value <= 0

    def __str__(self):
        return f"{self.value}" if self.value > 0 else "any"


class NaiveNAKernel:
    def __init__(
        self,
        dtype: DataType,
        operation: Operation,
        dim: Problem,
        kernel_size: TemplateIntParameter,
        dilation: TemplateIntParameter,
    ):
        self.dtype = dtype
        self.operation = operation
        self.dim = dim
        self.arguments = carg_map[dim][operation]
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.name_cc = (
            f"na{dim_to_cc[dim]}_{operation.name.lower()}_cuda_naive_{dtype.name}"
        )
        self.name_cc += f"_ks_{self.kernel_size}"
        self.name_cc += f"_di_{self.dilation}"
        self.template_name = naive_op_to_template_name(operation, dim)

    @property
    def path_to_header(self):
        pth = ""
        pth += "natten/cuda/"
        pth += "naive/"
        pth += f"{naive_op_to_filename(self.operation, self.dim)}"
        pth += ".cuh"
        return pth

    @property
    def filename(self):
        pth = ""
        pth += f"{self.name_cc}"
        pth += ".cu"
        return pth

    def parameters(self):
        def_str = ""
        for i, arg in enumerate(self.arguments):
            def_str += f"{arg.name}"
            if i < len(self.arguments) - 1:
                def_str += ", "
        return def_str

    def argpack_parameters(self):
        params_str = ""
        params_str += f"{self.dtype.natten_dtype}, "
        params_str += f"{self.kernel_size.value}, {self.dilation.value}"
        return params_str

    def method_decl(self):
        def_str = f"void {self.name_cc}(\n"
        for i, arg in enumerate(self.arguments):
            def_str += f"  {arg}"
            if i < len(self.arguments) - 1:
                def_str += ",\n"
            else:
                def_str += ")"
                break
        return def_str

    def method_def(self):
        launch_str = ""
        launch_str += f"  using Arguments = natten::naive::ArgumentPack<{self.argpack_parameters()}>;\n"
        launch_str += f"  using Kernel = {self.template_name}<Arguments>;\n"
        launch_str += "  Kernel kernel;\n"
        launch_str += "  kernel(\n"
        launch_str += self.parameters()
        launch_str += ");\n"
        return launch_str

    def header(self):
        header_str = ""
        header_str += self.method_decl()
        header_str += ";\n\n"
        return header_str

    def source(self):
        source_str = ""
        source_str += self.method_decl()
        source_str += " {\n"
        if self.dtype.name == "half":
            source_str += "\nif(cc >= 60) {\n"
        elif self.dtype.name == "bfloat16":
            source_str += "\nif(cc >= 80) {\n"
        source_str += self.method_def()
        if self.dtype.name in ["half", "bfloat16"]:
            source_str += "\n} else {\n"
            source_str += (
                'std::cerr << "This half type is not supported on the selected device.'
                + '"'
            )
            source_str += "  << std::endl; \n"
            source_str += "exit(EXIT_FAILURE); \n"
            source_str += "\n}\n"
        source_str += "}\n\n"
        return source_str

    def write_source_file(self, path):
        source_head = []
        source_head += ["#include <cuda_runtime.h>\n"]
        source_head += ["#include <iostream>\n"]
        source_head += ["#include <natten/dtypes.cuh>\n"]
        source_head += ["#include <natten/naive_argpack.h>\n"]

        source_head += [f"#include <{self.path_to_header}>\n"]

        source_head += ["namespace natten { \n"]
        source_head += ["namespace cuda { \n"]
        source_head += ["namespace naive { \n"]

        source_head = "".join(source_head)
        source_body = self.source()

        source_foot = "".join(
            [
                "} \n",
                "} \n",
                "} \n",
                "\n",
            ]
        )
        filename = f"{path}/{self.filename}"
        with open(filename, "w") as f:
            f.write(source_head)
            f.write(source_body)
            f.write(source_foot)


def write_combined_source_file(path, filename, headers, sources):
    source_head = []
    source_head += ["#include <cuda_runtime.h>\n"]
    source_head += ["#include <iostream>\n"]
    source_head += ["#include <natten/dtypes.cuh>\n"]
    source_head += ["#include <natten/naive_argpack.h>\n"]

    for header in headers:
        source_head += [f"#include <{header}>\n"]

    source_head += ["namespace natten { \n"]
    source_head += ["namespace cuda { \n"]
    source_head += ["namespace naive { \n\n"]

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


class DataTypeDispatcher:
    def __init__(
        self,
        operation: Operation,
        dim: Problem,
    ):
        self.dtypes: List[DataType] = []
        self.operation = operation
        self.dim = dim
        self.name_base = f"na{dim_to_cc[dim]}_{operation.name.lower()}_cuda_naive"
        self.name_cc = f"DISPATCH_DTYPE_{self.name_base}"
        self.name_target = f"DISPATCH_KERNEL_{self.name_base}"

    def append(self, dtype: DataType):
        self.dtypes.append(dtype)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += (
            f"#define {self.name_cc}(dtype, kernel_size, dilation, ...) \\\n"
        )
        dispatcher_str += "  [&] { \\\n"
        for i, dtype in enumerate(self.dtypes):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (std::is_same<dtype, {dtype.natten_dtype}>::value)"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name_target}_{dtype.name}(kernel_size, dilation, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "NATTEN kernel launch failed!" \\\n'
        dispatcher_str += (
            '                << "'
            + f"{self.name_base} does not support this data type."
            + '" \\\n'
        )
        dispatcher_str += "                << std::endl; \\\n"
        dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();"
        dispatcher_str += "\n\n"
        return dispatcher_str


class KernelSizeDispatcher:
    def __init__(
        self,
        dtype: DataType,
        operation: Operation,
        dim: Problem,
    ):
        self.dtype = dtype
        self.operation = operation
        self.dim = dim
        self.name_base = f"na{dim_to_cc[dim]}_{operation.name.lower()}_cuda_naive"
        self.name_base += f"_{dtype.name}"
        self.name_cc = f"DISPATCH_KERNEL_{self.name_base}"
        self.name_target = f"DISPATCH_DILATION_{self.name_base}"
        self.kernels: List[TemplateIntParameter] = []

    def append(self, kernel: TemplateIntParameter):
        self.kernels.append(kernel)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(kernel_size, dilation, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        i = 0
        for ks in self.kernels:
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            if ks.is_default():
                # Skip the default case, we'll handle that in the else
                continue
            i += 1
            dispatcher_str += f"if (kernel_size == {ks.value})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += (
                f"  {self.name_target}_ks_{ks}(dilation, __VA_ARGS__); \\\n"
            )
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += "    "
        dispatcher_str += f"  {self.name_target}_ks_any(dilation, __VA_ARGS__); \\\n"
        dispatcher_str += "    } \\\n"
        dispatcher_str += "}();"
        dispatcher_str += "\n\n"
        return dispatcher_str


class DilationDispatcher:
    def __init__(
        self,
        dtype: DataType,
        kernel_size: TemplateIntParameter,
        operation: Operation,
        dim: Problem,
    ):
        self.dtype = dtype
        self.operation = operation
        self.dim = dim
        self.kernel_size = kernel_size
        self.name_base = f"na{dim_to_cc[dim]}_{operation.name.lower()}_cuda_naive"
        self.name_base += f"_{dtype.name}_ks_{kernel_size}"
        self.name_cc = f"DISPATCH_DILATION_{self.name_base}"
        self.name_target = f"naive::{self.name_base}"  # prepends namespace
        self.dilations: List[TemplateIntParameter] = []

    def append(self, dilation: TemplateIntParameter):
        self.dilations.append(dilation)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(dilation, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        i = 0
        for di in self.dilations:
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            if di.is_default():
                # Skip the default case, we'll handle that in the else
                continue
            i += 1
            dispatcher_str += f"if (dilation == {di.value})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name_target}_di_{di}(__VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += "    "
        dispatcher_str += f"  {self.name_target}_di_any(__VA_ARGS__); \\\n"
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
        header_foot += ["} ", "// namespace {namespace}", " \n"]
    header_foot += [
        "\n",
    ]
    with open(path, "w") as f:
        f.write("".join(header_head))
        f.write(content)
        f.write("".join(header_foot))


def generate_cuda_kernels(path, num_splits=2):

    CUDA_DTYPES = [
        NATTEN_Double,
        NATTEN_Float,
        NATTEN_Half,
        NATTEN_BFloat,
    ]

    CUDA_OPS = ALL_OPS
    CUDA_SIZES = ALL_SIZES

    CUDA_KERNEL_SIZES = [-1] + [k for k in range(3, 15, 2)]
    CUDA_DILATIONS = [-1] + [k for k in range(1, 2)]

    dtype_dispatchers = []
    kernel_dispatchers = []
    dilation_dispatchers = []
    kernels = []

    for op, dim in product(CUDA_OPS, CUDA_SIZES):
        new_dispatcher = DataTypeDispatcher(operation=op, dim=dim)
        for dtype in CUDA_DTYPES:
            new_dispatcher.append(dtype)
            kernel_dispatcher = KernelSizeDispatcher(dtype=dtype, operation=op, dim=dim)
            for ks_int in CUDA_KERNEL_SIZES:
                kernel_dispatcher.append(TemplateIntParameter(ks_int))
                dilation_dispatcher = DilationDispatcher(
                    dtype=dtype,
                    kernel_size=TemplateIntParameter(ks_int),
                    operation=op,
                    dim=dim,
                )
                for di_int in CUDA_DILATIONS:
                    dilation_dispatcher.append(TemplateIntParameter(di_int))
                    new_kernel = NaiveNAKernel(
                        dtype=dtype,
                        operation=op,
                        dim=dim,
                        kernel_size=TemplateIntParameter(ks_int),
                        dilation=TemplateIntParameter(di_int),
                    )
                    kernels.append(new_kernel)
                dilation_dispatchers.append(dilation_dispatcher)
            kernel_dispatchers.append(kernel_dispatcher)
        dtype_dispatchers.append(new_dispatcher)

    #

    path_to_sources = f"{path}/autogen/src/cuda/naive/"
    rel_header = "natten_autogen/cuda/naive/"
    path_to_header_dir = f"{path}/autogen/include/{rel_header}"

    os.makedirs(path_to_sources, exist_ok=False)
    os.makedirs(path_to_header_dir, exist_ok=False)

    path_headers = f"{path_to_header_dir}/kernels.h"
    path_dtype = f"{path_to_header_dir}/interface.h"
    path_ks = f"{path_to_header_dir}/dispatch_ks.h"
    path_di = f"{path_to_header_dir}/dispatch_di.h"

    rel_path_headers = f"{rel_header}kernels.h"
    rel_path_ks = f"{rel_header}dispatch_ks.h"
    rel_path_di = f"{rel_header}dispatch_di.h"

    dtype_disp = ""
    for dispatcher in dtype_dispatchers:
        dtype_disp += dispatcher.get_dispatcher()

    ks_disp = ""
    for dispatcher in kernel_dispatchers:
        ks_disp += dispatcher.get_dispatcher()

    di_disp = ""
    for dispatcher in dilation_dispatchers:
        di_disp += dispatcher.get_dispatcher()

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
        write_combined_source_file(
            path_to_sources, f"source_{split_idx}.cu", pth_set, source_list
        )
    assert split_idx == num_splits - 1, f"Expected {split_idx=} == {num_splits=} - 1"
    assert len(kernels_emitted) == len(kernels) and sorted(kernels_emitted) == [
        x for x in range(len(kernels))
    ]

    namespaces = ["natten", "cuda", "naive"]
    cuda_headers = ["natten/dtypes.cuh"]
    write_header_file(dtype_disp, path_dtype, namespaces, cuda_headers + [rel_path_ks])
    write_header_file(ks_disp, path_ks, namespaces, cuda_headers + [rel_path_di])
    write_header_file(di_disp, path_di, namespaces, cuda_headers + [rel_path_headers])
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
def generate_cuda_naive(output_directory: str, num_splits: int):
    generate_cuda_kernels(output_directory, num_splits=num_splits)


if __name__ == "__main__":
    generate_cuda_naive()
