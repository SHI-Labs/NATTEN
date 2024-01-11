# Copyright (c) 2022-2024 Ali Hassani.
#
# This script is intended to emit naive kernel instantiations into
# a variable number of source files, generate appropriate headers
# and a single dispatcher interface, which will be used by the
# NATTEN API to call the kernels.
#
# NOTE: these scripts are heavily under-documented, and
# overly-repetitive, and will be replaced in future PRs.

import os
from enum import Enum
from typing import List

import click


DEFAULT_OUTPUT_DIR = "csrc/"


class Operation(Enum):
    PN = 0
    NN = 2
    IN = 3


ALL_OPS = [
    Operation.PN,
    Operation.NN,
    Operation.IN,
]


class DataType:
    def __init__(self, name, natten_dtype, bits, is_int=False):
        self.name = name
        self.natten_dtype = natten_dtype
        self.bits = bits
        self.check_tf32 = False if is_int else (bits == 32)
        self.multi_source = self.check_tf32

    def __str__(self):
        return self.name

    def source_a(self):
        assert self.check_tf32
        out = ""
        out += "    if (natten::kEnableGemmTF32) { \n"
        out += (
            "      using DConfig = natten::gemm::detail::DTypeConfig<natten::tf32>;\n"
        )
        return out

    def source_b(self):
        assert self.check_tf32
        out = "\n"
        out += "    } else { \n"
        out += "      using DConfig = natten::gemm::detail::DTypeConfig<natten::float32>;\n"
        return out

    def source_c(self):
        assert self.check_tf32
        out = "\n"
        out += "    }\n"
        return out

    def source(self):
        out = ""
        out += f"  using DConfig = natten::gemm::detail::DTypeConfig<{self.natten_dtype}>;\n"
        return out


NATTEN_Double = DataType("double", "natten::float64", 64)
NATTEN_Float = DataType("float", "natten::float32", 32)
NATTEN_Half = DataType("half", "natten::float16", 16)
NATTEN_BFloat = DataType("bfloat16", "natten::bfloat16", 16)


class GemmConfig:
    def __init__(
        self, M, N, K, warp_M, warp_N, warp_K, math_M, math_N, math_K, stages, sm
    ):
        self.M = M
        self.N = N
        self.K = K
        self.warp_M = warp_M
        self.warp_N = warp_N
        self.warp_K = warp_K
        self.math_M = math_M
        self.math_N = math_N
        self.math_K = math_K
        self.stages = stages
        self.sm = sm

    def is_simt(self):
        return self.math_M == 1

    def __str__(self):
        out = ""
        out += f"{self.M}x{self.N}x{self.K}_"
        out += f"{self.warp_M}x{self.warp_N}x{self.warp_K}_"
        out += f"{self.math_M}x{self.math_N}x{self.math_K}_"
        out += f"{self.stages}_"
        out += f"sm{self.sm}"
        return out

    def source(self, dtype):
        out = ""
        out += "  using GConfig = natten::gemm::detail::GemmConfig<"
        out += f"{self.M}, {self.N}, {self.K}, "
        out += f"{self.warp_M}, {self.warp_N}, {self.warp_K}, "
        out += f"{self.math_M}, {self.math_N}, {self.math_K}, "
        out += f"{self.stages}>;\n"
        out += f"  using ArchConfig = natten::gemm::detail::ArchArgs<{self.sm}, {dtype}>;\n"
        return out


class AlignmentConfig:
    def __init__(self, max_numel: int, operation: Operation):
        self.max_numel = max_numel
        if operation == Operation.PN:
            alignment_a = max_numel
            alignment_b = max_numel
            alignment_c = 1
        elif operation in [Operation.NN, Operation.IN]:
            alignment_a = 1
            alignment_b = max_numel
            alignment_c = max_numel
        else:
            raise ValueError()
        self.alignment_a = alignment_a
        self.alignment_b = alignment_b
        self.alignment_c = alignment_c

    def __str__(self):
        return f"align{self.max_numel}"

    def source(self):
        out = ""
        out += "  using AConfig = natten::gemm::detail::AlignmentConfig<"
        out += f"{self.alignment_a}, {self.alignment_b}, {self.alignment_c}"
        out += ">;\n"
        return out


def cutlass_op_to_template_name(operation: Operation):
    out = ""
    if operation == Operation.PN:
        out += "PointwiseNeighborhood"
    elif operation == Operation.NN:
        out += "NeighborhoodNeighborhood"
    elif operation == Operation.IN:
        out += "InverseNeighborhood"
    else:
        raise ValueError()
    out += "1D"
    return out


#####################################################################################


class CArg:
    def __init__(self, dtype: str, name: str):
        self.dtype = dtype
        self.name = name

    def __str__(self):
        return f"{self.dtype} {self.name}"


COMMON_ARGS = [
    CArg("int", "kernel_size"),
    CArg("int", "dilation"),
    CArg("float", "scale"),
    CArg("cudaStream_t", "stream"),
]

PN_COMMON_ARGS = [
    CArg("void *", "query_ptr"),
    CArg("void *", "key_ptr"),
    CArg("void *", "attn_ptr"),
    CArg(
        "void *", "bias_ptr"
    ),  # It's Q, K, A, then bias -- in naive it's Q, K, bias, A.
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

NA1D_PROBLEM_SIZE_ARGS = [
    CArg("int", "batch_size"),
    CArg("int", "heads"),
    CArg("int", "length"),
    CArg("int", "dim"),
    CArg("int64_t", "attn_stride_0"),
    CArg("int64_t", "attn_stride_1"),
    CArg("int64_t", "attn_stride_2"),
]


carg_map = {
    Operation.PN: PN_COMMON_ARGS + NA1D_PROBLEM_SIZE_ARGS + COMMON_ARGS,
    Operation.NN: NN_COMMON_ARGS + NA1D_PROBLEM_SIZE_ARGS + COMMON_ARGS,
    Operation.IN: IN_COMMON_ARGS + NA1D_PROBLEM_SIZE_ARGS + COMMON_ARGS,
}

op_and_dtype_2_config = {
    70: {
        Operation.PN: {
            NATTEN_Double: GemmConfig(
                M=128,
                N=128,
                K=8,
                warp_M=32,
                warp_N=64,
                warp_K=8,
                math_M=1,
                math_N=1,
                math_K=1,
                stages=4,
                sm=70,
            ),
            NATTEN_Float: GemmConfig(
                M=128,
                N=128,
                K=8,
                warp_M=32,
                warp_N=64,
                warp_K=8,
                math_M=1,
                math_N=1,
                math_K=1,
                stages=4,
                sm=70,
            ),
            NATTEN_Half: GemmConfig(
                M=128,
                N=128,
                K=32,
                warp_M=64,
                warp_N=64,
                warp_K=32,
                math_M=8,
                math_N=8,
                math_K=4,
                stages=2,
                sm=70,
            ),
        },
        Operation.NN: {
            NATTEN_Double: GemmConfig(
                M=64,
                N=32,
                K=8,
                warp_M=32,
                warp_N=16,
                warp_K=8,
                math_M=1,
                math_N=1,
                math_K=1,
                stages=4,
                sm=70,
            ),
            NATTEN_Float: GemmConfig(
                M=64,
                N=32,
                K=8,
                warp_M=32,
                warp_N=16,
                warp_K=8,
                math_M=1,
                math_N=1,
                math_K=1,
                stages=4,
                sm=70,
            ),
            NATTEN_Half: GemmConfig(
                M=64,
                N=64,
                K=32,
                warp_M=32,
                warp_N=32,
                warp_K=32,
                math_M=8,
                math_N=8,
                math_K=4,
                stages=2,
                sm=70,
            ),
        },
        Operation.IN: {
            NATTEN_Double: GemmConfig(
                M=64,
                N=32,
                K=8,
                warp_M=32,
                warp_N=16,
                warp_K=8,
                math_M=1,
                math_N=1,
                math_K=1,
                stages=4,
                sm=70,
            ),
            NATTEN_Float: GemmConfig(
                M=64,
                N=32,
                K=8,
                warp_M=32,
                warp_N=16,
                warp_K=8,
                math_M=1,
                math_N=1,
                math_K=1,
                stages=4,
                sm=70,
            ),
            NATTEN_Half: GemmConfig(
                M=64,
                N=64,
                K=32,
                warp_M=32,
                warp_N=32,
                warp_K=32,
                math_M=8,
                math_N=8,
                math_K=4,
                stages=2,
                sm=70,
            ),
        },
    },
    75: {
        Operation.PN: {
            NATTEN_Double: GemmConfig(
                M=128,
                N=128,
                K=8,
                warp_M=32,
                warp_N=64,
                warp_K=8,
                math_M=1,
                math_N=1,
                math_K=1,
                stages=4,
                sm=75,
            ),
            NATTEN_Float: GemmConfig(
                M=128,
                N=128,
                K=8,
                warp_M=32,
                warp_N=64,
                warp_K=8,
                math_M=1,
                math_N=1,
                math_K=1,
                stages=4,
                sm=75,
            ),
            NATTEN_Half: GemmConfig(
                M=128,
                N=128,
                K=32,
                warp_M=64,
                warp_N=64,
                warp_K=32,
                math_M=16,
                math_N=8,
                math_K=8,
                stages=2,
                sm=75,
            ),
        },
        Operation.NN: {
            NATTEN_Double: GemmConfig(
                M=64,
                N=32,
                K=8,
                warp_M=32,
                warp_N=16,
                warp_K=8,
                math_M=1,
                math_N=1,
                math_K=1,
                stages=4,
                sm=75,
            ),
            NATTEN_Float: GemmConfig(
                M=64,
                N=32,
                K=8,
                warp_M=32,
                warp_N=16,
                warp_K=8,
                math_M=1,
                math_N=1,
                math_K=1,
                stages=4,
                sm=75,
            ),
            NATTEN_Half: GemmConfig(
                M=64,
                N=64,
                K=32,
                warp_M=32,
                warp_N=32,
                warp_K=32,
                math_M=16,
                math_N=8,
                math_K=8,
                stages=2,
                sm=75,
            ),
        },
        Operation.IN: {
            NATTEN_Double: GemmConfig(
                M=64,
                N=32,
                K=8,
                warp_M=32,
                warp_N=16,
                warp_K=8,
                math_M=1,
                math_N=1,
                math_K=1,
                stages=4,
                sm=75,
            ),
            NATTEN_Float: GemmConfig(
                M=64,
                N=32,
                K=8,
                warp_M=32,
                warp_N=16,
                warp_K=8,
                math_M=1,
                math_N=1,
                math_K=1,
                stages=4,
                sm=75,
            ),
            NATTEN_Half: GemmConfig(
                M=64,
                N=64,
                K=32,
                warp_M=32,
                warp_N=32,
                warp_K=32,
                math_M=16,
                math_N=8,
                math_K=8,
                stages=2,
                sm=75,
            ),
        },
    },
    80: {
        Operation.PN: {
            NATTEN_Double: GemmConfig(
                M=128,
                N=128,
                K=16,
                warp_M=64,
                warp_N=64,
                warp_K=16,
                math_M=8,
                math_N=8,
                math_K=4,
                stages=3,
                sm=80,
            ),
            NATTEN_Float: GemmConfig(
                M=128,
                N=128,
                K=16,
                warp_M=64,
                warp_N=64,
                warp_K=16,
                math_M=16,
                math_N=8,
                math_K=8,
                stages=3,
                sm=80,
            ),
            NATTEN_Half: GemmConfig(
                M=128,
                N=128,
                K=32,
                warp_M=64,
                warp_N=64,
                warp_K=32,
                math_M=16,
                math_N=8,
                math_K=16,
                stages=3,
                sm=80,
            ),
            NATTEN_BFloat: GemmConfig(
                M=128,
                N=128,
                K=32,
                warp_M=64,
                warp_N=64,
                warp_K=32,
                math_M=16,
                math_N=8,
                math_K=16,
                stages=3,
                sm=80,
            ),
        },
        Operation.NN: {
            NATTEN_Double: GemmConfig(
                M=64,
                N=32,
                K=16,
                warp_M=32,
                warp_N=16,
                warp_K=16,
                math_M=8,
                math_N=8,
                math_K=4,
                stages=3,
                sm=80,
            ),
            NATTEN_Float: GemmConfig(
                M=64,
                N=32,
                K=16,
                warp_M=32,
                warp_N=16,
                warp_K=16,
                math_M=16,
                math_N=8,
                math_K=8,
                stages=3,
                sm=80,
            ),
            NATTEN_Half: GemmConfig(
                M=64,
                N=64,
                K=32,
                warp_M=32,
                warp_N=32,
                warp_K=32,
                math_M=16,
                math_N=8,
                math_K=16,
                stages=3,
                sm=80,
            ),
            NATTEN_BFloat: GemmConfig(
                M=64,
                N=64,
                K=32,
                warp_M=32,
                warp_N=32,
                warp_K=32,
                math_M=16,
                math_N=8,
                math_K=16,
                stages=3,
                sm=80,
            ),
        },
        Operation.IN: {
            NATTEN_Double: GemmConfig(
                M=64,
                N=32,
                K=16,
                warp_M=32,
                warp_N=16,
                warp_K=16,
                math_M=8,
                math_N=8,
                math_K=4,
                stages=3,
                sm=80,
            ),
            NATTEN_Float: GemmConfig(
                M=64,
                N=32,
                K=16,
                warp_M=32,
                warp_N=16,
                warp_K=16,
                math_M=16,
                math_N=8,
                math_K=8,
                stages=3,
                sm=80,
            ),
            NATTEN_Half: GemmConfig(
                M=64,
                N=64,
                K=32,
                warp_M=32,
                warp_N=32,
                warp_K=32,
                math_M=16,
                math_N=8,
                math_K=16,
                stages=3,
                sm=80,
            ),
            NATTEN_BFloat: GemmConfig(
                M=64,
                N=64,
                K=32,
                warp_M=32,
                warp_N=32,
                warp_K=32,
                math_M=16,
                math_N=8,
                math_K=16,
                stages=3,
                sm=80,
            ),
        },
    },
}


#####################################################################################


class NAGemmKernel:
    def __init__(
        self,
        operation: Operation,
        dtype: DataType,
        max_numel: int,
        gemm_config: GemmConfig,
    ):
        self.alignment = AlignmentConfig(max_numel=max_numel, operation=operation)
        self.gemm_config = gemm_config
        self.dtype = dtype
        self.operation = operation
        self.arguments = carg_map[operation]
        self.name_cc = f"na1d_{operation.name.lower()}_cuda_gemm_{dtype}"
        self.name_cc += f"_{self.gemm_config}"
        self.name_cc += f"_{self.alignment}"
        self.template_name = cutlass_op_to_template_name(operation)

    @property
    def header_files(self):
        return [
            "natten/config.h",
            "natten/dtypes.cuh",
            "natten/gemm_argpack.cuh",
            "natten/cuda/gemm/na1d.cuh",
        ]

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

    def method_def_inner(self):
        launch_str = ""
        launch_str += f"  using Kernel = {self.template_name}<GConfig, AConfig, DConfig, ArchConfig>;\n"
        launch_str += "  Kernel kernel;\n"
        launch_str += "  kernel(\n"
        launch_str += self.parameters()
        launch_str += ");\n"
        return launch_str

    def method_def(self):
        launch_str = ""
        launch_str += self.gemm_config.source(self.dtype)
        launch_str += self.alignment.source()
        if self.dtype.multi_source and self.gemm_config.sm >= 80:
            launch_str += self.dtype.source_a()
            launch_str += self.method_def_inner()
            launch_str += self.dtype.source_b()
            launch_str += self.method_def_inner()
            launch_str += self.dtype.source_c()
        else:
            launch_str += self.dtype.source()
            launch_str += self.method_def_inner()
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
        # if self.dtype.name == "half":
        #    source_str += "\n#ifdef NATTEN_ENABLE_FP16\n"
        # elif self.dtype.name == "bfloat16":
        #    source_str += "\n#ifdef NATTEN_ENABLE_BF16\n"
        source_str += self.method_def()
        # if self.dtype.name in ["half", "bfloat16"]:
        #    source_str += "\n#else\n"
        #    source_str += "std::cerr << \"NATTEN was not built with support for this half type." + "\""
        #    source_str += "  << std::endl; \n"
        #    source_str += "exit(EXIT_FAILURE); \n"
        #    source_str += "\n#endif\n"
        source_str += "}\n\n"
        return source_str

    def write_source_file(self, path):
        source_head = []

        source_head += ["#include <iostream>\n"]

        source_head += [f"#include <{f}>\n" for f in self.header_files]

        source_head += ["\nnamespace natten { \n"]
        source_head += ["namespace cuda { \n"]
        source_head += ["namespace gemm { \n"]

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

    for header in headers:
        source_head += [f"#include <{header}>\n"]

    source_head += ["namespace natten { \n"]
    source_head += ["namespace cuda { \n"]
    source_head += ["namespace gemm { \n\n"]

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
        sm: int,
    ):
        self.operation = operation
        self.name_base = f"na1d_{operation.name.lower()}_cuda_gemm"
        self.name_cc = f"DISPATCH_DTYPE_{self.name_base}_sm{sm}"
        self.name_target = f"DISPATCH_ALIGNMENT_{self.name_base}_sm{sm}"
        self.dtypes: List[DataType] = []

    def append(self, dtype: DataType):
        self.dtypes.append(dtype)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(dtype, dim, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, dtype in enumerate(self.dtypes):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (std::is_same<dtype, {dtype.natten_dtype}>::value)"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += (
                f"  {self.name_target}_{dtype.name}(dim, __VA_ARGS__); \\\n"
            )
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


class AlignmentDispatcher:
    def __init__(
        self,
        dtype: DataType,
        operation: Operation,
        gemm_config: GemmConfig,
    ):
        self.gemm_config = gemm_config
        self.dtype = dtype
        self.operation = operation
        name_base = f"na1d_{operation.name.lower()}_cuda_gemm"
        self.name_base = name_base + f"_sm{self.gemm_config.sm}_{dtype}"
        self.name_target_base = name_base + f"_{dtype}"
        self.name_cc = f"DISPATCH_ALIGNMENT_{self.name_base}"
        self.name_target = f"natten::cuda::gemm::{self.name_target_base}"
        self.name_target += f"_{self.gemm_config}"
        possible_aligments = [128, 64, 32]
        self.alignments = []
        if dtype.bits <= 32 and not gemm_config.is_simt():
            for b in possible_aligments:
                assert b >= dtype.bits and b % dtype.bits == 0
                self.alignments.append(b // dtype.bits)
        else:
            self.alignments = [1]

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(dim, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        i = 0
        if self.alignments is not None and len(self.alignments) > 1:
            for numel in self.alignments:
                alignment = AlignmentConfig(numel, self.operation)
                dispatcher_str += "    "
                if i > 0:
                    dispatcher_str += "else "
                i += 1
                dispatcher_str += f"if (dim % {numel} == 0)"
                dispatcher_str += " { \\\n"
                dispatcher_str += "    "
                dispatcher_str += f"  {self.name_target}_{alignment}(__VA_ARGS__); \\\n"
                dispatcher_str += "    } \\\n"
            dispatcher_str += "    else { \\\n"
            dispatcher_str += '      std::cerr << "NATTEN kernel launch failed!" \\\n'
            dispatcher_str += (
                '                << "'
                + f"{self.name_base} requires at least 32-bit alignment."
                + '" \\\n'
            )
            dispatcher_str += (
                '                << "Got dim=" << dim << ", dtype='
                + self.dtype.name
                + '. " \\\n'
            )
            dispatcher_str += "                << std::endl; \\\n"
            dispatcher_str += "      exit(EXIT_FAILURE); \\\n"
            dispatcher_str += "    } \\\n"
        else:
            # assert self.dtype.bits == 64
            alignment = AlignmentConfig(self.alignments[0], self.operation)
            dispatcher_str += f"  {self.name_target}_{alignment}(__VA_ARGS__); \\\n"
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


def generate_cuda_kernels(path, sm, num_splits=4):
    if sm < 70:
        raise NotImplementedError()
    if sm < 80:
        DTYPES = [
            NATTEN_Half,
        ]
    else:
        DTYPES = [
            NATTEN_Double,
            NATTEN_Float,
            NATTEN_Half,
            NATTEN_BFloat,
        ]

    dtype_dispatchers = []
    align_dispatchers = []
    kernels = []

    for op in ALL_OPS:
        new_dispatcher = DataTypeDispatcher(operation=op, sm=sm)
        for dtype in DTYPES:
            new_dispatcher.append(dtype)
            align_dispatcher = AlignmentDispatcher(
                dtype=dtype,
                operation=op,
                gemm_config=op_and_dtype_2_config[sm][op][dtype],
            )
            for numel in align_dispatcher.alignments:
                new_kernel = NAGemmKernel(
                    operation=op,
                    dtype=dtype,
                    max_numel=numel,
                    gemm_config=op_and_dtype_2_config[sm][op][dtype],
                )
                kernels.append(new_kernel)
            align_dispatchers.append(align_dispatcher)
        dtype_dispatchers.append(new_dispatcher)

    path_to_sources = f"{path}/autogen/src/cuda/gemm/1d/sm{sm}/"
    rel_header = f"natten_autogen/cuda/gemm/1d/sm{sm}/"
    path_to_header_dir = f"{path}/autogen/include/{rel_header}"

    os.makedirs(path_to_sources, exist_ok=False)
    os.makedirs(path_to_header_dir, exist_ok=False)

    path_headers = f"{path_to_header_dir}/kernels.h"
    path_dtype = f"{path_to_header_dir}/dispatch_dtype.h"
    path_align = f"{path_to_header_dir}/dispatch_align.h"

    rel_path_headers = f"{rel_header}kernels.h"
    rel_path_align = f"{rel_header}dispatch_align.h"

    dtype_disp = ""
    for dispatcher in dtype_dispatchers:
        dtype_disp += dispatcher.get_dispatcher()

    align_disp = ""
    for dispatcher in align_dispatchers:
        align_disp += dispatcher.get_dispatcher()

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
            for hdr in kernel.header_files:
                pth_set.add(hdr)
            source_list.append(kernel)
            kernels_emitted.append(kernel_idx)
        write_combined_source_file(
            path_to_sources, f"source_{split_idx}.cu", pth_set, source_list
        )
    assert split_idx == num_splits - 1, f"Expected {split_idx=} == {num_splits=} - 1"
    assert len(kernels_emitted) == len(kernels) and sorted(kernels_emitted) == [
        x for x in range(len(kernels))
    ]

    namespaces = ["natten", "cuda", "gemm"]
    cuda_headers = ["natten/dtypes.cuh", "natten/gemm_argpack.cuh", "natten/config.h"]
    write_header_file(
        dtype_disp, path_dtype, namespaces, cuda_headers + [rel_path_align]
    )
    write_header_file(
        align_disp, path_align, namespaces, cuda_headers + [rel_path_headers]
    )
    write_header_file(headers, path_headers, namespaces, cuda_headers)


class DeviceDispatcher:
    def __init__(self, operation: Operation, sm_list: List[int]):
        self.operation = operation
        self.name_base = f"na1d_{operation.name.lower()}_cuda_gemm"
        self.name_cc = f"LAUNCH_{self.name_base}"
        self.targets = {
            sm: DataTypeDispatcher(operation=operation, sm=sm).name_cc
            for sm in sorted(sm_list, reverse=True)
        }

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(cc, dtype, dim, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, (sm, target_name) in enumerate(self.targets.items()):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (cc >= {sm})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {target_name}(dtype, dim, __VA_ARGS__); \\\n"
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


def generate_interface(path, sm_list):
    if any(sm < 70 for sm in sm_list):
        raise NotImplementedError()

    dispatchers = [DeviceDispatcher(op, sm_list=sm_list) for op in ALL_OPS]

    rel_header = "natten_autogen/cuda/gemm/1d"
    path_to_header_dir = f"{path}/autogen/include/{rel_header}"

    dtype_paths = [f"{rel_header}/sm{sm}/dispatch_dtype.h" for sm in sm_list]

    interface_path = f"{path_to_header_dir}/interface.h"

    disp = ""
    for dispatcher in dispatchers:
        disp += dispatcher.get_dispatcher()

    namespaces = ["natten", "cuda", "gemm"]
    cuda_headers = ["natten/dtypes.cuh", "natten/config.h"]
    write_header_file(disp, interface_path, namespaces, cuda_headers + dtype_paths)


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
def generate_cuda_gemm_2d(output_directory: str, num_splits: int):
    SM_LIST = [70, 75, 80]
    for sm in SM_LIST:
        generate_cuda_kernels(output_directory, sm=sm, num_splits=num_splits)
    generate_interface(output_directory, SM_LIST)


if __name__ == "__main__":
    generate_cuda_gemm_2d()
