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
        assert not self.check_tf32
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


class GemmShape:
    def __init__(self, M, N, K, warp_M, warp_N, warp_K):
        self.M = M
        self.N = N
        self.K = K
        self.warp_M = warp_M
        self.warp_N = warp_N
        self.warp_K = warp_K


class InstructionShape:
    def __init__(self, math_M, math_N, math_K, stages, sm):
        self.math_M = math_M
        self.math_N = math_N
        self.math_K = math_K
        self.stages = stages
        self.sm = sm


def get_gemm_config(tile: GemmShape, inst: InstructionShape):
    return GemmConfig(
        M=tile.M,
        N=tile.N,
        K=tile.K,
        warp_M=tile.warp_M,
        warp_N=tile.warp_N,
        warp_K=tile.warp_K,
        math_M=inst.math_M,
        math_N=inst.math_N,
        math_K=inst.math_K,
        stages=inst.stages,
        sm=inst.sm,
    )


class KernelConfig:
    def __init__(self, gemm_config, tile, ext, neighborhood):
        self.gemm_config = gemm_config
        self.tile = tile
        self.ext = ext
        assert neighborhood > 0
        self.neighborhood = neighborhood
        self.kernel_size = neighborhood * 2 + 1

    def is_simt(self):
        return self.gemm_config.math_M == 1

    def __str__(self):
        out = ""
        out += f"{self.gemm_config.M}x{self.gemm_config.N}x{self.gemm_config.K}_"
        out += f"{self.gemm_config.warp_M}x{self.gemm_config.warp_N}x{self.gemm_config.warp_K}_"
        out += f"{self.gemm_config.math_M}x{self.gemm_config.math_N}x{self.gemm_config.math_K}_"
        out += f"{self.gemm_config.stages}_"
        out += f"sm{self.gemm_config.sm}_"
        out += f"ks{self.kernel_size}"
        return out

    def source(self, dtype):
        out = ""
        out += "  using GConfig = natten::gemm::detail::GemmConfig2D<"
        out += f"{self.gemm_config.M}, {self.gemm_config.N}, {self.gemm_config.K}, "
        out += f"{self.gemm_config.warp_M}, {self.gemm_config.warp_N}, {self.gemm_config.warp_K}, "
        out += f"{self.gemm_config.math_M}, {self.gemm_config.math_N}, {self.gemm_config.math_K}, "
        out += f"{self.gemm_config.stages}, "
        out += f"{self.tile}, {self.ext}, {self.neighborhood}"
        out += ">;\n"
        out += f"  using ArchConfig = natten::gemm::detail::ArchArgs<{self.gemm_config.sm}, {dtype}>;\n"
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
    out += "2D"
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


carg_map = {
    Operation.PN: PN_COMMON_ARGS + NA2D_PROBLEM_SIZE_ARGS + COMMON_ARGS,
    Operation.NN: NN_COMMON_ARGS + NA2D_PROBLEM_SIZE_ARGS + COMMON_ARGS,
    Operation.IN: IN_COMMON_ARGS + NA2D_PROBLEM_SIZE_ARGS + COMMON_ARGS,
}

inst_shape_map = {
    70: {
        NATTEN_Double: InstructionShape(1, 1, 1, stages=4, sm=70),
        NATTEN_Float: InstructionShape(1, 1, 1, stages=4, sm=70),
        NATTEN_Half: InstructionShape(8, 8, 4, stages=2, sm=70),
    },
    75: {
        NATTEN_Double: InstructionShape(1, 1, 1, stages=4, sm=75),
        NATTEN_Float: InstructionShape(1, 1, 1, stages=4, sm=75),
        NATTEN_Half: InstructionShape(16, 8, 8, stages=2, sm=75),
    },
    80: {
        NATTEN_Double: InstructionShape(8, 8, 4, stages=3, sm=80),
        NATTEN_Float: InstructionShape(16, 8, 8, stages=3, sm=80),
        NATTEN_Half: InstructionShape(16, 8, 16, stages=3, sm=80),
        NATTEN_BFloat: InstructionShape(16, 8, 16, stages=3, sm=80),
    },
}

ALL_DTYPES = {sm: d.keys() for sm, d in inst_shape_map.items()}

# I really don't have time to benchmark different shapes, and in 2D it gets super messy
Gemm_128 = {
    70: {
        64: GemmShape(M=128, N=128, K=8, warp_M=64, warp_N=64, warp_K=8),
        32: GemmShape(M=128, N=128, K=8, warp_M=64, warp_N=64, warp_K=8),
        16: GemmShape(M=128, N=128, K=32, warp_M=64, warp_N=64, warp_K=32),
    },
    75: {
        64: GemmShape(M=128, N=128, K=8, warp_M=64, warp_N=64, warp_K=8),
        32: GemmShape(M=128, N=128, K=8, warp_M=64, warp_N=64, warp_K=8),
        16: GemmShape(M=128, N=128, K=32, warp_M=64, warp_N=64, warp_K=32),
    },
    80: {
        64: GemmShape(M=128, N=128, K=16, warp_M=64, warp_N=64, warp_K=16),
        32: GemmShape(M=128, N=128, K=16, warp_M=64, warp_N=64, warp_K=16),
        16: GemmShape(M=128, N=128, K=32, warp_M=64, warp_N=64, warp_K=32),
    },
}
Gemm_64 = {
    70: {
        64: GemmShape(M=64, N=64, K=8, warp_M=32, warp_N=32, warp_K=8),
        32: GemmShape(M=64, N=64, K=8, warp_M=32, warp_N=16, warp_K=8),
        16: GemmShape(M=64, N=64, K=32, warp_M=32, warp_N=32, warp_K=32),
    },
    75: {
        64: GemmShape(M=64, N=64, K=8, warp_M=32, warp_N=32, warp_K=8),
        32: GemmShape(M=64, N=64, K=8, warp_M=32, warp_N=16, warp_K=8),
        16: GemmShape(M=64, N=64, K=32, warp_M=32, warp_N=32, warp_K=32),
    },
    80: {
        64: GemmShape(M=64, N=64, K=16, warp_M=32, warp_N=32, warp_K=16),
        32: GemmShape(M=64, N=64, K=16, warp_M=32, warp_N=16, warp_K=16),
        16: GemmShape(M=64, N=64, K=32, warp_M=32, warp_N=32, warp_K=32),
    },
}


PN_HALO = 1  # No haloing
NN_HALO = 2  # Regular haloing
IN_HALO = 4  # Double haloing


def generate_gemm_config(kernel_size, tile_size, tile_ext, gemm_m, sm):
    if gemm_m == 64:
        gemm_shape = Gemm_64[sm]
    elif gemm_m == 128:
        gemm_shape = Gemm_128[sm]
    else:
        raise ValueError()

    output = {}
    for dtype in ALL_DTYPES[sm]:
        gc = get_gemm_config(gemm_shape[dtype.bits], inst_shape_map[sm][dtype])
        assert kernel_size > 1 and kernel_size % 2 == 1
        output[dtype] = KernelConfig(gc, tile_size, tile_ext, kernel_size // 2)

    return output


MAX_KERNEL_SIZE_SUPPORTED = 63


def get_kernel_instance_map(sm):
    return {
        Operation.PN: {
            3: generate_gemm_config(3, 7, PN_HALO, 64, sm=sm),
            5: generate_gemm_config(5, 6, PN_HALO, 64, sm=sm),
            7: generate_gemm_config(7, 5, PN_HALO, 64, sm=sm),
            9: generate_gemm_config(9, 7, PN_HALO, 64, sm=sm),
            11: generate_gemm_config(11, 6, PN_HALO, 64, sm=sm),
            13: generate_gemm_config(13, 10, PN_HALO, 64, sm=sm),
            15: generate_gemm_config(15, 9, PN_HALO, 128, sm=sm),
            17: generate_gemm_config(17, 11, PN_HALO, 128, sm=sm),
            19: generate_gemm_config(19, 13, PN_HALO, 128, sm=sm),
            21: generate_gemm_config(21, 12, PN_HALO, 128, sm=sm),
            23: generate_gemm_config(23, 14, PN_HALO, 128, sm=sm),
            25: generate_gemm_config(25, 13, PN_HALO, 128, sm=sm),
            27: generate_gemm_config(27, 14, PN_HALO, 128, sm=sm),
            29: generate_gemm_config(29, 17, PN_HALO, 128, sm=sm),
            31: generate_gemm_config(31, 17, PN_HALO, 128, sm=sm),
            33: generate_gemm_config(33, 17, PN_HALO, 128, sm=sm),
            35: generate_gemm_config(35, 22, PN_HALO, 128, sm=sm),
            37: generate_gemm_config(37, 22, PN_HALO, 128, sm=sm),
            39: generate_gemm_config(39, 23, PN_HALO, 128, sm=sm),
            41: generate_gemm_config(41, 22, PN_HALO, 128, sm=sm),
            43: generate_gemm_config(43, 22, PN_HALO, 128, sm=sm),
            45: generate_gemm_config(45, 23, PN_HALO, 128, sm=sm),
            47: generate_gemm_config(47, 33, PN_HALO, 128, sm=sm),
            49: generate_gemm_config(49, 33, PN_HALO, 128, sm=sm),
            51: generate_gemm_config(51, 33, PN_HALO, 128, sm=sm),
            53: generate_gemm_config(53, 33, PN_HALO, 128, sm=sm),
            55: generate_gemm_config(55, 33, PN_HALO, 128, sm=sm),
            57: generate_gemm_config(57, 33, PN_HALO, 128, sm=sm),
            59: generate_gemm_config(59, 33, PN_HALO, 128, sm=sm),
            61: generate_gemm_config(61, 33, PN_HALO, 128, sm=sm),
            63: generate_gemm_config(63, 33, PN_HALO, 128, sm=sm),
        },
        Operation.NN: {
            3: generate_gemm_config(3, 8, NN_HALO, 64, sm=sm),
            5: generate_gemm_config(5, 8, NN_HALO, 64, sm=sm),
            7: generate_gemm_config(7, 8, NN_HALO, 64, sm=sm),
            9: generate_gemm_config(9, 8, NN_HALO, 64, sm=sm),
            11: generate_gemm_config(11, 8, NN_HALO, 64, sm=sm),
            13: generate_gemm_config(13, 8, NN_HALO, 64, sm=sm),
            15: generate_gemm_config(15, 8, NN_HALO, 64, sm=sm),
            17: generate_gemm_config(17, 8, NN_HALO, 64, sm=sm),
            19: generate_gemm_config(19, 8, NN_HALO, 64, sm=sm),
            21: generate_gemm_config(21, 11, NN_HALO, 128, sm=sm),
            23: generate_gemm_config(23, 11, NN_HALO, 128, sm=sm),
            25: generate_gemm_config(25, 11, NN_HALO, 128, sm=sm),
            27: generate_gemm_config(27, 11, NN_HALO, 128, sm=sm),
            29: generate_gemm_config(29, 11, NN_HALO, 128, sm=sm),
            31: generate_gemm_config(31, 11, NN_HALO, 128, sm=sm),
            33: generate_gemm_config(33, 11, NN_HALO, 128, sm=sm),
            35: generate_gemm_config(35, 11, NN_HALO, 128, sm=sm),
            37: generate_gemm_config(37, 11, NN_HALO, 128, sm=sm),
            39: generate_gemm_config(39, 11, NN_HALO, 128, sm=sm),
            41: generate_gemm_config(41, 11, NN_HALO, 128, sm=sm),
            43: generate_gemm_config(43, 11, NN_HALO, 128, sm=sm),
            45: generate_gemm_config(45, 11, NN_HALO, 128, sm=sm),
            47: generate_gemm_config(47, 11, NN_HALO, 128, sm=sm),
            49: generate_gemm_config(49, 11, NN_HALO, 128, sm=sm),
            51: generate_gemm_config(51, 11, NN_HALO, 128, sm=sm),
            53: generate_gemm_config(53, 11, NN_HALO, 128, sm=sm),
            55: generate_gemm_config(55, 11, NN_HALO, 128, sm=sm),
            57: generate_gemm_config(57, 11, NN_HALO, 128, sm=sm),
            59: generate_gemm_config(59, 11, NN_HALO, 128, sm=sm),
            61: generate_gemm_config(61, 11, NN_HALO, 128, sm=sm),
            63: generate_gemm_config(63, 11, NN_HALO, 128, sm=sm),
        },
        Operation.IN: {
            3: generate_gemm_config(3, 8, IN_HALO, 64, sm=sm),
            5: generate_gemm_config(5, 8, IN_HALO, 64, sm=sm),
            7: generate_gemm_config(7, 8, IN_HALO, 64, sm=sm),
            9: generate_gemm_config(9, 8, IN_HALO, 64, sm=sm),
            11: generate_gemm_config(11, 8, IN_HALO, 64, sm=sm),
            13: generate_gemm_config(13, 8, IN_HALO, 64, sm=sm),
            15: generate_gemm_config(15, 8, IN_HALO, 64, sm=sm),
            17: generate_gemm_config(17, 8, IN_HALO, 64, sm=sm),
            19: generate_gemm_config(19, 8, IN_HALO, 64, sm=sm),
            21: generate_gemm_config(21, 8, IN_HALO, 64, sm=sm),
            23: generate_gemm_config(23, 8, IN_HALO, 64, sm=sm),
            25: generate_gemm_config(25, 8, IN_HALO, 64, sm=sm),
            27: generate_gemm_config(27, 8, IN_HALO, 64, sm=sm),
            29: generate_gemm_config(29, 8, IN_HALO, 64, sm=sm),
            31: generate_gemm_config(31, 8, IN_HALO, 64, sm=sm),
            33: generate_gemm_config(33, 8, IN_HALO, 64, sm=sm),
            35: generate_gemm_config(35, 8, IN_HALO, 64, sm=sm),
            37: generate_gemm_config(37, 8, IN_HALO, 64, sm=sm),
            39: generate_gemm_config(39, 8, IN_HALO, 64, sm=sm),
            41: generate_gemm_config(41, 8, IN_HALO, 64, sm=sm),
            43: generate_gemm_config(43, 8, IN_HALO, 64, sm=sm),
            45: generate_gemm_config(45, 8, IN_HALO, 64, sm=sm),
            47: generate_gemm_config(47, 8, IN_HALO, 64, sm=sm),
            49: generate_gemm_config(49, 8, IN_HALO, 64, sm=sm),
            51: generate_gemm_config(51, 8, IN_HALO, 64, sm=sm),
            53: generate_gemm_config(53, 8, IN_HALO, 64, sm=sm),
            55: generate_gemm_config(55, 8, IN_HALO, 64, sm=sm),
            57: generate_gemm_config(57, 8, IN_HALO, 64, sm=sm),
            59: generate_gemm_config(59, 8, IN_HALO, 64, sm=sm),
            61: generate_gemm_config(61, 8, IN_HALO, 64, sm=sm),
            63: generate_gemm_config(63, 8, IN_HALO, 64, sm=sm),
        },
    }


#####################################################################################


class NAGemmKernel:
    def __init__(
        self,
        operation: Operation,
        dtype: DataType,
        max_numel: int,
        gemm_config: KernelConfig,
    ):
        self.alignment = AlignmentConfig(max_numel=max_numel, operation=operation)
        self.gemm_config = gemm_config
        self.dtype = dtype
        self.operation = operation
        self.arguments = carg_map[operation]
        self.name_cc = f"na2d_{operation.name.lower()}_cuda_gemm_{dtype}"
        self.name_cc += f"_{self.gemm_config}"
        self.name_cc += f"_{self.alignment}"
        self.template_name = cutlass_op_to_template_name(operation)

    @property
    def header_files(self):
        return [
            "natten/config.h",
            "natten/dtypes.cuh",
            "natten/gemm_argpack.cuh",
            "natten/cuda/gemm/na2d.cuh",
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
        if self.dtype.multi_source:
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
    def __init__(self, operation: Operation, sm: int):
        self.operation = operation
        self.name_base = f"na2d_{operation.name.lower()}_cuda_gemm_sm{sm}"
        self.name_cc = f"DISPATCH_DTYPE_{self.name_base}"
        self.name_target = f"DISPATCH_KERNELSIZE_{self.name_base}"
        self.dtypes: List[DataType] = []

    def append(self, dtype: DataType):
        self.dtypes.append(dtype)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(dtype, kernel_size, dim, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, dtype in enumerate(self.dtypes):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (std::is_same<dtype, {dtype.natten_dtype}>::value)"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name_target}_{dtype.name}(kernel_size, dim, __VA_ARGS__); \\\n"
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
    def __init__(self, dtype: DataType, operation: Operation, sm: int):
        self.operation = operation
        name_base = f"na2d_{operation.name.lower()}_cuda_gemm"
        self.name_base = name_base + f"_sm{sm}_{dtype}"
        self.name_target_base = name_base + f"_{dtype}"
        self.name_cc = f"DISPATCH_KERNELSIZE_{self.name_base}"
        self.name_target = f"DISPATCH_ALIGNMENT_{self.name_target_base}"
        self.configs: List[KernelConfig] = []

    def append(self, gemm_config: KernelConfig):
        self.configs.append(gemm_config)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name_cc}(kernel_size, dim, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, conf in enumerate(self.configs):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (kernel_size == {conf.kernel_size})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += f"  {self.name_target}_{conf}(dim, __VA_ARGS__); \\\n"
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += '      std::cerr << "NATTEN kernel launch failed! " \\\n'
        dispatcher_str += (
            '                << "'
            + f"{self.name_base} does not support implement "
            + '" \\\n'
        )
        dispatcher_str += (
            '                << " kernel size " << kernel_size << ". " \\\n'
        )
        dispatcher_str += (
            '                << "'
            + " You may try generating it manually and build from source."
            + '" \\\n'
        )
        dispatcher_str += (
            '                << "'
            + " Refer to NATTEN's github repository for more information."
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
        gemm_config: KernelConfig,
    ):
        self.gemm_config = gemm_config
        self.dtype = dtype
        self.operation = operation
        self.name_base = f"na2d_{operation.name.lower()}_cuda_gemm"
        self.name_base += f"_{dtype}"
        self.name_base += f"_{gemm_config}"
        self.name_cc = f"DISPATCH_ALIGNMENT_{self.name_base}"
        self.name_target = f"natten::cuda::gemm::{self.name_base}"
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


def generate_cuda_kernels(path, max_kernel_size, sm, num_splits=4):
    assert max_kernel_size >= 3
    assert max_kernel_size <= MAX_KERNEL_SIZE_SUPPORTED, (
        f"We only support up to kernel size {MAX_KERNEL_SIZE_SUPPORTED}"
        + " for 2D GEMM kernels. Please contact the developers or raise an issue if you are interested in trying larger kernels."
    )
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
    kernel_dispatchers = []
    align_dispatchers = []
    kernels = []

    kernel_instance_map = get_kernel_instance_map(sm)

    for op in ALL_OPS:
        new_dispatcher = DataTypeDispatcher(operation=op, sm=sm)
        for dtype in DTYPES:
            new_dispatcher.append(dtype)
            kernel_dispatcher = KernelSizeDispatcher(operation=op, dtype=dtype, sm=sm)
            for kernel_size in kernel_instance_map[op].keys():
                if kernel_size > max_kernel_size:
                    break
                kernel_dispatcher.append(kernel_instance_map[op][kernel_size][dtype])
                align_dispatcher = AlignmentDispatcher(
                    dtype=dtype,
                    operation=op,
                    gemm_config=kernel_instance_map[op][kernel_size][dtype],
                )
                for numel in align_dispatcher.alignments:
                    new_kernel = NAGemmKernel(
                        operation=op,
                        dtype=dtype,
                        max_numel=numel,
                        gemm_config=kernel_instance_map[op][kernel_size][dtype],
                    )
                    kernels.append(new_kernel)
                align_dispatchers.append(align_dispatcher)
            kernel_dispatchers.append(kernel_dispatcher)
        dtype_dispatchers.append(new_dispatcher)

    path_to_sources = f"{path}/autogen/src/cuda/gemm/2d/sm{sm}/"
    rel_header = f"natten_autogen/cuda/gemm/2d/sm{sm}/"
    path_to_header_dir = f"{path}/autogen/include/{rel_header}"

    os.makedirs(path_to_sources, exist_ok=False)
    os.makedirs(path_to_header_dir, exist_ok=False)

    path_headers = f"{path_to_header_dir}/kernels.h"
    path_dtype = f"{path_to_header_dir}/dispatch_dtype.h"
    path_kernel = f"{path_to_header_dir}/dispatch_kernel_size.h"
    path_align = f"{path_to_header_dir}/dispatch_align.h"

    rel_path_headers = f"{rel_header}kernels.h"
    rel_path_kernel = f"{rel_header}dispatch_kernel_size.h"
    rel_path_align = f"{rel_header}dispatch_align.h"

    dtype_disp = ""
    for dispatcher in dtype_dispatchers:
        dtype_disp += dispatcher.get_dispatcher()

    kernel_disp = ""
    for dispatcher in kernel_dispatchers:
        kernel_disp += dispatcher.get_dispatcher()

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
        dtype_disp, path_dtype, namespaces, cuda_headers + [rel_path_kernel]
    )
    write_header_file(
        kernel_disp, path_kernel, namespaces, cuda_headers + [rel_path_align]
    )
    write_header_file(
        align_disp, path_align, namespaces, cuda_headers + [rel_path_headers]
    )
    write_header_file(headers, path_headers, namespaces, cuda_headers)


class DeviceDispatcher:
    def __init__(self, operation: Operation, sm_list: List[int]):
        self.operation = operation
        self.name_base = f"na2d_{operation.name.lower()}_cuda_gemm"
        self.name_cc = f"LAUNCH_{self.name_base}"
        self.targets = {
            sm: DataTypeDispatcher(operation=operation, sm=sm).name_cc
            for sm in sorted(sm_list, reverse=True)
        }

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += (
            f"#define {self.name_cc}(cc, dtype, kernel_size, dim, ...) \\\n"
        )
        dispatcher_str += "  [&] { \\\n"
        for i, (sm, target_name) in enumerate(self.targets.items()):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (cc >= {sm})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += (
                f"  {target_name}(dtype, kernel_size, dim, __VA_ARGS__); \\\n"
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


def generate_interface(path, sm_list):
    if any(sm < 70 for sm in sm_list):
        raise NotImplementedError()

    dispatchers = [DeviceDispatcher(op, sm_list=sm_list) for op in ALL_OPS]

    rel_header = "natten_autogen/cuda/gemm/2d"
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
    "--max-kernel-size",
    default=33,
    help="Maximum kernel size to emit. Default: 33.",
)
def generate_cuda_gemm_2d(output_directory: str, max_kernel_size: int):
    SM_PROPS = {70: 4, 75: 4, 80: 32}
    for sm, num_splits in SM_PROPS.items():
        generate_cuda_kernels(
            output_directory, max_kernel_size, sm=sm, num_splits=num_splits
        )
    generate_interface(output_directory, list(SM_PROPS.keys()))


if __name__ == "__main__":
    generate_cuda_gemm_2d()
