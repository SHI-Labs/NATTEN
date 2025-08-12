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
from typing import List


DEFAULT_OUTPUT_DIR = "csrc/"


KERNEL_DECL_TEMPLATE = """
void {kernel_name}(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      {DimType} qkv_shape,
      {DimType} window_size,
      {DimType} stride,
      {DimType} dilation,
      float attn_scale,
      cudaStream_t stream,
      bool has_dot_product_min, bool has_dot_product_max,
      float dot_product_min, float dot_product_max);
"""


KERNEL_IMPL_TEMPLATE = """
void {kernel_name}(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      {DimType} qkv_shape,
      {DimType} window_size,
      {DimType} stride,
      {DimType} dilation,
      float attn_scale,
      cudaStream_t stream,
      bool has_dot_product_min, bool has_dot_product_max,
      float dot_product_min, float dot_product_max) {{

  using Causal = {Causal};

  fna_reference_forward(
    static_cast<{dtype}*>(ptr_Q),
    static_cast<{dtype}*>(ptr_K),
    static_cast<{dtype}*>(ptr_V),
    static_cast<{dtype}*>(ptr_O),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{{}},
    attn_scale,
    stream,
    has_dot_product_min, has_dot_product_max,
    dot_product_min, dot_product_max);
}}
"""


KERNEL_BWD_DECL_TEMPLATE = """
void {kernel_name}(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_DO,
      void* ptr_DQ,
      void* ptr_DK,
      void* ptr_DV,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      {DimType} qkv_shape,
      {DimType} window_size,
      {DimType} stride,
      {DimType} dilation,
      float attn_scale,
      cudaStream_t stream,
      bool has_dot_product_min, bool has_dot_product_max,
      float dot_product_min, float dot_product_max);
"""


KERNEL_BWD_IMPL_TEMPLATE = """
void {kernel_name}(
      void* ptr_Q,
      void* ptr_K,
      void* ptr_V,
      void* ptr_O,
      void* ptr_DO,
      void* ptr_DQ,
      void* ptr_DK,
      void* ptr_DV,
      void* ptr_LSE,
      int batch_size,
      int seqlen,
      int heads,
      int dim,
      int dim_value,
      int num_additional_kv,
      {DimType} qkv_shape,
      {DimType} window_size,
      {DimType} stride,
      {DimType} dilation,
      float attn_scale,
      cudaStream_t stream,
      bool has_dot_product_min, bool has_dot_product_max,
      float dot_product_min, float dot_product_max) {{

  using Causal = {Causal};

  fna_reference_backward(
    static_cast<{dtype}*>(ptr_Q),
    static_cast<{dtype}*>(ptr_K),
    static_cast<{dtype}*>(ptr_V),
    static_cast<{dtype}*>(ptr_O),
    static_cast<{dtype}*>(ptr_DO),
    static_cast<{dtype}*>(ptr_DQ),
    static_cast<{dtype}*>(ptr_DK),
    static_cast<{dtype}*>(ptr_DV),
    static_cast<float*>(ptr_LSE),
    batch_size,
    seqlen,
    heads,
    dim,
    dim_value,
    num_additional_kv,
    qkv_shape,
    window_size,
    stride,
    dilation,
    Causal{{}},
    attn_scale,
    stream,
    has_dot_product_min, has_dot_product_max,
    dot_product_min, dot_product_max);
}}
"""


class DataType:
    def __init__(self, name, short_name, torch_name, bits):
        self.name = name
        self.bits = bits
        self.short_name = short_name
        self.torch_name = torch_name


Float = DataType("float", "float32", "torch::kFloat32", 32)
Half = DataType("cutlass::half_t", "float16", "torch::kFloat16", 16)
BFloat = DataType("cutlass::bfloat16_t", "bfloat16", "torch::kBFloat16", 16)


def iterable_to_static_cute_tuple(shape_in) -> str:
    shape = ", ".join([f"cute::Int<{x}>" for x in shape_in])
    return f"cute::tuple<{shape}>"


def get_dim_type(na_dim: int) -> str:
    shape = ", ".join(["int" for _ in range(na_dim)])
    return f"cute::tuple<{shape}>"


class ReferenceFnaInstance:
    def __init__(
        self,
        na_dim: int,
        dtype: DataType,
        causal: tuple,
        is_backward: bool,
    ):
        assert 0 < na_dim <= 3
        assert na_dim == len(causal)
        self.na_dim = na_dim
        self.causal = causal
        self.dtype = dtype
        self.is_backward = is_backward

    def get_causal_cute(self) -> str:
        consts = ", ".join(
            ["cute::true_type" if c else "cute::false_type" for c in self.causal]
        )
        return f"cute::tuple<{consts}>"

    def get_name(self) -> str:
        backward_str = "" if not self.is_backward else "_backward"
        name = f"reference_fna{self.na_dim}d{backward_str}"
        name += f"_{self.dtype.short_name}"
        name += "_causal" + "x".join(["1" if c else "0" for c in self.causal])
        return name

    def get_decl(self) -> str:
        return (
            KERNEL_BWD_DECL_TEMPLATE if self.is_backward else KERNEL_DECL_TEMPLATE
        ).format(
            kernel_name=self.get_name(),
            DimType=get_dim_type(self.na_dim),
            dtype=self.dtype.name,
        )

    def get_impl(self) -> str:
        return (
            KERNEL_BWD_IMPL_TEMPLATE if self.is_backward else KERNEL_IMPL_TEMPLATE
        ).format(
            kernel_name=self.get_name(),
            DimType=get_dim_type(self.na_dim),
            Causal=self.get_causal_cute(),
            dtype=self.dtype.name,
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

    source_head += ["#include <natten/cuda/reference/fna_reference_forward.hpp>\n"]
    source_head += ["#include <natten/cuda/reference/fna_reference_backward.hpp>\n"]

    for header in headers:
        source_head += [f"#include <{header}>\n"]

    source_head += ["namespace natten { \n"]
    source_head += ["namespace cuda { \n"]
    source_head += ["namespace reference { \n\n"]

    source_head = "".join(source_head)

    source_body = []
    for kernel in kernels:
        source_body += "\n\n" + kernel.get_impl() + "\n\n"
    source_body = "".join(source_body)

    source_foot = "".join(
        [
            "} // namespace reference \n",
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


class NaDimDispatcher:
    def __init__(self, is_backward: bool):
        fwd_bwd_str = "BACKWARD" if is_backward else "FORWARD"
        self.name = f"DISPATCH_REFERENCE_FNA_{fwd_bwd_str}"
        self.dims: List[int] = []
        self.is_backward = is_backward

    def append(self, na_dim: int):
        self.dims.append(na_dim)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name}(rank, dtype, is_causal, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, na_dim in enumerate(self.dims):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if constexpr (rank == {na_dim})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += (
                f"  {self.name}_{na_dim}D(dtype, is_causal, __VA_ARGS__); \\\n"
            )
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += (
            '      std::cerr << "Reference FNA kernel launch failed!" \\\n'
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
    def __init__(self, is_backward: bool, na_dim: int):
        self.dtypes: List[DataType] = []
        self.na_dim = na_dim
        fwd_bwd_str = "BACKWARD" if is_backward else "FORWARD"
        self.name = f"DISPATCH_REFERENCE_FNA_{fwd_bwd_str}_{self.na_dim}D"
        self.is_backward = is_backward

    def append(self, dtype: DataType):
        self.dtypes.append(dtype)

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name}(dtype, is_causal, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        for i, dtype in enumerate(self.dtypes):
            dispatcher_str += "    "
            if i > 0:
                dispatcher_str += "else "
            dispatcher_str += f"if (dtype == {dtype.torch_name})"
            dispatcher_str += " { \\\n"
            dispatcher_str += "    "
            dispatcher_str += (
                f"  {self.name}_{dtype.short_name}(is_causal, __VA_ARGS__); \\\n"
            )
            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += (
            '      std::cerr << "Reference FNA kernel launch failed!" \\\n'
        )
        dispatcher_str += (
            '                << "'
            + f"Reference FNA-{self.na_dim}D does not support this data type."
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
        is_backward: bool,
        na_dim: int,
        dtype: DataType,
    ):
        self.na_dim = na_dim
        self.dtype = dtype

        fwd_bwd_str = "BACKWARD" if is_backward else "FORWARD"
        self.name = f"DISPATCH_REFERENCE_FNA_{fwd_bwd_str}_{self.na_dim}D_{self.dtype.short_name}"
        self.is_backward = is_backward

        self.cms: List = []

    def append(self, cm):
        assert len(cm) == self.na_dim, f"{cm} incompatible for {self.na_dim}D NA."
        self.cms.append(cm)

    def get_kernel_instance(self, cm):
        return ReferenceFnaInstance(
            na_dim=self.na_dim,
            dtype=self.dtype,
            causal=cm,
            is_backward=self.is_backward,
        )

    def get_target_name(self, cm):
        kernel = self.get_kernel_instance(cm)
        return kernel.get_name()

    def get_dispatcher(self):
        dispatcher_str = ""
        dispatcher_str += f"#define {self.name}(is_causal, ...) \\\n"
        dispatcher_str += "  [&] { \\\n"
        i = 0
        for cm in self.cms:
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

            dispatcher_str += f"    natten::cuda::reference::{self.get_target_name(cm)}(__VA_ARGS__); \\\n"

            dispatcher_str += "    } \\\n"
        dispatcher_str += "    else { \\\n"
        dispatcher_str += "    "
        dispatcher_str += (
            '      std::cerr << "Reference FNA kernel launch failed!" \\\n'
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


def write_header_file(content, path, namespaces, extra_includes=[]):
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


def generate_reference_fna_kernels(path, num_splits=2):

    NA_DIMS = [1, 2, 3]

    SUPPORTED_DTYPES = [
        Float,
        Half,
        BFloat,
    ]

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

    dtype_dispatchers = []
    cm_dispatchers = []
    kernels = []

    rank_dispatcher = NaDimDispatcher(is_backward=False)
    rank_dispatcher_bwd = NaDimDispatcher(is_backward=True)
    for na_dim in NA_DIMS:
        rank_dispatcher.append(na_dim)
        rank_dispatcher_bwd.append(na_dim)

        for is_backward in [False, True]:
            dtype_dispatcher = DTypeDispatcher(is_backward=is_backward, na_dim=na_dim)
            for dtype in SUPPORTED_DTYPES:
                dtype_dispatcher.append(dtype)

                cm_dispatcher = CausalMaskDispatcher(
                    is_backward=is_backward, na_dim=na_dim, dtype=dtype
                )
                for cm in CAUSAL_MASKS[na_dim]:
                    cm_dispatcher.append(cm)
                    kernels.append(cm_dispatcher.get_kernel_instance(cm))

                cm_dispatchers.append(cm_dispatcher)
            dtype_dispatchers.append(dtype_dispatcher)

    #

    path_to_sources = f"{path}/autogen/src/cuda/reference/"
    rel_header = "natten_autogen/cuda/reference/"
    path_to_header_dir = f"{path}/autogen/include/{rel_header}"

    os.makedirs(path_to_sources, exist_ok=False)
    os.makedirs(path_to_header_dir, exist_ok=False)

    path_headers = f"{path_to_header_dir}kernels.h"
    path_rank = f"{path_to_header_dir}interface.h"
    path_dtype = f"{path_to_header_dir}dispatch_dtype.h"
    path_cm = f"{path_to_header_dir}dispatch_cm.h"

    rel_path_headers = f"{rel_header}kernels.h"
    rel_path_dtype = f"{rel_header}dispatch_dtype.h"
    rel_path_cm = f"{rel_header}dispatch_cm.h"

    rank_disp = rank_dispatcher.get_dispatcher() + rank_dispatcher_bwd.get_dispatcher()

    dtype_disp = ""
    for dispatcher in dtype_dispatchers:
        dtype_disp += dispatcher.get_dispatcher()

    cm_disp = ""
    for dispatcher in cm_dispatchers:
        cm_disp += dispatcher.get_dispatcher()

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

    namespaces = ["natten", "cuda", "reference"]
    cuda_headers = [
        "natten/natten.h",
        "ATen/ATen.h",
        "ATen/cuda/CUDAContext.h",
        "c10/cuda/CUDAGuard.h",
        "c10/cuda/CUDAStream.h",
        "torch/extension.h",
        "natten/natten.h",
        "natten/helpers.h",
        "natten/cuda/reference/fna_reference_forward.hpp",
        "natten/cuda/reference/fna_reference_backward.hpp",
    ]
    write_header_file(rank_disp, path_rank, namespaces, cuda_headers + [rel_path_dtype])
    write_header_file(dtype_disp, path_dtype, namespaces, cuda_headers + [rel_path_cm])
    write_header_file(cm_disp, path_cm, namespaces, cuda_headers + [rel_path_headers])
    write_header_file(headers, path_headers, namespaces, cuda_headers)


def generate_reference_fna(output_directory: str, num_splits: int):
    generate_reference_fna_kernels(output_directory, num_splits=num_splits)


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
        default=2,
        help="Number of source files into which the kernels are split. Default: 2.",
    )
    args = parser.parse_args()
    generate_reference_fna(args.output_directory, args.num_splits)
