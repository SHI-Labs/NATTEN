#################################################################################################
# Copyright (c) 2022-2025 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################

from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
from torch.profiler import profile as torch_profile


class LibNattenOp(Enum):
    Unrecognized = 0

    # Fused
    FnaForward = 1
    FnaBackward = 2

    # FMHA
    FmhaForward = 3
    FmhaBackward = 4

    # Misc
    Reduction = 5
    Elementwise = 6


NATTEN_TAGS = {
    LibNattenOp.FnaForward: [
        "natten::cuda::fna::FusedNeighborhoodAttentionKernel",
        "cutlass::fna::kernel::Sm100FnaFwdKernelTmaWarpspecialized",
        "cutlass::fna::collective::FnaMainloopTmaWarpSpecializedSm90",
        "cutlass::fna::collective::FnaMainloopTmaSm90",
    ],
    LibNattenOp.FnaBackward: [
        "natten::cuda::fna::FusedNeighborhoodAttentionBackwardKernel<",
        "cutlass::fna::collective::FnaBwdMainloopTmaWarpSpecializedSm90",
        "cutlass::fna::kernel::Sm100FnaBwdKernelTmaWarpSpecialized",
    ],
    LibNattenOp.FmhaForward: [
        "cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized",
        "cutlass::fmha::collective::FmhaMainloopTmaWarpSpecializedSm90",
        "cutlass::fmha::collective::FmhaMainloopTmaSm90",
        "natten::cuda::fmha::FusedNeighborhoodAttentionKernel",
        "natten::cuda::fmha::AttentionKernel",
    ],
    LibNattenOp.FmhaBackward: [
        "cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized",
        "cutlass::fmha::collective::FmhaMainloopTmaWarpSpecializedSm90",
        "cutlass::fmha::collective::FmhaMainloopTmaSm90",
        "natten::cuda::fmha::AttentionBackwardKernel",
        "cutlass::fmha::collective::FmhaBwdMainloopTmaWarpSpecializedSm90",
        "cutlass::fmha::kernel::Sm100FmhaBwdKernelTmaWarpSpecialized",
    ],
    LibNattenOp.Reduction: [
        "natten::cuda::reduction::kernel::ComputeDelta",
        "cutlass::fmha::kernel::FmhaKernelBwdSumOdO",
    ],
    LibNattenOp.Elementwise: [
        "cutlass::fmha::kernel::FmhaKernelBwdConvert",
    ],
}


def _format_time(time_us: float) -> str:
    """
    Source: https://github.com/pytorch/pytorch/blob/orig/release/1.13/torch/autograd/profiler_util.py
    """
    US_IN_SECOND = 1000.0 * 1000.0
    US_IN_MS = 1000.0
    if time_us >= US_IN_SECOND:
        return "{:.3f}s".format(time_us / US_IN_SECOND)
    if time_us >= US_IN_MS:
        return "{:.3f}ms".format(time_us / US_IN_MS)
    return "{:.3f}us".format(time_us)


class Result:
    def __init__(
        self,
        op: Optional[LibNattenOp],
        op_str: str,
        time: float,
        index: int = -1,
        num_calls: int = 1,
        tag: str = "N/A",
        framework: str = "N/A",
        kernel_type: str = "N/A",
    ):
        self.framework = framework
        self.kernel_type = kernel_type
        self.tag = "N/A" if not tag else tag

        if op in [
            LibNattenOp.FnaForward,
            LibNattenOp.FnaBackward,
            LibNattenOp.FmhaForward,
            LibNattenOp.FmhaBackward,
        ]:
            self.kernel_type = "attention"
            self.framework = "CUTLASS"

        elif op in [LibNattenOp.Reduction]:
            self.kernel_type = "reduction"
            self.framework = "CUTLASS"
            self.tag = "-"

        elif op in [LibNattenOp.Elementwise]:
            self.kernel_type = "elementwise"
            self.framework = "CUTLASS"
            self.tag = "-"

        self.op = op
        if op is not None and isinstance(op, LibNattenOp):
            self.op_str = op.name
        else:
            self.op_str = op_str

        self.time = time
        self.index = index
        self.num_calls = num_calls

    def to_str_name(self):
        return (
            f"framework_{self.framework}_type{self.kernel_type}_"
            + f"tag_{self.tag}_op_str{self.op_str}"
        )

    @property
    def op_name_formatted(self):
        op_str = "-" if self.op_str is None else str(self.op_str)
        MAX_CHARS = 45
        if len(op_str) > MAX_CHARS:
            return op_str[:MAX_CHARS] + "..."
        return op_str

    @property
    def time_str(self) -> str:
        return _format_time(self.time)

    def __le__(self, other) -> bool:
        return self.index <= other.index

    def __ge__(self, other) -> bool:
        return self.index >= other.index

    def __lt__(self, other) -> bool:
        return self.index < other.index

    def __gt__(self, other) -> bool:
        return self.index > other.index

    def __eq__(self, other) -> bool:
        return self.index == other.index

    def __ne__(self, other) -> bool:
        return self.index != other.index

    def __radd__(self, other):
        return self

    def __add__(self, other):
        return Result(
            op=self.op if self.op == other.op else "INV",
            op_str=self.op_str if self.op_str == other.op_str else "INV",
            time=self.time + other.time,
            index=min(self.index, other.index),
            num_calls=self.num_calls + other.num_calls,
            tag=self.tag if self.tag == other.tag else "INV",
            framework=self.framework if self.framework == other.framework else "INV",
            kernel_type=(
                self.kernel_type if self.kernel_type == other.kernel_type else "INV"
            ),
        )


def get_natten_op(sym: str) -> Optional[LibNattenOp]:
    for natten_op_type, tags in NATTEN_TAGS.items():
        for tag in tags:
            if tag in sym:
                return natten_op_type
    return None


KERNEL_TYPE_LOOKUP = {
    "attention": ["sdpa", "fmha", "flash_attn"],
    "gemm": ["gemm", "matmul"],
    "gemv": ["gemv"],
    "elementwise": ["elementwise"],
    "softmax": ["softmax"],
}


# TODO: I really hate how we're parsing symbols in this entire file!
def strip_str_name(name: str) -> Tuple[str, str]:
    if name.startswith("cudnn_generated_fort_native_"):
        return "attention", name.replace("cudnn_generated_fort_native_sdpa_", "")

    def remove_wrapper(s, wrapper):
        if s.startswith(wrapper):
            return s[len(wrapper) :], True
        return s, False

    kernel_type = "-"
    found = False
    for kernel_type_, tag_list in KERNEL_TYPE_LOOKUP.items():
        for tag in tag_list:
            if tag in name.lower():
                kernel_type = kernel_type_
                found = True
                break
        if found:
            break

    name = name.strip()
    # name, is_cutlass = remove_wrapper(name, "void cutlass::Kernel<")
    name, _ = remove_wrapper(name, "void (anonymous namespace)::")
    name, _ = remove_wrapper(name, "void ")
    name = name.split("<")[0].split(">")[0]
    name = name.replace("(anonymous namespace)::", "")
    name = name.replace("(anonymous namespace)", "")
    name = name.replace("at::native::", "")

    if name.startswith("cudnn_generated_fort_native_"):
        name = name.replace("cudnn_generated_fort_native_", "")
    if name.startswith("triton_"):
        name = name.replace("triton_", "")

    namespace_split = name.split("::")

    if len(namespace_split) <= 1:
        return kernel_type, name

    return kernel_type, ".".join(namespace_split)


ARCH_LOOKUP = {
    "Sm50": ["sm50", "maxwell"],
    "Sm52": ["sm52"],
    "Sm53": ["sm53"],
    "Sm60": ["sm60", "pascal"],
    "Sm61": ["sm61"],
    "Sm62": ["sm62"],
    "Sm70": ["sm70", "volta"],
    "Sm72": ["sm72"],
    "Sm75": ["sm75", "turing"],
    "Sm80": ["sm80"],
    "Sm86": ["sm86"],
    "Sm87": ["sm87"],
    "Sm8X": ["ampere"],
    "Sm89": ["sm89"],
    "Sm90": ["sm90", "hopper"],
    "Sm100": ["sm100", "blackwell"],
    "Sm101": ["sm101"],
    "Sm120": ["sm120"],
}


FRAMEWORK_LOOKUP = {
    "CUTLASS": ["cutlass", "cute"],
    "cuDNN": ["cudnn"],
    "cuBLAS": ["cublas"],
    "PyTorch": ["c10", "aten", "at::native"],
    "Triton": ["triton"],
}


def get_arch(sym: str) -> str:
    for arch, tag_list in ARCH_LOOKUP.items():
        for tag in tag_list:
            if tag in sym.lower():
                return arch
    return "-"


def get_framework(sym: str) -> str:
    for k, tag_list in FRAMEWORK_LOOKUP.items():
        for tag in tag_list:
            if tag in sym.lower():
                return k
    return "-"


def get_op_prio(op: Optional[LibNattenOp]) -> int:
    if op is None or not isinstance(op, LibNattenOp):
        return 999

    return int(op.value)


def convert_to_natten_profiler_ops(
    profiler: torch_profile,
) -> Optional[List[Result]]:

    events = profiler.events()

    results: Dict[str, Result] = {}
    for evt in events:
        if evt.key.startswith("Memcpy") or evt.key.startswith("Memset"):
            continue

        time_total = (
            evt.device_time_total if torch.cuda.is_available() else evt.cpu_time_total
        )

        natten_op = get_natten_op(evt.key)
        arch_tag = get_arch(evt.key)
        framework = get_framework(evt.key)
        kernel_type, op_str = strip_str_name(evt.key)

        if time_total > 0:
            result = Result(
                op=natten_op,
                op_str=op_str,
                time=time_total,
                index=get_op_prio(natten_op),  # Put NATTEN ops at the top
                num_calls=1,
                tag=arch_tag,
                framework=framework,
                kernel_type=kernel_type,
            )

            key = result.to_str_name()
            if key in results:
                results[key] += result
            else:
                results[key] = result

    results_final = [v for v in results.values()]
    return None if results_final is None else sorted(results_final)
