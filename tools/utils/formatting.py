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

from typing import Dict, List, Optional, Tuple, Union

from torch.profiler import profile as torch_profile

from .mappings import get_kernel_map

from .ops import CustomOp, NAOp

# NOTE: switch to | when < 3.10 support is dropped
OpType = Union[NAOp, CustomOp]


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
        op: OpType,
        op_str: str,
        time: float,
        index: int = -1,
        num_calls: int = 1,
        tag: Optional[str] = None,
    ):
        self.kernel_type = "N/A"
        if isinstance(op, CustomOp):
            self.kernel_type = op.namespace
        elif op in [
            NAOp.QKRPB,
            NAOp.AV,
            NAOp.QGRAD,
            NAOp.KGRAD,
            NAOp.VGRAD,
            NAOp.AGRAD,
            NAOp.RPBGRAD,
            NAOp.RPB,
            NAOp.LegacyPN,
            NAOp.LegacyNN,
            NAOp.LegacyIN,
        ]:
            self.kernel_type = "natten.naive"
        elif op in [NAOp.PN, NAOp.NN, NAOp.IN]:
            self.kernel_type = "natten.gemm"
        elif op in [NAOp.FusedForward, NAOp.FusedBackward]:
            self.kernel_type = "natten.fused"
        self.op = op
        self.op_str = op_str
        self.time = time
        self.index = index
        self.num_calls = num_calls
        self.tag = "N/A" if not tag else tag

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
        op = self.op if self.op == other.op else -1
        op_str = self.op_str if self.op_str == other.op_str else "Sum"
        time = self.time + other.time
        index = min(self.index, other.index)
        calls = self.num_calls + other.num_calls
        res = Result(op, op_str, time, index, calls)
        res.kernel_type = (
            self.kernel_type
            if self.kernel_type == other.kernel_type
            else f"{self.kernel_type}/{other.kernel_type}"
        )
        res.tag = self.tag if self.tag == other.tag else f"{self.tag}/{other.tag}"
        return res

    def __str__(self) -> str:
        return f"{self.kernel_type} \t\t {self.tag} \t\t {self.op_str} \t\t {self.time_str}"


def str_to_na_op(
    sym: str,
    na_dim: int,
) -> Tuple[Optional[NAOp], bool, str]:
    kernel_map = get_kernel_map(na_dim)
    for v, keys in kernel_map.items():
        for k in keys:
            if k in sym:
                kernel_tag = "N/A"
                arch_tags = ["Sm80", "Sm75", "Sm70", "Sm60", "Sm50"]
                for tag in arch_tags:
                    if tag in sym:
                        kernel_tag = tag
                        break
                return v, True, kernel_tag
    return None, False, ""


def custom_op_to_name(name: str) -> Tuple[str, str]:
    # TODO: figure out a mapping between known backend kernels
    # (at least those that are more common) instead of this.
    def remove_wrapper(s, wrapper):
        if s.startswith(wrapper):
            return s[len(wrapper) :], True
        return s, False

    name = name.strip()
    name, is_cutlass = remove_wrapper(name, "void cutlass::Kernel<")
    name, _ = remove_wrapper(name, "void (anonymous namespace)::")
    name, _ = remove_wrapper(name, "void ")
    name = name.split("<")[0].split(">")[0]
    name = name.replace("(anonymous namespace)", "ANON")
    namespace_split = name.split("::")
    default_namespace = "-" if not is_cutlass else "cutlass"
    if len(namespace_split) <= 1:
        return default_namespace, name
    return ".".join(namespace_split[:-1]), namespace_split[-1]


def convert_ops(ops: Dict[OpType, List[float]], tags: Dict) -> Optional[List[Result]]:
    output = []
    for op, values in ops.items():
        if len(values) and isinstance(op, NAOp):
            if op == NAOp.FusedForward:
                assert len(values) == 1
                output.append(
                    Result(op, NAOp.FusedForward.name, values[0], 0, tag=tags[op])
                )
                continue
            elif op == NAOp.FusedBackward:
                assert len(values) == 1
                output.append(
                    Result(op, NAOp.FusedBackward.name, values[0], 0, tag=tags[op])
                )
                continue
            elif op == NAOp.PN:
                assert len(values) in [1, 2]
                output.append(Result(op, NAOp.QKRPB.name, max(values), 0, tag=tags[op]))
                if len(values) > 1:
                    output.append(
                        Result(op, NAOp.AGRAD.name, min(values), 5, tag=tags[op])
                    )
                continue
            elif op == NAOp.NN:
                assert len(values) in [1, 2]
                output.append(
                    Result(op, NAOp.AV.name, sum(values) / len(values), 1, tag=tags[op])
                )
                if len(values) > 1:
                    output.append(
                        Result(
                            op,
                            NAOp.QGRAD.name,
                            sum(values) / len(values),
                            2,
                            tag=tags[op],
                        )
                    )
                continue
            elif op == NAOp.IN:
                assert len(values) in [1, 2]
                output.append(
                    Result(
                        op, NAOp.VGRAD.name, sum(values) / len(values), 4, tag=tags[op]
                    )
                )
                if len(values) > 1:
                    output.append(
                        Result(
                            op,
                            NAOp.KGRAD.name,
                            sum(values) / len(values),
                            3,
                            tag=tags[op],
                        )
                    )
                continue
            elif op == NAOp.LegacyPN:
                assert len(values) in [1, 2]
                output.append(Result(op, NAOp.QKRPB.name, max(values), 0, tag=tags[op]))
                if len(values) > 1:
                    output.append(
                        Result(op, NAOp.AGRAD.name, min(values), 5, tag=tags[op])
                    )
                continue
            elif op == NAOp.LegacyNN:
                assert len(values) in [1, 2]
                output.append(
                    Result(op, NAOp.AV.name, sum(values) / len(values), 1, tag=tags[op])
                )
                if len(values) > 1:
                    output.append(
                        Result(
                            op,
                            NAOp.QGRAD.name,
                            sum(values) / len(values),
                            2,
                            tag=tags[op],
                        )
                    )
                continue
            elif op == NAOp.LegacyIN:
                assert len(values) in [1, 2]
                output.append(
                    Result(
                        op, NAOp.VGRAD.name, sum(values) / len(values), 4, tag=tags[op]
                    )
                )
                if len(values) > 1:
                    output.append(
                        Result(
                            op,
                            NAOp.KGRAD.name,
                            sum(values) / len(values),
                            3,
                            tag=tags[op],
                        )
                    )
                continue
            assert len(values) == 1
            output.append(Result(op, op.name, values[0], op.value, tag=tags[op]))
        elif len(values):
            assert isinstance(op, CustomOp)
            for value in values:
                output.append(Result(op, op.name, value, 999, tag="-"))
    filtered_output = []
    rpb_op = None
    for res in output:
        if res.op == NAOp.RPB:
            rpb_op = res
        else:
            filtered_output.append(res)
    if rpb_op is not None:
        for res in filtered_output:
            if res.op_str == NAOp.QKRPB.name:
                res.time += rpb_op.time
                break
    return filtered_output


def extract_na_ops(
    profiler: torch_profile,
    na_dim: int,
) -> Optional[List[Result]]:
    events = profiler.events()
    logged_ops: Dict[OpType, List[float]] = {na_op: [] for na_op in NAOp}
    tags: Dict[NAOp, Optional[str]] = {na_op: None for na_op in NAOp}
    for evt in events:
        op, valid, tag = str_to_na_op(sym=evt.key, na_dim=na_dim)
        if valid and isinstance(op, NAOp):
            if tags[op] is None:
                tags[op] = tag
            else:
                assert tags[op] == tag
            logged_ops[op].append(evt.device_time_total)
        elif evt.device_time_total > 0:
            op_namespace, op_name = custom_op_to_name(evt.key)
            op_key = CustomOp(op_name, op_namespace)
            if op_key not in logged_ops:
                logged_ops[op_key] = [evt.device_time_total]
            else:
                logged_ops[op_key].append(evt.device_time_total)

    converted_ops = convert_ops(logged_ops, tags)
    return None if converted_ops is None else sorted(converted_ops)
