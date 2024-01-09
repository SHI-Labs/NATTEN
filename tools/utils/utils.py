#################################################################################################
# Copyright (c) 2023 Ali Hassani.
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
from torch import Tensor
from torch.profiler import profile as torch_profile, ProfilerActivity, record_function


class NAOp(Enum):
    # GEMM
    PN = 0
    NN = 1
    IN = 2

    # Naive
    QKRPB = 3
    AV = 4
    QGRAD = 5
    KGRAD = 6
    VGRAD = 7
    AGRAD = 8

    RPB = 9
    RPBGRAD = 10

    LegacyPN = 11
    LegacyNN = 12
    LegacyIN = 13


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
        op: NAOp,
        op_str: str,
        time: float,
        index: int = -1,
        num_calls: int = 1,
        tag: Optional[str] = None,
    ):
        self.kernel_type = "N/A"
        if op in [
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
            self.kernel_type = "naive"
        elif op in [NAOp.PN, NAOp.NN, NAOp.IN]:
            self.kernel_type = "gemm"
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


def convert_ops(ops: Dict[NAOp, List[float]], tags: Dict) -> Optional[List[Result]]:
    output = []
    for op, values in ops.items():
        if len(values):
            if op == NAOp.PN:
                if len(values) != 2:
                    return None
                output.append(Result(op, NAOp.QKRPB.name, max(values), 0, tag=tags[op]))
                output.append(Result(op, NAOp.AGRAD.name, min(values), 5, tag=tags[op]))
                continue
            elif op == NAOp.NN:
                if len(values) != 2:
                    return None
                output.append(
                    Result(op, NAOp.AV.name, sum(values) / 2, 1, tag=tags[op])
                )
                output.append(
                    Result(op, NAOp.QGRAD.name, sum(values) / 2, 2, tag=tags[op])
                )
                continue
            elif op == NAOp.IN:
                if len(values) != 2:
                    return None
                output.append(
                    Result(op, NAOp.VGRAD.name, sum(values) / 2, 4, tag=tags[op])
                )
                output.append(
                    Result(op, NAOp.KGRAD.name, sum(values) / 2, 3, tag=tags[op])
                )
                continue
            elif op == NAOp.LegacyPN:
                if len(values) != 2:
                    return None
                output.append(Result(op, NAOp.QKRPB.name, max(values), 0, tag=tags[op]))
                output.append(Result(op, NAOp.AGRAD.name, min(values), 5, tag=tags[op]))
                continue
            elif op == NAOp.LegacyNN:
                if len(values) != 2:
                    return None
                output.append(
                    Result(op, NAOp.AV.name, sum(values) / 2, 1, tag=tags[op])
                )
                output.append(
                    Result(op, NAOp.QGRAD.name, sum(values) / 2, 2, tag=tags[op])
                )
                continue
            elif op == NAOp.LegacyIN:
                if len(values) != 2:
                    return None
                output.append(
                    Result(op, NAOp.VGRAD.name, sum(values) / 2, 4, tag=tags[op])
                )
                output.append(
                    Result(op, NAOp.KGRAD.name, sum(values) / 2, 3, tag=tags[op])
                )
                continue
            if len(values) != 1:
                return None
            output.append(Result(op, op.name, values[0], op.value, tag=tags[op]))
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


def str_to_na_op(
    sym: str,
    pn_keywords: List[str],
    nn_keywords: List[str],
    in_keywords: List[str],
    rpb_keywords: List[str],
    rpbgrad_keywords: List[str],
    qkrpb_keywords: List[str],
    av_keywords: List[str],
    qgrad_keywords: List[str],
    kgrad_keywords: List[str],
    vgrad_keywords: List[str],
    agrad_keywords: List[str],
    legacy_pn_keywords: List[str],
    legacy_nn_keywords: List[str],
    legacy_in_keywords: List[str],
) -> Tuple[Optional[NAOp], bool, str]:
    kernel_map = {
        NAOp.PN: pn_keywords,
        NAOp.NN: nn_keywords,
        NAOp.IN: in_keywords,
        NAOp.LegacyPN: legacy_pn_keywords,
        NAOp.LegacyNN: legacy_nn_keywords,
        NAOp.LegacyIN: legacy_in_keywords,
        NAOp.RPB: rpb_keywords,
        NAOp.RPBGRAD: rpbgrad_keywords,
        NAOp.QKRPB: qkrpb_keywords,
        NAOp.AV: av_keywords,
        NAOp.QGRAD: qgrad_keywords,
        NAOp.KGRAD: kgrad_keywords,
        NAOp.VGRAD: vgrad_keywords,
        NAOp.AGRAD: agrad_keywords,
    }
    for v, keys in kernel_map.items():
        for k in keys:
            if k in sym:
                kernel_tag = "N/A"
                arch_tags = ["Sm80", "Sm75", "Sm70"]
                for tag in arch_tags:
                    if tag in sym:
                        kernel_tag = tag
                        break
                return v, True, kernel_tag
    return None, False, ""


def extract_na_ops(
    profiler: torch_profile, _keywords: Dict[str, List[str]]
) -> Optional[List[Result]]:
    events = profiler.events()
    logged_ops: Dict[NAOp, List[float]] = {na_op: [] for na_op in NAOp}
    tags: Dict[NAOp, Optional[str]] = {na_op: None for na_op in NAOp}
    for evt in events:
        op, valid, tag = str_to_na_op(sym=evt.key, **_keywords)
        if valid and isinstance(op, NAOp):
            if tags[op] is None:
                tags[op] = tag
            else:
                assert tags[op] == tag
            logged_ops[op].append(evt.cuda_time_total)

    converted_ops = convert_ops(logged_ops, tags)
    return None if converted_ops is None else sorted(converted_ops)


def profile_with_torch(
    qk_op,
    av_op,
    qk_backward_op,
    av_backward_op,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn: Tensor,
    out: Tensor,
    d_attn: Tensor,
    d_query: Tensor,
    d_key: Tensor,
    d_value: Tensor,
    bias: Optional[Tensor],
    d_bias: Optional[Tensor],
    kernel_size: int,
    dilation: int,
    warmup_steps: int,
) -> torch_profile:
    def run_ops():
        qk_op(attn, query, key, bias, kernel_size, dilation)
        av_op(out, attn, value, kernel_size, dilation)
        av_backward_op(d_attn, d_value, out, attn, value, kernel_size, dilation)
        qk_backward_op(
            d_query, d_key, d_bias, d_attn, query, key, kernel_size, dilation
        )

    with torch.no_grad():
        for _ in range(warmup_steps):
            run_ops()

        with torch_profile(
            activities=[ProfilerActivity.CUDA], record_shapes=True
        ) as prof:
            with record_function("model_inference"):
                run_ops()

    return prof
