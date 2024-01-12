#################################################################################################
# Copyright (c) 2022-2024 Ali Hassani.
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

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.profiler import profile as torch_profile, ProfilerActivity, record_function


def qk_cross(query: Tensor, key: Tensor) -> Tensor:
    """
    Performs cross attention between arbitrary-rank tensors ([batch, heads, ..., dim]).
    """
    output_shape = [x for x in query.shape[:-1]] + [key.shape[-2]]
    query_bmm_view = query.view(query.shape[0], query.shape[1], -1, query.shape[-1])
    key_transposed_bmm_view = key.view(
        key.shape[0], key.shape[1], -1, key.shape[-1]
    ).transpose(-2, -1)
    output = torch.matmul(query_bmm_view, key_transposed_bmm_view)
    return output.reshape(*output_shape)


def av_cross(attn: Tensor, value: Tensor) -> Tensor:
    """
    Applies cross attention weights.
    """
    output_shape = [x for x in attn.shape[:-1]] + [value.shape[-1]]
    attn_bmm_view = attn.view(attn.shape[0], attn.shape[1], -1, attn.shape[-1])
    value_bmm_view = value.view(value.shape[0], value.shape[1], -1, value.shape[-1])
    output = torch.matmul(attn_bmm_view, value_bmm_view)
    return output.reshape(*output_shape)


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


@dataclass(frozen=True)
class CustomOp:
    name: str
    namespace: str


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
        op: NAOp | CustomOp,
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


def convert_ops(
    ops: Dict[NAOp | CustomOp, List[float]], tags: Dict
) -> Optional[List[Result]]:
    output = []
    for op, values in ops.items():
        if len(values) and isinstance(op, NAOp):
            if op == NAOp.PN:
                # if len(values) != 2:
                #    return None
                output.append(Result(op, NAOp.QKRPB.name, max(values), 0, tag=tags[op]))
                output.append(Result(op, NAOp.AGRAD.name, min(values), 5, tag=tags[op]))
                continue
            elif op == NAOp.NN:
                # if len(values) != 2:
                #    return None
                output.append(
                    Result(op, NAOp.AV.name, sum(values) / len(values), 1, tag=tags[op])
                )
                output.append(
                    Result(
                        op, NAOp.QGRAD.name, sum(values) / len(values), 2, tag=tags[op]
                    )
                )
                continue
            elif op == NAOp.IN:
                # if len(values) != 2:
                #    return None
                output.append(
                    Result(
                        op, NAOp.VGRAD.name, sum(values) / len(values), 4, tag=tags[op]
                    )
                )
                output.append(
                    Result(
                        op, NAOp.KGRAD.name, sum(values) / len(values), 3, tag=tags[op]
                    )
                )
                continue
            elif op == NAOp.LegacyPN:
                # if len(values) != 2:
                #    return None
                output.append(Result(op, NAOp.QKRPB.name, max(values), 0, tag=tags[op]))
                output.append(Result(op, NAOp.AGRAD.name, min(values), 5, tag=tags[op]))
                continue
            elif op == NAOp.LegacyNN:
                # if len(values) != 2:
                #    return None
                output.append(
                    Result(op, NAOp.AV.name, sum(values) / 2, 1, tag=tags[op])
                )
                output.append(
                    Result(op, NAOp.QGRAD.name, sum(values) / 2, 2, tag=tags[op])
                )
                continue
            elif op == NAOp.LegacyIN:
                # if len(values) != 2:
                #    return None
                output.append(
                    Result(op, NAOp.VGRAD.name, sum(values) / 2, 4, tag=tags[op])
                )
                output.append(
                    Result(op, NAOp.KGRAD.name, sum(values) / 2, 3, tag=tags[op])
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


def extract_na_ops(
    profiler: torch_profile, _keywords: Dict[str, List[str]]
) -> Optional[List[Result]]:
    events = profiler.events()
    logged_ops: Dict[NAOp | CustomOp, List[float]] = {na_op: [] for na_op in NAOp}
    tags: Dict[NAOp, Optional[str]] = {na_op: None for na_op in NAOp}
    for evt in events:
        op, valid, tag = str_to_na_op(sym=evt.key, **_keywords)
        if valid and isinstance(op, NAOp):
            if tags[op] is None:
                tags[op] = tag
            else:
                assert tags[op] == tag
            logged_ops[op].append(evt.cuda_time_total)
        elif evt.cuda_time_total > 0:
            op_namespace, op_name = custom_op_to_name(evt.key)
            op_key = CustomOp(op_name, op_namespace)
            if op_key not in logged_ops:
                logged_ops[op_key] = [evt.cuda_time_total]
            else:
                logged_ops[op_key].append(evt.cuda_time_total)

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


def profile_extra_tokens_with_torch(
    qk_func,
    av_func,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    extra_key: Tensor,
    extra_value: Tensor,
    kernel_size: int,
    dilation: int,
    warmup_steps: int,
    disable_concat_fusion: bool = False,
) -> torch_profile:
    def run_ops(query, key, value, extra_key, extra_value):
        if not disable_concat_fusion:
            attn = qk_func(query, key, kernel_size, dilation, additional_keys=extra_key)
            attn = attn.softmax(dim=-1)
            out = av_func(
                attn, value, kernel_size, dilation, additional_values=extra_value
            )
            return out

        attn_na = qk_func(query, key, kernel_size, dilation)
        attn_extra = qk_cross(query, extra_key)
        n_na_weights, n_extra_weights = attn_na.shape[-1], attn_extra.shape[-1]
        attn = torch.cat([attn_na, attn_extra], dim=-1)
        attn = attn.softmax(dim=-1)
        attn_na, attn_extra = attn.split([n_na_weights, n_extra_weights], dim=-1)
        attn_na = attn_na.contiguous()
        attn_extra = attn_extra.contiguous()
        out_na = av_func(attn_na, value, kernel_size, dilation)
        out_extra = av_cross(attn_extra, extra_value)
        out_na += out_extra
        return out_na

    with torch.no_grad():
        for _ in range(warmup_steps):
            run_ops(query, key, value, extra_key, extra_value)

        with torch_profile(
            activities=[ProfilerActivity.CUDA], record_shapes=True
        ) as prof:
            with record_function("model_inference"):
                run_ops(query, key, value, extra_key, extra_value)

    return prof
