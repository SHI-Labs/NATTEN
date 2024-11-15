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

from functools import partial

import click

import natten
import torch

from utils import (
    generate_1d_problem,
    print_table,
    profile_fmha_with_torch,
    profile_na_with_torch,
)


@click.command()
@click.option("-b", "--batch-size", default=128)
@click.option("-n", "--heads", default=16)
@click.option("-x", "--length", default=144)
@click.option("-d", "--dim", default=32)
@click.option("-k", "--kernel-size", default=49)
@click.option("--dilation", default=1)
@click.option("--fp16", is_flag=True)
@click.option("--bf16", is_flag=True)
@click.option("--bias", is_flag=True)
@click.option("--causal", is_flag=True)
@click.option("--disable-gemm", is_flag=True)
@click.option("--disable-tf32", is_flag=True)
@click.option("--disable-autotuner", is_flag=True)
@click.option("--warmup-steps", default=10)
@click.option("--fuse", is_flag=True)
@click.option("--fmha", is_flag=True)
@click.option("--fav2", is_flag=True)
@click.option("--backprop", is_flag=True)
@click.option("--add-kv", default=0)
def profile_1d(
    batch_size: int,
    heads: int,
    length: int,
    dim: int,
    kernel_size: int,
    dilation: int,
    fp16: bool,
    bf16: bool,
    bias: bool,
    causal: bool,
    disable_gemm: bool,
    disable_tf32: bool,
    disable_autotuner: bool,
    warmup_steps: int,
    fuse: bool,
    fmha: bool,
    fav2: bool,
    backprop: bool,
    add_kv: int,
):

    dtype = torch.float32
    if fp16:
        dtype = torch.float16
    if bf16:
        dtype = torch.bfloat16

    if disable_gemm:
        natten.libnatten.set_gemm_na(False)
    if disable_tf32:
        natten.libnatten.set_gemm_tf32(False)

    if fuse:
        natten.use_fused_na()
        natten.use_kv_parallelism_in_fused_na()
        natten.set_memory_usage_preference("unrestricted")

        if disable_autotuner:
            natten.use_autotuner(False, False, False, False)
        else:
            natten.use_autotuner(True, True)

    func = partial(profile_na_with_torch, fuse=fuse)
    if fmha:
        func = partial(profile_fmha_with_torch, xformers=True)
    if fav2:
        func = partial(profile_fmha_with_torch, xformers=False)

    problem = generate_1d_problem(
        batch_size=batch_size,
        heads=heads,
        length=length,
        dim=dim,
        kernel_size=kernel_size,
        dilation=dilation,
        dtype=dtype,
        has_bias=bias,
        is_causal=causal,
        additional_kv_length=add_kv,
    )
    logged_ops = func(
        problem=problem,
        warmup_steps=warmup_steps,
        disable_backward=not backprop,
    )

    title = "Profiler results"
    headers = ["Kernel type", "Arch", "Operation", "CUDA time"]
    values = [[x.kernel_type, x.tag, x.op_str, x.time_str] for x in logged_ops]
    total = sum(logged_ops)
    values.append(["", "", "Total", total.time_str])
    print_table(title, headers, values)


if __name__ == "__main__":
    profile_1d()
