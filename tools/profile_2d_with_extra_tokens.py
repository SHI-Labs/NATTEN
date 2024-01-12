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

import click

import natten
import torch
from utils.pretty_printer import print_table
from utils.profiler_2d import profile_na2d_extra_tokens


@click.command()
@click.option("-b", "--batch-size", default=2)
@click.option("-n", "--heads", default=12)
@click.option("-x", "--height", default=56)
@click.option("-y", "--width", default=56)
@click.option("-d", "--dim", default=32)
@click.option("-k", "--kernel-size", default=13)
@click.option("--dilation", default=1)
@click.option("-l", "--num-extra-tokens", default=77)
@click.option("--fp16", is_flag=True)
@click.option("--bf16", is_flag=True)
@click.option("--disable-tiled", is_flag=True)
@click.option("--disable-gemm", is_flag=True)
@click.option("--disable-tf32", is_flag=True)
@click.option("--warmup-steps", default=10)
@click.option("--disable-fusion", is_flag=True)
@click.option("--broadcast-batch", is_flag=True)
def profile_2d(
    batch_size: int,
    heads: int,
    height: int,
    width: int,
    dim: int,
    kernel_size: int,
    dilation: int,
    num_extra_tokens: int,
    fp16: bool,
    bf16: bool,
    disable_tiled: bool,
    disable_gemm: bool,
    disable_tf32: bool,
    warmup_steps: int,
    disable_fusion: bool,
    broadcast_batch: bool,
):

    dtype = torch.float32
    if fp16:
        dtype = torch.float16
    if bf16:
        dtype = torch.bfloat16
    if disable_tiled:
        natten.libnatten.set_tiled_na(False)
    if disable_gemm:
        natten.libnatten.set_gemm_na(False)
    if disable_tf32:
        natten.libnatten.set_gemm_tf32(False)

    logged_ops = profile_na2d_extra_tokens(
        batch_size=batch_size,
        heads=heads,
        height=height,
        width=width,
        dim=dim,
        kernel_size=kernel_size,
        dilation=dilation,
        num_extra_tokens=num_extra_tokens,
        dtype=dtype,
        warmup_steps=warmup_steps,
        disable_concat_fusion=disable_fusion,
        broadcast_batch=broadcast_batch,
    )

    title = "Profiler results"
    headers = ["Kernel type", "Arch", "Operation", "CUDA time"]
    values = [[x.kernel_type, x.tag, x.op_str, x.time_str] for x in logged_ops]
    total = sum(logged_ops)
    values.append(["", "", "Total", total.time_str])
    print_table(title, headers, values)


if __name__ == "__main__":
    profile_2d()
