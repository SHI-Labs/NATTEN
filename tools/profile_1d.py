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

import argparse

import natten
import torch
from utils.pretty_printer import print_table
from utils.profiler_1d import profile_na1d


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int, default=128)
    parser.add_argument("-n", "--heads", type=int, default=16)
    parser.add_argument("-x", "--length", type=int, default=144)
    parser.add_argument("-d", "--dim", type=int, default=32)
    parser.add_argument("-k", "--kernel-size", type=int, default=49)
    parser.add_argument("--dilation", type=int, default=1)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--bias", action="store_true", default=False)
    parser.add_argument("--disable-tiled", action="store_true", default=False)
    parser.add_argument("--disable-gemm", action="store_true", default=False)
    parser.add_argument("--disable-tf32", action="store_true", default=False)
    parser.add_argument("--warmup-steps", type=int, default=10)
    args = parser.parse_args()

    dtype = torch.float32
    if args.fp16:
        dtype = torch.float16
    if args.bf16:
        dtype = torch.bfloat16
    if args.disable_tiled:
        natten.libnatten.set_tiled_na(False)
    if args.disable_gemm:
        natten.libnatten.set_gemm_na(False)
    if args.disable_tf32:
        natten.libnatten.set_gemm_tf32(False)

    logged_ops = profile_na1d(
        batch_size=args.batch_size,
        heads=args.heads,
        length=args.length,
        dim=args.dim,
        kernel_size=args.kernel_size,
        dilation=args.dilation,
        dtype=dtype,
        warmup_steps=args.warmup_steps,
        enable_bias=args.bias,
    )
    title = "Profiler results"
    headers = ["Kernel type", "Arch", "Operation", "CUDA time"]
    values = [[x.kernel_type, x.tag, x.op_str, x.time_str] for x in logged_ops]
    total = sum(logged_ops)
    values.append(["", "", "Total", total.time_str])
    print_table(title, headers, values)
