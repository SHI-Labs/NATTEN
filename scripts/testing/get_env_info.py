# Copyright (c) 2022 - 2026 Ali Hassani.
# Made largely with Claude Code

import warnings

warnings.filterwarnings("ignore")

import natten
import torch
from natten.utils.device import get_device_cc

print("-------------------------------------------------")
print("Environment info")
print("-------------------------------------------------")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch built with CUDA Compiler version: {torch.version.cuda}")
print(f"CUDA available: {'✓' if torch.cuda.is_available() else '✗'}")

if torch.cuda.is_available():
    n = torch.cuda.device_count()
    ccs = sorted(set(f"SM{get_device_cc(torch.device('cuda', i))}" for i in range(n)))
    names = sorted(set(torch.cuda.get_device_name(i) for i in range(n)))
    print(
        f"GPU architectures detected: {', '.join(ccs)} ({n} device{'s' if n > 1 else ''})"
    )
    print(f"GPUs: {', '.join(names)}")

print("-------------------------------------------------")

print(f"NATTEN version: {natten.__version__}")
print(f"With libnatten: {'✓' if natten.HAS_LIBNATTEN else '✗'}")

print("-------------------------------------------------")
print()
