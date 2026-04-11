# Copyright (c) 2022 - 2026 Ali Hassani.
# Made largely with Claude Code

import warnings

warnings.filterwarnings("ignore")

import platform
import sys
from datetime import datetime

try:
    "✓✗━─".encode(sys.stdout.encoding or "ascii")
    SEPARATOR = "━" * 49
    THIN_SEP = "─" * 49
    YES = "✓ YES"
    NO = "✗ NO"
except (UnicodeEncodeError, LookupError):
    SEPARATOR = "=" * 49
    THIN_SEP = "-" * 49
    YES = "YES"
    NO = "NO"

system_info = []
torch_info = []
natten_info = []
paths_info = []

# System info
system_info.append(("Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
system_info.append(("OS", f"{platform.system()} {platform.release()}"))
system_info.append(("Platform", platform.platform()))
system_info.append(("CPU", platform.processor() or platform.machine()))
system_info.append(("Python version", sys.version.split()[0]))

# PyTorch info
try:
    import torch
except ImportError:
    torch = None

if torch is None:
    torch_info.append(("PyTorch", "not installed"))
else:
    torch_info.append(("PyTorch version", torch.__version__))
    cxx11_abi = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None)
    if cxx11_abi is not None:
        torch_info.append(("GLIBCXX_USE_CXX11_ABI", YES if cxx11_abi else NO))
    built_with_cuda = torch.version.cuda is not None
    torch_info.append(("Built with CUDA", torch.version.cuda if built_with_cuda else NO))
    torch_info.append(("CUDA available", YES if torch.cuda.is_available() else NO))
    if built_with_cuda and not torch.cuda.is_available():
        torch_info.append(("Note", "built with CUDA but no GPU detected"))

    if torch.cuda.is_available():
        try:
            from natten.utils.device import get_device_cc

            n = torch.cuda.device_count()
            ccs = sorted(
                set(f"SM{get_device_cc(torch.device('cuda', i))}" for i in range(n))
            )
            names = sorted(set(torch.cuda.get_device_name(i) for i in range(n)))
            torch_info.append((
                "GPU architectures",
                f"{', '.join(ccs)} ({n} device{'s' if n > 1 else ''})",
            ))
            torch_info.append(("GPUs", ", ".join(names)))
        except ImportError:
            n = torch.cuda.device_count()
            names = sorted(set(torch.cuda.get_device_name(i) for i in range(n)))
            torch_info.append((
                "CUDA devices",
                f"{n} device{'s' if n > 1 else ''}",
            ))
            torch_info.append(("GPUs", ", ".join(names)))

# NATTEN info
if torch is None:
    natten_info.append(("NATTEN", "skipped (no PyTorch)"))
else:
    try:
        import natten
    except ImportError:
        natten = None

    if natten is None:
        natten_info.append(("NATTEN", "not installed"))
    else:
        natten_info.append(("NATTEN version", natten.__version__))
        natten_info.append(("With libnatten", YES if natten.HAS_LIBNATTEN else NO))

# Paths
import os

paths_info.append(("Hostname", platform.node()))
paths_info.append(("Python executable", sys.executable))
if torch is not None:
    paths_info.append(("PyTorch path", os.path.dirname(torch.__file__)))
if torch is not None and natten is not None:
    paths_info.append(("NATTEN path", os.path.dirname(natten.__file__)))


# Print
sections = [
    ("System info", system_info),
    ("PyTorch info", torch_info),
    ("NATTEN info", natten_info),
    ("Paths", paths_info),
]

print()
print(SEPARATOR)
print("natten.org/info.py")
print("curl -sSL https://natten.org/info.py | python3 -")
print(SEPARATOR)
for i, (title, entries) in enumerate(sections):
    print()
    print(title)
    print(THIN_SEP)
    for label, value in entries:
        print(f"{label}: {value}")
print(SEPARATOR)
print()
