#!/usr/bin/env python3
"""Static guards for the intentionally narrow RelayVSR NATTEN build profile."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]

def require(path: str, pattern: str, label: str) -> None:
    source = (ROOT / path).read_text(encoding="utf-8")
    if not re.search(pattern, source, flags=re.MULTILINE | re.DOTALL):
        raise SystemExit(f"FAIL: {label} not found in {path}")
    print(f"PASS: {label}")

require(
    "src/natten/version.py",
    r'^__version__\s*=\s*"0\.21\.6\+relayvsr1\.sm75\.torch2110cu130"$',
    "unique wheel version",
)
require("scripts/autogen_fna.py", r"SUPPORTED_ARCHS\s*=\s*\[75\]", "SM75-only FNA")
require("scripts/autogen_fna.py", r"CUDA_DTYPES\s*=\s*\[NATTEN_Half\]", "FP16-only FNA")
require("scripts/autogen_fna.py", r"NA_RANKS\s*=\s*\[2\]", "2D-only FNA")
require("scripts/autogen_fna.py", r"CAUSAL_MASKS\s*=\s*\{2:\s*\[\[0,\s*0\]\]\}", "non-causal FNA")
require("scripts/autogen_fna.py", r"GEMM_SHAPES\s*=\s*\[\(64,\s*64,\s*64\)\]", "forward 64x64x64 tile")
require(
    "scripts/autogen_fna.py",
    r"BACKWARD_GEMM_SHAPES\s*=\s*\{\s*75:\s*\{NATTEN_Half:\s*\[\(64,\s*64,\s*64\)\]\}",
    "backward 64x64x64 tile",
)
require("setup.py", r'"fna"\s*:\s*1', "single focused FNA source split")

mask = (ROOT / "csrc/include/natten/cuda/fna/na_utils.cuh").read_text(encoding="utf-8")
if mask.count("RelayVSR Flash attention uses clipped query-centred windows at stride=1.") != 1:
    raise SystemExit("FAIL: clipped-window patch marker missing or duplicated")
if mask.count("return cutlass::fast_max(index - window_radius_left, 0);") != 1:
    raise SystemExit("FAIL: clipped start formula missing or duplicated")
if mask.count("index - window_radius_left + window_size, spatial_extent") != 1:
    raise SystemExit("FAIL: clipped end formula missing or duplicated")
print("PASS: stride-1 clipped-window patch")
