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

import os

import torch

from ._libnatten import HAS_LIBNATTEN  # noqa: F401
from .utils.device import get_device_cc

_IS_CUDA_AVAILABLE = torch.cuda.is_available()
_IS_XPU_AVAILABLE = torch.xpu.is_available()

_TORCH_VERSION = [int(x) for x in torch.__version__.split(".")[:2]]

# NOTE: Triton does not recognize SM103 yet.
# TODO: remove the extra condition once Triton adds SM103 support.
_IS_TORCH_COMPILE_SUPPORTED = (
    _TORCH_VERSION >= [2, 6] and get_device_cc() >= 70 and get_device_cc() not in [103]
)


def parse_env_flag(env_var: str, default: bool) -> bool:
    default_str = "1" if default else "0"
    out_str = os.getenv(env_var, default_str)
    if out_str.strip() == "":
        return default
    if out_str == "0":
        return False
    if out_str == "1":
        return True
    return default


def parse_env_int(env_var: str, default: int) -> int:
    out_str = os.getenv(env_var, str(default))
    if out_str.strip() == "":
        return default
    try:
        return int(out_str)
    except ValueError:
        return default


# Unit tests
_RUN_EXTENDED_TESTS = parse_env_flag("NATTEN_RUN_EXTENDED_TESTS", False)
_RUN_ADDITIONAL_KV_TESTS = parse_env_flag("NATTEN_RUN_ADDITIONAL_KV_TESTS", True)
_RUN_FLEX_TESTS = parse_env_flag("NATTEN_RUN_FLEX_TESTS", True)
_NUM_RAND_SWEEP_TESTS = parse_env_int("NATTEN_RAND_SWEEP_TESTS", 1000)

# Profiler
DISABLE_TQDM = parse_env_flag("NATTEN_DISABLE_TQDM", False)
