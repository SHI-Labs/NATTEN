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

_TORCH_VERSION = [int(x) for x in torch.__version__.split(".")[:2]]

_IS_TORCH_COMPILE_SUPPORTED = _TORCH_VERSION >= [2, 6] and get_device_cc() >= 70

# Unit tests
_RUN_EXTENDED_TESTS = bool(os.getenv("NATTEN_RUN_EXTENDED_TESTS", "0") == "1")

# Profiler
DISABLE_TQDM = bool(os.getenv("NATTEN_DISABLE_TQDM", "0") == "1")
