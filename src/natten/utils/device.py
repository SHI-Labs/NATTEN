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

from typing import Optional

import torch


def is_cuda(device: torch.device) -> bool:
    return torch.cuda.is_available() and torch.version.cuda and device.type == "cuda"  # type: ignore


def is_rocm(device: torch.device) -> bool:
    return torch.cuda.is_available() and torch.version.hip and device.type == "cuda"  # type: ignore


def is_cpu(device: torch.device) -> bool:
    return device.type == "cpu"


def get_device_cc(device: Optional[torch.device] = None) -> int:
    if (
        torch.cuda.is_available()
        and torch.version.cuda
        and (device is None or is_cuda(device))
    ):
        major, minor = torch.cuda.get_device_capability(device)
        return major * 10 + minor

    return 0
