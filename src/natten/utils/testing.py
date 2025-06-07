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

import torch

from .._environment import _IS_CUDA_AVAILABLE, _RUN_EXTENDED_TESTS, HAS_LIBNATTEN

from ..backends.flex import _FLEX_COMPILE_SUPPORTED, _FLEX_SUPPORTED
from .device import get_device_cc, is_cuda


def skip_if_libnatten_is_not_supported():
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            if not _IS_CUDA_AVAILABLE:
                self.skipTest("CUDA is not available.")
            elif not HAS_LIBNATTEN:
                self.skipTest("Libnatten is not available.")
            else:
                return f(self, *args, **kwargs)

        return wrapper

    return decorator


def skip_if_cuda_is_not_supported():
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            if not _IS_CUDA_AVAILABLE:
                self.skipTest("CUDA is not available.")
            else:
                return f(self, *args, **kwargs)

        return wrapper

    return decorator


def skip_if_flex_is_not_supported():
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            if not _FLEX_SUPPORTED or get_device_cc() < 70:
                self.skipTest("Flex backend is not supported.")
            else:
                return f(self, *args, **kwargs)

        return wrapper

    return decorator


def skip_if_flex_compile_is_not_supported():
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            if not _FLEX_COMPILE_SUPPORTED:
                self.skipTest("Flex (compiled) backend is not supported.")
            else:
                return f(self, *args, **kwargs)

        return wrapper

    return decorator


def skip_if_not_running_extended_tests():
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            if not _RUN_EXTENDED_TESTS:
                self.skipTest("Skipping extended test cases.")
            else:
                return f(self, *args, **kwargs)

        return wrapper

    return decorator


def skip_if_hopper_kernels_not_supported():
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            if get_device_cc() != 90:
                self.skipTest("Hopper kernels are only supported on SM90.")
            else:
                return f(self, *args, **kwargs)

        return wrapper

    return decorator


def skip_if_blackwell_kernels_not_supported():
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            if get_device_cc() != 100:
                self.skipTest("Blackwell kernels are only supported on SM100.")
            else:
                return f(self, *args, **kwargs)

        return wrapper

    return decorator


def supports_float16(device: torch.device) -> bool:
    if is_cuda(device):
        device_cc = get_device_cc(device)

        if device_cc < 50:
            return False

        return True

    # TODO:
    return True


def supports_bfloat16(device: torch.device) -> bool:
    if is_cuda(device):
        device_cc = get_device_cc(device)

        if device_cc < 80:
            return False

        return True

    # TODO:
    return False
