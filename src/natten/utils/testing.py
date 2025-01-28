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

import sys

import torch

from .. import has_cuda, has_fna, has_fp64_gemm, has_gemm

_IS_CUDA_AVAILABLE = torch.cuda.is_available() and has_cuda()

_PYTHON_SUPPORTS_DYNAMO = [sys.version_info[0], sys.version_info[1]] < [3, 12]
_IS_TORCH_COMPILE_SUPPORTED = _PYTHON_SUPPORTS_DYNAMO and [
    int(x) for x in torch.__version__.split(".")[:2]
] >= [2, 4]

_SUPPORTS_NESTED = [int(x) for x in torch.__version__.split(".")[:2]] >= [2, 1]
_SUPPORTS_EXPERIMENTAL_OPS = [int(x) for x in torch.__version__.split(".")[:2]] >= [
    2,
    4,
]
_HAS_GEMM_KERNELS = has_gemm()
_GEMM_WITH_DOUBLE_PRECISION = has_fp64_gemm()
_HAS_FNA_KERNELS = has_fna()

try:
    import fvcore  # type: ignore  # noqa: F401

    _IS_FVCORE_AVAILABLE = True
except ImportError:
    _IS_FVCORE_AVAILABLE = False

try:
    from xformers.ops.fmha import (  # type: ignore  # noqa: F401
        memory_efficient_attention_partial,
    )

    _SUPPORTS_FNA_WITH_ADDITIONAL_KV = True
except ImportError:
    _SUPPORTS_FNA_WITH_ADDITIONAL_KV = False


def skip_if_cuda_is_not_supported():
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            if not _IS_CUDA_AVAILABLE:
                self.skipTest("CUDA is not available.")
            else:
                return f(self, *args, **kwargs)

        return wrapper

    return decorator


def skip_if_gemm_is_not_supported():
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            if not _IS_CUDA_AVAILABLE or not _HAS_GEMM_KERNELS:
                self.skipTest("GEMM kernels are not supported.")
            else:
                return f(self, *args, **kwargs)

        return wrapper

    return decorator


def skip_if_fna_is_not_supported():
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            if not _IS_CUDA_AVAILABLE or not _HAS_FNA_KERNELS:
                self.skipTest("FNA kernels are not supported.")
            else:
                return f(self, *args, **kwargs)

        return wrapper

    return decorator


def skip_if_gemm_does_not_support_double_precision():
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            if (
                not _IS_CUDA_AVAILABLE
                or not _HAS_GEMM_KERNELS
                or not _GEMM_WITH_DOUBLE_PRECISION
            ):
                self.skipTest(
                    "GEMM kernels don't support double precision on this device."
                )
            else:
                return f(self, *args, **kwargs)

        return wrapper

    return decorator


def skip_if_nested_is_not_supported():
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            if not _SUPPORTS_NESTED:
                self.skipTest(
                    "Nested tensors are not supported with this torch version."
                )
            else:
                return f(self, *args, **kwargs)

        return wrapper

    return decorator


def skip_if_experimental_ops_are_not_supported():
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            if not _SUPPORTS_EXPERIMENTAL_OPS:
                self.skipTest(
                    "Experimental ops (registered with torch.library) are not supported with this torch version."
                )
            else:
                return f(self, *args, **kwargs)

        return wrapper

    return decorator


def skip_if_fvcore_is_not_available():
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            if not _IS_FVCORE_AVAILABLE:
                self.skipTest("fvcore is not installed.")
            else:
                return f(self, *args, **kwargs)

        return wrapper

    return decorator


def skip_if_torch_compile_is_not_supported():
    def decorator(f):
        def wrapper(self, *args, **kwargs):
            if not _IS_TORCH_COMPILE_SUPPORTED:
                self.skipTest("torch.compile is not supported.")
            else:
                return f(self, *args, **kwargs)

        return wrapper

    return decorator


def fna_supports_additional_kv():
    return _SUPPORTS_FNA_WITH_ADDITIONAL_KV
