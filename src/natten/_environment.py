#################################################################################################
# Copyright (c) 2022 - 2026 Ali Hassani.
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

from natten._libnatten import HAS_LIBNATTEN  # noqa: F401
from natten.utils.environment import (
    _IS_CUDA_AVAILABLE,
    _IS_TORCH_COMPILE_SUPPORTED,
    _TORCH_VERSION,
    parse_env_flag,
    parse_env_int,
    parse_env_str,
)


# Default tokperm implementation; choices:
# NATTEN_TOKPERM_DEFAULT_IMPL="cutlass"
# NATTEN_TOKPERM_DEFAULT_IMPL="torch"
USE_TORCH_IMPL_DEFAULT = (
    parse_env_str("NATTEN_TOKPERM_DEFAULT_IMPL", "cutlass") == "torch"
)

# Unit tests
_RUN_EXTENDED_TESTS = parse_env_flag("NATTEN_RUN_EXTENDED_TESTS", False)
_RUN_ADDITIONAL_KV_TESTS = parse_env_flag("NATTEN_RUN_ADDITIONAL_KV_TESTS", True)
_RUN_FLEX_TESTS = parse_env_flag("NATTEN_RUN_FLEX_TESTS", True)
_NUM_RAND_SWEEP_TESTS = parse_env_int("NATTEN_RAND_SWEEP_TESTS", 1000)

# Profiler
DISABLE_TQDM = parse_env_flag("NATTEN_DISABLE_TQDM", False)


__all__ = [
    "HAS_LIBNATTEN",
    "_IS_CUDA_AVAILABLE",
    "_IS_TORCH_COMPILE_SUPPORTED",
    "DISABLE_TQDM",
    "_RUN_FLEX_TESTS",
    "_RUN_ADDITIONAL_KV_TESTS",
    "_RUN_FLEX_TESTS",
    "_NUM_RAND_SWEEP_TESTS",
    "_TORCH_VERSION",
]
