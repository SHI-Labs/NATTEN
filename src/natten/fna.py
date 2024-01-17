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
import warnings


class FusedNAState:
    enabled: bool = False


def enable_fna():
    FusedNAState.enabled = True
    warnings.warn(
        "You're enabling the use of Fused Neighborhood Attention kernels. "
        "This is an experimental feature, and only implements forward pass "
        "at the moment. Proceed with caution.\n\n"
        "For improved runtime performance, consider enabling auto-tuning:\n"
        "from natten.functional import enable_autotuner\n"
        "enable_autotuner()"
    )


def disable_fna():
    FusedNAState.enabled = False


def is_fna_enabled() -> bool:
    return FusedNAState.enabled
