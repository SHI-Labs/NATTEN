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

from natten.token_permute.frontend import (
    token_permute_operation,
    token_permute_varlen_operation,
    token_unpermute_operation,
    token_unpermute_varlen_operation,
)
from natten.token_permute.varlen import (
    generate_fna_varlen_metadata,
    generate_tokperm_varlen_metadata,
    verify_fna_varlen_metadata,
    verify_tokperm_varlen_metadata,
)

__all__ = [
    "token_permute_operation",
    "token_unpermute_operation",
    "token_permute_varlen_operation",
    "token_unpermute_varlen_operation",
    "generate_tokperm_varlen_metadata",
    "generate_fna_varlen_metadata",
    "verify_fna_varlen_metadata",
    "verify_tokperm_varlen_metadata",
]
