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

import torch  # noqa: F401

try:
    from natten import libnatten  # noqa: F401

    HAS_LIBNATTEN = True

    from natten._libnatten.torch_wrappers import (
        blackwell_fmha_backward,
        blackwell_fmha_forward,
        blackwell_na1d_backward,
        blackwell_na1d_forward,
        blackwell_na2d_backward,
        blackwell_na2d_forward,
        blackwell_na3d_backward,
        blackwell_na3d_forward,
        compute_delta,
        fmha_backward,
        fmha_forward,
        hopper_fmha_backward,
        hopper_fmha_forward,
        hopper_na1d_backward,
        hopper_na1d_forward,
        hopper_na2d_backward,
        hopper_na2d_forward,
        hopper_na3d_backward,
        hopper_na3d_forward,
        na1d_backward,
        na1d_forward,
        na2d_backward,
        na2d_forward,
        na3d_backward,
        na3d_forward,
        reference_na1d_backward,
        reference_na1d_forward,
        reference_na2d_backward,
        reference_na2d_forward,
        reference_na3d_backward,
        reference_na3d_forward,
        token_permute_1d,
        token_permute_2d,
        token_permute_3d,
        token_permute_varlen_1d,
        token_permute_varlen_2d,
        token_permute_varlen_3d,
        token_unpermute_1d,
        token_unpermute_2d,
        token_unpermute_3d,
        token_unpermute_varlen_1d,
        token_unpermute_varlen_2d,
        token_unpermute_varlen_3d,
    )

except ImportError:
    HAS_LIBNATTEN = False

    from natten._libnatten.stubs import (
        blackwell_fmha_backward,
        blackwell_fmha_forward,
        blackwell_na1d_backward,
        blackwell_na1d_forward,
        blackwell_na2d_backward,
        blackwell_na2d_forward,
        blackwell_na3d_backward,
        blackwell_na3d_forward,
        compute_delta,
        fmha_backward,
        fmha_forward,
        hopper_fmha_backward,
        hopper_fmha_forward,
        hopper_na1d_backward,
        hopper_na1d_forward,
        hopper_na2d_backward,
        hopper_na2d_forward,
        hopper_na3d_backward,
        hopper_na3d_forward,
        na1d_backward,
        na1d_forward,
        na2d_backward,
        na2d_forward,
        na3d_backward,
        na3d_forward,
        reference_na1d_backward,
        reference_na1d_forward,
        reference_na2d_backward,
        reference_na2d_forward,
        reference_na3d_backward,
        reference_na3d_forward,
        token_permute_1d,
        token_permute_2d,
        token_permute_3d,
        token_permute_varlen_1d,
        token_permute_varlen_2d,
        token_permute_varlen_3d,
        token_unpermute_1d,
        token_unpermute_2d,
        token_unpermute_3d,
        token_unpermute_varlen_1d,
        token_unpermute_varlen_2d,
        token_unpermute_varlen_3d,
    )


__all__ = [
    "blackwell_fmha_backward",
    "blackwell_fmha_forward",
    "blackwell_na1d_backward",
    "blackwell_na1d_forward",
    "blackwell_na2d_backward",
    "blackwell_na2d_forward",
    "blackwell_na3d_backward",
    "blackwell_na3d_forward",
    "compute_delta",
    "fmha_backward",
    "fmha_forward",
    "hopper_fmha_backward",
    "hopper_fmha_forward",
    "hopper_na1d_backward",
    "hopper_na1d_forward",
    "hopper_na2d_backward",
    "hopper_na2d_forward",
    "hopper_na3d_backward",
    "hopper_na3d_forward",
    "na1d_backward",
    "na1d_forward",
    "na2d_backward",
    "na2d_forward",
    "na3d_backward",
    "na3d_forward",
    "reference_na1d_backward",
    "reference_na1d_forward",
    "reference_na2d_backward",
    "reference_na2d_forward",
    "reference_na3d_backward",
    "reference_na3d_forward",
    "token_permute_1d",
    "token_permute_2d",
    "token_permute_3d",
    "token_unpermute_1d",
    "token_unpermute_2d",
    "token_unpermute_3d",
    "token_permute_varlen_1d",
    "token_permute_varlen_2d",
    "token_permute_varlen_3d",
    "token_unpermute_varlen_1d",
    "token_unpermute_varlen_2d",
    "token_unpermute_varlen_3d",
]
