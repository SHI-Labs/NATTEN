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


def libnatten_import_error(*args, **kwargs):
    raise ImportError(
        "Failed to import libnatten. "
        "Did you build / download NATTEN with the CUDA backend (libnatten)? "
        "Please make sure you built NATTEN correctly, or refer to "
        "https://shi-labs.com/natten for more information."
    )


# Reference kernels
reference_na1d_forward = libnatten_import_error
reference_na2d_forward = libnatten_import_error
reference_na3d_forward = libnatten_import_error
reference_na1d_backward = libnatten_import_error
reference_na2d_backward = libnatten_import_error
reference_na3d_backward = libnatten_import_error

# CUTLASS 2.X kernels
fmha_forward = libnatten_import_error
fmha_backward = libnatten_import_error

## SM50/SM70/SM75/SM80 - Original FNA
na1d_forward = libnatten_import_error
na2d_forward = libnatten_import_error
na3d_forward = libnatten_import_error
na1d_backward = libnatten_import_error
na2d_backward = libnatten_import_error
na3d_backward = libnatten_import_error

# CUTLASS 3.X kernels
## SM90 - Hopper FMHA
hopper_fmha_forward = libnatten_import_error
hopper_fmha_backward = libnatten_import_error

## SM90 - Hopper FNA
hopper_na1d_forward = libnatten_import_error
hopper_na2d_forward = libnatten_import_error
hopper_na3d_forward = libnatten_import_error
hopper_na1d_backward = libnatten_import_error
hopper_na2d_backward = libnatten_import_error
hopper_na3d_backward = libnatten_import_error

## SM100 - Blackwell FMHA
blackwell_fmha_forward = libnatten_import_error
blackwell_fmha_backward = libnatten_import_error

## SM100 - Blackwell FNA
blackwell_na1d_forward = libnatten_import_error
blackwell_na2d_forward = libnatten_import_error
blackwell_na3d_forward = libnatten_import_error
blackwell_na1d_backward = libnatten_import_error
blackwell_na2d_backward = libnatten_import_error
blackwell_na3d_backward = libnatten_import_error

# Token permute kernels
token_permute_1d = libnatten_import_error
token_permute_2d = libnatten_import_error
token_permute_3d = libnatten_import_error
token_unpermute_1d = libnatten_import_error
token_unpermute_2d = libnatten_import_error
token_unpermute_3d = libnatten_import_error

token_permute_varlen_1d = libnatten_import_error
token_permute_varlen_2d = libnatten_import_error
token_permute_varlen_3d = libnatten_import_error
token_unpermute_varlen_1d = libnatten_import_error
token_unpermute_varlen_2d = libnatten_import_error
token_unpermute_varlen_3d = libnatten_import_error

# Misc kernels
compute_delta = libnatten_import_error
