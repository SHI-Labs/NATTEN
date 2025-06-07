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


from torch import Tensor
from torch.nn.attention import sdpa_kernel, SDPBackend

from torch.nn.functional import scaled_dot_product_attention


def sdpa(q: Tensor, k: Tensor, v: Tensor, backend: str) -> Tensor:
    backends = []

    if backend == "xformers":
        backends = [SDPBackend.EFFICIENT_ATTENTION]

    elif backend == "fav2":
        backends = [SDPBackend.FLASH_ATTENTION]

    elif backend == "cudnn":
        backends = [SDPBackend.CUDNN_ATTENTION]

    else:
        raise NotImplementedError(
            f"Unrecognized SDPA backend {backend}. Choices are "
            "xformers, fav2, cudnn."
        )

    with sdpa_kernel(backends=backends):
        return scaled_dot_product_attention(q, k, v)
