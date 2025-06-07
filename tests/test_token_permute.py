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

import unittest

import torch

from natten.token_permute import maybe_pad, maybe_unpad, token_permute, token_unpermute
from natten.utils import log
from natten.utils.testing import skip_if_cuda_is_not_supported, supports_float16


logger = log.get_logger(__name__)


class TokenPermuteTest(unittest.TestCase):
    def _test_permute_unpermute(
        self, B, H, S, D, tile_shape, dilation, flip_dims, eps, dtype, device="cuda"
    ):
        with torch.no_grad():
            tensor = torch.randn((B, *S, H, D), device=device, dtype=dtype)

            logger.debug(
                f"Testing token permute: {tensor.shape=}, {tile_shape=}, {dilation=}, {flip_dims=}, {dtype=}."
            )

            tensor_copy = tensor.clone()
            tensor_padded, padding = maybe_pad(
                tensor_copy, tile_shape, dilation=dilation
            )
            tensor_permuted, attn_shape, rest = token_permute(
                tensor_padded, tile_shape, dilation=dilation, flip_tiled_dims=flip_dims
            )
            tensor_out = token_unpermute(
                tensor_permuted,
                tile_shape,
                attn_shape,
                rest,
                dilation=dilation,
                flip_tiled_dims=flip_dims,
            )
            tensor_out = maybe_unpad(tensor_out, padding)

            torch.testing.assert_close(tensor, tensor_out, atol=eps, rtol=0)

    def test_permute_unpermute_cpu(self):
        problem_sizes = [
            # 1D
            (1, 1, (80,), 32, (8,), None),
            (1, 1, (16,), 32, (2,), None),
            (1, 1, (16,), 32, (8,), None),
            (1, 1, (8,), 32, (8,), None),
            (1, 1, (80,), 128, (20,), None),
            (1, 1, (80,), 32, (8,), (7,)),
            (1, 1, (16,), 32, (2,), (3,)),
            (1, 1, (16,), 32, (8,), (3,)),
            (1, 1, (8,), 32, (8,), (2,)),
            (1, 1, (8,), 32, (8,), (4,)),
            (1, 1, (80,), 128, (40,), (13,)),
            # 2D
            (1, 1, (48, 80), 32, (8, 8), None),
            (1, 1, (16, 16), 32, (8, 2), None),
            (1, 1, (16, 16), 32, (8, 2), None),
            (1, 1, (44, 8), 32, (22, 8), None),
            (1, 1, (48, 80), 128, (4, 16), None),
            (1, 1, (48, 80), 32, (8, 8), (2, 8)),
            (1, 1, (16, 16), 32, (8, 2), (2, 4)),
            (1, 1, (16, 16), 32, (8, 2), (2, 1)),
            (1, 1, (44, 8), 32, (22, 8), (4, 8)),
            # 3D
            (1, 1, (30, 48, 80), 32, (4, 8, 8), None),
            (1, 1, (16, 16, 16), 32, (4, 8, 2), None),
            (1, 1, (32, 16, 16), 32, (4, 8, 2), None),
            (1, 1, (32, 44, 8), 32, (16, 22, 8), None),
            (1, 1, (30, 48, 80), 128, (2, 4, 16), None),
            (1, 1, (30, 48, 80), 32, (4, 8, 8), (2, 5, 4)),
            (1, 1, (16, 16, 16), 32, (4, 8, 2), (1, 9, 3)),
        ]
        for B, H, S, D, tile_shape, dilation in problem_sizes:
            for flip_dims in [True, False]:
                self._test_permute_unpermute(
                    B=B,
                    H=H,
                    S=S,
                    D=D,
                    tile_shape=tile_shape,
                    dilation=dilation,
                    flip_dims=flip_dims,
                    eps=1e-3,
                    dtype=torch.float32,
                    device="cpu",
                )

    @skip_if_cuda_is_not_supported()
    def test_permute_unpermute_cuda(self):
        torch.set_default_device("cuda")
        problem_sizes = [
            # 1D
            (1, 1, (128,), 1, (2,), None),
            (1, 1, (80,), 32, (8,), None),
            (1, 1, (16,), 32, (2,), None),
            (1, 1, (16,), 32, (8,), None),
            (1, 1, (8,), 32, (8,), None),
            (1, 1, (80,), 128, (20,), None),
            (1, 1, (80,), 128, (40,), None),
            (1, 1, (80,), 32, (8,), (7,)),
            (1, 1, (16,), 32, (2,), (3,)),
            (1, 1, (16,), 32, (8,), (3,)),
            (1, 1, (8,), 32, (8,), (2,)),
            (1, 1, (8,), 32, (8,), (4,)),
            (1, 1, (80,), 128, (20,), (13,)),
            (1, 1, (80,), 128, (40,), (13,)),
            # 2D
            (1, 1, (48, 80), 32, (8, 8), None),
            (1, 1, (16, 16), 32, (8, 2), None),
            (1, 1, (16, 16), 32, (8, 2), None),
            (1, 1, (44, 8), 32, (22, 8), None),
            (1, 1, (48, 80), 128, (4, 16), None),
            (1, 1, (48, 80), 128, (4, 16), None),
            (1, 1, (48, 80), 32, (8, 8), (2, 8)),
            (1, 1, (16, 16), 32, (8, 2), (2, 4)),
            (1, 1, (16, 16), 32, (8, 2), (2, 1)),
            (1, 1, (44, 8), 32, (22, 8), (4, 8)),
            (1, 1, (48, 80), 128, (4, 16), (11, 7)),
            (1, 1, (48, 80), 128, (4, 16), (5, 11)),
            # 3D
            (1, 1, (30, 48, 80), 32, (4, 8, 8), None),
            (1, 1, (16, 16, 16), 32, (4, 8, 2), None),
            (1, 1, (32, 16, 16), 32, (4, 8, 2), None),
            (1, 1, (32, 44, 8), 32, (16, 22, 8), None),
            (1, 1, (30, 48, 80), 128, (2, 4, 16), None),
            (1, 1, (30, 48, 80), 32, (4, 8, 8), (2, 5, 4)),
            (1, 1, (16, 16, 16), 32, (4, 8, 2), (1, 9, 3)),
            (1, 1, (32, 16, 16), 32, (4, 8, 2), (7, 7, 7)),
            (1, 1, (32, 44, 8), 32, (16, 22, 8), (13, 2, 6)),
            (1, 1, (30, 48, 80), 128, (2, 4, 16), (7, 3, 1)),
            (1, 1, (57, 32, 32), 128, (8, 4, 8), (2, 1, 1)),
            (1, 1, (57, 32, 32), 128, (4, 8, 8), (2, 1, 1)),
            (1, 1, (57, 32, 32), 128, (2, 8, 16), (2, 1, 1)),
            (1, 1, (57, 32, 32), 128, (4, 4, 16), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (8, 4, 8), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (4, 8, 8), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (2, 8, 16), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (4, 4, 16), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (8, 4, 8), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (4, 8, 8), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (2, 8, 16), (2, 1, 1)),
            (1, 1, (57, 20, 88), 128, (4, 4, 16), (2, 1, 1)),
        ]
        for B, H, S, D, tile_shape, dilation in problem_sizes:
            for flip_dims in [True, False]:
                self._test_permute_unpermute(
                    B=B,
                    H=H,
                    S=S,
                    D=D,
                    tile_shape=tile_shape,
                    dilation=dilation,
                    flip_dims=flip_dims,
                    eps=1e-3,
                    dtype=(
                        torch.float16
                        if supports_float16(torch.get_default_device())
                        else torch.float32
                    ),
                    device="cuda",
                )


if __name__ == "__main__":
    torch.manual_seed(42)
    unittest.main()
