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

import random
import unittest

import torch
from natten._libnatten import compute_delta  # type: ignore
from natten.utils.testing import (
    skip_if_libnatten_is_not_supported,
    supports_bfloat16,
    supports_float16,
)

from .utils import logger


def _reset_everything(random_seed: int = 42, torch_seed: int = 42):
    random.seed(random_seed)
    torch.manual_seed(torch_seed)
    logger.debug(f"Reset seeds: {random_seed=}, {torch_seed=}")
    torch.use_deterministic_algorithms(True)


def compute_delta_pt(out: torch.Tensor, d_out: torch.Tensor, dtype) -> torch.Tensor:
    assert out.dim() == d_out.dim() == 4
    assert out.shape == d_out.shape
    with torch.no_grad():
        return (out.clone().to(dtype) * d_out.clone().to(dtype)).sum(-1)


class ComputeDeltaTests(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    def _test_against_reference(
        self, batch, seqlen, heads, head_dim, eps, dtype, dtype_out
    ):
        input_shape = (batch, seqlen, heads, head_dim)
        out = torch.randn(input_shape, device="cuda", dtype=dtype)
        d_out = torch.randn_like(out)

        with torch.no_grad():
            delta_ref = compute_delta_pt(out, d_out, dtype_out)

            delta = torch.empty(input_shape[:-1], device="cuda", dtype=dtype_out)
            compute_delta(out, d_out, delta)

            torch.testing.assert_close(delta, delta_ref, atol=eps, rtol=0)

    def _test_all_dtypes_against_reference(self, batch, seqlen, heads, head_dim):
        torch.set_default_device("cuda")
        self._test_against_reference(
            batch=batch,
            seqlen=seqlen,
            heads=heads,
            head_dim=head_dim,
            eps=1e-3,
            dtype=torch.float32,
            dtype_out=torch.float32,
        )
        if supports_float16(torch.get_default_device()):
            self._test_against_reference(
                batch=batch,
                seqlen=seqlen,
                heads=heads,
                head_dim=head_dim,
                eps=1e-1,
                dtype=torch.float16,
                dtype_out=torch.float32,
            )

        if supports_bfloat16(torch.get_default_device()):
            self._test_against_reference(
                batch=batch,
                seqlen=seqlen,
                heads=heads,
                head_dim=head_dim,
                eps=2e-1,
                dtype=torch.bfloat16,
                dtype_out=torch.float32,
            )

    @skip_if_libnatten_is_not_supported()
    def test_against_pt_reference(self):
        # (batch, seqlen, heads, head_dim)
        input_sizes = [
            (1, 64, 2, 128),
            (128, 56, 4, 2),
            (128, 56, 8, 16),
            (1, 1235, 56, 512),
            (2, 128, 4, 48),
            (5, 50, 1, 60),
            (1, 127, 1, 256),
        ]
        for i, (batch, seqlen, heads, head_dim) in enumerate(input_sizes):
            _reset_everything(random_seed=i, torch_seed=i)
            self._test_all_dtypes_against_reference(batch, seqlen, heads, head_dim)


if __name__ == "__main__":
    unittest.main()
