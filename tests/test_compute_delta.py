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

from natten._libnatten import compute_delta  # type: ignore
from natten.utils.testing import (
    skip_if_libnatten_is_not_supported,
    supports_bfloat16,
    supports_float16,
)


def _reset_everything():
    torch.use_deterministic_algorithms(False)


def compute_delta_pt(out: torch.Tensor, d_out: torch.Tensor, dtype) -> torch.Tensor:
    assert out.dim() == d_out.dim() and out.dim() >= 2
    for i in range(out.dim()):
        assert out.shape[i] == d_out.shape[i]
    with torch.no_grad():
        return (out.clone().to(dtype) * d_out.clone().to(dtype)).sum(-1)


class ComputeDeltaTests(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    def _test_against_reference(self, input_shape, eps, dtype, dtype_out):
        assert len(input_shape) >= 2
        out = torch.randn(input_shape, device="cuda", dtype=dtype)
        d_out = torch.randn_like(out)

        with torch.no_grad():
            delta_ref = compute_delta_pt(out, d_out, dtype_out)

            delta = torch.empty(input_shape[:-1], device="cuda", dtype=dtype_out)
            compute_delta(out, d_out, delta)

            torch.testing.assert_close(delta, delta_ref, atol=eps, rtol=0)

    def _test_all_dtypes_against_reference(self, input_shape):
        torch.set_default_device("cuda")
        self._test_against_reference(
            input_shape=input_shape,
            eps=1e-2,
            dtype=torch.float32,
            dtype_out=torch.float32,
        )
        if supports_float16(torch.get_default_device()):
            self._test_against_reference(
                input_shape=input_shape,
                eps=1e-1,
                dtype=torch.float16,
                dtype_out=torch.float32,
            )

        if supports_bfloat16(torch.get_default_device()):
            self._test_against_reference(
                input_shape=input_shape,
                eps=1e-1,
                dtype=torch.bfloat16,
                dtype_out=torch.float32,
            )

    @skip_if_libnatten_is_not_supported()
    def test_against_pt_reference(self):
        input_sizes = [
            (2, 4),
            (5, 4),
            (1, 4, 64, 32),
            (128, 49),
            (50, 60),
            (127, 61),
            (128, 4, 56, 56, 32),
            (128, 8, 56, 56, 128),
            (1, 1, 56, 56, 1024),
        ]
        for input_size in input_sizes:
            self._test_all_dtypes_against_reference(input_size)


if __name__ == "__main__":
    torch.manual_seed(42)
    unittest.main()
