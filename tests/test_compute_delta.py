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

import os
import unittest

import torch

from natten import has_bfloat, has_half
from natten.libnatten import compute_delta  # type: ignore
from natten.utils.testing import skip_if_cuda_is_not_supported


def _reset_everything():
    import natten
    from natten.context import AutotunerContext, NattenContext

    NattenContext.reset()
    AutotunerContext.reset()
    natten.use_tiled_na()
    natten.use_gemm_na()
    natten.use_tf32_in_gemm_na()
    torch.use_deterministic_algorithms(False)
    os.environ["NATTEN_LOG_LEVEL"] = "CRITICAL"

    # NOTE: It is important to ensure determinism in torch GEMMs since
    # we don't write our own. Therefore we have to force determinism in
    # CUBLAS, and turn off CUDNN benchmarking (in case that backend
    # is built).
    # PT's caching allocator should also be turned off in unit tests for
    # when we run memcheck.
    os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.empty_cache()


HAS_HALF = has_half()
HAS_BFLOAT = has_bfloat()


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
        self._test_against_reference(
            input_shape=input_shape,
            eps=1e-2,
            dtype=torch.float32,
            dtype_out=torch.float32,
        )
        if HAS_HALF:
            self._test_against_reference(
                input_shape=input_shape,
                eps=1e-1,
                dtype=torch.float16,
                dtype_out=torch.float32,
            )

        if HAS_BFLOAT:
            self._test_against_reference(
                input_shape=input_shape,
                eps=1e-1,
                dtype=torch.bfloat16,
                dtype_out=torch.float32,
            )

    @skip_if_cuda_is_not_supported()
    def test_against_bmm_style(self):
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
