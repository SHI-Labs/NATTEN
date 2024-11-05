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

from natten import (
    has_bfloat,
    has_half,
    set_memory_usage_preference,
    use_autotuner,
    use_kv_parallelism_in_fused_na,
)
from natten.functional import na1d, na1d_av, na1d_qk
from natten.utils import check_all_args
from natten.utils.testing import (
    skip_if_cuda_is_not_supported,
    skip_if_fna_is_not_supported,
)


def _reset_everything():
    import natten
    from natten.context import AutotunerContext, NattenContext

    NattenContext.reset()
    AutotunerContext.reset()
    natten.use_tiled_na()
    natten.use_gemm_na()
    natten.use_tf32_in_gemm_na()
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


HAS_HALF = has_half()
HAS_BFLOAT = has_bfloat()


def check_args(kernel_size, dilation, is_causal):
    return check_all_args(1, kernel_size, dilation, is_causal)


def compute_bmm_reference(
    B, H, L, D, kernel_size, dilation, has_bias, is_causal=None, dtype=torch.float32
):
    kernel_size, dilation, is_causal = check_args(kernel_size, dilation, is_causal)
    assert not has_bias or not any(is_causal)
    with torch.no_grad():
        q, k, v = (
            torch.randn((B, H, L, D), device="cuda", dtype=dtype),
            torch.randn((B, H, L, D), device="cuda", dtype=dtype),
            torch.randn((B, H, L, D), device="cuda", dtype=dtype),
        )
        rpb = (
            None
            if not has_bias
            else torch.randn(H, 2 * kernel_size[0] - 1, device="cuda", dtype=dtype)
        )
        q_, k_, v_ = (
            q.clone().permute(0, 2, 1, 3).contiguous(),
            k.clone().permute(0, 2, 1, 3).contiguous(),
            v.clone().permute(0, 2, 1, 3).contiguous(),
        )
        rpb_ = None if rpb is None else rpb.clone()

    q_scaled = q * (D**-0.5)
    attn_ref = na1d_qk(
        q_scaled,
        k,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
        rpb=rpb,
    )

    logsumexp_ref = torch.logsumexp(attn_ref, dim=-1).permute(0, 2 ,1)
    maximums_ref, _ = torch.max(attn_ref, dim=-1)
    maximums_ref = maximums_ref.permute(0, 2, 1)

    return (q_, k_, v_, rpb_, kernel_size, dilation, is_causal), (logsumexp_ref, maximums_ref)


class FNA1DStatisticsTests(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    def _test_against_reference(self, inputs, reference, eps, dtype):
        q, k, v, rpb, kernel_size, dilation, is_causal = inputs
        kernel_size, dilation, is_causal = check_args(kernel_size, dilation, is_causal)
        logsumexp_ref, maximums_ref = reference
        q_, k_, v_ = q.clone().to(dtype), k.clone().to(dtype), v.clone().to(dtype)
        rpb_ = rpb if rpb is None else rpb.clone().to(dtype)
        assert q_.is_cuda and k_.is_cuda and v_.is_cuda

        _, logsumexp, maximums = na1d(
            q_,
            k_,
            v_,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            rpb=rpb_,
            return_statistics=True
        )

        torch.testing.assert_close(logsumexp.float(), logsumexp_ref.float(), atol=eps, rtol=0)
        torch.testing.assert_close(maximums.float(), maximums_ref.float(), atol=eps, rtol=0)

    def _test_all_dtypes_against_bmm_style(
        self,
        B,
        H,
        L,
        D,
        kernel_size,
        dilation,
        has_bias=False,
        is_causal=None,
    ):
        kernel_size, dilation, is_causal = check_args(kernel_size, dilation, is_causal)
        assert not has_bias or not any(is_causal)
        inputs, reference = compute_bmm_reference(
            B=B,
            H=H,
            L=L,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            has_bias=has_bias,
            is_causal=is_causal,
            dtype=torch.float32,
        )

        for autotune in [False, True]:
            if autotune:
                torch.use_deterministic_algorithms(False)
                use_kv_parallelism_in_fused_na(True)
                set_memory_usage_preference("unrestricted")
                use_autotuner(
                    True,
                    True,
                    True,
                    True,
                    warmup_steps_forward=1,
                    warmup_steps_backward=1,
                    steps_forward=1,
                    steps_backward=1,
                )
            else:
                use_autotuner(False, False, False, False)

            self._test_against_reference(
                inputs=inputs,
                reference=reference,
                dtype=torch.float32,
                eps=1e-2,
            )
            if HAS_HALF:
                self._test_against_reference(
                    inputs=inputs,
                    reference=reference,
                    dtype=torch.float16,
                    eps=1e-1,
                )
            if HAS_BFLOAT:
                self._test_against_reference(
                    inputs=inputs,
                    reference=reference,
                    dtype=torch.bfloat16,
                    eps=1e-1,
                )

            if autotune:
                torch.use_deterministic_algorithms(True)

    @skip_if_cuda_is_not_supported()
    @skip_if_fna_is_not_supported()
    def test_against_bmm_style(self):
        problem_sizes = [
            (1, 1, 3, 16, 3, 1),
            (1, 1, 16, 24, 3, 1),
            (1, 2, 33, 32, 15, 1),
            (1, 2, 33, 40, 15, 2),
            (4, 3, 256, 56, 255, 1),
            (2, 2, 4096, 64, 2047, 1),
            (2, 4, 4096, 72, 2047, 2),
            (4, 3, 5000, 80, 511, 8),
            (4, 3, 5000, 88, 255, 16),
            (1, 12, 512, 96, 255, 1),
            (4, 24, 512, 128, 99, 1),
            (1, 48, 512, 256, 45, 4),
        ]
        for B, H, L, D, kernel_size, dilation in problem_sizes:
            for is_causal in [False, True]:
                self._test_all_dtypes_against_bmm_style(
                    B=B,
                    H=H,
                    L=L,
                    D=D,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    is_causal=is_causal,
                )
            self._test_all_dtypes_against_bmm_style(
                B=B,
                H=H,
                L=L,
                D=D,
                kernel_size=kernel_size,
                dilation=dilation,
                has_bias=True,
            )


if __name__ == "__main__":
    torch.manual_seed(42)
    unittest.main()

