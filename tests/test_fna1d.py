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
    torch.manual_seed(42)

    # Attention merge recompilation requires this
    torch._dynamo.config.cache_size_limit = 64


HAS_HALF = has_half()
HAS_BFLOAT = has_bfloat()


def check_args(kernel_size, dilation, is_causal):
    return check_all_args(1, kernel_size, dilation, is_causal)


def compute_bmm_reference(
    B,
    H,
    L,
    D,
    kernel_size,
    dilation,
    has_bias,
    is_causal=None,
    dtype=torch.float32,
    additional_kv_length=0,
):
    kernel_size, dilation, is_causal = check_args(kernel_size, dilation, is_causal)
    assert not has_bias or not any(is_causal)
    with torch.no_grad():
        q, k, v, d_out = (
            torch.randn((B, H, L, D), device="cuda", dtype=dtype),
            torch.randn((B, H, L, D), device="cuda", dtype=dtype),
            torch.randn((B, H, L, D), device="cuda", dtype=dtype),
            torch.randn((B, H, L, D), device="cuda", dtype=dtype) * 0.05,
        )
        rpb = (
            None
            if not has_bias
            else torch.randn(H, 2 * kernel_size[0] - 1, device="cuda", dtype=dtype)
        )
        q_, k_, v_, d_out_ = (
            q.clone().permute(0, 2, 1, 3).contiguous(),
            k.clone().permute(0, 2, 1, 3).contiguous(),
            v.clone().permute(0, 2, 1, 3).contiguous(),
            d_out.clone().permute(0, 2, 1, 3).contiguous(),
        )
        rpb_ = None if rpb is None else rpb.clone()

        additional_k, additional_v = None, None
        additional_k_, additional_v_ = None, None
        if additional_kv_length > 0:
            additional_k, additional_v = (
                torch.randn(
                    (B, H, additional_kv_length, D), device="cuda", dtype=dtype
                ),
                torch.randn(
                    (B, H, additional_kv_length, D), device="cuda", dtype=dtype
                ),
            )
            additional_k_, additional_v_ = (
                additional_k.clone().permute(0, 2, 1, 3).contiguous(),
                additional_v.clone().permute(0, 2, 1, 3).contiguous(),
            )

    if rpb is None:
        q = q.requires_grad_(True)
        k = k.requires_grad_(True)
        v = v.requires_grad_(True)
        d_out = d_out.requires_grad_(True)
        if additional_kv_length > 0:
            additional_k = additional_k.requires_grad_(True)
            additional_v = additional_v.requires_grad_(True)

    q_scaled = q * (D**-0.5)
    attn_ref = na1d_qk(
        q_scaled,
        k,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
        rpb=rpb,
        additional_keys=additional_k,
    )
    attn_ref = attn_ref.softmax(dim=-1)
    out = na1d_av(
        attn_ref,
        v,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
        additional_values=additional_v,
    )
    with torch.no_grad():
        out_ref = out.clone().permute(0, 2, 1, 3).contiguous()
    dq_ref, dk_ref, dv_ref = None, None, None
    d_additional_k, d_additional_v = None, None
    if rpb is None:
        out.backward(d_out)
        with torch.no_grad():
            dq_ref, dk_ref, dv_ref = (
                q.grad.clone().permute(0, 2, 1, 3).contiguous(),
                k.grad.clone().permute(0, 2, 1, 3).contiguous(),
                v.grad.clone().permute(0, 2, 1, 3).contiguous(),
            )
            if additional_kv_length > 0:
                d_additional_k = (
                    additional_k.grad.clone().permute(0, 2, 1, 3).contiguous()
                )
                d_additional_v = (
                    additional_v.grad.clone().permute(0, 2, 1, 3).contiguous()
                )

    return (
        q_,
        k_,
        v_,
        d_out_,
        rpb_,
        kernel_size,
        dilation,
        is_causal,
        additional_k_,
        additional_v_,
    ), (
        out_ref,
        dq_ref,
        dk_ref,
        dv_ref,
        d_additional_k,
        d_additional_v,
    )


def compute_sdpa_reference(B, H, L, D, is_causal=False, dtype=torch.float32):
    with torch.no_grad():
        q, k, v, d_out = (
            torch.randn((B, H, L, D), device="cuda", dtype=dtype),
            torch.randn((B, H, L, D), device="cuda", dtype=dtype),
            torch.randn((B, H, L, D), device="cuda", dtype=dtype),
            torch.randn((B, H, L, D), device="cuda", dtype=dtype),
        )
        q_, k_, v_, d_out_ = (
            q.clone().permute(0, 2, 1, 3).contiguous(),
            k.clone().permute(0, 2, 1, 3).contiguous(),
            v.clone().permute(0, 2, 1, 3).contiguous(),
            d_out.clone().permute(0, 2, 1, 3).contiguous(),
        )

    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    d_out = d_out.requires_grad_(True)

    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    with torch.no_grad():
        out_ref = out.clone().permute(0, 2, 1, 3).contiguous()

    out.backward(d_out)
    with torch.no_grad():
        dq_ref, dk_ref, dv_ref = (
            q.grad.clone().permute(0, 2, 1, 3).contiguous(),
            k.grad.clone().permute(0, 2, 1, 3).contiguous(),
            v.grad.clone().permute(0, 2, 1, 3).contiguous(),
        )
    return (q_, k_, v_, d_out_, is_causal), (out_ref, dq_ref, dk_ref, dv_ref)


class FNA1DTests(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    def _test_against_reference(self, inputs, reference, eps, dtype):
        (
            q,
            k,
            v,
            d_out,
            rpb,
            kernel_size,
            dilation,
            is_causal,
            additional_k,
            additional_v,
        ) = inputs
        kernel_size, dilation, is_causal = check_args(kernel_size, dilation, is_causal)
        out_ref, dq_ref, dk_ref, dv_ref, d_additional_k, d_additional_v = reference
        q_, k_, v_ = q.clone().to(dtype), k.clone().to(dtype), v.clone().to(dtype)
        d_out_ = d_out.clone()
        rpb_ = rpb if rpb is None else rpb.clone().to(dtype)
        assert q_.is_cuda and k_.is_cuda and v_.is_cuda

        has_additional_kv = additional_k is not None and additional_v is not None
        assert has_additional_kv or (additional_k is None and additional_v is None)
        additional_k_, additional_v_ = None, None
        if has_additional_kv:
            additional_k_, additional_v_ = additional_k.clone().to(
                dtype
            ), additional_v.clone().to(dtype)

        if rpb is None:
            q_.requires_grad_(True)
            k_.requires_grad_(True)
            v_.requires_grad_(True)
            d_out_.requires_grad_(True)
            if has_additional_kv:
                additional_k_.requires_grad_(True)
                additional_v_.requires_grad_(True)

        out = na1d(
            q_,
            k_,
            v_,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            rpb=rpb_,
            additional_keys=additional_k_,
            additional_values=additional_v_,
        )

        torch.testing.assert_close(out.float(), out_ref, atol=eps, rtol=0)
        if rpb is None:
            out.backward(d_out_)

            torch.testing.assert_close(q_.grad.float(), dq_ref, atol=eps, rtol=0)
            torch.testing.assert_close(k_.grad.float(), dk_ref, atol=eps, rtol=0)
            torch.testing.assert_close(v_.grad.float(), dv_ref, atol=eps, rtol=0)

            if has_additional_kv:
                torch.testing.assert_close(
                    additional_v_.grad.float(), d_additional_v, atol=eps, rtol=0
                )
                torch.testing.assert_close(
                    additional_k_.grad.float(), d_additional_k, atol=eps, rtol=0
                )

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
        additional_kv_length=0,
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
            additional_kv_length=additional_kv_length,
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

            # xFormers' cutlass backend doesn't support partial attention.
            if additional_kv_length == 0:
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
            for additional_kv_length in [0, 64, L]:
                for is_causal in [False, True]:
                    self._test_all_dtypes_against_bmm_style(
                        B=B,
                        H=H,
                        L=L,
                        D=D,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        is_causal=is_causal,
                        additional_kv_length=additional_kv_length,
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

    def _test_against_sdpa(self, inputs, reference, eps, dtype):
        q, k, v, d_out, is_causal = inputs
        kernel_size = (q.shape[1],)
        kernel_size, dilation, is_causal = check_args(kernel_size, 1, is_causal)
        out_ref, dq_ref, dk_ref, dv_ref = reference
        q_, k_, v_ = q.clone().to(dtype), k.clone().to(dtype), v.clone().to(dtype)
        d_out_ = d_out.clone()
        assert q_.is_cuda and k_.is_cuda and v_.is_cuda

        q_.requires_grad_(True)
        k_.requires_grad_(True)
        v_.requires_grad_(True)
        d_out_.requires_grad_(True)

        out = na1d(
            q_,
            k_,
            v_,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
        )

        torch.testing.assert_close(out.float(), out_ref, atol=eps, rtol=0)

        out.backward(d_out_)

        torch.testing.assert_close(q_.grad.float(), dq_ref, atol=eps, rtol=0)
        torch.testing.assert_close(k_.grad.float(), dk_ref, atol=eps, rtol=0)
        torch.testing.assert_close(v_.grad.float(), dv_ref, atol=eps, rtol=0)

    def _test_all_dtypes_against_sdpa(
        self,
        B,
        H,
        L,
        D,
        is_causal=False,
    ):
        inputs, reference = compute_sdpa_reference(
            B=B,
            H=H,
            L=L,
            D=D,
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

            self._test_against_sdpa(
                inputs=inputs,
                reference=reference,
                dtype=torch.float32,
                eps=1e-2,
            )
            if HAS_HALF:
                self._test_against_sdpa(
                    inputs=inputs,
                    reference=reference,
                    dtype=torch.float16,
                    eps=1e-1,
                )
            if HAS_BFLOAT:
                self._test_against_sdpa(
                    inputs=inputs,
                    reference=reference,
                    dtype=torch.bfloat16,
                    eps=1e-1,
                )

    @skip_if_cuda_is_not_supported()
    @skip_if_fna_is_not_supported()
    def test_against_sdpa(self):
        problem_sizes = [
            (1, 1, 3, 16),
            (2, 1, 5, 24),
            (1, 2, 7, 32),
            (2, 2, 9, 40),
            (1, 2, 11, 48),
            (1, 2, 13, 56),
            (1, 1, 15, 64),
            (1, 2, 33, 96),
            (1, 1, 4095, 128),
            (1, 1, 2047, 256),
            (1, 2, 1023, 256),
        ]
        for B, H, L, D in problem_sizes:
            for is_causal in [False, True]:
                self._test_all_dtypes_against_sdpa(
                    B=B,
                    H=H,
                    L=L,
                    D=D,
                    is_causal=is_causal,
                )


if __name__ == "__main__":
    torch.manual_seed(42)
    unittest.main()
