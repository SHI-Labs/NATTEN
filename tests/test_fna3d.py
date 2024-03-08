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

import unittest
from itertools import product

import torch

from natten import disable_autotuner, enable_autotuner, has_bfloat, has_half
from natten.functional import na3d, na3d_av, na3d_qk
from natten.utils import check_all_args
from natten.utils.testing import (
    skip_if_cuda_is_not_supported,
    skip_if_fna_is_not_supported,
)

# NOTE: It is important to disable CUDNN benchmarking and TF32
# because some tests are written with the assumption of relatively
# good determinism. This affects certain torch builds, in particular
# those in NGC images (mostly just built from source with different flags
# compared to PyPI.)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

HAS_HALF = has_half()
HAS_BFLOAT = has_bfloat()


def check_args(kernel_size, dilation, is_causal):
    return check_all_args(3, kernel_size, dilation, is_causal)


def compute_bmm_reference(
    B,
    H,
    X,
    Y,
    Z,
    D,
    kernel_size,
    dilation,
    has_bias,
    is_causal=None,
    dtype=torch.float32,
):
    kernel_size, dilation, is_causal = check_args(kernel_size, dilation, is_causal)
    assert not has_bias or not any(is_causal)
    with torch.no_grad():
        q, k, v = (
            torch.randn((B, H, X, Y, Z, D), device="cuda", dtype=dtype),
            torch.randn((B, H, X, Y, Z, D), device="cuda", dtype=dtype),
            torch.randn((B, H, X, Y, Z, D), device="cuda", dtype=dtype),
        )
        rpb = (
            None
            if not has_bias
            else torch.randn(
                H,
                2 * kernel_size[0] - 1,
                2 * kernel_size[1] - 1,
                2 * kernel_size[2] - 1,
                device="cuda",
                dtype=dtype,
            )
        )
        q_, k_, v_ = (
            q.clone().permute(0, 2, 3, 4, 1, 5).contiguous(),
            k.clone().permute(0, 2, 3, 4, 1, 5).contiguous(),
            v.clone().permute(0, 2, 3, 4, 1, 5).contiguous(),
        )
        rpb_ = None if rpb is None else rpb.clone()

        q = q * (D**-0.5)
        attn_ref = na3d_qk(
            q,
            k,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            rpb=rpb,
        )
        attn_ref = attn_ref.softmax(dim=-1)
        out_ref = (
            na3d_av(
                attn_ref,
                v,
                kernel_size=kernel_size,
                dilation=dilation,
                is_causal=is_causal,
            )
            .permute(0, 2, 3, 4, 1, 5)
            .contiguous()
        )
    return (q_, k_, v_, rpb_, kernel_size, dilation, is_causal), out_ref


def compute_sdpa_reference(B, H, X, Y, Z, D, dtype=torch.float32):
    with torch.no_grad():
        q, k, v = (
            torch.randn((B, H, X * Y * Z, D), device="cuda", dtype=dtype),
            torch.randn((B, H, X * Y * Z, D), device="cuda", dtype=dtype),
            torch.randn((B, H, X * Y * Z, D), device="cuda", dtype=dtype),
        )
        q_, k_, v_ = (
            q.clone().permute(0, 2, 1, 3).reshape(B, X, Y, Z, H, D).contiguous(),
            k.clone().permute(0, 2, 1, 3).reshape(B, X, Y, Z, H, D).contiguous(),
            v.clone().permute(0, 2, 1, 3).reshape(B, X, Y, Z, H, D).contiguous(),
        )

        out_ref = (
            torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
            .permute(0, 2, 1, 3)
            .reshape(B, X, Y, Z, H, D)
            .contiguous()
        )
    return (q_, k_, v_), out_ref


class NA3dTests(unittest.TestCase):
    def _test_against_reference(self, inputs, reference, eps, dtype):
        q, k, v, rpb, kernel_size, dilation, is_causal = inputs
        kernel_size, dilation, is_causal = check_args(kernel_size, dilation, is_causal)
        out_ref = reference
        with torch.no_grad():
            q_, k_, v_ = q.clone().to(dtype), k.clone().to(dtype), v.clone().to(dtype)
            rpb_ = rpb if rpb is None else rpb.clone().to(dtype)
            assert q_.is_cuda and k_.is_cuda and v_.is_cuda

            out = na3d(
                q_,
                k_,
                v_,
                kernel_size=kernel_size,
                dilation=dilation,
                is_causal=is_causal,
                rpb=rpb_,
            )

            torch.testing.assert_close(out.float(), out_ref, atol=eps, rtol=0)

    def _test_all_dtypes_against_bmm_style(
        self,
        B,
        H,
        X,
        Y,
        Z,
        D,
        kernel_size,
        dilation,
        has_bias=False,
        is_causal=None,
        autotune=False,
    ):
        kernel_size, dilation, is_causal = check_args(kernel_size, dilation, is_causal)
        assert not has_bias or not any(is_causal)
        inputs, reference = compute_bmm_reference(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            has_bias=has_bias,
            is_causal=is_causal,
            dtype=torch.float32,
        )

        if autotune:
            enable_autotuner()
        else:
            disable_autotuner()

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

    @skip_if_cuda_is_not_supported()
    @skip_if_fna_is_not_supported()
    def test_against_bmm_style(self):
        problem_sizes = [
            (1, 1, 3, 3, 3, 16, 3, 3, 3, 1, 1, 1),
            (1, 2, 6, 8, 12, 24, 5, 7, 11, 1, 1, 1),
            (1, 4, 6, 8, 12, 32, 3, 3, 3, 2, 2, 4),
            (2, 2, 6, 8, 12, 40, 3, 3, 3, 1, 1, 1),
            (1, 12, 32, 8, 8, 48, 7, 5, 5, 2, 1, 1),
            (4, 8, 32, 10, 10, 48, 7, 3, 3, 1, 2, 3),
        ]
        for (
            B,
            H,
            X,
            Y,
            Z,
            D,
            kernel_size_d,
            kernel_size_h,
            kernel_size_w,
            dilation_d,
            dilation_h,
            dilation_w,
        ) in problem_sizes:
            for causal_d, causal_h, causal_w in product(
                [True, False], [True, False], [True, False]
            ):
                kernel_size = (kernel_size_d, kernel_size_h, kernel_size_w)
                dilation = (dilation_d, dilation_h, dilation_w)
                is_causal = (causal_d, causal_h, causal_w)
                self._test_all_dtypes_against_bmm_style(
                    B=B,
                    H=H,
                    X=X,
                    Y=Y,
                    Z=Z,
                    D=D,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    is_causal=is_causal,
                )
                self._test_all_dtypes_against_bmm_style(
                    B=B,
                    H=H,
                    X=X,
                    Y=Y,
                    Z=Z,
                    D=D,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    is_causal=is_causal,
                    autotune=True,
                )
            self._test_all_dtypes_against_bmm_style(
                B=B,
                H=H,
                X=X,
                Y=Y,
                Z=Z,
                D=D,
                kernel_size=kernel_size,
                dilation=dilation,
                has_bias=True,
            )
            self._test_all_dtypes_against_bmm_style(
                B=B,
                H=H,
                X=X,
                Y=Y,
                Z=Z,
                D=D,
                kernel_size=kernel_size,
                dilation=dilation,
                has_bias=True,
                autotune=True,
            )

    def _test_against_sdpa(self, inputs, reference, eps, dtype):
        q, k, v = inputs
        kernel_size = (q.shape[1], q.shape[2], q.shape[3])
        kernel_size, dilation, is_causal = check_args(kernel_size, 1, False)
        out_ref = reference
        with torch.no_grad():
            q_, k_, v_ = q.clone().to(dtype), k.clone().to(dtype), v.clone().to(dtype)
            assert q_.is_cuda and k_.is_cuda and v_.is_cuda

            out = na3d(
                q_,
                k_,
                v_,
                kernel_size=kernel_size,
                dilation=dilation,
                is_causal=is_causal,
            )

            torch.testing.assert_close(out.float(), out_ref, atol=eps, rtol=0)

    def _test_all_dtypes_against_sdpa(
        self,
        B,
        H,
        X,
        Y,
        Z,
        D,
        autotune=False,
    ):
        inputs, reference = compute_sdpa_reference(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            dtype=torch.float32,
        )

        if autotune:
            enable_autotuner()
        else:
            disable_autotuner()

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
            (1, 1, 3, 3, 3, 16),
            (2, 1, 5, 7, 3, 24),
            (1, 2, 9, 5, 13, 32),
            (2, 2, 15, 3, 5, 40),
            (1, 2, 17, 19, 21, 48),
            (1, 2, 3, 5, 9, 56),
            (1, 1, 7, 9, 3, 64),
            (1, 2, 5, 3, 11, 96),
            (1, 1, 9, 11, 3, 128),
            (1, 1, 3, 5, 9, 256),
            (1, 2, 3, 5, 9, 256),
        ]
        for B, H, X, Y, Z, D in problem_sizes:
            self._test_all_dtypes_against_sdpa(
                B=B,
                H=H,
                X=X,
                Y=Y,
                Z=Z,
                D=D,
            )
            self._test_all_dtypes_against_sdpa(
                B=B,
                H=H,
                X=X,
                Y=Y,
                Z=Z,
                D=D,
                autotune=True,
            )


if __name__ == "__main__":
    torch.manual_seed(42)
    unittest.main()
