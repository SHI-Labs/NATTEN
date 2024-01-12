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

import logging
import unittest

import torch

from natten import has_bfloat, has_half
from natten.functional import na3d_av, na3d_qk, na3d_qk_with_bias
from natten.utils.testing import (
    skip_if_cuda_is_not_supported,
    skip_if_nested_is_not_supported,
)
from torch.autograd import gradcheck

HAS_HALF = has_half()
HAS_BFLOAT = has_bfloat()
logger = logging.getLogger(__name__)


def init_cpu_ref(B, H, X, Y, Z, D, kernel_size, dilation, has_bias):
    with torch.no_grad():
        q, k, v = (
            torch.randn((B, H, X, Y, Z, D)) * (D**-0.5),
            torch.randn((B, H, X, Y, Z, D)),
            torch.randn((B, H, X, Y, Z, D)),
        )
        rpb = (
            None
            if not has_bias
            else torch.randn(
                H, 2 * kernel_size - 1, 2 * kernel_size - 1, 2 * kernel_size - 1
            )
        )
        q_, k_, v_ = (
            q.clone().cuda(),
            k.clone().cuda(),
            v.clone().cuda(),
        )
        rpb_ = None if rpb is None else rpb.clone().cuda()

        attn_ref = na3d_qk_with_bias(
            q, k, rpb, kernel_size, kernel_size, dilation, dilation
        )
        attn_ref = attn_ref.softmax(dim=-1)
        out_ref = na3d_av(attn_ref, v, kernel_size, kernel_size, dilation, dilation)
    return (q_, k_, v_, rpb_, kernel_size, dilation), (attn_ref.cuda(), out_ref.cuda())


class NA3DTests(unittest.TestCase):
    def _test_against_cpu(self, inputs, reference, eps, dtype):
        q, k, v, rpb, kernel_size, dilation = inputs
        attn_ref, out_ref = reference
        with torch.no_grad():
            q_, k_, v_ = q.clone().to(dtype), k.clone().to(dtype), v.clone().to(dtype)
            rpb_ = rpb if rpb is None else rpb.clone().to(dtype)
            assert q_.is_cuda and k_.is_cuda and v_.is_cuda

            attn = na3d_qk_with_bias(
                q_, k_, rpb_, kernel_size, kernel_size, dilation, dilation
            )
            attn = attn.softmax(dim=-1)
            out = na3d_av(attn, v_, kernel_size, kernel_size, dilation, dilation)

            torch.testing.assert_close(attn.float(), attn_ref, atol=eps, rtol=0)
            torch.testing.assert_close(out.float(), out_ref, atol=eps, rtol=0)

    def _test_all_dtypes_against_cpu(
        self, B, H, X, Y, Z, D, kernel_size, dilation, has_bias=False
    ):
        inputs, reference = init_cpu_ref(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            has_bias=has_bias,
        )

        self._test_against_cpu(
            inputs=inputs,
            reference=reference,
            dtype=torch.float32,
            eps=1e-4,
        )
        if HAS_HALF:
            self._test_against_cpu(
                inputs=inputs,
                reference=reference,
                dtype=torch.float16,
                eps=1e-1,
            )
        if HAS_BFLOAT:
            self._test_against_cpu(
                inputs=inputs,
                reference=reference,
                dtype=torch.bfloat16,
                eps=1e-1,
            )

    @skip_if_cuda_is_not_supported()
    def test_cpu_vs_cuda(self):
        self._test_all_dtypes_against_cpu(
            B=2, H=3, X=9, Y=10, Z=11, D=32, kernel_size=3, dilation=1
        )
        self._test_all_dtypes_against_cpu(
            B=2, H=2, X=9, Y=10, Z=11, D=32, kernel_size=3, dilation=3
        )
        self._test_all_dtypes_against_cpu(
            B=2, H=1, X=9, Y=10, Z=11, D=32, kernel_size=7, dilation=1
        )

    def _test_autograd_qk(
        self, B, H, X, Y, Z, D, kernel_size, dilation, eps, device, L_extra=0
    ):
        assert L_extra >= 0
        kwargs = {"dtype": torch.float64, "device": device, "requires_grad": True}
        query = torch.randn((B, H, X, Y, Z, D), **kwargs)
        key = torch.randn((B, H, X, Y, Z, D), **kwargs)
        rpb = torch.randn(
            (H, 2 * kernel_size - 1, 2 * kernel_size - 1, 2 * kernel_size - 1), **kwargs
        )
        extra_key = None if L_extra <= 0 else torch.randn((B, H, L_extra, D), **kwargs)
        variables = [
            query,
            key,
            rpb,
            kernel_size,
            kernel_size,
            dilation,
            dilation,
            extra_key,
        ]

        assert gradcheck(
            na3d_qk_with_bias,
            variables,
            eps=1e-6,
            atol=eps,
            rtol=1e-4,
            nondet_tol=0,
            fast_mode=False,
        ), "Autograd check failed for NA3D: QK."

    def _test_autograd_av(
        self, B, H, X, Y, Z, D, kernel_size, dilation, eps, device, L_extra=0
    ):
        assert L_extra >= 0
        kwargs = {"dtype": torch.float64, "device": device, "requires_grad": True}
        attn = torch.randn((B, H, X, Y, Z, kernel_size**3 + L_extra), **kwargs)
        value = torch.randn((B, H, X, Y, Z, D), **kwargs)
        extra_value = (
            None if L_extra <= 0 else torch.randn((B, H, L_extra, D), **kwargs)
        )
        variables = [
            attn,
            value,
            kernel_size,
            kernel_size,
            dilation,
            dilation,
            extra_value,
        ]

        assert gradcheck(
            na3d_av,
            variables,
            eps=1e-6,
            atol=eps,
            rtol=1e-4,
            nondet_tol=0,
            fast_mode=False,
        ), "Autograd check failed for NA3D: AV."

    def _test_autograd(self, B, H, X, Y, Z, D, kernel_size, dilation, eps, device):
        self._test_autograd_qk(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            eps=eps,
            device=device,
        )
        self._test_autograd_qk(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            eps=eps,
            device=device,
            L_extra=9,
        )
        self._test_autograd_av(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            eps=eps,
            device=device,
        )
        self._test_autograd_av(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            eps=eps,
            device=device,
            L_extra=9,
        )

    def test_autograd_cpu(self):
        self._test_autograd(
            B=1,
            H=1,
            X=3,
            Y=3,
            Z=6,
            D=4,
            kernel_size=3,
            dilation=1,
            eps=1e-6,
            device="cpu",
        )

    @skip_if_cuda_is_not_supported()
    def test_autograd_cuda(self):
        self._test_autograd(
            B=1,
            H=2,
            X=6,
            Y=3,
            Z=8,
            D=8,
            kernel_size=3,
            dilation=1,
            eps=1e-6,
            device="cuda",
        )

    @unittest.expectedFailure
    def test_invalid_kernel_size(self):
        self._test_autograd(
            B=1,
            H=2,
            X=6,
            Y=3,
            Z=8,
            D=8,
            kernel_size=2,
            dilation=1,
            eps=1e-6,
            device="cpu",
        )

    @unittest.expectedFailure
    def test_invalid_dilation(self):
        self._test_autograd(
            B=1,
            H=2,
            X=6,
            Y=3,
            Z=8,
            D=8,
            kernel_size=3,
            dilation=0,
            eps=1e-6,
            device="cpu",
        )

    def _test_fwad_qk(self, B, H, X, Y, Z, D, kernel_size, dilation, device, L_extra=0):
        assert L_extra >= 0
        kwargs = {"dtype": torch.float64, "device": device, "requires_grad": True}
        query = torch.randn((B, H, X, Y, Z, D), **kwargs)
        key = torch.randn((B, H, X, Y, Z, D), **kwargs)
        extra_key = None if L_extra <= 0 else torch.randn((B, H, L_extra, D), **kwargs)
        variables = [
            query,
            key,
            kernel_size,
            kernel_size,
            dilation,
            dilation,
            extra_key,
        ]

        assert gradcheck(
            na3d_qk,
            variables,
            check_forward_ad=True,
            check_backward_ad=False,
            check_undefined_grad=False,
            check_batched_grad=False,
            check_grad_dtypes=False,
        ), "Forward mode autograd check failed for NA3D: QK."

    def _test_fwad_av(self, B, H, X, Y, Z, D, kernel_size, dilation, device, L_extra=0):
        assert L_extra >= 0
        kwargs = {"dtype": torch.float64, "device": device, "requires_grad": True}
        attn = torch.randn((B, H, X, Y, Z, kernel_size**3 + L_extra), **kwargs)
        value = torch.randn((B, H, X, Y, Z, D), **kwargs)
        extra_value = (
            None if L_extra <= 0 else torch.randn((B, H, L_extra, D), **kwargs)
        )
        variables = [
            attn,
            value,
            kernel_size,
            kernel_size,
            dilation,
            dilation,
            extra_value,
        ]

        assert gradcheck(
            na3d_av,
            variables,
            check_forward_ad=True,
            check_backward_ad=False,
            check_undefined_grad=False,
            check_batched_grad=False,
            check_grad_dtypes=False,
        ), "Forward mode autograd check failed for NA3D: AV."

    def _test_fwad(self, B, H, X, Y, Z, D, kernel_size, dilation, device):
        self._test_fwad_qk(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            device=device,
        )
        self._test_fwad_qk(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            device=device,
            L_extra=9,
        )
        self._test_fwad_av(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            device=device,
        )
        self._test_fwad_av(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            device=device,
            L_extra=9,
        )

    def test_fwad_cpu(self):
        self._test_fwad(
            B=1,
            H=1,
            X=3,
            Y=3,
            Z=6,
            D=4,
            kernel_size=3,
            dilation=1,
            device="cpu",
        )

    @skip_if_cuda_is_not_supported()
    def test_fwad_cuda(self):
        self._test_fwad(
            B=1,
            H=2,
            X=6,
            Y=3,
            Z=8,
            D=8,
            kernel_size=3,
            dilation=1,
            device="cuda",
        )

    def _test_nested_qk_forward(self, dtype, device, test_additional_tokens=True):
        kernel_size, dilation = 7, 2
        kernel_size_d, dilation_d = 3, 3
        kwargs = {"dtype": dtype, "device": device, "requires_grad": False}
        query = torch.nested.nested_tensor(
            [
                torch.randn(1, 2, 9, 14, 16, 16),
                torch.randn(2, 8, 9, 16, 18, 32),
                torch.randn(4, 1, 9, 32, 20, 16),
            ],
            **kwargs,
        )
        key = torch.nested.nested_tensor(
            [
                torch.randn(1, 2, 9, 14, 16, 16),
                torch.randn(2, 8, 9, 16, 18, 32),
                torch.randn(4, 1, 9, 32, 20, 16),
            ],
            **kwargs,
        )
        out_nested = na3d_qk(
            query, key, kernel_size_d, kernel_size, dilation_d, dilation
        )
        out_ref = []
        for q, k in zip(query, key):
            out_ref.append(
                na3d_qk(q, k, kernel_size_d, kernel_size, dilation_d, dilation)
            )

        for o, o_ref in zip(out_nested, out_ref):
            torch.testing.assert_close(o, o_ref, atol=1e-6, rtol=0)

        if test_additional_tokens:
            n_extra_tokens = 9
            additional_keys = torch.nested.nested_tensor(
                [
                    torch.randn(1, 2, n_extra_tokens, 16),
                    torch.randn(2, 8, n_extra_tokens, 32),
                    torch.randn(4, 1, n_extra_tokens, 16),
                ],
                **kwargs,
            )
            out_nested = na3d_qk(
                query,
                key,
                kernel_size_d,
                kernel_size,
                dilation_d,
                dilation,
                additional_keys=additional_keys,
            )
            out_ref = []
            for q, k, k_add in zip(query, key, additional_keys):
                out_ref.append(
                    na3d_qk(
                        q,
                        k,
                        kernel_size_d,
                        kernel_size,
                        dilation_d,
                        dilation,
                        additional_keys=k_add,
                    )
                )

            for o, o_ref in zip(out_nested, out_ref):
                torch.testing.assert_close(o, o_ref, atol=1e-6, rtol=0)

    def _test_nested_av_forward(self, dtype, device, test_additional_tokens=True):
        kernel_size, dilation = 7, 2
        kernel_size_d, dilation_d = 3, 3
        kwargs = {"dtype": dtype, "device": device, "requires_grad": False}
        attn = torch.nested.nested_tensor(
            [
                torch.randn(1, 2, 9, 14, 16, kernel_size_d * kernel_size * kernel_size),
                torch.randn(2, 8, 9, 16, 18, kernel_size_d * kernel_size * kernel_size),
                torch.randn(4, 1, 9, 32, 20, kernel_size_d * kernel_size * kernel_size),
            ],
            **kwargs,
        )
        value = torch.nested.nested_tensor(
            [
                torch.randn(1, 2, 9, 14, 16, 16),
                torch.randn(2, 8, 9, 16, 18, 32),
                torch.randn(4, 1, 9, 32, 20, 16),
            ],
            **kwargs,
        )
        out_nested = na3d_av(
            attn, value, kernel_size_d, kernel_size, dilation_d, dilation
        )
        out_ref = []
        for a, v in zip(attn, value):
            out_ref.append(
                na3d_av(a, v, kernel_size_d, kernel_size, dilation_d, dilation)
            )

        for o, o_ref in zip(out_nested, out_ref):
            torch.testing.assert_close(o, o_ref, atol=1e-6, rtol=0)

        if test_additional_tokens:
            n_extra_tokens = 9
            attn = torch.nested.nested_tensor(
                [
                    torch.randn(
                        1,
                        2,
                        9,
                        14,
                        16,
                        kernel_size_d * kernel_size * kernel_size + n_extra_tokens,
                    ),
                    torch.randn(
                        2,
                        8,
                        9,
                        16,
                        18,
                        kernel_size_d * kernel_size * kernel_size + n_extra_tokens,
                    ),
                    torch.randn(
                        4,
                        1,
                        9,
                        32,
                        20,
                        kernel_size_d * kernel_size * kernel_size + n_extra_tokens,
                    ),
                ],
                **kwargs,
            )
            additional_values = torch.nested.nested_tensor(
                [
                    torch.randn(1, 2, n_extra_tokens, 16),
                    torch.randn(2, 8, n_extra_tokens, 32),
                    torch.randn(4, 1, n_extra_tokens, 16),
                ],
                **kwargs,
            )
            out_nested = na3d_av(
                attn,
                value,
                kernel_size_d,
                kernel_size,
                dilation_d,
                dilation,
                additional_values=additional_values,
            )
            out_ref = []
            for a, v, v_add in zip(attn, value, additional_values):
                out_ref.append(
                    na3d_av(
                        a,
                        v,
                        kernel_size_d,
                        kernel_size,
                        dilation_d,
                        dilation,
                        additional_values=v_add,
                    )
                )

            for o, o_ref in zip(out_nested, out_ref):
                torch.testing.assert_close(o, o_ref, atol=1e-6, rtol=0)

    @skip_if_nested_is_not_supported()
    def test_nested_forward_cpu(self):
        self._test_nested_qk_forward(dtype=torch.float32, device="cpu")
        self._test_nested_av_forward(dtype=torch.float32, device="cpu")

    @skip_if_cuda_is_not_supported()
    @skip_if_nested_is_not_supported()
    def test_nested_forward_cuda(self):
        self._test_nested_qk_forward(dtype=torch.float16, device="cuda")
        self._test_nested_av_forward(dtype=torch.float16, device="cuda")

    def _test_with_extra_tokens(
        self,
        B,
        H,
        X,
        Y,
        Z,
        D,
        kernel_size,
        dilation,
        has_bias,
        dtype,
        device="cuda",
        eps=1e-6,
        L_extra=9,
        broadcast_extra_kv_batch=False,
    ):
        assert L_extra > 0
        kwargs = {"device": device, "dtype": dtype}
        with torch.no_grad():
            q, k, v = (
                torch.randn((B, H, X, Y, Z, D), **kwargs) * (D**-0.5),
                torch.randn((B, H, X, Y, Z, D), **kwargs),
                torch.randn((B, H, X, Y, Z, D), **kwargs),
            )
            q_ref, k_ref, v_ref = q.clone(), k.clone(), v.clone()

            rpb, rpb_ref = None, None
            if has_bias:
                rpb = torch.randn(
                    H,
                    2 * kernel_size - 1,
                    2 * kernel_size - 1,
                    2 * kernel_size - 1,
                    **kwargs,
                )
                rpb_ref = rpb.clone()

            extra_k = torch.randn(
                (1 if broadcast_extra_kv_batch else B, H, L_extra, D), **kwargs
            )
            extra_v = torch.randn(
                (1 if broadcast_extra_kv_batch else B, H, L_extra, D), **kwargs
            )
            extra_k_ref, extra_v_ref = extra_k.clone(), extra_v.clone()
            if broadcast_extra_kv_batch:
                extra_k, extra_v = extra_k.expand(B, H, L_extra, D), extra_v.expand(
                    B, H, L_extra, D
                )
                extra_k_ref, extra_v_ref = extra_k_ref.repeat(
                    B, 1, 1, 1
                ), extra_v_ref.repeat(B, 1, 1, 1)

            # Reference implementation
            attn_extra_ref = (
                (
                    q_ref.view(B * H, X * Y * Z, D).contiguous()
                    @ extra_k_ref.view(B * H, L_extra, D).transpose(-2, -1).contiguous()
                )
                .view(B, H, X, Y, Z, L_extra)
                .contiguous()
            )
            attn_na_ref = na3d_qk_with_bias(
                q_ref, k_ref, rpb_ref, kernel_size, kernel_size, dilation, dilation
            )
            attn_ref = torch.cat([attn_na_ref, attn_extra_ref], dim=-1)
            attn_ref_softmax = attn_ref.softmax(dim=-1)
            attn_na_softmax_ref, attn_extra_softmax_ref = attn_ref_softmax.split(
                [kernel_size**3, L_extra], dim=-1
            )
            attn_na_softmax_ref, attn_extra_softmax_ref = (
                attn_na_softmax_ref.contiguous(),
                attn_extra_softmax_ref.contiguous(),
            )
            out_na_ref = na3d_av(
                attn_na_softmax_ref, v_ref, kernel_size, kernel_size, dilation, dilation
            )
            out_extra_ref = (
                (
                    attn_extra_softmax_ref.view(B * H, X * Y * Z, L_extra).contiguous()
                    @ extra_v_ref.view(B * H, L_extra, D).contiguous()
                )
                .view(B, H, X, Y, Z, D)
                .contiguous()
            )
            out_ref = out_extra_ref + out_na_ref

            # Op
            attn = na3d_qk_with_bias(
                q,
                k,
                rpb,
                kernel_size,
                kernel_size,
                dilation,
                dilation,
                additional_keys=extra_k,
            )
            attn_na, attn_extra = attn.split([kernel_size**3, L_extra], dim=-1)
            attn_na, attn_extra = attn_na.contiguous(), attn_extra.contiguous()
            attn_softmax = attn.softmax(dim=-1)
            attn_na_softmax, attn_extra_softmax = attn_softmax.split(
                [kernel_size**3, L_extra], dim=-1
            )
            attn_na_softmax, attn_extra_softmax = (
                attn_na_softmax.contiguous(),
                attn_extra_softmax.contiguous(),
            )
            out = na3d_av(
                attn_softmax,
                v,
                kernel_size,
                kernel_size,
                dilation,
                dilation,
                additional_values=extra_v,
            )

            # Elementwise checks
            torch.testing.assert_close(attn_na, attn_na_ref, atol=eps, rtol=eps)
            torch.testing.assert_close(
                attn_na_softmax, attn_na_softmax_ref, atol=eps, rtol=eps
            )
            torch.testing.assert_close(attn_extra, attn_extra_ref, atol=eps, rtol=eps)
            torch.testing.assert_close(
                attn_extra_softmax, attn_extra_softmax_ref, atol=eps, rtol=eps
            )
            torch.testing.assert_close(attn, attn_ref, atol=eps, rtol=eps)
            torch.testing.assert_close(out, out_ref, atol=eps, rtol=eps)

    def _test_cpu_with_extra_tokens(
        self, B, H, X, Y, Z, D, kernel_size, dilation, has_bias=False
    ):
        self._test_with_extra_tokens(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            has_bias=has_bias,
            dtype=torch.float32,
            device="cpu",
        )
        self._test_with_extra_tokens(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            has_bias=has_bias,
            dtype=torch.float32,
            device="cpu",
            broadcast_extra_kv_batch=True,
        )

    def _test_cuda_with_extra_tokens(
        self, B, H, X, Y, Z, D, kernel_size, dilation, has_bias=False
    ):
        self._test_with_extra_tokens(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            has_bias=has_bias,
            dtype=torch.float32,
            device="cuda",
        )
        self._test_with_extra_tokens(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            kernel_size=kernel_size,
            dilation=dilation,
            has_bias=has_bias,
            dtype=torch.float32,
            device="cuda",
            broadcast_extra_kv_batch=True,
        )
        if HAS_HALF:
            self._test_with_extra_tokens(
                B=B,
                H=H,
                X=X,
                Y=Y,
                Z=Z,
                D=D,
                kernel_size=kernel_size,
                dilation=dilation,
                has_bias=has_bias,
                dtype=torch.float16,
                device="cuda",
            )
        if HAS_BFLOAT:
            self._test_with_extra_tokens(
                B=B,
                H=H,
                X=X,
                Y=Y,
                Z=Z,
                D=D,
                kernel_size=kernel_size,
                dilation=dilation,
                has_bias=has_bias,
                dtype=torch.bfloat16,
                device="cuda",
            )

    def test_cpu_with_extra_tokens(self):
        self._test_cpu_with_extra_tokens(
            B=2, H=3, X=9, Y=10, Z=11, D=32, kernel_size=3, dilation=1
        )
        self._test_cpu_with_extra_tokens(
            B=2, H=2, X=9, Y=10, Z=11, D=32, kernel_size=3, dilation=3
        )
        self._test_cpu_with_extra_tokens(
            B=2, H=1, X=9, Y=10, Z=11, D=32, kernel_size=7, dilation=1
        )

    @skip_if_cuda_is_not_supported()
    def test_cuda_with_extra_tokens(self):
        self._test_cuda_with_extra_tokens(
            B=2, H=3, X=9, Y=10, Z=11, D=32, kernel_size=3, dilation=1
        )
        self._test_cuda_with_extra_tokens(
            B=2, H=2, X=9, Y=10, Z=11, D=32, kernel_size=3, dilation=3
        )
        self._test_cuda_with_extra_tokens(
            B=2, H=1, X=9, Y=10, Z=11, D=32, kernel_size=7, dilation=1
        )
        self._test_cuda_with_extra_tokens(
            B=1, H=2, X=28, Y=28, Z=28, D=32, has_bias=True, kernel_size=13, dilation=2
        )

    def _test_against_self_attention(
        self,
        B,
        H,
        X,
        Y,
        Z,
        D,
        dtype,
        device="cuda",
        eps=1e-6,
    ):
        # Only odd-sized kernels are valid in NA,
        # and NA == SA when kernel size equals
        # input size. NATTEN only allows square-sized
        # kernels, so we need to make sure that the
        # input size is an odd value across every
        # spatial dim.
        assert X == Y == Z and X % 2 == 1
        kernel_size = X
        dilation = 1
        kwargs = {"device": device, "dtype": dtype}
        with torch.no_grad():
            q, k, v = (
                torch.randn((B, H, X, Y, Z, D), **kwargs) * (D**-0.5),
                torch.randn((B, H, X, Y, Z, D), **kwargs),
                torch.randn((B, H, X, Y, Z, D), **kwargs),
            )
            q_ref, k_ref, v_ref = q.clone(), k.clone(), v.clone()

            # Reference implementation
            attn_ref = (
                (
                    q_ref.view(B * H, X * Y * Z, D).contiguous()
                    @ k_ref.view(B * H, X * Y * Z, D).transpose(-2, -1).contiguous()
                )
                .view(B, H, X, Y, Z, X * Y * Z)
                .contiguous()
            )
            attn_ref_softmax = attn_ref.softmax(dim=-1)
            out_ref = (
                (
                    attn_ref_softmax.view(B * H, X * Y * Z, X * Y * Z).contiguous()
                    @ v_ref.view(B * H, X * Y * Z, D).contiguous()
                )
                .view(B, H, X, Y, Z, D)
                .contiguous()
            )

            # Op
            attn = na3d_qk(q, k, kernel_size, kernel_size, dilation, dilation)
            attn_softmax = attn.softmax(dim=-1)
            out = na3d_av(attn_softmax, v, kernel_size, kernel_size, dilation, dilation)

            # We can only check the outputs against each other, and
            # not attention weights, because they are stored in a
            # different order with NATTEN.
            torch.testing.assert_close(out, out_ref, atol=eps, rtol=eps)

    def _test_cpu_against_self_attention(self, B, H, X, Y, Z, D):
        assert X == Y == Z and X % 2 == 1
        self._test_against_self_attention(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            dtype=torch.float32,
            device="cpu",
        )

    def _test_cuda_against_self_attention(self, B, H, X, Y, Z, D):
        assert X == Y == Z and X % 2 == 1
        self._test_against_self_attention(
            B=B,
            H=H,
            X=X,
            Y=Y,
            Z=Z,
            D=D,
            dtype=torch.float32,
            device="cuda",
            eps=1e-4,
        )
        if HAS_HALF:
            self._test_against_self_attention(
                B=B,
                H=H,
                X=X,
                Y=Y,
                Z=Z,
                D=D,
                dtype=torch.float16,
                device="cuda",
                eps=1e-1,
            )
        if HAS_BFLOAT:
            self._test_against_self_attention(
                B=B,
                H=H,
                X=X,
                Y=Y,
                Z=Z,
                D=D,
                dtype=torch.bfloat16,
                device="cuda",
                eps=1e-1,
            )

    def test_cpu_against_self_attention(self):
        self._test_cpu_against_self_attention(B=1, H=2, X=7, Y=7, Z=7, D=32)

    @skip_if_cuda_is_not_supported()
    def test_cuda_against_self_attention(self):
        self._test_cuda_against_self_attention(B=2, H=4, X=3, Y=3, Z=3, D=32)
        self._test_cuda_against_self_attention(B=2, H=4, X=5, Y=5, Z=5, D=32)
        self._test_cuda_against_self_attention(B=2, H=4, X=7, Y=7, Z=7, D=32)
        self._test_cuda_against_self_attention(B=2, H=4, X=9, Y=9, Z=9, D=32)


if __name__ == "__main__":
    torch.manual_seed(42)
    unittest.main()
