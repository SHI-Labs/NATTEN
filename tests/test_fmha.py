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

import random
import unittest
from typing import Optional, Tuple, Union

import natten  # noqa: F401

import torch
from natten.backends.configs.cutlass import (
    get_all_fmha_backward_configs,
    get_all_fmha_forward_configs,
)
from natten.backends.configs.cutlass_blackwell import (
    get_all_fmha_backward_configs as get_all_blackwell_fmha_backward_configs,
    get_all_fmha_forward_configs as get_all_blackwell_fmha_forward_configs,
)
from natten.backends.configs.cutlass_hopper import (
    get_all_fmha_backward_configs as get_all_hopper_fmha_backward_configs,
    get_all_fmha_forward_configs as get_all_hopper_fmha_forward_configs,
)
from natten.functional import attention
from natten.types import KernelSchedule
from natten.utils import log
from natten.utils.testing import (
    skip_if_blackwell_kernels_not_supported,
    skip_if_hopper_kernels_not_supported,
    skip_if_libnatten_is_not_supported,
    skip_if_not_running_extended_tests,
    supports_bfloat16,
    supports_float16,
)
from torch import Tensor


logger = log.get_logger(__name__)


def _reset_everything():
    from natten.context import (
        NattenContext,
        set_memory_usage_preference,
        use_kv_parallelism_in_fused_na,
    )

    NattenContext.reset()
    set_memory_usage_preference("unrestricted")
    use_kv_parallelism_in_fused_na(True)

    torch.manual_seed(42)
    torch.cuda.empty_cache()

    # Hopper and Blackwell FMHA bwd don't have deterministic option.
    torch.use_deterministic_algorithms(False)


def compute_sdpa_reference(
    batch, heads, dim, dim_v, seqlen_q, seqlen_kv, dtype=torch.float32
):
    dim_v = dim_v or dim
    with torch.no_grad():
        q, k, v, d_out = (
            torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=dtype),
            torch.randn((batch, heads, seqlen_kv, dim), device="cuda", dtype=dtype),
            torch.randn((batch, heads, seqlen_kv, dim_v), device="cuda", dtype=dtype),
            torch.randn((batch, heads, seqlen_q, dim_v), device="cuda", dtype=dtype),
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

    with torch.nn.attention.sdpa_kernel(
        backends=[torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]
    ):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

    lse_ref = None

    with torch.no_grad():
        out_ref = out.clone().permute(0, 2, 1, 3).contiguous().float()

    out.backward(d_out)
    with torch.no_grad():
        dq_ref, dk_ref, dv_ref = (
            q.grad.clone().permute(0, 2, 1, 3).contiguous().float(),
            k.grad.clone().permute(0, 2, 1, 3).contiguous().float(),
            v.grad.clone().permute(0, 2, 1, 3).contiguous().float(),
        )
    return (q_, k_, v_, d_out_), (out_ref, lse_ref, dq_ref, dk_ref, dv_ref)


def compute_natten_fmha_reference(
    batch, heads, dim, dim_v, seqlen_q, seqlen_kv, dtype=torch.float32
):
    dim_v = dim_v or dim
    with torch.no_grad():
        q, k, v, d_out = (
            torch.randn((batch, seqlen_q, heads, dim), device="cuda", dtype=dtype),
            torch.randn((batch, seqlen_kv, heads, dim), device="cuda", dtype=dtype),
            torch.randn((batch, seqlen_kv, heads, dim_v), device="cuda", dtype=dtype),
            torch.randn((batch, seqlen_q, heads, dim_v), device="cuda", dtype=dtype),
        )
        q_, k_, v_, d_out_ = (
            q.clone(),
            k.clone(),
            v.clone(),
            d_out.clone(),
        )

    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)
    d_out = d_out.requires_grad_(True)

    out, lse = attention(q, k, v, backend="cutlass-fmha", return_lse=True)

    with torch.no_grad():
        out_ref = out.clone().float()
        lse_ref = lse.clone().float()

    out.backward(d_out)
    with torch.no_grad():
        dq_ref, dk_ref, dv_ref = (
            q.grad.clone().float(),
            k.grad.clone().float(),
            v.grad.clone().float(),
        )
    return (q_, k_, v_, d_out_), (out_ref, lse_ref, dq_ref, dk_ref, dv_ref)


# TODO: write a class like the FNA tests
class FMHABackendTest(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    def _test_against_reference_inputs(
        self,
        inputs,
        reference,
        batch: int,
        heads: int,
        head_dim: int,
        seqlen_q: int,
        seqlen_kv: int,
        atol: Union[float, Tuple[float, float]],
        rtol: Union[float, Tuple[float, float]],
        dtype: torch.dtype,
        test_backprop: bool,
        backend: str,
        q_tile_size: Optional[int] = None,
        kv_tile_size: Optional[int] = None,
        backward_q_tile_size: Optional[int] = None,
        backward_kv_tile_size: Optional[int] = None,
        backward_kv_splits: Optional[int] = None,
        backward_use_pt_reduction: bool = False,
        run_persistent_kernel: bool = True,
        kernel_schedule: Optional[KernelSchedule] = None,
        torch_compile: bool = False,
        test_lse: bool = False,
        reference_str: str = "unspecified reference",
        head_dim_v: Optional[int] = None,
    ):
        head_dim_v = head_dim_v or head_dim
        logger.debug(
            f"Testing FMHA ({backend}) vs {reference_str}: {batch=}, {heads=}, {head_dim=}, {head_dim_v=}, "
            f"{seqlen_q=}, {seqlen_kv=}, {dtype=}, "
            f"{q_tile_size=}, {kv_tile_size=}, {kernel_schedule=}, {run_persistent_kernel=}, {torch_compile=}"
            + (
                f", {backward_q_tile_size=}, {backward_kv_tile_size=}, "
                f"{backward_kv_splits=}, {backward_use_pt_reduction=}."
                if test_backprop
                else "."
            )
        )

        q, k, v, d_out = inputs
        out_ref, lse_ref, dq_ref, dk_ref, dv_ref = reference

        # Run target
        q.requires_grad_(test_backprop)
        k.requires_grad_(test_backprop)
        v.requires_grad_(test_backprop)
        d_out.requires_grad_(test_backprop)

        outputs = attention(
            q,
            k,
            v,
            backend=backend,
            q_tile_size=q_tile_size,
            kv_tile_size=kv_tile_size,
            backward_q_tile_size=backward_q_tile_size,
            backward_kv_tile_size=backward_kv_tile_size,
            backward_kv_splits=backward_kv_splits,
            backward_use_pt_reduction=backward_use_pt_reduction,
            run_persistent_kernel=run_persistent_kernel,
            kernel_schedule=kernel_schedule,
            torch_compile=torch_compile,
            return_lse=test_lse,
        )

        if test_lse:
            assert (
                isinstance(outputs, tuple)
                and len(outputs) == 2
                and all(isinstance(x, Tensor) for x in outputs)
            )
            out_, lse_ = outputs

            out = out_.data.clone().float()
            lse = lse_.clone().float()
        else:
            assert isinstance(outputs, Tensor)
            out_ = outputs

            out = out_.data.clone().float()

        if test_backprop:
            dq, dk, dv = None, None, None
            out_.backward(d_out)
            with torch.no_grad():
                dq, dk, dv = (
                    q.grad.clone().float(),
                    k.grad.clone().float(),
                    v.grad.clone().float(),
                )

        if isinstance(atol, tuple):
            atol_forward, atol_backward = atol
        else:
            atol_forward, atol_backward = atol, atol

        if isinstance(rtol, tuple):
            rtol_forward, rtol_backward = rtol
        else:
            rtol_forward, rtol_backward = rtol, rtol

        torch.testing.assert_close(out, out_ref, atol=atol_forward, rtol=rtol_forward)
        if test_lse:
            assert (
                lse_ref is not None
            ), "Reference did not return LSE. If reference is PyTorch SDPA, it does not have an API for returning LSE. Use CUTLASS FMHA instead!"

            torch.testing.assert_close(
                lse, lse_ref, atol=atol_forward, rtol=rtol_forward
            )

        if test_backprop:
            torch.testing.assert_close(
                dq, dq_ref, atol=atol_backward, rtol=rtol_backward
            )
            torch.testing.assert_close(
                dk, dk_ref, atol=atol_backward, rtol=rtol_backward
            )
            torch.testing.assert_close(
                dv, dv_ref, atol=atol_backward, rtol=rtol_backward
            )

    def _test_against_torch_sdpa(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        seqlen_q: int,
        seqlen_kv: int,
        atol: Union[float, Tuple[float, float]],
        rtol: Union[float, Tuple[float, float]],
        dtype: torch.dtype,
        test_backprop: bool,
        backend: str,
        q_tile_size: Optional[int] = None,
        kv_tile_size: Optional[int] = None,
        backward_q_tile_size: Optional[int] = None,
        backward_kv_tile_size: Optional[int] = None,
        backward_kv_splits: Optional[int] = None,
        backward_use_pt_reduction: bool = False,
        run_persistent_kernel: bool = True,
        kernel_schedule: Optional[KernelSchedule] = None,
        torch_compile: bool = False,
        head_dim_v: Optional[int] = None,
    ):
        head_dim_v = head_dim_v or head_dim
        inputs, reference = compute_sdpa_reference(
            batch, heads, head_dim, head_dim_v, seqlen_q, seqlen_kv, dtype=dtype
        )
        self._test_against_reference_inputs(
            inputs=inputs,
            reference=reference,
            batch=batch,
            heads=heads,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            atol=atol,
            rtol=rtol,
            dtype=dtype,
            test_backprop=test_backprop,
            backend=backend,
            q_tile_size=q_tile_size,
            kv_tile_size=kv_tile_size,
            backward_q_tile_size=backward_q_tile_size,
            backward_kv_tile_size=backward_kv_tile_size,
            backward_kv_splits=backward_kv_splits,
            backward_use_pt_reduction=backward_use_pt_reduction,
            run_persistent_kernel=run_persistent_kernel,
            kernel_schedule=kernel_schedule,
            torch_compile=torch_compile,
            reference_str="torch sdpa",
        )

    def _test_against_natten_cutlass_fmha(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        seqlen_q: int,
        seqlen_kv: int,
        atol: Union[float, Tuple[float, float]],
        rtol: Union[float, Tuple[float, float]],
        dtype: torch.dtype,
        test_backprop: bool,
        test_lse: bool,
        backend: str,
        q_tile_size: Optional[int] = None,
        kv_tile_size: Optional[int] = None,
        backward_q_tile_size: Optional[int] = None,
        backward_kv_tile_size: Optional[int] = None,
        backward_kv_splits: Optional[int] = None,
        backward_use_pt_reduction: bool = False,
        run_persistent_kernel: bool = True,
        kernel_schedule: Optional[KernelSchedule] = None,
        torch_compile: bool = False,
        head_dim_v: Optional[int] = None,
    ):
        head_dim_v = head_dim_v or head_dim
        inputs, reference = compute_natten_fmha_reference(
            batch, heads, head_dim, head_dim_v, seqlen_q, seqlen_kv, dtype=dtype
        )
        self._test_against_reference_inputs(
            inputs=inputs,
            reference=reference,
            batch=batch,
            heads=heads,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            atol=atol,
            rtol=rtol,
            dtype=dtype,
            test_backprop=test_backprop,
            backend=backend,
            q_tile_size=q_tile_size,
            kv_tile_size=kv_tile_size,
            backward_q_tile_size=backward_q_tile_size,
            backward_kv_tile_size=backward_kv_tile_size,
            backward_kv_splits=backward_kv_splits,
            backward_use_pt_reduction=backward_use_pt_reduction,
            run_persistent_kernel=run_persistent_kernel,
            kernel_schedule=kernel_schedule,
            torch_compile=torch_compile,
            test_lse=test_lse,
            reference_str="cutlass-fmha",
        )

    def _test_cutlass_fmha_against_torch_sdpa(
        self,
        batch,
        heads,
        head_dim,
        head_dim_v,
        seqlen_q,
        seqlen_kv,
    ):
        torch.set_default_device("cuda")

        ALLOWED_DTYPES = [
            (torch.float32, (1e-4, 1e-4), (0, 0)),
        ]

        if supports_float16(torch.get_default_device()):
            ALLOWED_DTYPES.append((torch.float16, (1e-2, 4e-2), (0, 1e-3)))

        if supports_bfloat16(torch.get_default_device()):
            ALLOWED_DTYPES.append((torch.bfloat16, (1e-1, 2e-1), (0, 1e-2)))

        for dtype, atol, rtol in ALLOWED_DTYPES:

            dummy_fwd = torch.randn(
                (batch, seqlen_q, heads, max(head_dim, head_dim_v)),
                device="cuda",
                dtype=dtype,
            )

            dummy_bwd = torch.randn(
                (batch, seqlen_kv, heads, max(head_dim, head_dim_v)),
                device="cuda",
                dtype=dtype,
            )

            forward_configs = get_all_fmha_forward_configs(dummy_fwd)
            backward_configs = get_all_fmha_backward_configs(dummy_bwd)

            random.shuffle(forward_configs)
            random.shuffle(backward_configs)

            for i in range(max(len(forward_configs), len(backward_configs))):
                q_tile_size, kv_tile_size = forward_configs[i % len(forward_configs)]
                (
                    backward_q_tile_size,
                    backward_kv_tile_size,
                    kv_splits,
                    use_pt_reduction,
                ) = backward_configs[i % len(backward_configs)]

                self._test_against_torch_sdpa(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    head_dim_v=head_dim_v,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    dtype=dtype,
                    test_backprop=True,
                    atol=atol,
                    rtol=rtol,
                    backend="cutlass-fmha",
                    q_tile_size=q_tile_size,
                    kv_tile_size=kv_tile_size,
                    backward_q_tile_size=backward_q_tile_size,
                    backward_kv_tile_size=backward_kv_tile_size,
                    backward_kv_splits=kv_splits,
                    backward_use_pt_reduction=use_pt_reduction,
                    torch_compile=False,
                )

    def _test_cutlass_blackwell_fmha_against_torch_sdpa(
        self,
        batch,
        heads,
        head_dim,
        seqlen_q,
        seqlen_kv,
    ):
        torch.set_default_device("cuda")

        ALLOWED_DTYPES = [
            (torch.float16, (1e-2, 4e-2), (0, 1e-3)),
            (torch.bfloat16, (1e-1, 2e-1), (0, 1e-2)),
        ]

        for dtype, atol, rtol in ALLOWED_DTYPES:

            dummy = torch.randn(
                (batch, seqlen_kv, heads, head_dim), device="cuda", dtype=dtype
            )

            forward_configs = get_all_blackwell_fmha_forward_configs(dummy)
            backward_configs = get_all_blackwell_fmha_backward_configs(dummy)

            random.shuffle(forward_configs)
            random.shuffle(backward_configs)

            for i in range(max(len(forward_configs), len(backward_configs))):
                q_tile_size, kv_tile_size = forward_configs[i]
                backward_q_tile_size, backward_kv_tile_size = backward_configs[
                    i % len(backward_configs)
                ]

                self._test_against_torch_sdpa(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    dtype=dtype,
                    test_backprop=True,
                    atol=atol,
                    rtol=rtol,
                    backend="blackwell-fmha",
                    q_tile_size=q_tile_size,
                    kv_tile_size=kv_tile_size,
                    backward_q_tile_size=backward_q_tile_size,
                    backward_kv_tile_size=backward_kv_tile_size,
                    torch_compile=False,
                )

    def _test_cutlass_hopper_fmha_against_torch_sdpa(
        self,
        batch,
        heads,
        head_dim,
        seqlen_q,
        seqlen_kv,
    ):
        torch.set_default_device("cuda")

        ALLOWED_DTYPES = [
            (torch.float16, (1e-2, 4e-2), (0, 1e-3)),
            (torch.bfloat16, (1e-1, 2e-1), (0, 1e-2)),
        ]

        for dtype, atol, rtol in ALLOWED_DTYPES:

            dummy = torch.randn(
                (batch, seqlen_kv, heads, head_dim), device="cuda", dtype=dtype
            )

            forward_configs = get_all_hopper_fmha_forward_configs(dummy)
            backward_configs = get_all_hopper_fmha_backward_configs(dummy)

            random.shuffle(forward_configs)
            random.shuffle(backward_configs)

            for i in range(max(len(forward_configs), len(backward_configs))):
                (q_tile_size, kv_tile_size), schedule = forward_configs[
                    i % len(forward_configs)
                ]
                backward_q_tile_size, backward_kv_tile_size = backward_configs[
                    i % len(backward_configs)
                ]

                self._test_against_torch_sdpa(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    dtype=dtype,
                    test_backprop=True,
                    atol=atol,
                    rtol=rtol,
                    backend="hopper-fmha",
                    q_tile_size=q_tile_size,
                    kv_tile_size=kv_tile_size,
                    kernel_schedule=schedule,
                    backward_q_tile_size=backward_q_tile_size,
                    backward_kv_tile_size=backward_kv_tile_size,
                    torch_compile=False,
                )

    def _test_cutlass_hopper_fmha_against_cutlass_2x_fmha(
        self,
        batch,
        heads,
        head_dim,
        seqlen_q,
        seqlen_kv,
    ):
        torch.set_default_device("cuda")

        ALLOWED_DTYPES = [
            (torch.float16, 1e-2, 0),
            (torch.bfloat16, 1e-1, 0),
        ]

        for dtype, atol, rtol in ALLOWED_DTYPES:

            dummy = torch.randn(
                (batch, seqlen_kv, heads, head_dim), device="cuda", dtype=dtype
            )

            forward_configs = get_all_hopper_fmha_forward_configs(dummy)
            backward_configs = get_all_hopper_fmha_backward_configs(dummy)

            random.shuffle(forward_configs)
            random.shuffle(backward_configs)

            for i in range(max(len(forward_configs), len(backward_configs))):
                (q_tile_size, kv_tile_size), schedule = forward_configs[
                    i % len(forward_configs)
                ]
                backward_q_tile_size, backward_kv_tile_size = backward_configs[
                    i % len(backward_configs)
                ]

                self._test_against_natten_cutlass_fmha(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    dtype=dtype,
                    test_backprop=True,
                    test_lse=True,
                    atol=atol,
                    rtol=rtol,
                    backend="hopper-fmha",
                    q_tile_size=q_tile_size,
                    kv_tile_size=kv_tile_size,
                    kernel_schedule=schedule,
                    backward_q_tile_size=backward_q_tile_size,
                    backward_kv_tile_size=backward_kv_tile_size,
                    torch_compile=False,
                )

    def _test_cutlass_blackwell_fmha_against_cutlass_2x_fmha(
        self,
        batch,
        heads,
        head_dim,
        seqlen_q,
        seqlen_kv,
    ):
        torch.set_default_device("cuda")

        ALLOWED_DTYPES = [
            (torch.float16, (1e-2, 4e-2), (0, 1e-3)),
            (torch.bfloat16, (1e-1, 2e-1), (0, 1e-2)),
        ]

        for dtype, atol, rtol in ALLOWED_DTYPES:

            dummy = torch.randn(
                (batch, seqlen_kv, heads, head_dim), device="cuda", dtype=dtype
            )

            forward_configs = get_all_blackwell_fmha_forward_configs(dummy)
            backward_configs = get_all_blackwell_fmha_backward_configs(dummy)

            random.shuffle(forward_configs)
            random.shuffle(backward_configs)

            for i in range(max(len(forward_configs), len(backward_configs))):
                q_tile_size, kv_tile_size = forward_configs[i % len(forward_configs)]
                backward_q_tile_size, backward_kv_tile_size = backward_configs[
                    i % len(backward_configs)
                ]

                self._test_against_natten_cutlass_fmha(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    dtype=dtype,
                    test_backprop=True,
                    test_lse=True,
                    atol=atol,
                    rtol=rtol,
                    backend="blackwell-fmha",
                    q_tile_size=q_tile_size,
                    kv_tile_size=kv_tile_size,
                    backward_q_tile_size=backward_q_tile_size,
                    backward_kv_tile_size=backward_kv_tile_size,
                    torch_compile=False,
                )

    def _test_backend_against_torch_sdpa(
        self,
        batch,
        heads,
        head_dim,
        seqlen_q,
        seqlen_kv,
        backend,
        head_dim_v=None,
    ):
        if backend == "cutlass-fmha":
            return self._test_cutlass_fmha_against_torch_sdpa(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
            )
        elif backend == "blackwell-fmha":
            assert (
                head_dim_v is None or head_dim_v == head_dim
            ), "Blackwell FMHA does not allow head_dim_v."

            return self._test_cutlass_blackwell_fmha_against_torch_sdpa(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
            )
        elif backend == "hopper-fmha":
            assert (
                head_dim_v is None or head_dim_v == head_dim
            ), "Hopper FMHA does not allow head_dim_v."

            return self._test_cutlass_hopper_fmha_against_torch_sdpa(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
            )
        else:
            raise NotImplementedError(f"Add {backend=} to tests.")

    # Intended to test LSE specifically
    def _test_backend_against_natten_cutlass_fmha(
        self,
        batch,
        heads,
        head_dim,
        seqlen_q,
        seqlen_kv,
        backend,
    ):
        if backend == "hopper-fmha":
            return self._test_cutlass_hopper_fmha_against_cutlass_2x_fmha(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
            )
        elif backend == "blackwell-fmha":
            return self._test_cutlass_blackwell_fmha_against_cutlass_2x_fmha(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
            )
        else:
            raise NotImplementedError(f"Add {backend=} to tests.")

    def _test_randsweep_against_torch_sdpa(self, backend, max_tests=1000):
        random.seed(42)

        max_qk = 2**21
        for i in range(max_tests):
            batch = random.choice(range(1, 4))
            heads = random.choice(range(1, 4))

            supports_dim_v = False
            if backend == "blackwell-fmha":
                head_dim_choices = [32, 64, 128]
            elif backend == "hopper-fmha":
                head_dim_choices = [32, 64, 128]
            else:
                assert backend == "cutlass-fmha"
                head_dim_choices = range(8, 1024 + 1, 8)
                supports_dim_v = True

            head_dim = random.choice(head_dim_choices)
            head_dim_v = (
                head_dim if not supports_dim_v else random.choice(head_dim_choices)
            )

            seqlen_q = random.choice(range(8, 2**14, 1))
            seqlen_kv = random.choice(range(8, 2**14, 1))

            while seqlen_q * seqlen_kv > max_qk:
                cut_kv = random.choice([True, False])
                if cut_kv:
                    seqlen_kv = int(seqlen_kv * 0.1)
                else:
                    seqlen_q = int(seqlen_q * 0.1)

            self._test_backend_against_torch_sdpa(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                backend=backend,
            )

    @skip_if_libnatten_is_not_supported()
    def test_cutlass_fmha_fast_against_torch_sdpa(self):
        problem_sizes = [
            (1, 1, 32, 64, 128, 128),
            (2, 2, 64, 32, 128, 128),
            (1, 1, 32, 16, 32, 32),
            (1, 1, 32, 48, 128, 128),
            (1, 1, 192, 64, 3584, 381),
            (1, 1, 288, 256, 12072, 1680),
            (1, 1, 32, 64, 128, 128),
            (1, 1, 32, 128, 128, 4096),
            #
            (2, 2, 64, 64, 128, 128),
            (1, 1, 32, 32, 32, 32),
            (1, 1, 32, 32, 128, 128),
            (1, 1, 192, 192, 3584, 381),
            (1, 1, 288, 288, 12072, 1680),
            (1, 1, 32, 32, 128, 128),
            (1, 1, 32, 32, 128, 4096),
            (1, 1, 32, 32, 128, 258),
            (1, 2, 64, 64, 128, 15),
            (1, 1, 32, 32, 8, 17),
            (1, 1, 64, 64, 17, 49),
            (2, 4, 16, 16, 128, 237),
            (4, 3, 48, 48, 256, 33),
            (1, 1, 128, 128, 128, 75),
            (1, 1, 32, 32, 125, 444),
            (1, 2, 64, 64, 125, 231),
            (1, 1, 128, 128, 256, 10240),
        ]
        for (
            batch,
            heads,
            head_dim,
            head_dim_v,
            seqlen_q,
            seqlen_kv,
        ) in problem_sizes:
            self._test_backend_against_torch_sdpa(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                backend="cutlass-fmha",
            )

    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_cutlass_hopper_fmha_fast(self):
        problem_sizes = [
            (1, 1, 128, 128, 128),
            (2, 1, 128, 128, 128),
            (1, 2, 128, 128, 128),
            (2, 2, 128, 128, 128),
            (2, 2, 64, 128, 128),
            (1, 1, 32, 32, 32),
            (1, 1, 32, 128, 128),
            (1, 1, 128, 3584, 381),
            (1, 1, 128, 12072, 1680),
            (1, 1, 32, 128, 128),
            (1, 1, 32, 128, 4096),
            (1, 1, 32, 128, 258),
            (1, 2, 64, 128, 15),
            (1, 1, 32, 8, 17),
            (1, 1, 64, 17, 49),
            (2, 4, 32, 128, 237),
            (4, 3, 64, 256, 33),
            (1, 1, 128, 128, 75),
            (1, 1, 32, 125, 444),
            (1, 2, 64, 125, 231),
            (1, 1, 128, 256, 10240),
        ]
        for (
            batch,
            heads,
            head_dim,
            seqlen_q,
            seqlen_kv,
        ) in problem_sizes:
            self._test_backend_against_torch_sdpa(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                backend="hopper-fmha",
            )
            self._test_backend_against_natten_cutlass_fmha(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                backend="hopper-fmha",
            )

    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_cutlass_blackwell_fmha_fast(self):
        problem_sizes = [
            (1, 1, 32, 32, 13496),
            (3, 1, 32, 77, 8504),
            (1, 1, 32, 77, 8504),
            (1, 1, 64, 40, 12296),
            (1, 2, 64, 40, 12296),
            (1, 1, 128, 128, 128),
            (2, 2, 64, 128, 128),
            (1, 1, 32, 32, 32),
            (1, 1, 32, 128, 128),
            (1, 1, 128, 3584, 381),
            (1, 1, 128, 12072, 1680),
            (1, 1, 32, 128, 128),
            (1, 1, 32, 128, 4096),
            (1, 1, 32, 128, 258),
            (1, 2, 64, 128, 15),
            (1, 1, 32, 8, 17),
            (1, 1, 64, 17, 49),
            (2, 4, 32, 128, 237),
            (4, 3, 64, 256, 33),
            (1, 1, 128, 128, 75),
            (1, 1, 32, 125, 444),
            (1, 2, 64, 125, 231),
            (1, 1, 128, 256, 10240),
        ]
        for (
            batch,
            heads,
            head_dim,
            seqlen_q,
            seqlen_kv,
        ) in problem_sizes:
            self._test_backend_against_torch_sdpa(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                backend="blackwell-fmha",
            )
            self._test_backend_against_natten_cutlass_fmha(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                backend="blackwell-fmha",
            )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_cutlass_fmha_randsweep_against_torch_sdpa(self):
        self._test_randsweep_against_torch_sdpa(backend="cutlass-fmha")

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_cutlass_blackwell_fmha_randsweep_against_torch_sdpa(self):
        self._test_randsweep_against_torch_sdpa(backend="blackwell-fmha")

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_cutlass_hopper_fmha_randsweep_against_torch_sdpa(self):
        self._test_randsweep_against_torch_sdpa(backend="hopper-fmha")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    unittest.main()
