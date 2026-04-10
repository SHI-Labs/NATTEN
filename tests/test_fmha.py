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
from typing import Optional, Tuple, Union

import natten  # noqa: F401
import pytest
import torch
from natten._environment import _NUM_RAND_SWEEP_TESTS as RAND_SWEEP_TESTS
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
from natten.utils.dtype import is_fp8
from natten.utils.testing import (
    skip_if_blackwell_kernels_not_supported,
    skip_if_hopper_kernels_not_supported,
    skip_if_libnatten_is_not_supported,
    skip_if_not_running_extended_tests,
    supports_bfloat16,
    supports_float16,
)
from torch import Tensor

from .utils import logger


def _reset_everything(random_seed: int = 42, torch_seed: int = 42):
    from natten.context import (
        NattenContext,
        set_memory_usage_preference,
        use_kv_parallelism_in_fused_na,
    )

    NattenContext.reset()
    set_memory_usage_preference("unrestricted")
    use_kv_parallelism_in_fused_na(True)

    random.seed(random_seed)
    torch.manual_seed(torch_seed)
    logger.debug(f"Reset seeds: {random_seed=}, {torch_seed=}")
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(False)


def compute_sdpa_reference(
    batch,
    heads,
    heads_kv,
    dim,
    dim_v,
    seqlen_q,
    seqlen_kv,
    is_causal,
    dtype=torch.float32,
):
    dim_v = dim_v or dim
    assert heads % heads_kv == 0
    h_k = heads // heads_kv
    with torch.no_grad():
        q, k, v, d_out = (
            torch.randn((batch, heads, seqlen_q, dim), device="cuda", dtype=dtype),
            torch.randn((batch, heads_kv, seqlen_kv, dim), device="cuda", dtype=dtype),
            torch.randn(
                (batch, heads_kv, seqlen_kv, dim_v), device="cuda", dtype=dtype
            ),
            torch.randn((batch, heads, seqlen_q, dim_v), device="cuda", dtype=dtype)
            * 0.05,
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

    k_final, v_final = k, v
    # Decomposed GQA/MQA implementation for torch SDPA via explicit repeats
    if h_k > 1:
        k_final = torch.repeat_interleave(k, repeats=h_k, dim=1, output_size=heads)
        v_final = torch.repeat_interleave(v, repeats=h_k, dim=1, output_size=heads)

        assert k_final.shape[:2] == q.shape[:2]
        assert v_final.shape[:2] == q.shape[:2]
        assert k_final.shape[-1] == q.shape[-1]
        assert v_final.shape[-1] == q.shape[-1]
        assert k_final.shape[2] == seqlen_kv
        assert v_final.shape[2] == seqlen_kv

    with torch.nn.attention.sdpa_kernel(
        backends=[torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION]
    ):
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k_final, v_final, is_causal=is_causal
        )

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
    batch,
    heads,
    dim,
    dim_v,
    seqlen_q,
    seqlen_kv,
    is_causal,
    dtype=torch.float32,
    backend="cutlass-fmha",
    heads_kv=None,
    q_tile_size=None,
    kv_tile_size=None,
    backward_q_tile_size=None,
    backward_kv_tile_size=None,
    backward_kv_splits=None,
    backward_use_pt_reduction=False,
):
    dim_v = dim_v or dim
    heads_kv = heads_kv or heads
    with torch.no_grad():
        q, k, v, d_out = (
            torch.randn((batch, seqlen_q, heads, dim), device="cuda", dtype=dtype),
            torch.randn((batch, seqlen_kv, heads_kv, dim), device="cuda", dtype=dtype),
            torch.randn(
                (batch, seqlen_kv, heads_kv, dim_v), device="cuda", dtype=dtype
            ),
            torch.randn((batch, seqlen_q, heads, dim_v), device="cuda", dtype=dtype)
            * 0.05,
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

    out, lse = attention(
        q,
        k,
        v,
        is_causal=is_causal,
        backend=backend,
        q_tile_size=q_tile_size,
        kv_tile_size=kv_tile_size,
        backward_q_tile_size=backward_q_tile_size,
        backward_kv_tile_size=backward_kv_tile_size,
        backward_kv_splits=backward_kv_splits,
        backward_use_pt_reduction=backward_use_pt_reduction,
        return_lse=True,
    )

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
        is_causal: bool,
        atol: Union[float, Tuple[float, float]],
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
        test_lse: bool = False,
        reference_str: str = "unspecified reference",
        heads_kv: Optional[int] = None,
        head_dim_v: Optional[int] = None,
    ):
        heads_kv = heads_kv or heads
        head_dim_v = head_dim_v or head_dim
        determinism = torch.are_deterministic_algorithms_enabled()
        logger.debug(
            f"Testing FMHA ({backend}) vs {reference_str}:\n"
            f"{batch=}, {heads=}, {heads_kv=}, {head_dim=}, {head_dim_v=},\n"
            f"{seqlen_q=}, {seqlen_kv=}, {is_causal=}, {dtype=},\n"
            f"{q_tile_size=}, {kv_tile_size=}, {kernel_schedule=}, {run_persistent_kernel=}"
            + (
                f",\n({determinism=}) {backward_q_tile_size=}, {backward_kv_tile_size=}, "
                f"{backward_kv_splits=}, {backward_use_pt_reduction=}."
                if test_backprop
                else "."
            )
        )

        q, k, v, d_out = inputs
        out_ref, lse_ref, dq_ref, dk_ref, dv_ref = reference
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
        d_out = d_out.to(dtype)

        # Run target
        q.requires_grad_(test_backprop)
        k.requires_grad_(test_backprop)
        v.requires_grad_(test_backprop)
        d_out.requires_grad_(test_backprop)

        outputs = attention(
            q,
            k,
            v,
            is_causal=is_causal,
            backend=backend,
            q_tile_size=q_tile_size,
            kv_tile_size=kv_tile_size,
            backward_q_tile_size=backward_q_tile_size,
            backward_kv_tile_size=backward_kv_tile_size,
            backward_kv_splits=backward_kv_splits,
            backward_use_pt_reduction=backward_use_pt_reduction,
            run_persistent_kernel=run_persistent_kernel,
            kernel_schedule=kernel_schedule,
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

        torch.testing.assert_close(out, out_ref, atol=atol_forward, rtol=0)
        if test_lse:
            assert (
                lse_ref is not None
            ), "Reference did not return LSE. If reference is PyTorch SDPA, it does not have an API for returning LSE. Use CUTLASS FMHA instead!"

            torch.testing.assert_close(lse, lse_ref, atol=atol_forward, rtol=0)

        if test_backprop:
            if isinstance(atol_backward, tuple):
                assert len(atol_backward) == 3
                atol_dq, atol_dk, atol_dv = atol_backward
            else:
                assert isinstance(atol_backward, float) or isinstance(
                    atol_backward, int
                )
                atol_dq, atol_dk, atol_dv = atol_backward, atol_backward, atol_backward
            torch.testing.assert_close(dq, dq_ref, atol=atol_dq, rtol=0)
            torch.testing.assert_close(dk, dk_ref, atol=atol_dk, rtol=0)
            torch.testing.assert_close(dv, dv_ref, atol=atol_dv, rtol=0)

    def _test_against_torch_sdpa(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        seqlen_q: int,
        seqlen_kv: int,
        is_causal: bool,
        atol: Union[float, Tuple[float, float]],
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
        heads_kv: Optional[int] = None,
        head_dim_v: Optional[int] = None,
    ):
        heads_kv = heads_kv or heads
        head_dim_v = head_dim_v or head_dim
        sdpa_dtype = dtype if not is_fp8(dtype) else torch.float16
        inputs, reference = compute_sdpa_reference(
            batch,
            heads,
            heads_kv,
            head_dim,
            head_dim_v,
            seqlen_q,
            seqlen_kv,
            is_causal=is_causal,
            dtype=sdpa_dtype,
        )
        self._test_against_reference_inputs(
            inputs=inputs,
            reference=reference,
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            is_causal=is_causal,
            atol=atol,
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
            reference_str="torch sdpa",
        )

    def _test_against_natten_fmha(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        seqlen_q: int,
        seqlen_kv: int,
        is_causal: bool,
        atol: Union[float, Tuple[float, float]],
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
        head_dim_v: Optional[int] = None,
        heads_kv: Optional[int] = None,
        reference_backend: str = "cutlass-fmha",
        reference_q_tile_size: Optional[int] = None,
        reference_kv_tile_size: Optional[int] = None,
        reference_backward_q_tile_size: Optional[int] = None,
        reference_backward_kv_tile_size: Optional[int] = None,
        reference_backward_kv_splits: Optional[int] = None,
        reference_backward_use_pt_reduction: bool = False,
    ):
        head_dim_v = head_dim_v or head_dim
        heads_kv = heads_kv or heads
        inputs, reference = compute_natten_fmha_reference(
            batch,
            heads,
            head_dim,
            head_dim_v,
            seqlen_q,
            seqlen_kv,
            is_causal=is_causal,
            dtype=dtype,
            backend=reference_backend,
            heads_kv=heads_kv,
            q_tile_size=reference_q_tile_size,
            kv_tile_size=reference_kv_tile_size,
            backward_q_tile_size=reference_backward_q_tile_size,
            backward_kv_tile_size=reference_backward_kv_tile_size,
            backward_kv_splits=reference_backward_kv_splits,
            backward_use_pt_reduction=reference_backward_use_pt_reduction,
        )
        self._test_against_reference_inputs(
            inputs=inputs,
            reference=reference,
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            seqlen_q=seqlen_q,
            seqlen_kv=seqlen_kv,
            is_causal=is_causal,
            atol=atol,
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
            test_lse=test_lse,
            reference_str=reference_backend,
        )

    def _test_cutlass_fmha_against_torch_sdpa(
        self,
        batch,
        heads,
        head_dim,
        head_dim_v,
        seqlen_q,
        seqlen_kv,
        is_causal,
        max_configs=10,
    ):
        torch.set_default_device("cuda")

        is_mla = head_dim_v is not None and head_dim != head_dim_v
        ALLOWED_DTYPES = [
            (torch.float32, (1e-4, (1e-2, 1e-4, 1e-4))),
        ]

        if supports_float16(torch.get_default_device()):
            ALLOWED_DTYPES.append((torch.float16, (1e-2, (1e-2, 1e-2, 1e-2))))

        if supports_bfloat16(torch.get_default_device()):
            ALLOWED_DTYPES.append(
                (
                    torch.bfloat16,
                    (5e-2, (2e-2, 5e-2, 5e-2) if is_mla else (1e-2, 1e-2, 1e-2)),
                )
            )

        for dtype, atol in ALLOWED_DTYPES:

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
            assert len(forward_configs) > 0
            assert len(backward_configs) > 0

            random.shuffle(forward_configs)
            random.shuffle(backward_configs)

            n_configs_to_test = min(
                max_configs, max(len(forward_configs), len(backward_configs))
            )

            for i in range(n_configs_to_test):
                q_tile_size, kv_tile_size = forward_configs[i % len(forward_configs)]
                backward_q_tile_size, backward_kv_tile_size = backward_configs[
                    i % len(backward_configs)
                ]

                self._test_against_torch_sdpa(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    head_dim_v=head_dim_v,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    is_causal=is_causal,
                    dtype=dtype,
                    test_backprop=True,
                    atol=atol,
                    backend="cutlass-fmha",
                    q_tile_size=q_tile_size,
                    kv_tile_size=kv_tile_size,
                    backward_q_tile_size=backward_q_tile_size,
                    backward_kv_tile_size=backward_kv_tile_size,
                )

    def _test_cutlass_blackwell_fmha_against_torch_sdpa(
        self,
        batch,
        heads,
        head_dim,
        seqlen_q,
        seqlen_kv,
        is_causal,
        heads_kv=None,
    ):
        torch.set_default_device("cuda")

        is_gqa = heads_kv is not None and heads != heads_kv
        ALLOWED_DTYPES = [
            (torch.float16, (1e-2, (1e-2, 1e-2, 1e-2))),
            (
                torch.bfloat16,
                (5e-2, (1e-2, 1e-2, 1e-2) if not is_gqa else (1e-2, 5e-2, 5e-2)),
            ),
            (torch.float8_e4m3fn, (5e-1, None)),
            (torch.float8_e5m2, (9e-1, None)),
        ]

        for dtype, atol in ALLOWED_DTYPES:
            if is_fp8(dtype) and head_dim % 16 != 0:
                continue

            dummy = torch.empty(
                (batch, seqlen_kv, heads, head_dim), device="cuda", dtype=dtype
            )

            forward_configs = get_all_blackwell_fmha_forward_configs(dummy)
            backward_configs = get_all_blackwell_fmha_backward_configs(dummy)
            assert len(forward_configs) > 0
            assert len(backward_configs) > 0

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
                    heads_kv=heads_kv,
                    head_dim=head_dim,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    is_causal=is_causal,
                    dtype=dtype,
                    test_backprop=not is_fp8(dtype),
                    atol=atol,
                    backend="blackwell-fmha",
                    q_tile_size=q_tile_size,
                    kv_tile_size=kv_tile_size,
                    backward_q_tile_size=backward_q_tile_size,
                    backward_kv_tile_size=backward_kv_tile_size,
                )

    def _test_cutlass_hopper_fmha_against_torch_sdpa(
        self,
        batch,
        heads,
        head_dim,
        seqlen_q,
        seqlen_kv,
        is_causal,
    ):
        torch.set_default_device("cuda")

        ALLOWED_DTYPES = [
            (torch.float16, (1e-2, (1e-2, 1e-2, 1e-2))),
            (torch.bfloat16, (5e-2, (1e-2, 2e-2, 2e-2))),
        ]

        for dtype, atol in ALLOWED_DTYPES:

            dummy = torch.randn(
                (batch, seqlen_kv, heads, head_dim), device="cuda", dtype=dtype
            )

            forward_configs = get_all_hopper_fmha_forward_configs(dummy)
            backward_configs = get_all_hopper_fmha_backward_configs(dummy)
            assert len(forward_configs) > 0
            assert len(backward_configs) > 0

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
                    is_causal=is_causal,
                    dtype=dtype,
                    test_backprop=True,
                    atol=atol,
                    backend="hopper-fmha",
                    q_tile_size=q_tile_size,
                    kv_tile_size=kv_tile_size,
                    kernel_schedule=schedule,
                    backward_q_tile_size=backward_q_tile_size,
                    backward_kv_tile_size=backward_kv_tile_size,
                )

    def _test_cutlass_hopper_fmha_against_natten(
        self,
        batch,
        heads,
        head_dim,
        seqlen_q,
        seqlen_kv,
        is_causal,
    ):
        torch.set_default_device("cuda")

        ALLOWED_DTYPES = [
            (torch.float16, (1e-2, (1e-2, 1e-2, 1e-2))),
            (torch.bfloat16, (5e-2, (1e-2, 1e-2, 1e-2))),
        ]

        for dtype, atol in ALLOWED_DTYPES:

            dummy = torch.randn(
                (batch, seqlen_kv, heads, head_dim), device="cuda", dtype=dtype
            )

            forward_configs = get_all_hopper_fmha_forward_configs(dummy)
            backward_configs = get_all_hopper_fmha_backward_configs(dummy)
            assert len(forward_configs) > 0
            assert len(backward_configs) > 0

            random.shuffle(forward_configs)
            random.shuffle(backward_configs)

            for i in range(max(len(forward_configs), len(backward_configs))):
                (q_tile_size, kv_tile_size), schedule = forward_configs[
                    i % len(forward_configs)
                ]
                backward_q_tile_size, backward_kv_tile_size = backward_configs[
                    i % len(backward_configs)
                ]

                self._test_against_natten_fmha(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    is_causal=is_causal,
                    dtype=dtype,
                    test_backprop=True,
                    test_lse=True,
                    atol=atol,
                    backend="hopper-fmha",
                    q_tile_size=q_tile_size,
                    kv_tile_size=kv_tile_size,
                    kernel_schedule=schedule,
                    backward_q_tile_size=backward_q_tile_size,
                    backward_kv_tile_size=backward_kv_tile_size,
                )

    def _test_cutlass_fmha_determinism(
        self,
        batch,
        heads,
        head_dim,
        seqlen_q,
        seqlen_kv,
        is_causal,
        max_configs=5,
        fail_case=False,
    ):
        torch.set_default_device("cuda")

        torch.use_deterministic_algorithms(not fail_case)
        ALLOWED_DTYPES = [
            (torch.float32, (0, 0)),
            (torch.float16, (0, 0)),
            (torch.bfloat16, (0, 0)),
        ]

        for dtype, atol in ALLOWED_DTYPES:
            dummy = torch.empty(
                (batch, seqlen_kv, heads, head_dim), device="cuda", dtype=dtype
            )

            forward_configs = get_all_fmha_forward_configs(dummy)
            backward_configs = get_all_fmha_backward_configs(dummy)
            assert len(forward_configs) > 0
            assert len(backward_configs) > 0

            random.shuffle(forward_configs)
            random.shuffle(backward_configs)

            n_configs_to_test = min(
                max_configs, max(len(forward_configs), len(backward_configs))
            )

            for i in range(n_configs_to_test):
                q_tile_size, kv_tile_size = forward_configs[i % len(forward_configs)]
                backward_q_tile_size, backward_kv_tile_size = backward_configs[
                    i % len(backward_configs)
                ]

                self._test_against_natten_fmha(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    is_causal=is_causal,
                    dtype=dtype,
                    test_backprop=True,
                    test_lse=True,
                    atol=atol,
                    backend="cutlass-fmha",
                    q_tile_size=q_tile_size,
                    kv_tile_size=kv_tile_size,
                    backward_q_tile_size=backward_q_tile_size,
                    backward_kv_tile_size=backward_kv_tile_size,
                    backward_kv_splits=1,
                    backward_use_pt_reduction=False,
                    reference_backend="cutlass-fmha",
                    reference_q_tile_size=q_tile_size,
                    reference_kv_tile_size=kv_tile_size,
                    reference_backward_q_tile_size=backward_q_tile_size,
                    reference_backward_kv_tile_size=backward_kv_tile_size,
                    reference_backward_kv_splits=1,
                    reference_backward_use_pt_reduction=False,
                )

        torch.use_deterministic_algorithms(False)

    def _test_cutlass_blackwell_fmha_against_natten(
        self,
        batch,
        heads,
        head_dim,
        seqlen_q,
        seqlen_kv,
        is_causal,
        heads_kv=None,
        determinism=False,
        fail_case=False,
    ):
        torch.set_default_device("cuda")

        ALLOWED_DTYPES = [
            (torch.float16, (1e-2, (1e-2, 1e-2, 1e-2))),
            (torch.bfloat16, (5e-2, (1e-2, 1e-2, 1e-2))),
        ]

        if determinism:
            ALLOWED_DTYPES = [
                (torch.float16, (0, 0)),
                (torch.bfloat16, (0, 0)),
            ]
            torch.use_deterministic_algorithms(not fail_case)

        for dtype, atol in ALLOWED_DTYPES:

            dummy = torch.randn(
                (batch, seqlen_kv, heads, head_dim), device="cuda", dtype=dtype
            )

            forward_configs = get_all_blackwell_fmha_forward_configs(dummy)
            backward_configs = get_all_blackwell_fmha_backward_configs(dummy)
            assert len(forward_configs) > 0
            assert len(backward_configs) > 0

            random.shuffle(forward_configs)
            random.shuffle(backward_configs)

            for i in range(max(len(forward_configs), len(backward_configs))):
                q_tile_size, kv_tile_size = forward_configs[i % len(forward_configs)]
                backward_q_tile_size, backward_kv_tile_size = backward_configs[
                    i % len(backward_configs)
                ]

                reference_kwargs = {}
                if determinism:
                    reference_kwargs["reference_backend"] = "blackwell-fmha"
                    reference_kwargs["reference_q_tile_size"] = q_tile_size
                    reference_kwargs["reference_kv_tile_size"] = kv_tile_size
                    reference_kwargs["reference_backward_q_tile_size"] = (
                        backward_q_tile_size
                    )
                    reference_kwargs["reference_backward_kv_tile_size"] = (
                        backward_kv_tile_size
                    )

                self._test_against_natten_fmha(
                    batch=batch,
                    heads=heads,
                    heads_kv=heads_kv,
                    head_dim=head_dim,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    is_causal=is_causal,
                    dtype=dtype,
                    test_backprop=True,
                    test_lse=True,
                    atol=atol,
                    backend="blackwell-fmha",
                    q_tile_size=q_tile_size,
                    kv_tile_size=kv_tile_size,
                    backward_q_tile_size=backward_q_tile_size,
                    backward_kv_tile_size=backward_kv_tile_size,
                    **reference_kwargs,
                )

        torch.use_deterministic_algorithms(False)

    def _test_backend_against_torch_sdpa(
        self,
        batch,
        heads,
        head_dim,
        seqlen_q,
        seqlen_kv,
        is_causal,
        backend,
        heads_kv=None,
        head_dim_v=None,
    ):
        if backend == "cutlass-fmha":
            assert heads_kv is None or heads_kv == heads
            return self._test_cutlass_fmha_against_torch_sdpa(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                is_causal=is_causal,
            )
        elif backend == "blackwell-fmha":
            assert (
                head_dim_v is None or head_dim_v == head_dim
            ), "Blackwell FMHA does not allow head_dim_v."

            return self._test_cutlass_blackwell_fmha_against_torch_sdpa(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                head_dim=head_dim,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                is_causal=is_causal,
            )
        elif backend == "hopper-fmha":
            assert heads_kv is None or heads_kv == heads
            assert (
                head_dim_v is None or head_dim_v == head_dim
            ), "Hopper FMHA does not allow head_dim_v."

            return self._test_cutlass_hopper_fmha_against_torch_sdpa(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                is_causal=is_causal,
            )
        else:
            raise NotImplementedError(f"Add {backend=} to tests.")

    # Intended to test LSE and determinism
    def _test_backend_against_natten(
        self,
        batch,
        heads,
        head_dim,
        seqlen_q,
        seqlen_kv,
        is_causal,
        backend,
        heads_kv=None,
        head_dim_v=None,
        determinism: bool = False,
        fail_case: bool = False,
    ):
        if backend == "hopper-fmha":
            assert heads_kv is None or heads_kv == heads
            assert (
                head_dim_v is None or head_dim_v == head_dim
            ), "Hopper FMHA does not allow head_dim_v."
            assert not determinism
            assert not fail_case
            return self._test_cutlass_hopper_fmha_against_natten(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                is_causal=is_causal,
            )
        elif backend == "cutlass-fmha":
            assert heads_kv is None or heads_kv == heads
            assert determinism
            return self._test_cutlass_fmha_determinism(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                is_causal=is_causal,
                fail_case=fail_case,
            )
        elif backend == "blackwell-fmha":
            assert (
                head_dim_v is None or head_dim_v == head_dim
            ), "Blackwell FMHA does not allow head_dim_v."

            return self._test_cutlass_blackwell_fmha_against_natten(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                head_dim=head_dim,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                is_causal=is_causal,
                determinism=determinism,
                fail_case=fail_case,
            )
        else:
            raise NotImplementedError(f"Add {backend=} to tests.")

    def _test_randsweep(
        self,
        backend,
        max_tests=1000,
        reference: str = "sdpa",
        determinism: bool = False,
    ):
        assert reference in ("sdpa", "natten")
        max_qk = 2**21
        for i in range(max_tests):
            # to help with reproducibility of use cases
            _reset_everything(random_seed=i, torch_seed=i)
            batch = random.choice(range(1, 4))

            supports_dim_v = False
            supports_gqa_mqa = False
            if backend == "blackwell-fmha":
                head_dim_choices = list(range(8, 129, 8))
                heads_choices = range(1, 8 + 1)
                supports_gqa_mqa = True
            elif backend == "hopper-fmha":
                head_dim_choices = [32, 64, 128]
                heads_choices = range(1, 4)
            else:
                assert backend == "cutlass-fmha"
                head_dim_choices = range(8, 1024 + 1, 8)  # type: ignore[assignment]
                heads_choices = range(1, 4)
                supports_dim_v = True

            heads = random.choice(heads_choices)
            heads_kv = (
                heads
                if not supports_gqa_mqa
                else random.choice(
                    [1] + [i for i in range(1, heads + 1) if heads % i == 0]
                )
            )
            assert heads >= heads_kv and heads % heads_kv == 0

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

            for is_causal in [False, True]:
                if reference == "sdpa":
                    assert not determinism
                    self._test_backend_against_torch_sdpa(
                        batch=batch,
                        heads=heads,
                        heads_kv=heads_kv,
                        head_dim=head_dim,
                        head_dim_v=head_dim_v,
                        seqlen_q=seqlen_q,
                        seqlen_kv=seqlen_kv,
                        is_causal=is_causal,
                        backend=backend,
                    )
                elif reference == "natten":
                    self._test_backend_against_natten(
                        batch=batch,
                        heads=heads,
                        heads_kv=heads_kv,
                        head_dim=head_dim,
                        head_dim_v=head_dim_v,
                        seqlen_q=seqlen_q,
                        seqlen_kv=seqlen_kv,
                        is_causal=is_causal,
                        backend=backend,
                        determinism=determinism,
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
        for i, (
            batch,
            heads,
            head_dim,
            head_dim_v,
            seqlen_q,
            seqlen_kv,
        ) in enumerate(problem_sizes):
            _reset_everything(random_seed=i, torch_seed=i)
            for is_causal in [False, True]:
                self._test_backend_against_torch_sdpa(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    head_dim_v=head_dim_v,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    is_causal=is_causal,
                    backend="cutlass-fmha",
                )

    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_cutlass_hopper_fmha_fast(self):
        problem_sizes = [
            (3, 2, 64, 256, 7000),
            (3, 2, 64, 256, 8192),
            (3, 2, 64, 163, 11012),
            (1, 1, 32, 32, 128),
            (1, 1, 32, 128, 128),
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
        for i, (
            batch,
            heads,
            head_dim,
            seqlen_q,
            seqlen_kv,
        ) in enumerate(problem_sizes):
            _reset_everything(random_seed=i, torch_seed=i)
            for is_causal in [True, False]:
                self._test_backend_against_torch_sdpa(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    is_causal=is_causal,
                    backend="hopper-fmha",
                )
                self._test_backend_against_natten(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    is_causal=is_causal,
                    backend="hopper-fmha",
                )

    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_cutlass_blackwell_fmha_fast(self):
        problem_sizes = [
            #
            (1, 1, 1, 128, 128, 128),
            (1, 4, 1, 120, 128, 128),
            (1, 4, 2, 72, 128, 128),
            (1, 4, 4, 96, 128, 128),
            (1, 1, 1, 48, 128, 1024),
            (1, 1, 1, 128, 128, 13568),
            (1, 1, 1, 16, 128, 13496),
            (1, 1, 1, 32, 128, 13496),
            (1, 1, 1, 32, 32, 13496),
            (3, 1, 1, 32, 77, 8504),
            (1, 1, 1, 32, 77, 8504),
            (1, 1, 1, 64, 40, 12296),
            (1, 2, 1, 64, 40, 12296),
            (1, 2, 2, 64, 40, 12296),
            (1, 1, 1, 128, 128, 128),
            (2, 2, 1, 64, 128, 128),
            (2, 2, 2, 64, 128, 128),
            (1, 1, 1, 32, 32, 32),
            (1, 1, 1, 32, 128, 128),
            (1, 1, 1, 128, 3584, 381),
            (1, 1, 1, 112, 12072, 1680),
            (1, 1, 1, 32, 128, 128),
            (1, 1, 1, 32, 128, 4096),
            (1, 1, 1, 32, 128, 258),
            (1, 2, 1, 64, 128, 15),
            (1, 2, 2, 64, 128, 15),
            (1, 1, 1, 32, 8, 17),
            (1, 1, 1, 64, 17, 49),
            (2, 4, 1, 32, 128, 237),
            (2, 4, 2, 32, 128, 237),
            (2, 4, 4, 32, 128, 237),
            (4, 3, 1, 64, 256, 33),
            (4, 3, 3, 64, 256, 33),
            (1, 1, 1, 128, 128, 75),
            (1, 1, 1, 32, 125, 444),
            (1, 2, 1, 64, 125, 231),
            (1, 2, 2, 64, 125, 231),
            (1, 1, 1, 128, 256, 10240),
        ]
        for i, (
            batch,
            heads_q,
            heads_kv,
            head_dim,
            seqlen_q,
            seqlen_kv,
        ) in enumerate(problem_sizes):
            _reset_everything(random_seed=i, torch_seed=i)
            for is_causal in [False, True]:
                self._test_backend_against_torch_sdpa(
                    batch=batch,
                    heads=heads_q,
                    heads_kv=heads_kv,
                    head_dim=head_dim,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    is_causal=is_causal,
                    backend="blackwell-fmha",
                )
                if heads_q == heads_kv:
                    self._test_backend_against_natten(
                        batch=batch,
                        heads=heads_q,
                        head_dim=head_dim,
                        seqlen_q=seqlen_q,
                        seqlen_kv=seqlen_kv,
                        is_causal=is_causal,
                        backend="blackwell-fmha",
                    )

    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_cutlass_blackwell_fmha_determinism(self):
        problem_sizes = [
            (3, 7, 1, 128, 3584, 2077),
            (2, 4, 2, 32, 1639, 4000),
            (2, 4, 4, 32, 4096, 4096),
        ]

        for i, (
            batch,
            heads_q,
            heads_kv,
            head_dim,
            seqlen_q,
            seqlen_kv,
        ) in enumerate(problem_sizes):
            _reset_everything(random_seed=i, torch_seed=i)
            for is_causal in [False, True]:
                self._test_backend_against_natten(
                    batch=batch,
                    heads=heads_q,
                    heads_kv=heads_kv,
                    head_dim=head_dim,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    is_causal=is_causal,
                    backend="blackwell-fmha",
                    determinism=True,
                )

    @pytest.mark.xfail(
        reason="exact match must fail when determinism is disabled.", strict=True
    )
    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_cutlass_blackwell_fmha_determinism_xfail0(self):
        _reset_everything(random_seed=0, torch_seed=0)
        problem_sizes = [
            (3, 7, 1, 128, 3584, 2077),
        ]

        for (
            batch,
            heads_q,
            heads_kv,
            head_dim,
            seqlen_q,
            seqlen_kv,
        ) in problem_sizes:
            self._test_backend_against_natten(
                batch=batch,
                heads=heads_q,
                heads_kv=heads_kv,
                head_dim=head_dim,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                is_causal=False,
                backend="blackwell-fmha",
                determinism=True,
                fail_case=True,
            )

    @pytest.mark.xfail(
        reason="exact match must fail when determinism is disabled.", strict=True
    )
    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_cutlass_blackwell_fmha_determinism_xfail1(self):
        _reset_everything(random_seed=0, torch_seed=0)
        problem_sizes = [
            (2, 4, 2, 32, 11283, 9873),
        ]

        for (
            batch,
            heads_q,
            heads_kv,
            head_dim,
            seqlen_q,
            seqlen_kv,
        ) in problem_sizes:
            self._test_backend_against_natten(
                batch=batch,
                heads=heads_q,
                heads_kv=heads_kv,
                head_dim=head_dim,
                seqlen_q=seqlen_q,
                seqlen_kv=seqlen_kv,
                is_causal=True,
                backend="blackwell-fmha",
                determinism=True,
                fail_case=True,
            )

    @skip_if_libnatten_is_not_supported()
    def test_cutlass_fmha_determinism(self):
        problem_sizes = [
            (3, 7, 1, 128, 3584, 2077),
            (2, 4, 2, 32, 1639, 4000),
            (2, 4, 4, 32, 4096, 4096),
        ]

        for i, (
            batch,
            heads_q,
            heads_kv,
            head_dim,
            seqlen_q,
            seqlen_kv,
        ) in enumerate(problem_sizes):
            _reset_everything(random_seed=i, torch_seed=i)
            for is_causal in [False, True]:
                self._test_backend_against_natten(
                    batch=batch,
                    heads=heads_q,
                    head_dim=head_dim,
                    seqlen_q=seqlen_q,
                    seqlen_kv=seqlen_kv,
                    is_causal=is_causal,
                    backend="cutlass-fmha",
                    determinism=True,
                )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_cutlass_fmha_determinism_randsweep(self):
        self._test_randsweep(
            backend="cutlass-fmha",
            max_tests=max(2, RAND_SWEEP_TESTS // 20),
            reference="natten",
            determinism=True,
        )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_cutlass_fmha_randsweep_against_torch_sdpa(self):
        self._test_randsweep(
            backend="cutlass-fmha", max_tests=max(1, RAND_SWEEP_TESTS // 2)
        )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_cutlass_blackwell_fmha_randsweep_against_torch_sdpa(self):
        self._test_randsweep(backend="blackwell-fmha", max_tests=RAND_SWEEP_TESTS)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_cutlass_blackwell_fmha_determinism_randsweep(self):
        self._test_randsweep(
            backend="blackwell-fmha",
            max_tests=max(2, RAND_SWEEP_TESTS // 20),
            reference="natten",
            determinism=True,
        )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_hopper_kernels_not_supported()
    def test_cutlass_hopper_fmha_randsweep_against_torch_sdpa(self):
        self._test_randsweep(backend="hopper-fmha", max_tests=RAND_SWEEP_TESTS)


if __name__ == "__main__":
    unittest.main()
