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
from typing import List, Optional, Tuple

import natten  # noqa: F401
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
from natten.functional import attention
from natten.utils import log
from natten.utils.testing import (
    skip_if_blackwell_kernels_not_supported,
    skip_if_libnatten_is_not_supported,
    skip_if_not_running_extended_tests,
)


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


def compute_split_reference(
    batch: int,
    heads: int,
    head_dim: int,
    seqlens_Q_list: List[int],
    seqlens_KV_list: List[int],
    is_causal: bool,
    backend: str,
    test_backprop: bool,
    dtype: torch.dtype = torch.float32,
    heads_kv: Optional[int] = None,
    head_dim_v: Optional[int] = None,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
    backward_q_tile_size: Optional[int] = None,
    backward_kv_tile_size: Optional[int] = None,
    backward_kv_splits: Optional[int] = None,
    backward_use_pt_reduction: bool = False,
    run_persistent_kernel: bool = True,
):
    heads_kv = heads_kv or heads
    head_dim_v = head_dim_v or head_dim

    assert len(seqlens_Q_list) == len(seqlens_KV_list) == batch

    seqlen_q_total = sum(seqlens_Q_list)
    seqlen_kv_total = sum(seqlens_KV_list)
    dtype_safe = torch.float16
    with torch.no_grad():
        q_ref, k_ref, v_ref, d_out_ref = (
            torch.randn(
                (1, seqlen_q_total, heads, head_dim), device="cuda", dtype=dtype_safe
            ).to(dtype),
            torch.randn(
                (1, seqlen_kv_total, heads_kv, head_dim),
                device="cuda",
                dtype=dtype_safe,
            ).to(dtype),
            torch.randn(
                (1, seqlen_kv_total, heads_kv, head_dim_v),
                device="cuda",
                dtype=dtype_safe,
            ).to(dtype),
            torch.randn(
                (1, seqlen_q_total, heads, head_dim_v), device="cuda", dtype=dtype_safe
            ).to(dtype),
        )
        q, k, v, d_out = (
            q_ref.clone(),
            k_ref.clone(),
            v_ref.clone(),
            d_out_ref.clone(),
        )

    out_list = []
    lse_list = []
    d_q_list = []
    d_k_list = []
    d_v_list = []

    q_start, kv_start = 0, 0
    for b in range(batch):
        seqlen_q = seqlens_Q_list[b]
        seqlen_kv = seqlens_KV_list[b]

        q_ = q_ref[:, q_start : q_start + seqlen_q, :, :].clone()
        k_ = k_ref[:, kv_start : kv_start + seqlen_kv, :, :].clone()
        v_ = v_ref[:, kv_start : kv_start + seqlen_kv, :, :].clone()

        if test_backprop:
            q_ = q_.requires_grad_(True)
            k_ = k_.requires_grad_(True)
            v_ = v_.requires_grad_(True)
            d_out_ = (
                d_out_ref[:, q_start : q_start + seqlen_q, :, :]
                .clone()
                .requires_grad_(True)
            )

        out_, lse_ = attention(
            q_,
            k_,
            v_,
            is_causal=is_causal,
            backend=backend,
            return_lse=True,
            q_tile_size=q_tile_size,
            kv_tile_size=kv_tile_size,
            backward_q_tile_size=backward_q_tile_size,
            backward_kv_tile_size=backward_kv_tile_size,
            backward_kv_splits=backward_kv_splits,
            backward_use_pt_reduction=backward_use_pt_reduction,
            run_persistent_kernel=run_persistent_kernel,
        )

        if test_backprop:
            out_.backward(d_out_)

        with torch.no_grad():
            out_list.append(out_.data.clone().float())
            lse_list.append(lse_.data.clone().float())
            if test_backprop:
                assert q_.grad is not None
                assert k_.grad is not None
                assert v_.grad is not None
                d_q_list.append(q_.grad.clone().float())
                d_k_list.append(k_.grad.clone().float())
                d_v_list.append(v_.grad.clone().float())

        q_start += seqlen_q
        kv_start += seqlen_kv

    assert q_start == seqlen_q_total
    assert kv_start == seqlen_kv_total

    out_ref = torch.cat(out_list, dim=1)
    lse_ref = torch.cat(lse_list, dim=1)
    assert out_ref.shape[:3] == q_ref.shape[:3]
    dq_ref = None
    dk_ref = None
    dv_ref = None
    if test_backprop:
        dq_ref = torch.cat(d_q_list, dim=1)
        dk_ref = torch.cat(d_k_list, dim=1)
        dv_ref = torch.cat(d_v_list, dim=1)

        assert dq_ref.shape == q_ref.shape
        assert dk_ref.shape == k_ref.shape
        assert dv_ref.shape == v_ref.shape

    return (q, k, v, d_out), (out_ref, lse_ref, dq_ref, dk_ref, dv_ref)


class FMHAVarlenTest(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    def _test_against_manual_varlen(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        seqlens_Q_list: List[int],
        seqlens_KV_list: List[int],
        is_causal: bool,
        dtype: torch.dtype,
        atol_fwd: Tuple[float, float],
        atol_bwd: Optional[Tuple[float, float, float]],
        backend: str,
        reference_backend: str,
        test_backprop: bool,
        q_tile_size: Optional[int] = None,
        kv_tile_size: Optional[int] = None,
        backward_q_tile_size: Optional[int] = None,
        backward_kv_tile_size: Optional[int] = None,
        backward_kv_splits: Optional[int] = None,
        backward_use_pt_reduction: bool = False,
        run_persistent_kernel: bool = True,
        reference_q_tile_size: Optional[int] = None,
        reference_kv_tile_size: Optional[int] = None,
        reference_backward_q_tile_size: Optional[int] = None,
        reference_backward_kv_tile_size: Optional[int] = None,
        reference_backward_kv_splits: Optional[int] = None,
        reference_backward_use_pt_reduction: bool = False,
        reference_run_persistent_kernel: bool = True,
        heads_kv: Optional[int] = None,
        head_dim_v: Optional[int] = None,
    ):
        heads_kv = heads_kv or heads
        head_dim_v = head_dim_v or head_dim

        logger.debug(
            f"Testing FMHA varlen ({backend}) vs {reference_backend}: {batch=}, {heads=}, {heads_kv=}, {head_dim=}, {head_dim_v=}, "
            f"{seqlens_Q_list=}, {seqlens_KV_list=}, {is_causal=}, {dtype=}, "
            f"{q_tile_size=}, {kv_tile_size=}, {run_persistent_kernel=}"
            + (
                f", {backward_q_tile_size=}, {backward_kv_tile_size=}, "
                f"{backward_kv_splits=}, {backward_use_pt_reduction=}."
                if test_backprop
                else "."
            )
        )

        inputs, reference = compute_split_reference(
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            seqlens_Q_list=seqlens_Q_list,
            seqlens_KV_list=seqlens_KV_list,
            is_causal=is_causal,
            dtype=dtype,
            backend=reference_backend,
            q_tile_size=reference_q_tile_size,
            kv_tile_size=reference_kv_tile_size,
            backward_q_tile_size=reference_backward_q_tile_size,
            backward_kv_tile_size=reference_backward_kv_tile_size,
            backward_kv_splits=reference_backward_kv_splits,
            backward_use_pt_reduction=reference_backward_use_pt_reduction,
            run_persistent_kernel=reference_run_persistent_kernel,
            test_backprop=test_backprop,
        )

        q, k, v, d_out = inputs
        out_ref, lse_ref, dq_ref, dk_ref, dv_ref = reference
        q = q.to(dtype)
        k = k.to(dtype)
        v = v.to(dtype)
        d_out = d_out.to(dtype)

        # Run target
        if test_backprop:
            q.requires_grad_(test_backprop)
            k.requires_grad_(test_backprop)
            v.requires_grad_(test_backprop)
            d_out.requires_grad_(test_backprop)

        seqlens_Q = torch.tensor(seqlens_Q_list, dtype=torch.int32, device=q.device)
        seqlens_KV = torch.tensor(seqlens_KV_list, dtype=torch.int32, device=q.device)

        out_, lse_ = attention(
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
            return_lse=True,
            seqlens_Q=seqlens_Q,
            seqlens_KV=seqlens_KV,
        )
        out = out_.float()
        lse = lse_.float()

        if test_backprop:
            dq, dk, dv = None, None, None
            out_.backward(d_out)
            with torch.no_grad():
                dq, dk, dv = (
                    q.grad.clone().float(),
                    k.grad.clone().float(),
                    v.grad.clone().float(),
                )

        atol_out, atol_lse = atol_fwd
        assert out.shape == out_ref.shape

        torch.testing.assert_close(out, out_ref, atol=atol_out, rtol=0)
        torch.testing.assert_close(lse, lse_ref, atol=atol_lse, rtol=0)

        if test_backprop:
            assert atol_bwd is not None
            atol_dq, atol_dk, atol_dv = atol_bwd
            torch.testing.assert_close(dq, dq_ref, atol=atol_dq, rtol=0)
            torch.testing.assert_close(dk, dk_ref, atol=atol_dk, rtol=0)
            torch.testing.assert_close(dv, dv_ref, atol=atol_dv, rtol=0)

    def _test_cutlass_fmha_varlen(
        self,
        batch,
        heads,
        head_dim,
        head_dim_v,
        seqlens_Q_list,
        seqlens_KV_list,
        is_causal,
        max_configs=5,
    ):
        head_dim_v = head_dim_v or head_dim
        torch.set_default_device("cuda")

        # We're testing against the same backend and same dtype,
        # but with varlen implemented as multiple kernel calls, so
        # error thresholds should be much smaller here.
        # This is therefore only a test of the varlen functionality.
        # Correctness per dtype is expected to be verified in the main
        # fmha tests.
        # dQ still needs a more relaxed threshold because of the non-determinism
        ALLOWED_DTYPES = [
            # dtype, (atol_out, atol_lse), (atol_dq, atol_dk, atol_dv)
            (torch.float32, (1e-6, 1e-6), (1e-2, 1e-6, 1e-6)),
            (torch.float16, (1e-6, 1e-6), (5e-2, 5e-6, 5e-6)),
            (torch.bfloat16, (1e-6, 1e-6), (5e-2, 5e-6, 5e-6)),
        ]

        for dtype, atol_fwd, atol_bwd in ALLOWED_DTYPES:
            dummy_fwd = torch.empty(
                (1, min(seqlens_Q_list), heads, max(head_dim, head_dim_v)),
                device="cuda",
                dtype=dtype,
            )
            dummy_bwd = torch.empty(
                (1, min(seqlens_KV_list), heads, max(head_dim, head_dim_v)),
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
                (
                    backward_q_tile_size,
                    backward_kv_tile_size,
                    kv_splits,
                    use_pt_reduction,
                ) = backward_configs[i % len(backward_configs)]

                self._test_against_manual_varlen(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    head_dim_v=head_dim_v,
                    seqlens_Q_list=seqlens_Q_list,
                    seqlens_KV_list=seqlens_KV_list,
                    is_causal=is_causal,
                    dtype=dtype,
                    atol_fwd=atol_fwd,
                    atol_bwd=atol_bwd,
                    backend="cutlass-fmha",
                    reference_backend="cutlass-fmha",
                    test_backprop=True,
                    q_tile_size=q_tile_size,
                    kv_tile_size=kv_tile_size,
                    backward_q_tile_size=backward_q_tile_size,
                    backward_kv_tile_size=backward_kv_tile_size,
                    backward_kv_splits=kv_splits,
                    backward_use_pt_reduction=use_pt_reduction,
                    reference_q_tile_size=q_tile_size,
                    reference_kv_tile_size=kv_tile_size,
                    reference_backward_q_tile_size=backward_q_tile_size,
                    reference_backward_kv_tile_size=backward_kv_tile_size,
                    reference_backward_kv_splits=kv_splits,
                    reference_backward_use_pt_reduction=use_pt_reduction,
                )

    def _test_cutlass_blackwell_fmha_varlen(
        self,
        batch,
        heads,
        head_dim,
        seqlens_Q_list,
        seqlens_KV_list,
        is_causal,
        heads_kv: Optional[int] = None,
    ):
        torch.set_default_device("cuda")

        # We're testing against the same backend and same dtype,
        # but with varlen implemented as multiple kernel calls, so
        # error thresholds should be much smaller here.
        # This is therefore only a test of the varlen functionality.
        # Correctness per dtype is expected to be verified in the main
        # fmha tests.
        # dQ still needs a more relaxed threshold because of the non-determinism
        ALLOWED_DTYPES = [
            # dtype, (atol_out, atol_lse), (atol_dq, atol_dk, atol_dv)
            (torch.float16, (1e-6, 1e-6), (1e-2, 5e-6, 5e-6)),
            (torch.bfloat16, (1e-6, 1e-6), (1e-2, 5e-6, 5e-6)),
            (torch.float8_e4m3fn, (1e-6, 1e-6), None),
            (torch.float8_e5m2, (1e-6, 1e-6), None),
        ]

        for dtype, atol_fwd, atol_bwd in ALLOWED_DTYPES:
            dummy = torch.empty((1, 128, heads, head_dim), device="cuda", dtype=dtype)

            forward_configs = get_all_blackwell_fmha_forward_configs(dummy)
            backward_configs = get_all_blackwell_fmha_backward_configs(dummy)
            assert len(forward_configs) > 0
            test_backprop = len(backward_configs) > 0

            random.shuffle(forward_configs)
            random.shuffle(backward_configs)

            for i in range(max(len(forward_configs), len(backward_configs))):
                q_tile_size, kv_tile_size = forward_configs[i]
                backward_q_tile_size, backward_kv_tile_size = None, None
                if test_backprop:
                    backward_q_tile_size, backward_kv_tile_size = backward_configs[
                        i % len(backward_configs)
                    ]

                for run_persistent_kernel in [True, False]:
                    self._test_against_manual_varlen(
                        batch=batch,
                        heads=heads,
                        heads_kv=heads_kv,
                        head_dim=head_dim,
                        seqlens_Q_list=seqlens_Q_list,
                        seqlens_KV_list=seqlens_KV_list,
                        is_causal=is_causal,
                        dtype=dtype,
                        atol_fwd=atol_fwd,
                        atol_bwd=atol_bwd,
                        backend="blackwell-fmha",
                        reference_backend="blackwell-fmha",
                        test_backprop=test_backprop,
                        q_tile_size=q_tile_size,
                        kv_tile_size=kv_tile_size,
                        backward_q_tile_size=backward_q_tile_size,
                        backward_kv_tile_size=backward_kv_tile_size,
                        run_persistent_kernel=run_persistent_kernel,
                    )

    def _test_varlen_backend(
        self,
        batch,
        heads,
        head_dim,
        seqlens_Q_list,
        seqlens_KV_list,
        is_causal,
        backend,
        heads_kv=None,
        head_dim_v=None,
    ):
        if backend == "cutlass-fmha":
            assert heads_kv is None or heads_kv == heads
            return self._test_cutlass_fmha_varlen(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                seqlens_Q_list=seqlens_Q_list,
                seqlens_KV_list=seqlens_KV_list,
                is_causal=is_causal,
            )
        elif backend == "blackwell-fmha":
            assert (
                head_dim_v is None or head_dim_v == head_dim
            ), "Blackwell FMHA does not allow head_dim_v."

            return self._test_cutlass_blackwell_fmha_varlen(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                head_dim=head_dim,
                seqlens_Q_list=seqlens_Q_list,
                seqlens_KV_list=seqlens_KV_list,
                is_causal=is_causal,
            )
        else:
            raise NotImplementedError(f"Add {backend=} to tests.")

    def _test_varlen_randsweep(self, backend, max_tests=1000):
        random.seed(42)

        max_qk = 2**17
        for i in range(max_tests):
            batch = random.choice(range(1, 12))

            supports_dim_v = False
            supports_gqa_mqa = False
            if backend == "blackwell-fmha":
                head_dim_choices = [32, 64, 128]
                heads_choices = range(1, 8 + 1)
                supports_gqa_mqa = True
            elif backend == "hopper-fmha":
                head_dim_choices = [32, 64, 128]
                heads_choices = range(1, 4)
            else:
                assert backend == "cutlass-fmha"
                head_dim_choices = range(8, 256 + 1, 8)
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

            seqlens_Q_list = []
            seqlens_KV_list = []
            for i in range(batch):
                max_q = min(2**12, max(max_qk - sum(seqlens_Q_list), 24))
                max_k = min(2**12, max(max_qk - sum(seqlens_KV_list), 24))
                new_q = random.choice(range(8, max_q, 1))
                new_k = random.choice(range(8, max_k, 1))
                seqlens_Q_list.append(new_q)
                seqlens_KV_list.append(new_k)

            for is_causal in [False, True]:
                # TODO
                if backend not in ["blackwell-fmha", "cutlass-fmha"] and is_causal:
                    continue

                self._test_varlen_backend(
                    batch=batch,
                    heads=heads,
                    heads_kv=heads_kv,
                    head_dim=head_dim,
                    head_dim_v=head_dim_v,
                    seqlens_Q_list=seqlens_Q_list,
                    seqlens_KV_list=seqlens_KV_list,
                    is_causal=is_causal,
                    backend=backend,
                )

    @skip_if_libnatten_is_not_supported()
    def test_cutlass_varlen_fmha_fast(self):
        random.seed(42)
        problem_sizes = [
            (
                7,
                7,
                56,
                [126, 1127, 3130, 3782, 1445, 751, 594],
                [746, 2886, 1273, 1399, 33, 3570, 2327],
            ),
            (
                8,
                6,
                240,
                [1912, 1057, 1518, 2489, 3955, 2950, 3193, 678],
                [2873, 3149, 695, 2495, 3071, 477, 3363, 3959],
            ),
            (6, 1, 128, [128, 128, 135, 121, 128, 128], [128, 128, 135, 121, 128, 128]),
            (5, 1, 128, [128, 128, 135, 128, 128], [128, 128, 135, 128, 128]),
            (2, 1, 128, [135, 200], [128, 768]),
            (2, 1, 128, [1024, 200], [128, 768]),
            (2, 1, 128, [135, 200], [135, 768]),
            (2, 1, 128, [1024, 200], [135, 768]),
            (2, 1, 128, [1024, 256], [128, 768]),
            (4, 1, 128, [1024, 8, 17, 2048], [10, 20, 512, 16]),
            (3, 2, 128, [268, 1584, 1571], [2448, 4088, 1925]),
            (2, 1, 128, [1024, 256], [512, 768]),
        ]
        for (
            batch,
            heads,
            head_dim,
            seqlens_Q_list,
            seqlens_KV_list,
        ) in problem_sizes:
            for is_causal in [False, True]:
                self._test_varlen_backend(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    seqlens_Q_list=seqlens_Q_list,
                    seqlens_KV_list=seqlens_KV_list,
                    is_causal=is_causal,
                    backend="cutlass-fmha",
                )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_cutlass_varlen_fmha_extended(self):
        self._test_varlen_randsweep(backend="cutlass-fmha", max_tests=RAND_SWEEP_TESTS)

    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_cutlass_blackwell_varlen_fmha_fast(self):
        problem_sizes = [
            (
                9,
                4,
                128,
                [2669, 2240, 910, 2421, 3323, 34, 3308, 2867, 1401],
                [2880, 1726, 1847, 1147, 3568, 3116, 661, 1739, 1146],
            ),
            (6, 1, 128, [128, 128, 135, 121, 128, 128], [128, 128, 135, 121, 128, 128]),
            (5, 1, 128, [128, 128, 135, 128, 128], [128, 128, 135, 128, 128]),
            (2, 1, 128, [135, 200], [128, 768]),
            (2, 1, 128, [1024, 200], [128, 768]),
            (2, 1, 128, [135, 200], [135, 768]),
            (2, 1, 128, [1024, 200], [135, 768]),
            (2, 1, 128, [1024, 256], [128, 768]),
            (4, 1, 128, [1024, 8, 17, 2048], [10, 20, 512, 16]),
            (3, 2, 128, [268, 1584, 1571], [2448, 4088, 1925]),
            (2, 1, 128, [1024, 256], [512, 768]),
        ]
        for (
            batch,
            heads,
            head_dim,
            seqlens_Q_list,
            seqlens_KV_list,
        ) in problem_sizes:
            for is_causal in [False, True]:
                self._test_varlen_backend(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    seqlens_Q_list=seqlens_Q_list,
                    seqlens_KV_list=seqlens_KV_list,
                    is_causal=is_causal,
                    backend="blackwell-fmha",
                )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_cutlass_blackwell_varlen_fmha_extended(self):
        self._test_varlen_randsweep(
            backend="blackwell-fmha", max_tests=RAND_SWEEP_TESTS
        )


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    unittest.main()
