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
from typing import Tuple

import natten  # noqa: F401
import torch
from natten._environment import _NUM_RAND_SWEEP_TESTS as RAND_SWEEP_TESTS
from natten.functional import attention, merge_attentions
from natten.utils import log
from natten.utils.testing import (
    skip_if_blackwell_kernels_not_supported,
    skip_if_hopper_kernels_not_supported,
    skip_if_libnatten_is_not_supported,
    skip_if_not_running_extended_tests,
)
from torch import Tensor

from .utils import reset_torch_compile


logger = log.get_logger(__name__)


def _reset_everything():
    torch.manual_seed(42)
    torch.cuda.empty_cache()

    # Hopper and Blackwell FMHA bwd don't have deterministic option.
    torch.use_deterministic_algorithms(False)


def sdpa_split(
    q: Tensor,
    k_0: Tensor,
    v_0: Tensor,
    k_1: Tensor,
    v_1: Tensor,
    do: Tensor,
    backend: str,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    q = q.requires_grad_(True)
    k_0 = k_0.requires_grad_(True)
    v_0 = v_0.requires_grad_(True)
    k_1 = k_1.requires_grad_(True)
    v_1 = v_1.requires_grad_(True)

    out1, lse1 = attention(q, k_0, v_0, return_lse=True, backend=backend)
    out2, lse2 = attention(q, k_1, v_1, return_lse=True, backend=backend)

    out, lse = merge_attentions([out1, out2], [lse1, lse2], torch_compile=False)

    # lse = lse.squeeze(-1)
    assert lse.shape == lse1.shape, f"{lse.shape=} != {lse1.shape=}"
    assert lse.shape == lse2.shape, f"{lse.shape=} != {lse2.shape=}"

    out.backward(do)

    with torch.no_grad():
        output = out.data
        assert q.grad is not None
        assert k_0.grad is not None
        assert k_1.grad is not None
        assert v_0.grad is not None
        assert v_1.grad is not None
        dq, dk1, dv1 = q.grad.data, k_0.grad.data, v_0.grad.data
        dk2, dv2 = k_1.grad.data, v_1.grad.data

        dk = torch.cat([dk1, dk2], dim=1)
        dv = torch.cat([dv1, dv2], dim=1)

        return output, dq, dk, dv


def sdpa_ref(
    q: Tensor,
    k_0: Tensor,
    v_0: Tensor,
    k_1: Tensor,
    v_1: Tensor,
    do: Tensor,
    backend: str,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    with torch.no_grad():
        k = torch.cat([k_0, k_1], dim=1)
        v = torch.cat([v_0, v_1], dim=1)

    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)

    out: Tensor = attention(q, k, v, backend=backend)  #  type: ignore[assignment]
    out.backward(do)

    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None

    return out.data, q.grad.data, k.grad.data, v.grad.data


class AttentionMergeTest(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    def _test(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        seqlen_Q: int,
        seqlen_KV_0: int,
        seqlen_KV_1: int,
        backend: str,
    ):

        ALLOWED_DTYPES = [
            # dtype, (atol_out, (atol_dq, atol_dk, atol_dv))
            (torch.float32, 1e-3, (1e-2, 1e-3, 1e-3)),
            (torch.float16, 1e-2, (1e-2, 1e-2, 1e-2)),
            (torch.bfloat16, 5e-2, (5e-2, 5e-2, 5e-2)),
        ]

        SUPPORTED_DTYPES = [torch.float16, torch.bfloat16]
        # TODO: query backends directly for supported dtypes
        if backend == "cutlass-fmha":
            SUPPORTED_DTYPES += [torch.float32]

        for dtype, atol_out, (atol_dq, atol_dk, atol_dv) in ALLOWED_DTYPES:
            if dtype not in SUPPORTED_DTYPES:
                continue

            reset_torch_compile(1)

            logger.debug(
                f"Testing Attention Merging: {batch=}, {heads=}, {head_dim=}, "
                f"{seqlen_Q=}, {seqlen_KV_0=}, {seqlen_KV_1=}, "
                f"{dtype=}, {backend=}."
            )

            q = torch.randn(
                batch, seqlen_Q, heads, head_dim, device="cuda", dtype=dtype
            )
            k_0 = torch.randn(
                batch, seqlen_KV_0, heads, head_dim, device="cuda", dtype=dtype
            )
            v_0 = torch.randn_like(k_0)
            k_1 = torch.randn(
                batch, seqlen_KV_1, heads, head_dim, device="cuda", dtype=dtype
            )
            v_1 = torch.randn_like(k_1)
            do = torch.randn_like(q)

            q_ref = q.clone()
            k_0_ref = k_0.clone()
            v_0_ref = v_0.clone()
            k_1_ref = k_1.clone()
            v_1_ref = v_1.clone()
            do_ref = do.clone()

            output_ref, dq_ref, dk_ref, dv_ref = sdpa_ref(
                q_ref,
                k_0_ref,
                v_0_ref,
                k_1_ref,
                v_1_ref,
                do_ref,
                backend=backend,
            )

            output, dq, dk, dv = sdpa_split(q, k_0, v_0, k_1, v_1, do, backend=backend)

            torch.testing.assert_close(
                output.float(), output_ref.float(), atol=atol_out, rtol=0
            )
            torch.testing.assert_close(dk.float(), dk_ref.float(), atol=atol_dk, rtol=0)
            torch.testing.assert_close(dv.float(), dv_ref.float(), atol=atol_dv, rtol=0)
            torch.testing.assert_close(dq.float(), dq_ref.float(), atol=atol_dq, rtol=0)

    def _test_randsweep(self, backend, num_tests=1000):
        random.seed(42)

        max_Q = 16384
        max_KV = 16384
        for i in range(num_tests):
            batch = random.choice(range(1, 12))
            heads = random.choice(range(1, 8))

            if backend == "blackwell-fmha":
                head_dim_choices = [32, 64, 128]
            elif backend == "hopper-fmha":
                head_dim_choices = [32, 64, 128]
            else:
                assert backend == "cutlass-fmha"
                head_dim_choices = range(8, 256 + 1, 8)

            head_dim = random.choice(head_dim_choices)

            seqlen_Q = random.choice(range(8, max_Q + 1))
            seqlen_KV_0 = random.choice(range(8, max_KV + 1))
            seqlen_KV_1 = random.choice(range(8, max_KV + 1))

            self._test(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                seqlen_Q=seqlen_Q,
                seqlen_KV_0=seqlen_KV_0,
                seqlen_KV_1=seqlen_KV_1,
                backend=backend,
            )

    @skip_if_libnatten_is_not_supported()
    def test_attention_merge_cutlass_fmha_fast(self):
        self._test_randsweep(backend="cutlass-fmha", num_tests=10)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_attention_merge_cutlass_fmha_extended(self):
        self._test_randsweep(backend="cutlass-fmha", num_tests=RAND_SWEEP_TESTS)

    @skip_if_hopper_kernels_not_supported()
    @skip_if_libnatten_is_not_supported()
    def test_attention_merge_hopper_fmha_fast(self):
        self._test_randsweep(backend="hopper-fmha", num_tests=10)

    @skip_if_hopper_kernels_not_supported()
    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_attention_merge_hopper_fmha_extended(self):
        self._test_randsweep(backend="hopper-fmha", num_tests=RAND_SWEEP_TESTS)

    @skip_if_blackwell_kernels_not_supported()
    @skip_if_libnatten_is_not_supported()
    def test_attention_merge_blackwell_fmha_fast(self):
        self._test_randsweep(backend="blackwell-fmha", num_tests=10)

    @skip_if_blackwell_kernels_not_supported()
    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    def test_attention_merge_blackwell_fmha_extended(self):
        self._test_randsweep(backend="blackwell-fmha", num_tests=RAND_SWEEP_TESTS)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    unittest.main()
