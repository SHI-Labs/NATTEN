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
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(False)


def sdpa_split(
    q: Tensor,
    k_list: list,
    v_list: list,
    do: Tensor,
    backend: str,
    torch_compile: bool,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    q = q.requires_grad_(True)
    k_list = [k.requires_grad_(True) for k in k_list]
    v_list = [v.requires_grad_(True) for v in v_list]

    outputs, lses = [], []
    for k, v in zip(k_list, v_list):
        out, lse = attention(q, k, v, return_lse=True, backend=backend)
        outputs.append(out)
        lses.append(lse)

    out, _ = merge_attentions(outputs, lses, torch_compile=torch_compile)

    out.backward(do)

    with torch.no_grad():
        output = out.data
        assert q.grad is not None
        dq = q.grad.data

        dk_list = []
        dv_list = []
        for k, v in zip(k_list, v_list):
            assert k.grad is not None
            assert v.grad is not None
            dk_list.append(k.grad.data)
            dv_list.append(v.grad.data)

        dk = torch.cat(dk_list, dim=1)
        dv = torch.cat(dv_list, dim=1)

        return output, dq, dk, dv


def sdpa_ref(
    q: Tensor,
    k_list: list,
    v_list: list,
    do: Tensor,
    backend: str,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    with torch.no_grad():
        k = torch.cat(k_list, dim=1)
        v = torch.cat(v_list, dim=1)

    q = q.requires_grad_(True)
    k = k.requires_grad_(True)
    v = v.requires_grad_(True)

    out: Tensor = attention(q, k, v, backend=backend)  # type: ignore[assignment]
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
        seqlen_KV_list: list,
        backend: str,
        torch_compile: bool,
    ):

        ALLOWED_DTYPES = [
            # (dtype, atol_out, (atol_dq, atol_dk, atol_dv))
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

            if torch_compile:
                reset_torch_compile(1)

            logger.debug(
                f"Testing Attention Merging: {batch=}, {heads=}, {head_dim=}, "
                f"{seqlen_Q=}, seqlen_KV_list={seqlen_KV_list}, "
                f"{dtype=}, {backend=}, {torch_compile=}."
            )

            q = torch.randn(
                batch, seqlen_Q, heads, head_dim, device="cuda", dtype=dtype
            )

            k_list = []
            v_list = []
            for seqlen_KV in seqlen_KV_list:
                k = torch.randn(
                    batch, seqlen_KV, heads, head_dim, device="cuda", dtype=dtype
                )
                v = torch.randn_like(k)
                k_list.append(k)
                v_list.append(v)

            do = torch.randn_like(q)

            q_ref = q.clone()
            k_list_ref = [k.clone() for k in k_list]
            v_list_ref = [v.clone() for v in v_list]
            do_ref = do.clone()

            output_ref, dq_ref, dk_ref, dv_ref = sdpa_ref(
                q_ref,
                k_list_ref,
                v_list_ref,
                do_ref,
                backend=backend,
            )

            output, dq, dk, dv = sdpa_split(
                q, k_list, v_list, do, backend=backend, torch_compile=torch_compile
            )

            torch.testing.assert_close(
                output.float(), output_ref.float(), atol=atol_out, rtol=0
            )
            torch.testing.assert_close(dk.float(), dk_ref.float(), atol=atol_dk, rtol=0)
            torch.testing.assert_close(dv.float(), dv_ref.float(), atol=atol_dv, rtol=0)
            torch.testing.assert_close(dq.float(), dq_ref.float(), atol=atol_dq, rtol=0)

    def _test_randsweep(self, backend, num_tests=1000):
        random.seed(42)

        max_Q_ = 16384
        max_KV_total_ = 2**15
        for i in range(num_tests):
            batch = random.choice(range(1, 4))
            heads = random.choice(range(1, 4))

            # Adjust max seqlens accordingly so we get somewhat more consistent runtimes
            max_Q = max_Q_ // batch
            max_KV_total = max_KV_total_ // heads

            if backend == "blackwell-fmha":
                head_dim_choices = [32, 64, 128]
            elif backend == "hopper-fmha":
                head_dim_choices = [32, 64, 128]
            else:
                assert backend == "cutlass-fmha"
                head_dim_choices = range(8, 256 + 1, 8)

            head_dim = random.choice(head_dim_choices)

            seqlen_Q = random.choice(range(8, max_Q + 1))

            num_splits = random.choice(range(2, 9))

            # Generate split lengths that don't exceed max_KV_total when summed
            seqlen_KV_list = []
            remaining = max_KV_total
            for j in range(num_splits - 1):
                # Each split gets at least 8, up to remaining
                max_for_split = max(8, remaining - 8 * (num_splits - j - 1))
                seqlen_KV = random.choice(range(8, min(max_for_split, remaining) + 1))
                seqlen_KV_list.append(seqlen_KV)
                remaining -= seqlen_KV

            # Last split gets what's left (at least 8)
            seqlen_KV_list.append(max(8, remaining))

            # torch compile only affects the merge op, and it's slow, so we skew the likelihood
            torch_compile = random.choice([True, False, False, False])

            self._test(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                seqlen_Q=seqlen_Q,
                seqlen_KV_list=seqlen_KV_list,
                backend=backend,
                torch_compile=torch_compile,
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
    unittest.main()
