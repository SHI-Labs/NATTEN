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

import math
import random
import unittest
from itertools import product

import torch
from natten._environment import _NUM_RAND_SWEEP_TESTS as RAND_SWEEP_TESTS
from natten.backends.configs.cutlass_blackwell import (
    get_all_backward_configs,
    get_all_forward_configs,
)
from natten.functional import neighborhood_attention_generic
from natten.utils import log
from natten.utils.testing import (
    skip_if_blackwell_kernels_not_supported,
    skip_if_libnatten_is_not_supported,
    skip_if_not_running_extended_tests,
)
from natten.varlen import neighborhood_attention_varlen

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

    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.empty_cache()

    # Blackwell FNA bwd doesn't have deterministic option.
    torch.use_deterministic_algorithms(False)


class BlackwellFNAVarlenBackendTest(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    def _test_dtype(
        self,
        token_layout_list,
        heads,
        head_dim,
        kernel_size,
        stride,
        dilation,
        is_causal,
        #
        q_tile_shape,
        kv_tile_shape,
        backward_q_tile_shape,
        backward_kv_tile_shape,
        run_persistent_kernel,
        #
        dtype,
        atol_fwd,
        atol_bwd,
        atol_dq,
        test_backprop: bool,
        #
        backend: str,
        #
        heads_kv=None,
        head_dim_v=None,
    ):
        torch.set_default_device("cuda")
        na_dim = len(token_layout_list[0])
        assert na_dim in [1, 2, 3], "Only supports NA1D, 2D, 3D."
        heads_kv = heads_kv or heads
        head_dim_v = head_dim_v or head_dim

        logger.debug(
            f"Testing {backend} varlen:\n"
            f"{token_layout_list=},\n"
            f"{heads=}, {heads_kv=}, {head_dim=}, {head_dim_v=}, {dtype=}, {test_backprop=}\n"
            f"{kernel_size=}, {stride=}, {dilation=}, {is_causal=}\n"
            f"{q_tile_shape=}, {kv_tile_shape=}, {backward_q_tile_shape=}, {backward_kv_tile_shape=}\n"
            f"{run_persistent_kernel=}."
        )

        total_seqlen_pre_permute = sum([math.prod(x) for x in token_layout_list])
        with torch.no_grad():
            q_ref, k_ref, v_ref = (
                torch.randn(
                    (1, total_seqlen_pre_permute, heads, head_dim),
                    device="cuda",
                    dtype=torch.float32,
                ),
                torch.randn(
                    (1, total_seqlen_pre_permute, heads_kv, head_dim),
                    device="cuda",
                    dtype=torch.float32,
                ),
                torch.randn(
                    (1, total_seqlen_pre_permute, heads_kv, head_dim_v),
                    device="cuda",
                    dtype=torch.float32,
                ),
            )
            q_ref = q_ref.to(dtype)
            k_ref = k_ref.to(dtype)
            v_ref = v_ref.to(dtype)

            q, k, v = (
                q_ref.clone(),
                k_ref.clone(),
                v_ref.clone(),
            )
            if test_backprop:
                d_out = (
                    torch.randn(
                        (1, total_seqlen_pre_permute, heads, head_dim_v),
                        device="cuda",
                        dtype=torch.float32,
                    )
                    * 0.05
                ).to(dtype)
                d_out_ref = d_out.clone()
                d_out_ref = d_out_ref

        # Reference impl
        output_list = []
        lse_list = []
        dq_list = []
        dk_list = []
        dv_list = []
        for i, token_layout in enumerate(token_layout_list):
            seq_start = sum([math.prod(x) for x in token_layout_list[:i]])
            seq_end = seq_start + math.prod(token_layout)

            q_batch = q_ref[:, seq_start:seq_end, :, :].reshape(
                1, *token_layout, heads, head_dim
            )
            k_batch = k_ref[:, seq_start:seq_end, :, :].reshape(
                1, *token_layout, heads_kv, head_dim
            )
            v_batch = v_ref[:, seq_start:seq_end, :, :].reshape(
                1, *token_layout, heads_kv, head_dim_v
            )

            if test_backprop:
                q_batch = q_batch.requires_grad_(True)
                k_batch = k_batch.requires_grad_(True)
                v_batch = v_batch.requires_grad_(True)
                d_o_batch = d_out_ref[:, seq_start:seq_end, :, :].reshape(
                    1, *token_layout, heads, head_dim
                )

            o_batch, lse_batch = neighborhood_attention_generic(
                q_batch,
                k_batch,
                v_batch,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                backend=backend,
                q_tile_shape=q_tile_shape,
                kv_tile_shape=kv_tile_shape,
                backward_q_tile_shape=backward_q_tile_shape,
                backward_kv_tile_shape=backward_kv_tile_shape,
                run_persistent_kernel=run_persistent_kernel,
                return_lse=True,
            )
            with torch.no_grad():
                output_list.append(o_batch.clone().reshape(1, -1, heads, head_dim_v))
                lse_list.append(lse_batch.clone().reshape(1, -1, heads))

            if test_backprop:
                o_batch.backward(d_o_batch)  # type: ignore[union-attr]
                with torch.no_grad():
                    assert q_batch.grad is not None
                    assert k_batch.grad is not None
                    assert v_batch.grad is not None
                    dq_list.append(q_batch.grad.clone().reshape(1, -1, heads, head_dim))
                    dk_list.append(
                        k_batch.grad.clone().reshape(1, -1, heads_kv, head_dim)
                    )
                    dv_list.append(
                        v_batch.grad.clone().reshape(1, -1, heads_kv, head_dim_v)
                    )

        output_ref = torch.cat(output_list, dim=1)
        lse_ref = torch.cat(lse_list, dim=1)
        if test_backprop:
            dq_ref = torch.cat(dq_list, dim=1)
            dk_ref = torch.cat(dk_list, dim=1)
            dv_ref = torch.cat(dv_list, dim=1)

        # Target
        if test_backprop:
            q = q.requires_grad_(True)
            k = k.requires_grad_(True)
            v = v.requires_grad_(True)

        output, lse = neighborhood_attention_varlen(
            q,
            k,
            v,
            token_layout_list=token_layout_list,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            backend=backend,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            backward_q_tile_shape=backward_q_tile_shape,
            backward_kv_tile_shape=backward_kv_tile_shape,
            run_persistent_kernel=run_persistent_kernel,
            return_lse=True,
        )
        torch.testing.assert_close(
            output.float(), output_ref.float(), atol=atol_fwd, rtol=0
        )
        torch.testing.assert_close(lse, lse_ref, atol=atol_fwd, rtol=0)
        if test_backprop:
            output.backward(d_out)  # type: ignore[union-attr]
            with torch.no_grad():
                assert q.grad is not None
                assert k.grad is not None
                assert v.grad is not None
                dq = q.grad
                dk = k.grad
                dv = v.grad

                torch.testing.assert_close(
                    dq.float(), dq_ref.float(), atol=atol_dq, rtol=0
                )
                torch.testing.assert_close(
                    dk.float(), dk_ref.float(), atol=atol_bwd, rtol=0
                )
                torch.testing.assert_close(
                    dv.float(), dv_ref.float(), atol=atol_bwd, rtol=0
                )

    def _test_all_dtypes(
        self,
        token_layout_list,
        heads,
        head_dim,
        kernel_size,
        stride,
        dilation,
        is_causal,
        heads_kv=None,
        head_dim_v=None,
        n_configs_to_test=None,
    ):
        na_dim = len(token_layout_list[0])
        ALLOWED_DTYPES = [
            (torch.float16, (1e-6, 1e-6, 1e-2)),
            (torch.bfloat16, (1e-6, 1e-6, 1e-2)),
            (torch.float8_e4m3fn, (1e-6, None, None)),
            (torch.float8_e5m2, (1e-6, None, None)),
        ]

        for dtype, (atol_fwd, atol_bwd, atol_dq) in ALLOWED_DTYPES:

            forward_configs = get_all_forward_configs(
                na_dim=na_dim, head_dim=head_dim, dtype=dtype, device="cuda"
            )
            backward_configs = get_all_backward_configs(
                na_dim=na_dim, head_dim=head_dim, dtype=dtype, device="cuda"
            )
            assert len(forward_configs) > 0
            test_backprop = len(backward_configs) > 0

            random.shuffle(forward_configs)
            random.shuffle(backward_configs)

            n_configs = max(len(forward_configs), len(backward_configs))
            if n_configs_to_test is not None:
                n_configs = min(n_configs_to_test, n_configs)

            for i in range(n_configs):
                q_tile_shape, kv_tile_shape = forward_configs[i % len(forward_configs)]
                backward_q_tile_shape, backward_kv_tile_shape = None, None
                if test_backprop:
                    backward_q_tile_shape, backward_kv_tile_shape = backward_configs[
                        i % len(backward_configs)
                    ]

                for run_persistent_kernel in [True, False]:
                    self._test_dtype(
                        token_layout_list=token_layout_list,
                        heads=heads,
                        head_dim=head_dim,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        is_causal=is_causal,
                        #
                        q_tile_shape=q_tile_shape,
                        kv_tile_shape=kv_tile_shape,
                        backward_q_tile_shape=backward_q_tile_shape,
                        backward_kv_tile_shape=backward_kv_tile_shape,
                        run_persistent_kernel=run_persistent_kernel,
                        #
                        dtype=dtype,
                        atol_fwd=atol_fwd,
                        atol_bwd=atol_bwd,
                        atol_dq=atol_dq,
                        test_backprop=test_backprop,
                        #
                        backend="blackwell-fna",
                        #
                        heads_kv=heads_kv,
                        head_dim_v=head_dim_v,
                    )

    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_fixed_use_cases(self):
        # token_layout_list=[(37, 21), (35, 75), (72, 37), (78, 58), (78, 55), (50, 32), (21, 69), (67, 15)],
        # heads=3, heads_kv=3, head_dim=32, head_dim_v=32, dtype=torch.float16, test_backprop=True
        # kernel_size=(3, 15), stride=(1, 3), dilation=(6, 1), is_causal=(True, False)
        problem_sizes = [
            # ([(9, 2), (9, 2), (28, 2), (28, 2)], 3, 3, 32, (2, 2), (1, 1), (1, 1)),
            ([(18, 2), (56, 2)], 3, 3, 32, (2, 2), (1, 1), (2, 1)),
            ([(48, 2), (56, 2)], 3, 3, 32, (2, 2), (1, 1), (2, 1)),
            ([(18, 3), (50, 3)], 3, 3, 32, (3, 3), (1, 1), (2, 1)),
            ([(18, 3), (54, 3)], 3, 3, 32, (3, 3), (1, 1), (6, 1)),
            ([(37, 21), (72, 37)], 3, 3, 32, (3, 15), (1, 3), (6, 1)),
            (
                [
                    (37, 21),
                    (35, 75),
                    (72, 37),
                    (78, 58),
                    (78, 55),
                    (50, 32),
                    (21, 69),
                    (67, 15),
                ],
                3,
                3,
                32,
                (3, 15),
                (1, 3),
                (6, 1),
            ),
            ([(50, 17)], 3, 3, 32, (3, 17), (1, 1), (6, 1)),
            ([(50, 30)], 3, 3, 32, (8, 3), (1, 1), (6, 1)),
            ([(50, 30)], 3, 3, 32, (3, 3), (1, 1), (9, 1)),
            ([(50, 30)], 3, 3, 32, (3, 3), (1, 1), (8, 1)),
            ([(50, 30)], 3, 3, 32, (3, 3), (1, 1), (7, 1)),
            ([(50, 30)], 3, 3, 32, (3, 3), (1, 1), (5, 1)),
            ([(50, 30)], 3, 3, 32, (3, 3), (1, 1), (6, 1)),
            ([(50, 32)], 3, 3, 32, (3, 3), (1, 1), (6, 1)),
            ([(50, 32)], 3, 3, 32, (3, 15), (1, 3), (6, 1)),
            ([(14, 29)], 2, 1, 64, (3, 7), (2, 4), (2, 3)),
            ([(14, 29)], 2, 1, 64, (3, 7), (2, 4), (1, 1)),
            ([(14, 29), (8, 31), (17, 23), (23, 37)], 2, 1, 64, (3, 7), (2, 4), (2, 3)),
            ([(15,), (127,), (49,), (256,), (24,)], 2, 2, 128, (3,), (2,), (2,)),
            ([(29,), (127,), (49,), (256,), (37,)], 2, 1, 64, (9,), (2,), (3,)),
        ]
        for (
            token_layout_list,
            heads,
            heads_kv,
            head_dim,
            kernel_size,
            stride,
            dilation,
        ) in problem_sizes:
            na_dim = len(token_layout_list[0])
            for is_causal_ in product(*[[True, False] for _ in range(na_dim)]):
                is_causal = tuple(is_causal_)
                self._test_all_dtypes(
                    token_layout_list=token_layout_list,
                    heads=heads,
                    heads_kv=heads_kv,
                    head_dim=head_dim,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    is_causal=is_causal,
                    n_configs_to_test=5,
                )

    def _test_rand_sweep(self, na_dim, max_tests=1000, configs_to_test=None):
        random.seed(42)
        max_seqlen = 2**16
        for i in range(max_tests):
            batch = random.choice(range(2, 16))
            heads = random.choice(range(1, 4))
            heads_kv = random.choice([i for i in range(1, heads + 1) if heads % i == 0])
            head_dim = random.choice([32, 64, 128])

            token_layout_list = []
            min_input_shape = None
            for b in range(batch):
                input_shape = []
                for j in range(na_dim):
                    input_shape.append(random.choice(range(4, 97)))

                while math.prod(input_shape) > max_seqlen:
                    dim_to_cut = random.choice(range(na_dim))
                    input_shape[dim_to_cut] = max(4, int(input_shape[dim_to_cut] * 0.1))

                input_shape = tuple(input_shape)
                token_layout_list.append(input_shape)
                if min_input_shape is None:
                    min_input_shape = input_shape
                else:
                    min_input_shape = tuple(
                        min(x, y) for x, y in zip(input_shape, min_input_shape)
                    )

            kernel_size = tuple(random.choice(range(2, x + 1)) for x in min_input_shape)
            stride = tuple(random.choice(range(1, k + 1)) for k in kernel_size)
            dilation = tuple(
                random.choice(range(1, x // k + 1))
                for x, k in zip(min_input_shape, kernel_size)
            )
            is_causal = tuple(random.choice([False, True]) for _ in range(na_dim))

            self._test_all_dtypes(
                token_layout_list=token_layout_list,
                heads=heads,
                heads_kv=heads_kv,
                head_dim=head_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                n_configs_to_test=configs_to_test,
            )

    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_rand_sweep_quick(self):
        self._test_rand_sweep(1, max_tests=10, configs_to_test=3)
        self._test_rand_sweep(2, max_tests=10, configs_to_test=3)
        self._test_rand_sweep(3, max_tests=10, configs_to_test=3)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_rand_sweep_extended(self):
        self._test_rand_sweep(1, max_tests=RAND_SWEEP_TESTS, configs_to_test=3)
        self._test_rand_sweep(2, max_tests=RAND_SWEEP_TESTS, configs_to_test=3)
        self._test_rand_sweep(3, max_tests=RAND_SWEEP_TESTS, configs_to_test=3)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    unittest.main()
