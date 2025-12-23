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
from natten._environment import (
    _NUM_RAND_SWEEP_TESTS as RAND_SWEEP_TESTS,
    _RUN_ADDITIONAL_KV_TESTS as ENABLE_ADDITIONAL_KV_TESTS,
)
from natten.backends.configs.cutlass_blackwell import (
    get_all_backward_configs,
    get_all_forward_configs,
)
from natten.utils.testing import (
    skip_if_blackwell_kernels_not_supported,
    skip_if_libnatten_is_not_supported,
    skip_if_not_running_extended_tests,
)

from .utils import NattenBackendTester, reset_torch_compile

ADDITIONAL_KV_LENGTHS = [0, 64] if ENABLE_ADDITIONAL_KV_TESTS else [0]


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

    # Blackwell FNA bwd doesn't have deterministic option.
    torch.use_deterministic_algorithms(False)


class BlackwellFNABackendTest(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    def _test_all_dtypes_against_cutlass_2x_fna(
        self,
        batch,
        heads,
        head_dim,
        input_shape,
        kernel_size,
        stride,
        dilation,
        is_causal=None,
        additional_kv_length=0,
        configs_to_test=None,
        heads_kv=None,
    ):
        torch.set_default_device("cuda")
        assert isinstance(input_shape, tuple)
        na_dim = len(input_shape)
        assert na_dim in [1, 2, 3], "Only supports NA1D, 2D, 3D."

        if additional_kv_length > 0:
            # cutlass-fna doesn't fuse additional KV, uses merge_attentions
            reset_torch_compile(1)

        tester = NattenBackendTester(
            batch=batch,
            heads=heads,
            heads_kv=heads_kv,
            head_dim=head_dim,
            input_shape=input_shape,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            additional_kv_length=additional_kv_length,
            test_backprop=True,
            reference_backend="cutlass-fna",
            reference_fmha_backend="cutlass-fmha",
            dtype=torch.float16,
        )

        ALLOWED_DTYPES = [
            (torch.float16, (1e-2, 2e-2)),
            (torch.bfloat16, (1e-1, 1e-1)),
            (torch.float8_e4m3fn, (45e-2, None)),
            (torch.float8_e5m2, (9e-1, None)),
        ]

        test_id = 0
        for dtype, atol in ALLOWED_DTYPES:

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

            for i in range(max(len(forward_configs), len(backward_configs))):
                q_tile_shape, kv_tile_shape = forward_configs[i % len(forward_configs)]
                backward_q_tile_shape, backward_kv_tile_shape = None, None
                if test_backprop:
                    backward_q_tile_shape, backward_kv_tile_shape = backward_configs[
                        i % len(backward_configs)
                    ]

                if additional_kv_length > 0:
                    reset_torch_compile(2)
                for persistent in [True, False]:
                    tester.test(
                        eps=atol,
                        dtype=dtype,
                        target_backend="blackwell-fna",
                        target_fmha_backend="blackwell-fmha",
                        q_tile_shape=q_tile_shape,
                        kv_tile_shape=kv_tile_shape,
                        backward_q_tile_shape=backward_q_tile_shape,
                        backward_kv_tile_shape=backward_kv_tile_shape,
                        run_persistent_kernel=persistent,
                        test_backprop=test_backprop,
                    )
                    test_id += 1
                    if configs_to_test is not None and test_id > configs_to_test:
                        return

    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_1d_against_cutlass_2x(self):
        problem_sizes = [
            (1, 4, 2, 64, (128,), (127,), (1,), (1)),
            (1, 4, 2, 64, (128,), (15,), (1,), (1)),
            (1, 4, 1, 64, (128,), (15,), (1,), (1)),
            (1, 2, 1, 64, (128,), (15,), (1,), (1)),
            (1, 2, 2, 64, (128,), (15,), (1,), (1)),
            (2, 4, 4, 128, (128,), (63,), (31,), (1)),
            (2, 4, 2, 128, (128,), (63,), (31,), (1)),
            (2, 4, 1, 128, (128,), (63,), (31,), (1)),
            (4, 3, 3, 128, (256,), (255,), (82,), (1)),
            (4, 3, 1, 128, (256,), (255,), (82,), (1)),
            (1, 1, 1, 128, (32768,), (2048,), (2048,), (1)),
            (1, 1, 1, 128, (32768,), (2048,), (256,), (1)),
            (1, 1, 1, 128, (32768,), (2048,), (128,), (1)),
            (1, 1, 1, 128, (32768,), (2048,), (1,), (1)),
            (1, 1, 1, 32, (128,), (3,), (2,), (5)),
            (1, 1, 1, 32, (128,), (3,), (1,), (1)),
            (1, 1, 1, 32, (128,), (3,), (2,), (10)),
            (1, 1, 1, 64, (128,), (8,), (7,), (5)),
            (1, 1, 1, 128, (128,), (61,), (33,), (1)),
            (1, 1, 1, 32, (125,), (3,), (1,), (1)),
            (1, 2, 2, 64, (125,), (15,), (1,), (1)),
            (1, 1, 1, 128, (256,), (3,), (2,), (10)),
        ]
        for (
            batch,
            heads,
            heads_kv,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in problem_sizes:
            for additional_kv_length in ADDITIONAL_KV_LENGTHS:
                for causal in [True, False]:
                    is_causal = (causal,)
                    self._test_all_dtypes_against_cutlass_2x_fna(
                        batch=batch,
                        heads=heads,
                        heads_kv=heads_kv,
                        head_dim=head_dim,
                        input_shape=input_shape,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        is_causal=is_causal,
                        additional_kv_length=additional_kv_length,
                    )

    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_2d_against_cutlass_2x(self):
        problem_sizes = [
            (1, 12, 12, 32, (17, 16), (3, 8), (1, 2), (1, 1)),
            (1, 12, 6, 32, (17, 16), (3, 8), (1, 2), (1, 1)),
            (1, 12, 4, 32, (17, 16), (3, 8), (1, 2), (1, 1)),
            (1, 2, 2, 32, (5, 25), (5, 16), (1, 16), (1, 1)),
            (1, 2, 1, 32, (5, 25), (5, 16), (1, 16), (1, 1)),
            (1, 1, 1, 32, (84, 19), (7, 3), (1, 1), (1, 1)),
            (1, 1, 1, 32, (84, 19), (7, 3), (1, 1), (5, 1)),
            (1, 1, 1, 128, (19, 29), (8, 8), (1, 1), (2, 3)),
            (1, 1, 1, 128, (48, 17), (24, 16), (1, 1), (2, 1)),
            (1, 1, 1, 128, (67, 80), (12, 7), (8, 4), (5, 11)),
            (1, 1, 1, 128, (8, 8), (8, 8), (1, 1), (1, 1)),
            (1, 1, 1, 128, (33, 33), (24, 16), (1, 1), (1, 1)),
            (1, 1, 1, 32, (16, 16), (16, 16), (1, 1), (1, 1)),
            (1, 1, 1, 128, (44, 80), (44, 80), (1, 1), (1, 1)),
            (1, 1, 1, 32, (40, 20), (3, 7), (1, 1), (1, 1)),
            (1, 1, 1, 32, (16, 16), (3, 3), (1, 1), (1, 1)),
            (1, 1, 1, 128, (44, 80), (9, 10), (1, 1), (1, 1)),
            (1, 1, 1, 64, (28, 40), (17, 31), (1, 1), (1, 1)),
            (1, 1, 1, 64, (36, 40), (36, 40), (12, 13), (1, 1)),
            (1, 1, 1, 128, (44, 80), (44, 80), (4, 8), (1, 1)),
        ]
        for (
            batch,
            heads,
            heads_kv,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in problem_sizes:
            for additional_kv_length in ADDITIONAL_KV_LENGTHS:
                for causal_x, causal_y in product([True, False], [True, False]):
                    is_causal = (causal_x, causal_y)
                    self._test_all_dtypes_against_cutlass_2x_fna(
                        batch=batch,
                        heads=heads,
                        heads_kv=heads_kv,
                        head_dim=head_dim,
                        input_shape=input_shape,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        is_causal=is_causal,
                        additional_kv_length=additional_kv_length,
                    )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_2d_against_cutlass_2x_extended(self):
        problem_sizes = [
            (1, 6, 2, 32, (5, 25), (5, 16), (1, 16), (1, 1)),
            (1, 2, 2, 32, (5, 25), (5, 16), (1, 16), (1, 1)),
            (1, 2, 1, 32, (5, 25), (5, 16), (4, 16), (1, 1)),
            (1, 1, 1, 32, (84, 69), (7, 68), (1, 1), (1, 1)),
            (1, 1, 1, 32, (84, 69), (7, 23), (1, 1), (1, 1)),
            (1, 1, 1, 32, (84, 69), (7, 20), (1, 6), (1, 1)),
            (1, 1, 1, 32, (84, 69), (7, 68), (1, 1), (5, 1)),
            (1, 1, 1, 32, (84, 69), (7, 23), (1, 1), (5, 1)),
            (1, 1, 1, 32, (84, 69), (7, 20), (1, 6), (5, 1)),
            (1, 1, 1, 128, (128, 128), (8, 8), (1, 1), (1, 1)),
            (1, 1, 1, 128, (128, 128), (8, 8), (1, 1), (4, 4)),
            (1, 1, 1, 128, (64, 64), (32, 32), (1, 1), (2, 2)),
            (1, 1, 1, 128, (64, 64), (32, 32), (1, 1), (1, 2)),
            (1, 1, 1, 128, (48, 48), (24, 24), (1, 1), (1, 2)),
            (1, 1, 1, 128, (17, 48), (16, 24), (1, 1), (1, 2)),
            (1, 1, 1, 128, (48, 48), (24, 24), (1, 1), (2, 2)),
            (1, 1, 1, 128, (48, 17), (24, 16), (1, 1), (1, 1)),
            (1, 1, 1, 128, (72, 80), (24, 16), (1, 1), (3, 5)),
            (1, 1, 1, 128, (48, 17), (24, 16), (1, 1), (1, 1)),
            (1, 1, 1, 128, (44, 80), (32, 32), (22, 16), (1, 1)),
            (1, 1, 1, 128, (44, 80), (24, 24), (1, 1), (1, 1)),
            (1, 1, 1, 128, (44, 80), (24, 24), (8, 16), (1, 1)),
            (1, 1, 1, 128, (44, 80), (24, 16), (1, 1), (1, 1)),
            (1, 1, 1, 128, (44, 80), (24, 16), (8, 16), (1, 1)),
            (1, 1, 1, 64, (28, 40), (28, 40), (1, 1), (1, 1)),
            (1, 1, 1, 32, (16, 16), (16, 16), (4, 5), (1, 1)),
            (1, 1, 1, 32, (16, 16), (16, 16), (4, 8), (1, 1)),
            (1, 1, 1, 32, (16, 16), (15, 15), (1, 1), (1, 1)),
            (1, 1, 1, 32, (16, 16), (14, 14), (1, 1), (1, 1)),
            (1, 1, 1, 64, (36, 40), (36, 40), (1, 1), (1, 1)),
            (1, 1, 1, 128, (48, 80), (24, 24), (8, 8), (1, 1)),
        ]
        for (
            batch,
            heads,
            heads_kv,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in problem_sizes:
            for additional_kv_length in ADDITIONAL_KV_LENGTHS:
                for causal_x, causal_y in product([True, False], [True, False]):
                    is_causal = (causal_x, causal_y)
                    self._test_all_dtypes_against_cutlass_2x_fna(
                        batch=batch,
                        heads=heads,
                        heads_kv=heads_kv,
                        head_dim=head_dim,
                        input_shape=input_shape,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        is_causal=is_causal,
                        additional_kv_length=additional_kv_length,
                    )

    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_3d_against_cutlass_2x(self):
        problem_sizes = [
            (4, 8, 4, 64, (32, 10, 10), (7, 3, 3), (5, 1, 1), (1, 2, 3)),
            (4, 8, 2, 64, (32, 10, 10), (7, 3, 3), (5, 1, 1), (1, 2, 3)),
            (4, 8, 1, 64, (32, 10, 10), (7, 3, 3), (5, 1, 1), (1, 2, 3)),
            (1, 1, 1, 64, (18, 37, 12), (14, 16, 12), (12, 8, 1), (1, 2, 1)),
            (1, 1, 1, 32, (13, 11, 9), (3, 4, 3), (2, 3, 3), (1, 1, 1)),
            (1, 1, 1, 32, (13, 11, 9), (3, 4, 3), (2, 3, 3), (3, 2, 2)),
            (1, 4, 4, 32, (8, 8, 16), (3, 3, 3), (2, 1, 2), (2, 2, 4)),
            (1, 1, 1, 64, (18, 37, 12), (14, 16, 12), (12, 8, 1), (1, 2, 1)),
            (1, 1, 1, 128, (57, 20, 88), (20, 4, 6), (1, 1, 1), (2, 1, 1)),
            (1, 1, 1, 128, (57, 32, 32), (10, 32, 32), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (32, 64, 64), (16, 16, 16), (1, 2, 2), (1, 1, 1)),
            (1, 1, 1, 128, (30, 48, 80), (18, 24, 24), (16, 8, 8), (1, 1, 1)),
            (1, 1, 1, 128, (16, 44, 80), (12, 32, 32), (8, 22, 16), (1, 1, 1)),
            (1, 1, 1, 128, (31, 32, 32), (10, 32, 32), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (30, 32, 32), (10, 32, 32), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 32, (16, 16, 16), (16, 16, 16), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 32, (8, 4, 8), (7, 3, 7), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 32, (16, 16, 16), (15, 15, 15), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 64, (24, 28, 40), (24, 28, 40), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 32, (16, 16, 16), (16, 16, 16), (2, 4, 5), (1, 1, 1)),
        ]
        for (
            batch,
            heads,
            heads_kv,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in problem_sizes:
            for additional_kv_length in ADDITIONAL_KV_LENGTHS:
                for causal_x, causal_y, causal_z in product(
                    [True, False], [True, False], [True, False]
                ):
                    is_causal = (causal_x, causal_y, causal_z)
                    self._test_all_dtypes_against_cutlass_2x_fna(
                        batch=batch,
                        heads=heads,
                        heads_kv=heads_kv,
                        head_dim=head_dim,
                        input_shape=input_shape,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        is_causal=is_causal,
                        additional_kv_length=additional_kv_length,
                    )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_3d_against_cutlass_2x_extended(self):
        problem_sizes = [
            (1, 12, 12, 64, (32, 8, 8), (7, 5, 5), (2, 1, 3), (2, 1, 1)),
            (1, 12, 6, 64, (32, 8, 8), (7, 5, 5), (2, 1, 3), (2, 1, 1)),
            (1, 12, 4, 64, (32, 8, 8), (7, 5, 5), (2, 1, 3), (2, 1, 1)),
            (1, 12, 1, 64, (32, 8, 8), (7, 5, 5), (2, 1, 3), (2, 1, 1)),
            (2, 2, 2, 32, (8, 8, 10), (3, 4, 3), (3, 4, 1), (1, 1, 1)),
            (2, 2, 1, 32, (8, 8, 10), (3, 4, 3), (3, 4, 1), (1, 1, 1)),
            (1, 1, 1, 64, (18, 37, 12), (14, 16, 12), (12, 8, 1), (1, 1, 1)),
            (1, 1, 1, 32, (13, 11, 9), (3, 4, 3), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 32, (13, 11, 9), (3, 4, 3), (1, 1, 1), (3, 2, 2)),
            (1, 1, 1, 32, (8, 8, 4), (3, 4, 3), (1, 1, 1), (1, 1, 1)),
            (1, 2, 2, 32, (8, 8, 12), (5, 8, 11), (2, 3, 4), (1, 1, 1)),
            (1, 1, 1, 64, (18, 37, 12), (14, 16, 12), (12, 8, 1), (1, 1, 1)),
            (2, 3, 3, 32, (5, 45, 73), (2, 7, 32), (1, 4, 32), (2, 2, 2)),
            (1, 1, 1, 128, (57, 20, 88), (19, 4, 6), (13, 1, 6), (2, 2, 7)),
            (2, 3, 3, 128, (57, 20, 88), (19, 4, 6), (13, 1, 6), (2, 2, 7)),
            (2, 3, 1, 128, (57, 20, 88), (19, 4, 6), (13, 1, 6), (2, 2, 7)),
            (1, 1, 1, 128, (32, 64, 64), (16, 16, 16), (2, 1, 1), (2, 2, 3)),
            (1, 1, 1, 128, (61, 61, 61), (10, 10, 10), (1, 1, 1), (1, 1, 2)),
            (1, 1, 1, 128, (61, 61, 61), (10, 10, 10), (1, 1, 1), (1, 2, 1)),
            (1, 1, 1, 128, (61, 61, 61), (10, 10, 10), (1, 1, 1), (2, 2, 2)),
            (1, 1, 1, 128, (32, 64, 64), (16, 16, 16), (2, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (32, 64, 64), (16, 16, 16), (1, 1, 2), (1, 1, 1)),
            (1, 1, 1, 32, (16, 16, 16), (16, 16, 16), (8, 4, 8), (1, 1, 1)),
            (1, 1, 1, 64, (24, 36, 40), (24, 36, 40), (10, 12, 13), (1, 1, 1)),
            (1, 1, 1, 128, (16, 44, 80), (16, 44, 80), (8, 4, 8), (1, 1, 1)),
            (1, 1, 1, 128, (30, 44, 80), (18, 24, 24), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (30, 44, 80), (18, 24, 24), (2, 8, 16), (1, 1, 1)),
            (1, 1, 1, 128, (30, 44, 80), (24, 24, 16), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (30, 44, 80), (24, 24, 16), (2, 8, 16), (1, 1, 1)),
            (1, 1, 1, 32, (16, 16, 16), (14, 14, 14), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 32, (16, 16, 16), (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (16, 44, 80), (8, 9, 10), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 64, (24, 28, 40), (11, 17, 31), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 64, (24, 36, 40), (24, 36, 40), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (16, 44, 80), (16, 44, 80), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (30, 48, 18), (30, 48, 18), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (8, 8, 8), (8, 8, 8), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (30, 48, 17), (18, 24, 16), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (33, 33, 33), (18, 24, 16), (1, 1, 1), (1, 1, 1)),
            (1, 1, 1, 128, (30, 48, 18), (3, 2, 8), (1, 1, 1), (8, 18, 2)),
            (1, 1, 1, 128, (61, 32, 32), (10, 32, 32), (1, 1, 1), (2, 1, 1)),
            (1, 1, 1, 128, (57, 32, 32), (10, 32, 32), (1, 1, 1), (2, 1, 1)),
            (1, 1, 1, 128, (57, 20, 88), (20, 8, 16), (1, 1, 1), (2, 1, 1)),
            (3, 1, 1, 64, (18, 37, 12), (14, 16, 12), (12, 8, 6), (1, 2, 1)),
            (1, 1, 1, 128, (32, 64, 64), (16, 16, 16), (1, 1, 2), (1, 3, 2)),
            (1, 1, 1, 128, (48, 64, 64), (7, 15, 11), (1, 2, 2), (5, 3, 2)),
        ]
        for (
            batch,
            heads,
            heads_kv,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in problem_sizes:
            for additional_kv_length in ADDITIONAL_KV_LENGTHS:
                for causal_x, causal_y, causal_z in product(
                    [True, False], [True, False], [True, False]
                ):
                    is_causal = (causal_x, causal_y, causal_z)
                    self._test_all_dtypes_against_cutlass_2x_fna(
                        batch=batch,
                        heads=heads,
                        heads_kv=heads_kv,
                        head_dim=head_dim,
                        input_shape=input_shape,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        is_causal=is_causal,
                        additional_kv_length=additional_kv_length,
                    )

    def _test_rand_sweep_against_cutlass_2x(
        self, na_dim, max_tests=1000, configs_to_test=None
    ):
        random.seed(42)
        max_seqlen = 2**17
        for i in range(max_tests):
            batch = random.choice(range(1, 4))
            heads = random.choice(range(1, 4))
            heads_kv = random.choice([i for i in range(1, heads + 1) if heads % i == 0])
            head_dim = random.choice([32, 64, 128])

            input_shape = []
            for j in range(na_dim):
                input_shape.append(random.choice(range(4, 97)))

            while math.prod(input_shape) > max_seqlen:
                dim_to_cut = random.choice(range(na_dim))
                input_shape[dim_to_cut] = max(4, int(input_shape[dim_to_cut] * 0.1))

            input_shape = tuple(input_shape)
            kernel_size = tuple(random.choice(range(2, x + 1)) for x in input_shape)
            stride = tuple(random.choice(range(1, k + 1)) for k in kernel_size)
            dilation = tuple(
                random.choice(range(1, x // k + 1))
                for x, k in zip(input_shape, kernel_size)
            )
            is_causal = tuple(random.choice([False, True]) for _ in range(na_dim))

            additional_kv_length = (
                random.choice(range(8, 513, 8)) if ENABLE_ADDITIONAL_KV_TESTS else 0
            )

            self._test_all_dtypes_against_cutlass_2x_fna(
                batch=batch,
                heads=heads,
                heads_kv=heads_kv,
                head_dim=head_dim,
                input_shape=input_shape,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                additional_kv_length=additional_kv_length,
                configs_to_test=configs_to_test,
            )

    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_rand_sweep_1d_against_cutlass_2x_quick(self):
        self._test_rand_sweep_against_cutlass_2x(1, max_tests=10, configs_to_test=3)

    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_rand_sweep_2d_against_cutlass_2x_quick(self):
        self._test_rand_sweep_against_cutlass_2x(2, max_tests=10, configs_to_test=3)

    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_rand_sweep_3d_against_cutlass_2x_quick(self):
        self._test_rand_sweep_against_cutlass_2x(3, max_tests=10, configs_to_test=3)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_rand_sweep_1d_against_cutlass_2x(self):
        self._test_rand_sweep_against_cutlass_2x(1, max_tests=RAND_SWEEP_TESTS)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_rand_sweep_2d_against_cutlass_2x(self):
        self._test_rand_sweep_against_cutlass_2x(2, max_tests=RAND_SWEEP_TESTS)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_rand_sweep_3d_against_cutlass_2x(self):
        self._test_rand_sweep_against_cutlass_2x(3, max_tests=RAND_SWEEP_TESTS)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    unittest.main()
