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
import os
import random
import time
import unittest
from itertools import product

import torch

from natten import allow_flex_compile
from natten.backends.configs.flex import FLEX_FORWARD_TILE_SHAPES
from natten.utils import log
from natten.utils.testing import (
    skip_if_flex_compile_is_not_supported,
    skip_if_flex_is_not_supported,
    skip_if_libnatten_is_not_supported,
    skip_if_not_running_extended_tests,
    supports_bfloat16,
    supports_float16,
)

from .utils import NattenBackendTester, reset_torch_compile


logger = log.get_logger(__name__)


# TODO: enable when Flex is stable / check with new PT releases
ENABLE_FLEX_COMPILE_TESTS = False
ENABLE_FLEX_COMPILE_BACKPROP_TESTS = False


def _reset_everything():
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
    torch.cuda.empty_cache()

    reset_torch_compile(1024)

    allow_flex_compile(
        ENABLE_FLEX_COMPILE_TESTS, backprop=ENABLE_FLEX_COMPILE_BACKPROP_TESTS
    )


class FlexBackendTest(unittest.TestCase):
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
        torch_compile=False,
        constrain_torch_compile_cache=True,
        max_runs=None,
    ):
        torch.set_default_device("cuda")
        assert isinstance(input_shape, tuple)
        na_dim = len(input_shape)
        assert na_dim in [1, 2, 3], "Only supports NA1D, 2D, 3D."

        test_backprop = ENABLE_FLEX_COMPILE_BACKPROP_TESTS if torch_compile else True

        if additional_kv_length > 0:
            # cutlass-fna doesn't fuse additional KV, uses merge_attentions
            reset_torch_compile(1)

        tester = NattenBackendTester(
            batch=batch,
            heads=heads,
            head_dim=head_dim,
            input_shape=input_shape,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            additional_kv_length=additional_kv_length,
            test_backprop=test_backprop,
            reference_backend="cutlass-fna",
            reference_fmha_backend="cutlass-fmha",
            dtype=torch.float32,
        )

        # TODO: write note on why backprop eps is different when additional_kv_length > 0
        run_idx = 0
        no_token_permute_config = (None, None)
        configs = FLEX_FORWARD_TILE_SHAPES[na_dim] + [no_token_permute_config]
        for q_tile_shape, kv_tile_shape in configs:
            if constrain_torch_compile_cache:
                if torch_compile or additional_kv_length > 0:
                    reset_torch_compile(4 if additional_kv_length > 0 else 2)
                else:
                    reset_torch_compile(0)

            assert supports_float16(
                torch.get_default_device()
            ), "Flex only supports SM70 and above, and it should have FP16!"

            tester.test(
                eps=(1e-2, 1e-2 if additional_kv_length == 0 else 3e-1),
                dtype=torch.float16,
                target_backend="flex-fna",
                target_fmha_backend="flex-fmha",
                q_tile_shape=q_tile_shape,
                kv_tile_shape=kv_tile_shape,
                torch_compile=torch_compile,
            )
            run_idx += 1
            if max_runs is not None and run_idx > max_runs:
                return

            if supports_bfloat16(torch.get_default_device()):
                tester.test(
                    eps=(1e-1, 1e-1 if additional_kv_length == 0 else 5e-1),
                    dtype=torch.bfloat16,
                    target_backend="flex-fna",
                    target_fmha_backend="flex-fmha",
                    q_tile_shape=q_tile_shape,
                    kv_tile_shape=kv_tile_shape,
                    torch_compile=torch_compile,
                )
                run_idx += 1
                if max_runs is not None and run_idx > max_runs:
                    return

    @skip_if_libnatten_is_not_supported()
    @skip_if_flex_compile_is_not_supported()
    def test_0_compile_caching(self):
        if not ENABLE_FLEX_COMPILE_TESTS:
            self.skipTest("Flex compile tests have been disabled.")

        # Verify torch compile caching works by rerunning use cases a second time,
        # and checking runtimes.
        # Might be a little flaky...
        # Torch compile autotuner might also have a separate cache of its own...

        def run_tests(problem_sizes, max_runs_per_use_case):
            for (
                batch,
                heads,
                head_dim,
                input_shape,
                kernel_size,
                stride,
                dilation,
            ) in problem_sizes:
                self._test_all_dtypes_against_cutlass_2x_fna(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    input_shape=input_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    is_causal=False,
                    additional_kv_length=0,
                    torch_compile=True,
                    constrain_torch_compile_cache=False,
                    max_runs=max_runs_per_use_case,
                )

        max_runs_per_use_case = 10
        problem_sizes = [
            (1, 1, 32, (128,), (3,), (2,), (5)),
            (1, 1, 32, (128,), (3,), (1,), (1)),
            (1, 1, 32, (4, 4, 4), (2, 2, 2), (1, 1, 1), (1, 1, 1)),
            (1, 1, 128, (16, 16), (4, 4), (2, 3), (1, 1)),
            (4, 3, 32, (32, 32), (31, 31), (15, 15), (1, 1)),
            (1, 1, 32, (32,), (32,), (1,), (1)),
        ]

        reset_torch_compile(len(problem_sizes) * max_runs_per_use_case)
        # NOTE: this might need adjustment based on device, and certainly number of use cases
        # On H100 with ctk 12.8, torch 2.7+cu128, this was 0.4s if we use LRU cache.
        # It's around 3.5 seconds if don't use LRU cache -- which we don't by default.
        # First run was around 55 seconds.
        MAX_RUNTIME_S = 5
        # I.e. second run should take at most 10% of the first run
        EXPECTED_PCT_OF_RUNTIME = 0.1

        logger.debug("Testing torch compile cache.")
        start_time = time.time()
        run_tests(problem_sizes, max_runs_per_use_case)
        elapsed_first_run = time.time() - start_time
        logger.debug(
            f"First run of {len(problem_sizes)} use cases finished in {elapsed_first_run:.1f} seconds."
        )

        logger.debug("Second run.")
        start_time = time.time()
        run_tests(problem_sizes, max_runs_per_use_case)
        elapsed_second_run = time.time() - start_time
        logger.debug(
            f"Second run of {len(problem_sizes)} use cases finished in {elapsed_second_run:.1f} seconds."
        )

        assert elapsed_second_run <= min(
            elapsed_first_run * EXPECTED_PCT_OF_RUNTIME, MAX_RUNTIME_S
        )

    @skip_if_libnatten_is_not_supported()
    @skip_if_flex_is_not_supported()
    def test_1d_against_cutlass_2x(self):
        problem_sizes = [
            (1, 1, 32, (128,), (3,), (2,), (5)),
            (1, 1, 32, (128,), (8,), (7,), (5)),
            (1, 1, 32, (125,), (3,), (1,), (1)),
            (1, 2, 32, (125,), (15,), (1,), (1)),
            (1, 1, 32, (256,), (3,), (2,), (10)),
        ]
        for (
            batch,
            heads,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in problem_sizes:
            for additional_kv_length in [0, 64]:
                for causal in [True, False]:
                    is_causal = (causal,)
                    self._test_all_dtypes_against_cutlass_2x_fna(
                        batch=batch,
                        heads=heads,
                        head_dim=head_dim,
                        input_shape=input_shape,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        is_causal=is_causal,
                        additional_kv_length=additional_kv_length,
                        torch_compile=False,
                    )

    @skip_if_libnatten_is_not_supported()
    @skip_if_flex_compile_is_not_supported()
    def test_1d_against_cutlass_2x_compiled(self):
        if not ENABLE_FLEX_COMPILE_TESTS:
            self.skipTest("Flex compile tests have been disabled.")

        problem_sizes = [
            (1, 2, 32, (128,), (15,), (1,), (1)),
            (1, 1, 32, (128,), (8,), (7,), (5)),
            (2, 4, 128, (128,), (63,), (31,), (1)),
            (4, 3, 128, (256,), (255,), (82,), (1)),
            (1, 1, 128, (128,), (61,), (33,), (1)),
            (1, 1, 32, (256,), (3,), (2,), (10)),
        ]
        for (
            batch,
            heads,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in problem_sizes:
            for additional_kv_length in [0, 64]:
                for causal in [True, False]:
                    is_causal = (causal,)
                    self._test_all_dtypes_against_cutlass_2x_fna(
                        batch=batch,
                        heads=heads,
                        head_dim=head_dim,
                        input_shape=input_shape,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        is_causal=is_causal,
                        additional_kv_length=additional_kv_length,
                        torch_compile=True,
                    )

    @skip_if_libnatten_is_not_supported()
    @skip_if_flex_is_not_supported()
    def test_2d_against_cutlass_2x(self):
        problem_sizes = [
            (1, 1, 32, (84, 69), (7, 20), (1, 6), (5, 1)),
            (1, 1, 128, (19, 29), (8, 8), (1, 1), (2, 3)),
            (1, 1, 32, (128, 128), (19, 24), (2, 3), (2, 2)),
            (1, 1, 64, (56, 56), (17, 4), (2, 1), (3, 2)),
            (2, 2, 64, (32, 64), (25, 31), (10, 20), (1, 2)),
            (2, 4, 64, (64, 128), (21, 29), (10, 12), (3, 4)),
            (4, 3, 128, (56, 56), (7, 7), (1, 1), (2, 4)),
        ]
        for (
            batch,
            heads,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in problem_sizes:
            for additional_kv_length in [0, 64]:
                for causal_x, causal_y in product([True, False], [True, False]):
                    is_causal = (causal_x, causal_y)
                    self._test_all_dtypes_against_cutlass_2x_fna(
                        batch=batch,
                        heads=heads,
                        head_dim=head_dim,
                        input_shape=input_shape,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        is_causal=is_causal,
                        additional_kv_length=additional_kv_length,
                        torch_compile=False,
                    )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_flex_is_not_supported()
    def test_2d_against_cutlass_2x_extended(self):
        problem_sizes = [
            (1, 1, 32, (84, 69), (7, 68), (1, 1), (5, 1)),
            (1, 1, 32, (84, 69), (7, 23), (1, 1), (5, 1)),
            (1, 1, 32, (128, 128), (4, 4), (1, 1), (2, 2)),
            (1, 1, 32, (128, 128), (4, 4), (1, 1), (1, 1)),
            (1, 1, 64, (56, 56), (17, 4), (2, 1), (1, 2)),
            (1, 1, 64, (128, 128), (93, 78), (2, 2), (1, 1)),
            (1, 1, 128, (16, 16), (3, 3), (2, 2), (1, 1)),
            (1, 1, 128, (16, 16), (4, 4), (2, 3), (1, 1)),
            (4, 3, 32, (32, 32), (31, 31), (15, 15), (1, 1)),
            (2, 2, 64, (32, 64), (26, 30), (1, 1), (1, 2)),
            (2, 4, 64, (64, 128), (55, 101), (1, 1), (1, 1)),
            (4, 3, 128, (28, 46), (11, 13), (1, 1), (1, 1)),
        ]
        for (
            batch,
            heads,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in problem_sizes:
            for additional_kv_length in [0, 64]:
                for causal_x, causal_y in product([True, False], [True, False]):
                    is_causal = (causal_x, causal_y)
                    self._test_all_dtypes_against_cutlass_2x_fna(
                        batch=batch,
                        heads=heads,
                        head_dim=head_dim,
                        input_shape=input_shape,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        is_causal=is_causal,
                        additional_kv_length=additional_kv_length,
                        torch_compile=False,
                    )

    @skip_if_libnatten_is_not_supported()
    @skip_if_flex_compile_is_not_supported()
    def test_2d_against_cutlass_2x_compiled(self):
        if not ENABLE_FLEX_COMPILE_TESTS:
            self.skipTest("Flex compile tests have been disabled.")

        problem_sizes = [
            (1, 1, 32, (84, 69), (7, 68), (1, 1), (5, 1)),
            (1, 1, 128, (19, 29), (8, 8), (1, 1), (2, 3)),
            (1, 1, 32, (128, 128), (4, 4), (1, 1), (2, 2)),
            (1, 1, 64, (56, 56), (17, 4), (2, 1), (3, 2)),
            (1, 1, 64, (128, 128), (93, 78), (2, 2), (1, 1)),
            (1, 1, 128, (16, 16), (3, 3), (2, 2), (1, 1)),
            (2, 2, 64, (32, 64), (25, 31), (10, 20), (1, 2)),
            (2, 4, 64, (64, 128), (55, 101), (1, 1), (1, 1)),
            (2, 4, 64, (64, 128), (21, 29), (10, 12), (3, 4)),
            (4, 3, 128, (56, 56), (7, 7), (1, 1), (2, 4)),
        ]
        for (
            batch,
            heads,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in problem_sizes:
            for causal_x, causal_y in product([True, False], [True, False]):
                is_causal = (causal_x, causal_y)
                self._test_all_dtypes_against_cutlass_2x_fna(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    input_shape=input_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    is_causal=is_causal,
                    additional_kv_length=0,
                    torch_compile=True,
                )

    @skip_if_libnatten_is_not_supported()
    @skip_if_flex_compile_is_not_supported()
    def test_3d_against_cutlass_2x_compiled(self):
        if not ENABLE_FLEX_COMPILE_TESTS:
            self.skipTest("Flex compile tests have been disabled.")

        problem_sizes = [
            (1, 1, 128, (24, 44, 80), (24, 12, 24), (1, 4, 8), (1, 1, 1)),
            (1, 1, 128, (20, 40, 40), (20, 12, 16), (1, 4, 16), (1, 1, 1)),
            (1, 1, 64, (18, 37, 12), (14, 16, 12), (12, 8, 1), (1, 2, 1)),
            (1, 1, 128, (8, 8, 4), (3, 4, 3), (1, 1, 1), (1, 1, 1)),
            (1, 2, 128, (8, 8, 12), (5, 8, 11), (2, 3, 4), (1, 1, 1)),
            (4, 8, 64, (32, 10, 10), (7, 3, 3), (5, 1, 1), (1, 2, 3)),
            (1, 4, 32, (8, 8, 16), (3, 3, 3), (2, 1, 2), (2, 2, 4)),
            (2, 2, 32, (8, 8, 10), (3, 4, 3), (3, 4, 1), (1, 1, 1)),
            (1, 12, 64, (32, 8, 8), (7, 5, 5), (2, 1, 3), (2, 1, 1)),
        ]
        for (
            batch,
            heads,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in problem_sizes:
            for causal_x, causal_y, causal_z in product(
                [True, False], [True, False], [True, False]
            ):
                is_causal = (causal_x, causal_y, causal_z)
                self._test_all_dtypes_against_cutlass_2x_fna(
                    batch=batch,
                    heads=heads,
                    head_dim=head_dim,
                    input_shape=input_shape,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    is_causal=is_causal,
                    additional_kv_length=0,
                    torch_compile=True,
                )

    # This use case fails on H100, ctk 12.8, cuda driver 550.54.15 and torch stable 2.7.0 + ctk 12.8.
    # Runs perfectly fine without torch compile.
    @unittest.skip("Failing use case.")
    def test_3d_against_cutlass_2x_compiled_failing(self):
        problem_sizes = [
            (1, 1, 128, (13, 11, 9), (3, 4, 3), (2, 3, 3), (3, 2, 2)),
        ]
        for (
            batch,
            heads,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in problem_sizes:
            is_causal = (False, True, True)
            self._test_all_dtypes_against_cutlass_2x_fna(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                input_shape=input_shape,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                additional_kv_length=0,
                torch_compile=True,
            )

    @skip_if_libnatten_is_not_supported()
    @skip_if_flex_is_not_supported()
    def test_3d_against_cutlass_2x(self):
        problem_sizes = [
            (1, 1, 128, (13, 11, 9), (3, 4, 3), (2, 3, 3), (3, 2, 2)),
            (1, 1, 128, (13, 11, 9), (3, 4, 3), (1, 1, 1), (3, 2, 2)),
            (1, 2, 128, (8, 8, 12), (5, 8, 11), (2, 3, 4), (1, 1, 1)),
            (1, 1, 64, (32, 10, 10), (7, 3, 3), (5, 1, 1), (1, 2, 3)),
            (1, 4, 32, (8, 8, 16), (3, 3, 3), (2, 1, 2), (2, 2, 4)),
        ]
        for (
            batch,
            heads,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in problem_sizes:
            for additional_kv_length in [0, 64]:
                for causal_x, causal_y, causal_z in product(
                    [True, False], [True, False], [True, False]
                ):
                    is_causal = (causal_x, causal_y, causal_z)
                    self._test_all_dtypes_against_cutlass_2x_fna(
                        batch=batch,
                        heads=heads,
                        head_dim=head_dim,
                        input_shape=input_shape,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        is_causal=is_causal,
                        additional_kv_length=additional_kv_length,
                        torch_compile=False,
                    )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_flex_is_not_supported()
    def test_3d_against_cutlass_2x_extended(self):
        problem_sizes = [
            (1, 1, 32, (4, 4, 4), (2, 2, 2), (1, 1, 1), (1, 1, 1)),
            (1, 2, 32, (4, 4, 4), (2, 2, 2), (1, 1, 1), (1, 1, 1)),
            (1, 2, 32, (4, 4, 4), (2, 2, 2), (2, 2, 1), (1, 1, 1)),
            (1, 1, 128, (8, 8, 4), (3, 4, 3), (1, 1, 1), (1, 1, 1)),
            (1, 2, 128, (8, 8, 12), (5, 8, 11), (2, 3, 4), (1, 1, 1)),
            (2, 2, 32, (8, 8, 10), (3, 4, 3), (3, 4, 1), (1, 1, 1)),
            (1, 12, 64, (32, 8, 8), (7, 5, 5), (2, 1, 3), (2, 1, 1)),
            (4, 8, 64, (32, 10, 10), (7, 3, 3), (5, 1, 1), (1, 2, 3)),
            (1, 1, 64, (18, 37, 12), (14, 16, 12), (12, 8, 1), (1, 2, 1)),
        ]
        for (
            batch,
            heads,
            head_dim,
            input_shape,
            kernel_size,
            stride,
            dilation,
        ) in problem_sizes:
            for additional_kv_length in [0, 64]:
                for causal_x, causal_y, causal_z in product(
                    [True, False], [True, False], [True, False]
                ):
                    is_causal = (causal_x, causal_y, causal_z)
                    self._test_all_dtypes_against_cutlass_2x_fna(
                        batch=batch,
                        heads=heads,
                        head_dim=head_dim,
                        input_shape=input_shape,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        is_causal=is_causal,
                        additional_kv_length=additional_kv_length,
                        torch_compile=False,
                    )

    def _test_rand_sweep_against_cutlass_2x(self, na_dim, torch_compile: bool = False):
        random.seed(42)

        max_tests = 1000
        max_seqlen = 2**17 if torch_compile else 2**13
        max_kernel_size = None if torch_compile else 2**10

        for i in range(max_tests):
            batch = random.choice(range(1, 4))
            heads = random.choice(range(1, 4))
            head_dim = random.choice([32, 64, 128])

            input_shape_ = []
            for j in range(na_dim):
                input_shape_.append(random.choice(range(4, 97)))

            while math.prod(input_shape_) > max_seqlen:
                dim_to_cut = random.choice(range(na_dim))
                input_shape_[dim_to_cut] = max(4, int(input_shape_[dim_to_cut] * 0.1))

            input_shape = tuple(input_shape_)

            kernel_size_ = [random.choice(range(2, x + 1)) for x in input_shape]
            if max_kernel_size is not None:
                while math.prod(kernel_size_) > max_kernel_size:
                    dim_to_cut = random.choice(range(na_dim))
                    kernel_size_[dim_to_cut] = max(
                        2, int(kernel_size_[dim_to_cut] * 0.1)
                    )

            kernel_size = tuple(kernel_size_)
            stride = tuple(random.choice(range(1, k + 1)) for k in kernel_size)
            dilation = tuple(
                random.choice(range(1, x // k + 1))
                for x, k in zip(input_shape, kernel_size)
            )
            is_causal = tuple(random.choice([False, True]) for _ in range(na_dim))

            self._test_all_dtypes_against_cutlass_2x_fna(
                batch=batch,
                heads=heads,
                head_dim=head_dim,
                input_shape=input_shape,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                additional_kv_length=0,
                torch_compile=torch_compile,
            )

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_flex_is_not_supported()
    def test_rand_sweep_1d_against_cutlass_2x(self):
        self._test_rand_sweep_against_cutlass_2x(1)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_flex_is_not_supported()
    def test_rand_sweep_2d_against_cutlass_2x(self):
        self._test_rand_sweep_against_cutlass_2x(2)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_flex_is_not_supported()
    def test_rand_sweep_3d_against_cutlass_2x(self):
        self._test_rand_sweep_against_cutlass_2x(3)


if __name__ == "__main__":
    torch.manual_seed(42)
    unittest.main()
