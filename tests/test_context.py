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

import os
import unittest

import natten

import torch
from natten.utils.testing import (
    skip_if_cuda_is_not_supported,
    skip_if_fna_is_not_supported,
    skip_if_gemm_is_not_supported,
)


def _reset_everything():
    from natten.context import AutotunerContext, NattenContext

    NattenContext.reset()
    AutotunerContext.reset()
    natten.use_tiled_na()
    natten.use_gemm_na()
    natten.use_tf32_in_gemm_na()
    torch.use_deterministic_algorithms(False)
    os.environ["NATTEN_LOG_LEVEL"] = "CRITICAL"


class AutotunerContextTests(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    @skip_if_fna_is_not_supported()
    def test_set_flags(self):
        from natten.context import AutotunerContext

        assert (
            not natten.is_autotuner_enabled_for_forward()
            and not natten.is_autotuner_enabled_for_backward()
            and not natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
        )

        natten.use_autotuner(True)
        assert (
            natten.is_autotuner_enabled_for_forward()
            and not natten.is_autotuner_enabled_for_backward()
            and not natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
        )

        natten.use_autotuner(False, True)
        assert (
            not natten.is_autotuner_enabled_for_forward()
            and natten.is_autotuner_enabled_for_backward()
            and not natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
        )

        natten.use_autotuner(False, True, False, True)
        assert (
            not natten.is_autotuner_enabled_for_forward()
            and natten.is_autotuner_enabled_for_backward()
            and not natten.is_autotuner_thorough_for_forward()
            and natten.is_autotuner_thorough_for_backward()
        )

        natten.use_autotuner(False, False, False, True)
        assert (
            not natten.is_autotuner_enabled_for_forward()
            and not natten.is_autotuner_enabled_for_backward()
            and not natten.is_autotuner_thorough_for_forward()
            and natten.is_autotuner_thorough_for_backward()
        )

        natten.use_autotuner(thorough_mode_backward=False)
        assert (
            not natten.is_autotuner_enabled_for_forward()
            and not natten.is_autotuner_enabled_for_backward()
            and not natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
        )

        natten.use_autotuner(thorough_mode_forward=True)
        assert (
            not natten.is_autotuner_enabled_for_forward()
            and not natten.is_autotuner_enabled_for_backward()
            and natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
        )

        natten.use_autotuner(backward_pass=True)
        assert (
            not natten.is_autotuner_enabled_for_forward()
            and natten.is_autotuner_enabled_for_backward()
            and natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
        )

        natten.use_autotuner(
            warmup_steps_forward=8,
            warmup_steps_backward=3,
            steps_forward=2,
            steps_backward=1,
        )
        assert (
            not natten.is_autotuner_enabled_for_forward()
            and natten.is_autotuner_enabled_for_backward()
            and natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
            and AutotunerContext.warmup_steps_forward == 8
            and AutotunerContext.warmup_steps_backward == 3
            and AutotunerContext.steps_forward == 2
            and AutotunerContext.steps_backward == 1
        )

        natten.use_autotuner(steps_backward=17)
        assert (
            not natten.is_autotuner_enabled_for_forward()
            and natten.is_autotuner_enabled_for_backward()
            and natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
            and AutotunerContext.warmup_steps_forward == 8
            and AutotunerContext.warmup_steps_backward == 3
            and AutotunerContext.steps_forward == 2
            and AutotunerContext.steps_backward == 17
        )

        natten.use_autotuner(steps_forward=25)
        assert (
            not natten.is_autotuner_enabled_for_forward()
            and natten.is_autotuner_enabled_for_backward()
            and natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
            and AutotunerContext.warmup_steps_forward == 8
            and AutotunerContext.warmup_steps_backward == 3
            and AutotunerContext.steps_forward == 25
            and AutotunerContext.steps_backward == 17
        )

        natten.use_autotuner(warmup_steps_backward=1)
        assert (
            not natten.is_autotuner_enabled_for_forward()
            and natten.is_autotuner_enabled_for_backward()
            and natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
            and AutotunerContext.warmup_steps_forward == 8
            and AutotunerContext.warmup_steps_backward == 1
            and AutotunerContext.steps_forward == 25
            and AutotunerContext.steps_backward == 17
        )

        natten.use_autotuner(warmup_steps_forward=10)
        assert (
            not natten.is_autotuner_enabled_for_forward()
            and natten.is_autotuner_enabled_for_backward()
            and natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
            and AutotunerContext.warmup_steps_forward == 10
            and AutotunerContext.warmup_steps_backward == 1
            and AutotunerContext.steps_forward == 25
            and AutotunerContext.steps_backward == 17
        )

        # Reset
        natten.disable_autotuner()
        assert (
            not natten.is_autotuner_enabled_for_forward()
            and not natten.is_autotuner_enabled_for_backward()
            and not natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
        )

    @unittest.expectedFailure
    def test_fail1(self):
        assert not natten.are_deterministic_algorithms_enabled()
        natten.use_deterministic_algorithms()
        assert natten.are_deterministic_algorithms_enabled()

        assert (
            not natten.is_autotuner_enabled_for_forward()
            and not natten.is_autotuner_enabled_for_backward()
            and not natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
        )
        natten.use_autotuner(True)

    @unittest.expectedFailure
    def test_fail2(self):
        assert not natten.are_deterministic_algorithms_enabled()
        natten.use_deterministic_algorithms()
        assert natten.are_deterministic_algorithms_enabled()

        assert (
            not natten.is_autotuner_enabled_for_forward()
            and not natten.is_autotuner_enabled_for_backward()
            and not natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
        )
        natten.use_autotuner(False, True)

    @unittest.expectedFailure
    def test_fail3(self):
        assert not natten.are_deterministic_algorithms_enabled()
        natten.use_deterministic_algorithms()
        assert natten.are_deterministic_algorithms_enabled()

        assert (
            not natten.is_autotuner_enabled_for_forward()
            and not natten.is_autotuner_enabled_for_backward()
            and not natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
        )
        natten.use_autotuner(backward_pass=True)

    @unittest.expectedFailure
    def test_fail4(self):
        assert not natten.are_deterministic_algorithms_enabled()
        natten.use_deterministic_algorithms()
        assert natten.are_deterministic_algorithms_enabled()

        assert (
            not natten.is_autotuner_enabled_for_forward()
            and not natten.is_autotuner_enabled_for_backward()
            and not natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
        )
        natten.use_autotuner(warmup_steps_backward=(False,))

    @unittest.expectedFailure
    def test_fail5(self):
        assert not natten.are_deterministic_algorithms_enabled()
        natten.use_deterministic_algorithms()
        assert natten.are_deterministic_algorithms_enabled()

        assert (
            not natten.is_autotuner_enabled_for_forward()
            and not natten.is_autotuner_enabled_for_backward()
            and not natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
        )
        natten.use_autotuner(steps_forward="abc")

    def test_warn_skip(self):
        assert not natten.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        assert not natten.are_deterministic_algorithms_enabled()

        assert (
            not natten.is_autotuner_enabled_for_forward()
            and not natten.is_autotuner_enabled_for_backward()
            and not natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
        )

        natten.use_autotuner(True)
        assert (
            not natten.is_autotuner_enabled_for_forward()
            and not natten.is_autotuner_enabled_for_backward()
            and not natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
        )

        natten.use_autotuner(False, True)
        assert (
            not natten.is_autotuner_enabled_for_forward()
            and not natten.is_autotuner_enabled_for_backward()
            and not natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
        )

        natten.use_autotuner(backward_pass=True)
        assert (
            not natten.is_autotuner_enabled_for_forward()
            and not natten.is_autotuner_enabled_for_backward()
            and not natten.is_autotuner_thorough_for_forward()
            and not natten.is_autotuner_thorough_for_backward()
        )
        torch.use_deterministic_algorithms(False)


class LibnattenContextTests(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    @skip_if_cuda_is_not_supported()
    def test_tiled_na_flag(self):
        assert natten.is_tiled_na_enabled()
        natten.use_tiled_na(False)
        assert not natten.is_tiled_na_enabled()
        natten.use_tiled_na()
        assert natten.is_tiled_na_enabled()
        natten.use_tiled_na(False)
        assert not natten.is_tiled_na_enabled()
        natten.use_tiled_na(True)
        assert natten.is_tiled_na_enabled()

    @skip_if_gemm_is_not_supported()
    def test_gemm_na_flags(self):
        assert natten.is_gemm_na_enabled()
        natten.use_gemm_na(False)
        assert not natten.is_gemm_na_enabled()
        natten.use_gemm_na()
        assert natten.is_gemm_na_enabled()
        natten.use_gemm_na(False)
        assert not natten.is_gemm_na_enabled()
        natten.use_gemm_na(True)
        assert natten.is_gemm_na_enabled()

        if natten.has_tf32_gemm():
            # SM80 and above
            assert natten.has_fp64_gemm()
            assert natten.is_tf32_in_gemm_na_enabled()
            assert natten.is_gemm_na_enabled()

            natten.use_tf32_in_gemm_na(False)
            assert not natten.is_tf32_in_gemm_na_enabled()
            natten.use_tf32_in_gemm_na()
            assert natten.is_tf32_in_gemm_na_enabled()
            natten.use_tf32_in_gemm_na(False)
            assert not natten.is_tf32_in_gemm_na_enabled()
            natten.use_tf32_in_gemm_na(True)
            assert natten.is_tf32_in_gemm_na_enabled()


class MainContextTests(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    @skip_if_fna_is_not_supported()
    def test_mem_usage_perf(self):
        from natten.context import MemoryUsagePreference

        assert natten.get_memory_usage_preference() == MemoryUsagePreference.Default
        natten.set_memory_usage_preference()
        assert natten.get_memory_usage_preference() == MemoryUsagePreference.Default
        natten.set_memory_usage_preference("default")
        assert natten.get_memory_usage_preference() == MemoryUsagePreference.Default
        natten.set_memory_usage_preference("strict")
        assert natten.get_memory_usage_preference() == MemoryUsagePreference.Strict
        natten.set_memory_usage_preference("default")
        assert natten.get_memory_usage_preference() == MemoryUsagePreference.Default
        natten.set_memory_usage_preference("unrestricted")
        assert (
            natten.get_memory_usage_preference() == MemoryUsagePreference.Unrestricted
        )
        natten.set_memory_usage_preference()
        assert natten.get_memory_usage_preference() == MemoryUsagePreference.Default

    @unittest.expectedFailure
    @skip_if_fna_is_not_supported()
    def test_mem_usage_perf_fail1(self):
        natten.set_memory_usage_preference(0)

    @unittest.expectedFailure
    @skip_if_fna_is_not_supported()
    def test_mem_usage_perf_fail2(self):
        natten.set_memory_usage_preference("invalid")

    def test_deterministic_mode(self):
        assert not natten.are_deterministic_algorithms_enabled()
        natten.use_deterministic_algorithms()
        assert natten.are_deterministic_algorithms_enabled()
        natten.use_deterministic_algorithms(False)
        assert not natten.are_deterministic_algorithms_enabled()
        natten.use_deterministic_algorithms(True)
        assert natten.are_deterministic_algorithms_enabled()
        natten.use_deterministic_algorithms(False)
        assert not natten.are_deterministic_algorithms_enabled()

    @skip_if_fna_is_not_supported()
    def test_use_fna_mode(self):
        assert not natten.is_fused_na_enabled()
        natten.use_fused_na()
        assert natten.is_fused_na_enabled()
        natten.use_fused_na(False)
        assert not natten.is_fused_na_enabled()
        natten.use_fused_na(True)
        assert natten.is_fused_na_enabled()
        natten.use_fused_na(False)
        assert not natten.is_fused_na_enabled()

    @skip_if_fna_is_not_supported()
    def test_use_fna_mode_with_kv_parallel(self):
        assert (
            not natten.is_fused_na_enabled()
            and not natten.is_kv_parallelism_in_fused_na_enabled()
        )
        natten.use_fused_na()
        assert (
            natten.is_fused_na_enabled()
            and not natten.is_kv_parallelism_in_fused_na_enabled()
        )
        natten.use_fused_na(False)
        assert (
            not natten.is_fused_na_enabled()
            and not natten.is_kv_parallelism_in_fused_na_enabled()
        )
        natten.use_fused_na(True)
        assert (
            natten.is_fused_na_enabled()
            and not natten.is_kv_parallelism_in_fused_na_enabled()
        )
        natten.use_fused_na(False)
        assert (
            not natten.is_fused_na_enabled()
            and not natten.is_kv_parallelism_in_fused_na_enabled()
        )
        natten.use_fused_na(False, kv_parallel=True)
        assert (
            not natten.is_fused_na_enabled()
            and not natten.is_kv_parallelism_in_fused_na_enabled()
        )
        natten.use_fused_na(True, kv_parallel=True)
        assert (
            natten.is_fused_na_enabled()
            and natten.is_kv_parallelism_in_fused_na_enabled()
        )
        natten.use_fused_na(True, kv_parallel=False)
        assert (
            natten.is_fused_na_enabled()
            and not natten.is_kv_parallelism_in_fused_na_enabled()
        )
        natten.use_fused_na(False, False)
        assert (
            not natten.is_fused_na_enabled()
            and not natten.is_kv_parallelism_in_fused_na_enabled()
        )
        natten.use_fused_na(True, kv_parallel=True)
        assert (
            natten.is_fused_na_enabled()
            and natten.is_kv_parallelism_in_fused_na_enabled()
        )
        natten.use_fused_na(False)
        assert (
            not natten.is_fused_na_enabled()
            and not natten.is_kv_parallelism_in_fused_na_enabled()
        )

        # Trying to turn on KV parallelism while NATTEN's deterministic mode is on will trigger
        # a runtime error. But the `kv_parallel` arg should be ignored when mode (first pos arg)
        # is False, so this should work without any errors.
        assert not natten.are_deterministic_algorithms_enabled()
        natten.use_deterministic_algorithms()
        assert natten.are_deterministic_algorithms_enabled()
        natten.use_fused_na(False, kv_parallel=True)
        assert (
            not natten.is_fused_na_enabled()
            and not natten.is_kv_parallelism_in_fused_na_enabled()
        )

        # Reset
        natten.use_fused_na(False)
        assert (
            not natten.is_fused_na_enabled()
            and not natten.is_kv_parallelism_in_fused_na_enabled()
        )
        natten.use_deterministic_algorithms(False)
        assert not natten.are_deterministic_algorithms_enabled()

    @unittest.expectedFailure
    @skip_if_fna_is_not_supported()
    def test_use_fna_mode_with_kv_parallel_fail(self):
        assert (
            not natten.is_fused_na_enabled()
            and not natten.is_kv_parallelism_in_fused_na_enabled()
        )
        assert not natten.are_deterministic_algorithms_enabled()

        # Trying to turn on KV parallelism while NATTEN's deterministic mode is on should trigger
        # a runtime error
        natten.use_deterministic_algorithms()
        assert natten.are_deterministic_algorithms_enabled()
        natten.use_fused_na(True, kv_parallel=True)


if __name__ == "__main__":
    unittest.main()
