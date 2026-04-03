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
import re
import unittest
from functools import partial

import torch
from natten.functional import attention
from natten.utils import log
from natten.utils.testing import (
    skip_if_blackwell_kernels_not_supported,
    skip_if_hopper_kernels_not_supported,
    skip_if_libnatten_is_not_supported,
)
from natten.utils.varlen import generate_varlen_parameters
from torch.profiler import profile as torch_profile, ProfilerActivity

logger = log.get_logger(__name__)

DEVICE = "cuda"
DTYPE = torch.float16
SEQLEN_PAD_MULTIPLE = 64

KERNEL_PATTERN = re.compile(r"fmha|fna", re.IGNORECASE)
BWD_PATTERN = re.compile(r"backward|bwd", re.IGNORECASE)
BWD_HELPER_PATTERN = re.compile(r"convert|sum_odo|sumodo|delta", re.IGNORECASE)


def _generate_problem_shapes():
    """Generate random but deterministic problem shapes for the test.

    Returns heads, head_dim, two normal seqlen lists, two empty seqlen lists,
    and the padded total_seqlen.
    """
    heads = random.choice([1, 2, 4, 8, 16])
    head_dim = random.choice([32, 64, 128])

    # Normal batches: random batch sizes and odd sequence lengths
    num_seqs_a = random.randint(2, 8)
    num_seqs_b = random.randint(2, 6)
    # Use odd/prime-ish lengths in a reasonable range to exercise non-aligned paths
    normal_seqlens_a = [random.randrange(17, 513, 2) for _ in range(num_seqs_a)]
    normal_seqlens_b = [random.randrange(17, 513, 2) for _ in range(num_seqs_b)]

    # Empty batches: random batch sizes, all zeros
    num_empty_a = random.randint(1, 10)
    num_empty_b = random.randint(1, 5)
    empty_seqlens_a = [0] * num_empty_a
    empty_seqlens_b = [0] * num_empty_b

    max_sum = max(sum(normal_seqlens_a), sum(normal_seqlens_b))
    total_seqlen = (
        (max_sum + SEQLEN_PAD_MULTIPLE - 1) // SEQLEN_PAD_MULTIPLE * SEQLEN_PAD_MULTIPLE
    )

    return (
        heads,
        head_dim,
        normal_seqlens_a,
        normal_seqlens_b,
        empty_seqlens_a,
        empty_seqlens_b,
        total_seqlen,
    )


def _reset(max_recompiles=1):
    random.seed(42)
    torch.manual_seed(42)
    torch.compiler.reset()
    torch._dynamo.config.accumulated_recompile_limit = max_recompiles
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    torch.cuda.empty_cache()


def _make_qkv(total_seqlen, heads, head_dim):
    shape = (1, total_seqlen, heads, head_dim)
    q = torch.randn(shape, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k = torch.randn(shape, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v = torch.randn(shape, device=DEVICE, dtype=DTYPE, requires_grad=True)
    return q, k, v


def _attention_fn(
    q, k, v, cumseq_q, cumseq_kv, max_q, max_kv, backend, is_causal=False
):
    return attention(
        q,
        k,
        v,
        cumulative_seqlen_Q=cumseq_q,
        cumulative_seqlen_KV=cumseq_kv,
        max_seqlen_Q=max_q,
        max_seqlen_KV=max_kv,
        backend=backend,
        is_causal=is_causal,
    )


def _run_fwd_bwd(fn, q, k, v, cumseq_q, cumseq_kv, max_q, max_kv):
    out = fn(q, k, v, cumseq_q, cumseq_kv, max_q, max_kv)
    out.sum().backward()


def _profile_case(label, fn, q, k, v, cumseq_q, cumseq_kv, max_q, max_kv):
    with torch.no_grad():
        qq = q.clone().requires_grad_(True)
        kk = k.clone().requires_grad_(True)
        vv = v.clone().requires_grad_(True)

    torch.cuda.synchronize()
    logger.debug(f"Profiling [{label}]: capturing CUDA trace")
    with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
        _run_fwd_bwd(fn, qq, kk, vv, cumseq_q, cumseq_kv, max_q, max_kv)
        torch.cuda.synchronize()

    logger.debug(
        f"Profiling [{label}]: CUDA events:\n{prof.key_averages().table(sort_by='cuda_time_total', row_limit=20)}"
    )
    return prof


def _find_attention_kernels(prof):
    """Return (all_names, fwd_dict, bwd_dict) of attention kernels.

    Each dict maps kernel name to its call count from the profiler.
    Forward kernels may not have an explicit forward/fwd keyword in their name,
    so we identify them as any attention kernel that is NOT a backward kernel.
    """
    fwd_kernels = {}
    bwd_kernels = {}
    for evt in prof.key_averages():
        if KERNEL_PATTERN.search(evt.key):
            if BWD_PATTERN.search(evt.key) and not BWD_HELPER_PATTERN.search(evt.key):
                bwd_kernels[evt.key] = evt.count
            elif not BWD_PATTERN.search(evt.key):
                fwd_kernels[evt.key] = evt.count
    all_names = set(fwd_kernels) | set(bwd_kernels)
    return all_names, fwd_kernels, bwd_kernels


def _make_fn(backend, use_compile, is_causal):
    fn = partial(_attention_fn, backend=backend, is_causal=is_causal)
    if use_compile:
        fn = torch.compile(fn, fullgraph=True, dynamic=True)
    return fn


def _prepare_case(seqlens_list, total_seqlen, heads, head_dim):
    q, k, v = _make_qkv(total_seqlen, heads, head_dim)
    seqlens = torch.tensor(seqlens_list, dtype=torch.int32, device=DEVICE)
    cumseq_q, cumseq_kv, max_q, max_kv = generate_varlen_parameters(
        query=q,
        key=k,
        value=v,
        seqlens_Q=seqlens,
        seqlens_KV=seqlens,
    )
    return q, k, v, cumseq_q, cumseq_kv, max_q, max_kv


def _run_empty_batch_kernel_check(backend, use_compile, test_causal=False):
    _reset(max_recompiles=2 if test_causal else 1)
    mode = "compiled" if use_compile else "eager"
    causal_tag = "test_causal" if test_causal else "noncausal"
    logger.debug(f"=== Testing backend={backend}, mode={mode}, {causal_tag} ===")

    (
        heads,
        head_dim,
        normal_seqlens_a,
        normal_seqlens_b,
        empty_seqlens_a,
        empty_seqlens_b,
        total_seqlen,
    ) = _generate_problem_shapes()

    logger.debug(
        f"{heads=}, {head_dim=}, {total_seqlen=} "
        f"(padded to multiple of {SEQLEN_PAD_MULTIPLE})"
    )

    # Prepare all cases
    normal_a = _prepare_case(normal_seqlens_a, total_seqlen, heads, head_dim)
    normal_b = _prepare_case(normal_seqlens_b, total_seqlen, heads, head_dim)
    empty_a = _prepare_case(empty_seqlens_a, total_seqlen, heads, head_dim)
    empty_b = _prepare_case(empty_seqlens_b, total_seqlen, heads, head_dim)

    logger.debug(
        f"Normal A: seqlens={normal_seqlens_a}, max_q={normal_a[5]}, max_kv={normal_a[6]}"
    )
    logger.debug(
        f"Normal B: seqlens={normal_seqlens_b}, max_q={normal_b[5]}, max_kv={normal_b[6]}"
    )
    logger.debug(
        f"Empty A: seqlens={empty_seqlens_a}, max_q={empty_a[5]}, max_kv={empty_a[6]}"
    )
    logger.debug(
        f"Empty B: seqlens={empty_seqlens_b}, max_q={empty_b[5]}, max_kv={empty_b[6]}"
    )

    def _build_cases():
        if test_causal:
            fn_causal = _make_fn(backend, use_compile, is_causal=True)
            fn_noncausal = _make_fn(backend, use_compile, is_causal=False)
            return [
                ("normal_a/causal", fn_causal, normal_a),
                ("empty_a/noncausal", fn_noncausal, empty_a),
                ("normal_b/noncausal", fn_noncausal, normal_b),
                ("empty_b/causal", fn_causal, empty_b),
            ]
        else:
            fn = _make_fn(backend, use_compile, is_causal=False)
            return [
                ("normal_a/noncausal", fn, normal_a),
                ("empty_a/noncausal", fn, empty_a),
                ("normal_b/noncausal", fn, normal_b),
                ("empty_b/noncausal", fn, empty_b),
            ]

    cases = _build_cases()

    if use_compile:
        logger.debug(
            "Running interleaved compile check: normal_a -> empty_a -> normal_b -> empty_b"
        )
        for label, fn, args in cases:
            logger.debug(f"Running compile check: {label}")
            fn(*args)
            logger.debug(f"PASS: {label} (no recompile)")
        # Reset compiler before profiling
        torch.compiler.reset()
        cases = _build_cases()

    # Profile normal batches: must contain exactly 1 forward and 1 backward attention
    # kernel, each called exactly once.
    all_normal_kernels = set()
    for label, fn, args in cases:
        if not label.startswith("normal"):
            continue
        full_label = f"{backend}/{mode}/{label}"
        logger.debug(f"Checking normal batch kernels: {full_label}")
        prof = _profile_case(full_label, fn, *args)
        all_names, fwd_kernels, bwd_kernels = _find_attention_kernels(prof)
        all_events = [evt.key for evt in prof.key_averages()]
        logger.debug(
            f"[{full_label}] attention kernels: fwd={fwd_kernels}, bwd={bwd_kernels}"
        )
        assert len(fwd_kernels) == 1, (
            f"[{full_label}] expected exactly 1 forward attention kernel, "
            f"got {len(fwd_kernels)}: {fwd_kernels}. CUDA events: {all_events}"
        )
        assert len(bwd_kernels) == 1, (
            f"[{full_label}] expected exactly 1 backward attention kernel, "
            f"got {len(bwd_kernels)}: {bwd_kernels}. CUDA events: {all_events}"
        )
        fwd_name, fwd_count = next(iter(fwd_kernels.items()))
        bwd_name, bwd_count = next(iter(bwd_kernels.items()))
        assert fwd_count == 1, (
            f"[{full_label}] forward kernel {fwd_name!r} called {fwd_count} times, "
            f"expected 1"
        )
        assert bwd_count == 1, (
            f"[{full_label}] backward kernel {bwd_name!r} called {bwd_count} times, "
            f"expected 1"
        )
        all_normal_kernels |= all_names

    logger.debug(f"Combined normal attention kernels: {all_normal_kernels}")

    # Profile empty batches: must NOT contain any of the kernels found in normal
    failures = []
    for label, fn, args in cases:
        if not label.startswith("empty"):
            continue
        full_label = f"{backend}/{mode}/{label}"
        logger.debug(f"Checking empty batch kernels: {full_label}")
        prof = _profile_case(full_label, fn, *args)
        all_names, _fwd_kernels, _bwd_kernels = _find_attention_kernels(prof)
        leaked = all_normal_kernels & all_names
        logger.debug(
            f"[{full_label}] attention kernels found: {all_names}, "
            f"leaked from normal: {leaked}"
        )
        if leaked:
            failures.append(
                f"[{full_label}] should skip attention kernels but found: {leaked}"
            )

    assert not failures, "\n".join(failures)

    logger.debug(f"=== PASSED backend={backend}, mode={mode}, {causal_tag} ===")


class VarlenRecompileTest(unittest.TestCase):
    # Cutlass FMHA (works on Hopper and Blackwell)

    @skip_if_libnatten_is_not_supported()
    def test_cutlass_fmha_eager(self):
        _run_empty_batch_kernel_check("cutlass-fmha", use_compile=False)

    @skip_if_libnatten_is_not_supported()
    def test_cutlass_fmha_compiled(self):
        _run_empty_batch_kernel_check("cutlass-fmha", use_compile=True)

    @skip_if_libnatten_is_not_supported()
    def test_cutlass_fmha_eager_test_causal(self):
        _run_empty_batch_kernel_check(
            "cutlass-fmha", use_compile=False, test_causal=True
        )

    @skip_if_libnatten_is_not_supported()
    def test_cutlass_fmha_compiled_test_causal(self):
        _run_empty_batch_kernel_check(
            "cutlass-fmha", use_compile=True, test_causal=True
        )

    # Hopper FMHA

    @skip_if_hopper_kernels_not_supported()
    def test_hopper_fmha_eager(self):
        _run_empty_batch_kernel_check("hopper-fmha", use_compile=False)

    @skip_if_hopper_kernels_not_supported()
    def test_hopper_fmha_compiled(self):
        _run_empty_batch_kernel_check("hopper-fmha", use_compile=True)

    @skip_if_hopper_kernels_not_supported()
    def test_hopper_fmha_eager_test_causal(self):
        _run_empty_batch_kernel_check(
            "hopper-fmha", use_compile=False, test_causal=True
        )

    @skip_if_hopper_kernels_not_supported()
    def test_hopper_fmha_compiled_test_causal(self):
        _run_empty_batch_kernel_check("hopper-fmha", use_compile=True, test_causal=True)

    # Blackwell FMHA

    @skip_if_blackwell_kernels_not_supported()
    def test_blackwell_fmha_eager(self):
        _run_empty_batch_kernel_check("blackwell-fmha", use_compile=False)

    @skip_if_blackwell_kernels_not_supported()
    def test_blackwell_fmha_compiled(self):
        _run_empty_batch_kernel_check("blackwell-fmha", use_compile=True)

    @skip_if_blackwell_kernels_not_supported()
    def test_blackwell_fmha_eager_test_causal(self):
        _run_empty_batch_kernel_check(
            "blackwell-fmha", use_compile=False, test_causal=True
        )

    @skip_if_blackwell_kernels_not_supported()
    def test_blackwell_fmha_compiled_test_causal(self):
        _run_empty_batch_kernel_check(
            "blackwell-fmha", use_compile=True, test_causal=True
        )
