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
        #
        kernel_size,
        stride,
        dilation,
        is_causal,
        #
        kernel_size_list,
        stride_list,
        dilation_list,
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
            f"{heads=}, {heads_kv=}, {head_dim=}, {head_dim_v=}, {dtype=}, {test_backprop=}\n"
            f"{kernel_size=}, {stride=}, {dilation=}, {is_causal=}\n"
            f"{token_layout_list=},\n"
            f"{kernel_size_list=},\n"
            f"{stride_list=},\n"
            f"{dilation_list=},\n"
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

            kernel_size_ = (
                kernel_size if kernel_size_list is None else kernel_size_list[i]
            )
            stride_ = stride if stride_list is None else stride_list[i]
            dilation_ = dilation if dilation_list is None else dilation_list[i]
            o_batch, lse_batch = neighborhood_attention_generic(
                q_batch,
                k_batch,
                v_batch,
                kernel_size=kernel_size_,
                stride=stride_,
                dilation=dilation_,
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
            #
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            #
            kernel_size_list=kernel_size_list,
            stride_list=stride_list,
            dilation_list=dilation_list,
            #
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
        kernel_size=None,
        stride=None,
        dilation=None,
        is_causal=None,
        #
        kernel_size_list=None,
        stride_list=None,
        dilation_list=None,
        #
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
                        kernel_size_list=kernel_size_list,
                        stride_list=stride_list,
                        dilation_list=dilation_list,
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
        problem_sizes = [
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

    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_fixed_use_cases_varparam(self):
        problem_sizes = [
            (
                [(32, 32, 32), (128, 64, 16), (32, 32, 32)],
                3,
                3,
                128,
                [(2, 2, 2) for _ in range(3)],
                [1 for _ in range(3)],
                [(1, 1, 1), (1, 1, 1), (8, 8, 2)],
            ),
            (
                [
                    (72, 23, 5),
                    (57, 7, 67),
                    (42, 5, 17),
                    (78, 4, 71),
                    (4, 75, 72),
                    (5, 43, 43),
                    (90, 4, 9),
                    (53, 26, 7),
                    (67, 60, 8),
                    (65, 35, 4),
                    (45, 5, 85),
                    (40, 55, 4),
                    (19, 80, 5),
                ],
                2,
                1,
                128,
                [(2, 2, 2) for _ in range(13)],
                [1 for _ in range(13)],
                [
                    (20, 2, 1),
                    (10, 1, 17),
                    (18, 1, 2),
                    (19, 2, 33),
                    (1, 6, 31),
                    (2, 5, 14),
                    (38, 2, 1),
                    (21, 3, 3),
                    (13, 4, 1),
                    (4, 8, 2),
                    (14, 2, 2),
                    (6, 23, 1),
                    (1, 1, 2),
                ],
            ),
            (
                [
                    (47, 38, 12),
                    (76, 7, 64),
                    (30, 48, 13),
                    (87, 4, 68),
                    (59, 8, 73),
                    (29, 5, 55),
                    (83, 69, 6),
                    (74, 27, 6),
                    (8, 94, 70),
                    (77, 67, 4),
                    (63, 38, 8),
                    (82, 59, 7),
                    (84, 11, 70),
                ],
                3,
                3,
                128,
                [(8, 3, 3) for _ in range(13)],
                [(6, 1, 3) for _ in range(13)],
                [
                    (2, 7, 3),
                    (5, 1, 9),
                    (3, 8, 4),
                    (4, 1, 5),
                    (2, 1, 8),
                    (2, 1, 16),
                    (9, 14, 2),
                    (1, 4, 1),
                    (1, 5, 1),
                    (5, 9, 1),
                    (6, 1, 2),
                    (1, 2, 1),
                    (6, 3, 10),
                ],
            ),
            (
                [(128,), (256,), (58,)],
                3,
                3,
                32,
                [(32,), (64,), (28,)],
                [(1,), (1,), (2,)],
                [(2,), (4,), (1,)],
            ),
            (
                [
                    (5, 73),
                    (26, 11),
                    (7, 37),
                    (73, 9),
                    (57, 64),
                    (44, 64),
                    (89, 15),
                    (27, 28),
                    (52, 76),
                    (14, 42),
                    (80, 55),
                ],
                3,
                3,
                128,
                [
                    (5, 5),
                    (18, 4),
                    (5, 27),
                    (63, 2),
                    (19, 4),
                    (39, 24),
                    (31, 13),
                    (5, 10),
                    (52, 13),
                    (14, 39),
                    (56, 9),
                ],
                [
                    (3, 4),
                    (2, 4),
                    (5, 21),
                    (16, 1),
                    (16, 3),
                    (36, 18),
                    (8, 6),
                    (4, 6),
                    (7, 13),
                    (8, 38),
                    (52, 7),
                ],
                [1 for _ in range(11)],
            ),
        ]
        for (
            token_layout_list,
            heads,
            heads_kv,
            head_dim,
            kernel_size_list,
            stride_list,
            dilation_list,
        ) in problem_sizes:
            na_dim = len(token_layout_list[0])
            for is_causal_ in product(*[[False, True] for _ in range(na_dim)]):
                is_causal = tuple(is_causal_)
                self._test_all_dtypes(
                    token_layout_list=token_layout_list,
                    heads=heads,
                    heads_kv=heads_kv,
                    head_dim=head_dim,
                    kernel_size_list=kernel_size_list,
                    stride_list=stride_list,
                    dilation_list=dilation_list,
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

            def gen_kernel_size(input_shape):
                kernel_size = tuple(random.choice(range(2, x + 1)) for x in input_shape)
                # Never generate cases that could be self attn, because reference will map them to
                # FMHA and throw off the high error thresholds in this test.
                if all(k == x for k, x in zip(kernel_size, input_shape)):
                    max_size = max(kernel_size)
                    assert max_size > 2
                    kernel_size = tuple(
                        k if k != max_size else k - 1 for k in kernel_size
                    )

                return kernel_size

            def gen_stride(kernel_size):
                return tuple(
                    random.choice(range(1, max(2, k + 1))) for k in kernel_size
                )

            def gen_dilation(token_layout_shape, kernel_size):
                return tuple(
                    random.choice(range(1, max(2, x // k + 1)))
                    for x, k in zip(token_layout_shape, kernel_size)
                )

            def reduce_params(param_list, fn):
                min_shape = param_list[0]
                for shape in param_list[1:]:
                    min_shape = tuple(fn(x, y) for x, y in zip(min_shape, shape))
                return min_shape

            kernel_size = None
            stride = 1
            dilation = 1
            is_causal = tuple(random.choice([False, True]) for _ in range(na_dim))

            # if each parameter's variability is sampled independently, the likelihood that all
            # three parameters are variable is too little (1/8).
            if random.choice([False, True]):
                var_kernel_size, var_stride, var_dilation = True, True, True
            else:
                var_kernel_size = random.choice([False, True])
                var_stride = random.choice([False, True])
                var_dilation = random.choice([False, True])

            kernel_size_list = None
            stride_list = None
            dilation_list = None

            if var_kernel_size:
                kernel_size_list = []
                for b in range(batch):
                    kernel_size_list.append(gen_kernel_size(token_layout_list[b]))
            else:
                kernel_size = gen_kernel_size(min_input_shape)

            if var_stride:
                stride_list = []
                for b in range(batch):
                    kernel_size_batch = (
                        kernel_size if kernel_size_list is None else kernel_size_list[b]
                    )
                    stride_list.append(gen_stride(kernel_size_batch))
            else:
                min_kernel_size = (
                    kernel_size
                    if kernel_size_list is None
                    else reduce_params(kernel_size_list, fn=min)
                )
                stride = gen_stride(min_kernel_size)

            if var_dilation:
                dilation_list = []
                for b in range(batch):
                    kernel_size = (
                        kernel_size if kernel_size_list is None else kernel_size_list[b]
                    )
                    dilation_list.append(
                        gen_dilation(token_layout_list[b], kernel_size)
                    )
            else:
                max_kernel_size = (
                    kernel_size
                    if kernel_size_list is None
                    else reduce_params(kernel_size_list, fn=max)
                )
                dilation = gen_dilation(min_input_shape, max_kernel_size)

            self._test_all_dtypes(
                token_layout_list=token_layout_list,
                heads=heads,
                heads_kv=heads_kv,
                head_dim=head_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                #
                kernel_size_list=kernel_size_list,
                stride_list=stride_list,
                dilation_list=dilation_list,
                #
                n_configs_to_test=configs_to_test,
            )

    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_rand_sweep_quick(self):
        self._test_rand_sweep(3, max_tests=10, configs_to_test=3)
        self._test_rand_sweep(2, max_tests=10, configs_to_test=3)
        self._test_rand_sweep(1, max_tests=10, configs_to_test=3)

    @skip_if_not_running_extended_tests()
    @skip_if_libnatten_is_not_supported()
    @skip_if_blackwell_kernels_not_supported()
    def test_rand_sweep_extended(self):
        self._test_rand_sweep(3, max_tests=RAND_SWEEP_TESTS, configs_to_test=10)
        self._test_rand_sweep(2, max_tests=RAND_SWEEP_TESTS, configs_to_test=10)
        self._test_rand_sweep(1, max_tests=RAND_SWEEP_TESTS, configs_to_test=10)


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    unittest.main()
