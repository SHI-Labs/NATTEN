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

import math
import os
import unittest

import natten

import torch
from natten.utils.testing import (
    skip_if_cuda_is_not_supported,
    skip_if_experimental_ops_are_not_supported,
    skip_if_fna_is_not_supported,
    skip_if_fvcore_is_not_available,
    skip_if_torch_flop_count_is_not_supported,
)


def _reset_everything():
    natten.use_tiled_na()
    natten.use_gemm_na()
    natten.use_tf32_in_gemm_na()
    natten.use_fused_na(False, kv_parallel=False)
    os.environ["NATTEN_LOG_LEVEL"] = "CRITICAL"


class FlopCounterTests(unittest.TestCase):
    def setUp(self):
        _reset_everything()

    def tearDown(self):
        _reset_everything()

    def _build_model(
        self,
        dim_per_head,
        heads,
        kernel_size,
        dilation,
        is_causal,
        qkv_bias,
        use_experimental_ops,
    ):
        mod = natten.NeighborhoodAttention1D
        if hasattr(kernel_size, "__len__") and len(kernel_size) == 2:
            mod = natten.NeighborhoodAttention2D
        elif hasattr(kernel_size, "__len__") and len(kernel_size) == 3:
            mod = natten.NeighborhoodAttention3D

        return mod(
            dim=dim_per_head * heads,
            num_heads=heads,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            qkv_bias=qkv_bias,
            use_experimental_ops=use_experimental_ops,
        )

    def _build_input(self, batch, spatial_extent, heads, dim_per_head):
        shape = [batch, *spatial_extent, heads * dim_per_head]
        return torch.randn(shape)

    def _compute_flops(
        self,
        batch,
        spatial_extent,
        heads,
        dim_per_head,
        kernel_size,
        dilation,
        is_causal,
        qkv_bias,
        return_macs,
    ):
        # FVCore reports MACs, not FLOPs
        # PT reports FLOPs, not MACs
        c = 1 if return_macs else 2

        qkv_M, qkv_N, qkv_K = (
            batch * math.prod(spatial_extent),
            dim_per_head * heads * 3,
            dim_per_head * heads,
        )
        qkv_flops = qkv_M * qkv_N * qkv_K * c
        # FVCore doesn't count bias FLOPs
        # if qkv_bias:
        #     qkv_flops += qkv_M * qkv_N

        proj_M, proj_N, proj_K = (
            batch * math.prod(spatial_extent),
            dim_per_head * heads,
            dim_per_head * heads,
        )
        proj_flops = proj_M * proj_N * proj_K * c

        attn_0_M, attn_0_N, attn_0_K = (
            batch * heads * math.prod(spatial_extent),
            math.prod(kernel_size),
            dim_per_head,
        )
        attn_1_M, attn_1_N, attn_1_K = (
            batch * heads * math.prod(spatial_extent),
            dim_per_head,
            math.prod(kernel_size),
        )

        attn_0_flops = attn_0_M * attn_0_N * attn_0_K * c
        attn_1_flops = attn_1_M * attn_1_N * attn_1_K * c

        attn_flops = attn_0_flops + attn_1_flops

        return qkv_flops + attn_flops + proj_flops

    def _test_natten_flops_with_fvcore(
        self,
        batch,
        spatial_extent,
        heads,
        dim_per_head,
        kernel_size,
        dilation,
        is_causal,
        qkv_bias,
        run_on_cuda=False,
    ):
        try:
            from natten.flops import get_flops
        except ImportError:
            raise ImportError(
                "FVCore not found, but related tests are still being run. "
                "Did you forget to use @skip_if_fvcore_is_not_available ?"
            )
        m = self._build_model(
            dim_per_head=dim_per_head,
            heads=heads,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            qkv_bias=qkv_bias,
            use_experimental_ops=False,
        )
        x = self._build_input(
            batch=batch,
            spatial_extent=spatial_extent,
            heads=heads,
            dim_per_head=dim_per_head,
        )
        if run_on_cuda:
            m = m.cuda()
            x = x.cuda()

        natten_flops = get_flops(m, x)

        reference_flops = self._compute_flops(
            batch=batch,
            spatial_extent=spatial_extent,
            heads=heads,
            dim_per_head=dim_per_head,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            qkv_bias=qkv_bias,
            return_macs=True,  # fvcore reports MACs
        )

        assert reference_flops == natten_flops, (
            "FLOPs are incorrect. "
            + f"Expected {reference_flops} FLOPs, FVCore computed {natten_flops} FLOPs. "
            + f"Difference: {abs(reference_flops - natten_flops)} FLOPs. "
        )

    def _test_natten_flops_with_torch(
        self,
        batch,
        spatial_extent,
        heads,
        dim_per_head,
        kernel_size,
        dilation,
        is_causal,
        qkv_bias,
        run_on_cuda=False,
    ):
        try:
            from torch.utils.flop_counter import FlopCounterMode
        except ImportError:
            raise ImportError(
                "Torch flop utilities not found, but related tests are still being run. "
                "Did you forget to use @skip_if_experimental_ops_are_not_supported ?"
            )
        m = self._build_model(
            dim_per_head=dim_per_head,
            heads=heads,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            qkv_bias=qkv_bias,
            use_experimental_ops=True,  # Experimental ops required for PT's FLOP counter to work
        )
        x = self._build_input(
            batch=batch,
            spatial_extent=spatial_extent,
            heads=heads,
            dim_per_head=dim_per_head,
        )
        if run_on_cuda:
            m = m.cuda()
            x = x.cuda()

        flop_counter = FlopCounterMode(display=False)

        with flop_counter:
            _ = m(x)

        natten_flops = flop_counter.get_total_flops()

        reference_flops = self._compute_flops(
            batch=batch,
            spatial_extent=spatial_extent,
            heads=heads,
            dim_per_head=dim_per_head,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            qkv_bias=qkv_bias,
            return_macs=False,  # PT reports FLOPs
        )

        assert reference_flops == natten_flops, (
            "FLOPs are incorrect. "
            + f"Expected {reference_flops} FLOPs, FVCore computed {natten_flops} FLOPs. "
            + f"Difference: {abs(reference_flops - natten_flops)} FLOPs. "
        )

    @skip_if_fvcore_is_not_available()
    def test_fvcore_flops_unfused(self):
        natten.use_fused_na(False, kv_parallel=False)

        self._test_natten_flops_with_fvcore(
            batch=4,
            spatial_extent=(128,),
            heads=4,
            dim_per_head=64,
            kernel_size=(5,),
            dilation=(2,),
            is_causal=(False,),
            qkv_bias=False,
        )

        self._test_natten_flops_with_fvcore(
            batch=1,
            spatial_extent=(16, 10),
            heads=2,
            dim_per_head=32,
            kernel_size=(5, 3),
            dilation=(2, 1),
            is_causal=(False, False),
            qkv_bias=True,
        )

        self._test_natten_flops_with_fvcore(
            batch=1,
            spatial_extent=(6, 5, 8),
            heads=2,
            dim_per_head=32,
            kernel_size=(5, 3, 7),
            dilation=(1, 1, 1),
            is_causal=(True, False, False),
            qkv_bias=True,
        )

    @skip_if_fvcore_is_not_available()
    @skip_if_cuda_is_not_supported()
    @skip_if_fna_is_not_supported()
    def test_fvcore_flops_fna(self):
        natten.use_fused_na(True, kv_parallel=True)

        self._test_natten_flops_with_fvcore(
            batch=4,
            spatial_extent=(128,),
            heads=4,
            dim_per_head=64,
            kernel_size=(5,),
            dilation=(2,),
            is_causal=(False,),
            qkv_bias=False,
            run_on_cuda=True,
        )

        self._test_natten_flops_with_fvcore(
            batch=1,
            spatial_extent=(16, 10),
            heads=2,
            dim_per_head=32,
            kernel_size=(5, 3),
            dilation=(2, 1),
            is_causal=(False, False),
            qkv_bias=True,
            run_on_cuda=True,
        )

        self._test_natten_flops_with_fvcore(
            batch=1,
            spatial_extent=(6, 5, 8),
            heads=2,
            dim_per_head=32,
            kernel_size=(5, 3, 7),
            dilation=(1, 1, 1),
            is_causal=(True, False, False),
            qkv_bias=True,
            run_on_cuda=True,
        )

    # Expected failure since experimental ops only include FNA for now
    @unittest.expectedFailure
    @skip_if_torch_flop_count_is_not_supported()
    @skip_if_experimental_ops_are_not_supported()
    def test_torch_flops_unfused(self):
        natten.use_fused_na(False, kv_parallel=False)

        self._test_natten_flops_with_torch(
            batch=4,
            spatial_extent=(128,),
            heads=4,
            dim_per_head=64,
            kernel_size=(5,),
            dilation=(2,),
            is_causal=(False,),
            qkv_bias=False,
        )

        self._test_natten_flops_with_torch(
            batch=1,
            spatial_extent=(16, 10),
            heads=2,
            dim_per_head=32,
            kernel_size=(5, 3),
            dilation=(2, 1),
            is_causal=(False, False),
            qkv_bias=True,
        )

        self._test_natten_flops_with_torch(
            batch=1,
            spatial_extent=(6, 5, 8),
            heads=2,
            dim_per_head=32,
            kernel_size=(5, 3, 7),
            dilation=(1, 1, 1),
            is_causal=(True, False, False),
            qkv_bias=True,
        )

    @skip_if_torch_flop_count_is_not_supported()
    @skip_if_experimental_ops_are_not_supported()
    @skip_if_cuda_is_not_supported()
    @skip_if_fna_is_not_supported()
    def test_torch_flops_fna(self):
        natten.use_fused_na(True, kv_parallel=True)

        self._test_natten_flops_with_torch(
            batch=4,
            spatial_extent=(128,),
            heads=4,
            dim_per_head=64,
            kernel_size=(5,),
            dilation=(2,),
            is_causal=(False,),
            qkv_bias=False,
            run_on_cuda=True,
        )

        self._test_natten_flops_with_torch(
            batch=1,
            spatial_extent=(16, 10),
            heads=2,
            dim_per_head=32,
            kernel_size=(5, 3),
            dilation=(2, 1),
            is_causal=(False, False),
            qkv_bias=True,
            run_on_cuda=True,
        )

        self._test_natten_flops_with_torch(
            batch=1,
            spatial_extent=(6, 5, 8),
            heads=2,
            dim_per_head=32,
            kernel_size=(5, 3, 7),
            dilation=(1, 1, 1),
            is_causal=(True, False, False),
            qkv_bias=True,
            run_on_cuda=True,
        )


if __name__ == "__main__":
    unittest.main()
