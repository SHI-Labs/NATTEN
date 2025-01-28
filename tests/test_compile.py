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
    skip_if_experimental_ops_are_not_supported,
    skip_if_fna_is_not_supported,
    skip_if_torch_compile_is_not_supported,
)


def _reset_everything():
    natten.use_tiled_na()
    natten.use_gemm_na()
    natten.use_tf32_in_gemm_na()
    natten.use_fused_na(False, kv_parallel=False)
    os.environ["NATTEN_LOG_LEVEL"] = "CRITICAL"


class TorchCompileTests(unittest.TestCase):
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
        ).cuda()

    def _build_input(self, batch, spatial_extent, heads, dim_per_head):
        shape = [batch, *spatial_extent, heads * dim_per_head]
        return torch.randn(shape, device="cuda")

    def _test_torch_compile(
        self,
        batch,
        spatial_extent,
        heads,
        dim_per_head,
        kernel_size,
        dilation,
        is_causal,
        qkv_bias,
        use_experimental_ops,
    ):
        m = self._build_model(
            dim_per_head=dim_per_head,
            heads=heads,
            kernel_size=kernel_size,
            dilation=dilation,
            is_causal=is_causal,
            qkv_bias=qkv_bias,
            use_experimental_ops=use_experimental_ops,
        )
        m_compiled = torch.compile(m, fullgraph=True)
        x = self._build_input(
            batch=batch,
            spatial_extent=spatial_extent,
            heads=heads,
            dim_per_head=dim_per_head,
        )

        with torch.inference_mode():
            _ = m_compiled(x)

    # Expected failure since only experimental ops support torch.compile
    @unittest.expectedFailure
    @skip_if_torch_compile_is_not_supported()
    @skip_if_experimental_ops_are_not_supported()
    @skip_if_cuda_is_not_supported()
    @skip_if_fna_is_not_supported()
    def test_torch_compile_0(self):
        natten.use_fused_na(False)

        self._test_torch_compile(
            batch=4,
            spatial_extent=(128,),
            heads=4,
            dim_per_head=64,
            kernel_size=(5,),
            dilation=(2,),
            is_causal=(False,),
            qkv_bias=False,
            use_experimental_ops=False,
        )

    # Expected failure since experimental ops only include FNA for now
    @unittest.expectedFailure
    @skip_if_torch_compile_is_not_supported()
    @skip_if_experimental_ops_are_not_supported()
    @skip_if_cuda_is_not_supported()
    @skip_if_fna_is_not_supported()
    def test_torch_compile_1(self):
        natten.use_fused_na(False)

        self._test_torch_compile(
            batch=4,
            spatial_extent=(128,),
            heads=4,
            dim_per_head=64,
            kernel_size=(5,),
            dilation=(2,),
            is_causal=(False,),
            qkv_bias=False,
            use_experimental_ops=True,
        )

    # Expected failure since only experimental ops support torch.compile
    @unittest.expectedFailure
    @skip_if_torch_compile_is_not_supported()
    @skip_if_experimental_ops_are_not_supported()
    @skip_if_cuda_is_not_supported()
    @skip_if_fna_is_not_supported()
    def test_torch_compile_2(self):
        natten.use_fused_na(True, kv_parallel=True)
        self._test_torch_compile(
            batch=4,
            spatial_extent=(128,),
            heads=4,
            dim_per_head=64,
            kernel_size=(5,),
            dilation=(2,),
            is_causal=(False,),
            qkv_bias=False,
            use_experimental_ops=False,
        )

    @skip_if_torch_compile_is_not_supported()
    @skip_if_experimental_ops_are_not_supported()
    @skip_if_cuda_is_not_supported()
    @skip_if_fna_is_not_supported()
    def test_torch_compile_3(self):
        natten.use_fused_na(True, kv_parallel=True)
        self._test_torch_compile(
            batch=1,
            spatial_extent=(16, 10),
            heads=2,
            dim_per_head=32,
            kernel_size=(5, 3),
            dilation=(2, 1),
            is_causal=(False, False),
            qkv_bias=True,
            use_experimental_ops=True,
        )

        self._test_torch_compile(
            batch=1,
            spatial_extent=(6, 5, 8),
            heads=2,
            dim_per_head=32,
            kernel_size=(5, 3, 7),
            dilation=(1, 1, 1),
            is_causal=(True, False, False),
            qkv_bias=True,
            use_experimental_ops=True,
        )


if __name__ == "__main__":
    unittest.main()
