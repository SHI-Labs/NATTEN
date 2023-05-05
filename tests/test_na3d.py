#################################################################################################
# Copyright (c) 2023 Ali Hassani.
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

import logging
import os
import unittest

import torch
from torch.autograd import gradcheck
from torch.utils.cpp_extension import CUDA_HOME

from natten import NeighborhoodAttention3D
from natten.functional import natten3dav, natten3dqkrpb

HAS_CUDA = (
    torch.cuda.is_available()
    and (CUDA_HOME is not None)
    or os.getenv("FORCE_CUDA", "0") == "1"
)
logger = logging.getLogger(__name__)


def _priv_test_gradcheck_natten3dqk(
    batch_size,
    heads,
    depth,
    height,
    width,
    dim,
    kernel_size_d,
    kernel_size,
    dilation_d,
    dilation,
    dtype,
    device,
    eps,
    atol,
    rtol,
    ndtol,
    fast_mode,
):
    torch.manual_seed(42)
    md = "FAST" if fast_mode else "SLOW"
    kwargs = {"dtype": dtype, "device": device, "requires_grad": True}
    query = torch.randn((batch_size, heads, depth, height, width, dim), **kwargs)
    key = torch.randn((batch_size, heads, depth, height, width, dim), **kwargs)
    rpb = None
    variables = [query, key, rpb, kernel_size_d, kernel_size, dilation_d, dilation]

    assert gradcheck(
        natten3dqkrpb,
        variables,
        eps=eps,
        atol=atol,
        rtol=rtol,
        nondet_tol=ndtol,
        fast_mode=fast_mode,
    ), f"Gradcheck failed for `NATTEN3D-QK` with {dtype} on {device}, {md} MODE."


def _priv_test_gradcheck_natten3dqkrpb(
    batch_size,
    heads,
    depth,
    height,
    width,
    dim,
    kernel_size_d,
    kernel_size,
    dilation_d,
    dilation,
    dtype,
    device,
    eps,
    atol,
    rtol,
    ndtol,
    fast_mode,
):
    torch.manual_seed(42)
    md = "FAST" if fast_mode else "SLOW"
    kwargs = {"dtype": dtype, "device": device, "requires_grad": True}
    query = torch.randn((batch_size, heads, depth, height, width, dim), **kwargs)
    key = torch.randn((batch_size, heads, depth, height, width, dim), **kwargs)
    rpb = torch.randn(
        (heads, 2 * kernel_size_d - 1, 2 * kernel_size - 1, 2 * kernel_size - 1),
        **kwargs,
    )
    variables = [query, key, rpb, kernel_size_d, kernel_size, dilation_d, dilation]

    assert gradcheck(
        natten3dqkrpb,
        variables,
        eps=eps,
        atol=atol,
        rtol=rtol,
        nondet_tol=ndtol,
        fast_mode=fast_mode,
    ), f"Gradcheck failed for `NATTEN3D-QKRPB` with {dtype} on {device}, {md} MODE."


def _priv_test_gradcheck_natten3dav(
    batch_size,
    heads,
    depth,
    height,
    width,
    dim,
    kernel_size_d,
    kernel_size,
    dilation_d,
    dilation,
    dtype,
    device,
    eps,
    atol,
    rtol,
    ndtol,
    fast_mode,
):
    torch.manual_seed(42)
    md = "FAST" if fast_mode else "SLOW"
    kwargs = {"dtype": dtype, "device": device, "requires_grad": True}
    attn = torch.randn(
        (
            batch_size,
            heads,
            depth,
            height,
            width,
            kernel_size_d * kernel_size * kernel_size,
        ),
        **kwargs,
    )
    value = torch.randn((batch_size, heads, depth, height, width, dim), **kwargs)
    variables = [attn, value, kernel_size_d, kernel_size, dilation_d, dilation]

    assert gradcheck(
        natten3dav,
        variables,
        eps=eps,
        atol=atol,
        rtol=rtol,
        nondet_tol=ndtol,
        fast_mode=fast_mode,
    ), f"Gradcheck failed for `NATTEN3D-AV` with {dtype} on {device}, {md} MODE."


def _priv_test_allclose_cpu_cuda(
    batch_size,
    depth,
    height,
    width,
    depth_kernel_sizes=[3, 5, 7, 9, 11, 13],
    kernel_sizes=[3, 5, 7, 9, 11, 13],
    dims=[2, 4, 8],
    heads=[1, 2, 4],
    tol=1e-6,
):
    for kernel_size_d in depth_kernel_sizes:
        for kernel_size in kernel_sizes:
            for dim in dims:
                for num_heads in heads:
                    for rpb in [True, False]:
                        model_kwargs = {
                            "dim": dim * num_heads,
                            "kernel_size_d": kernel_size_d,
                            "kernel_size": kernel_size,
                            "num_heads": num_heads,
                            "qkv_bias": True,
                            "bias": rpb,
                        }

                        base_state_dict = NeighborhoodAttention3D(
                            **model_kwargs
                        ).state_dict()

                        x1 = torch.randn(
                            (batch_size, depth, height, width, dim * num_heads)
                        )
                        x2 = x1.clone().detach().cuda(0)

                        nat1 = NeighborhoodAttention3D(**model_kwargs).eval()
                        nat1.load_state_dict(base_state_dict, strict=False)

                        nat2 = NeighborhoodAttention3D(**model_kwargs).cuda(0).eval()
                        nat2.load_state_dict(base_state_dict, strict=False)

                        y1 = nat1(x1)
                        y2 = nat2(x2)

                        forward_mse = ((y1.data - y2.cpu().data) ** 2).mean()

                        assert forward_mse < tol, (
                            f"FAIL: Forward MSE ({forward_mse}) was above the specified"
                            f" tolerance (tol) for heads={num_heads}, dim={dim},"
                            f" kernel_size_d={kernel_size_d},"
                            f" kernel_size={kernel_size}, rpb={rpb}."
                        )


class NA3DTest(unittest.TestCase):
    def test_natten3dqk_gradcheck_cpu_fast(self):
        (
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
        ) = (1, 2, 15, 16, 17, 32, 3, 5, 2, 3)
        _priv_test_gradcheck_natten3dqk(
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
            torch.float64,
            "cpu",
            1e-6,
            1e-5,
            1e-3,
            0,
            True,
        )

    def test_natten3dqk_gradcheck_cuda_slow(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        (
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
        ) = (1, 2, 15, 16, 17, 4, 3, 5, 2, 3)
        _priv_test_gradcheck_natten3dqk(
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
            torch.float64,
            "cuda",
            1e-6,
            1e-5,
            1e-3,
            1e-8,
            False,
        )

    def test_natten3dqk_gradcheck_cuda_fast(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        (
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
        ) = (1, 2, 15, 16, 17, 32, 3, 5, 2, 3)
        _priv_test_gradcheck_natten3dqk(
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
            torch.float64,
            "cuda",
            1e-6,
            1e-5,
            1e-3,
            1e-8,
            True,
        )

    def test_natten3dqkrpb_gradcheck_cpu_fast(self):
        (
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
        ) = (1, 2, 15, 16, 17, 32, 3, 5, 2, 3)
        _priv_test_gradcheck_natten3dqkrpb(
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
            torch.float64,
            "cpu",
            1e-6,
            1e-5,
            1e-3,
            0,
            True,
        )

    def test_natten3dqkrpb_gradcheck_cuda_slow(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        (
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
        ) = (1, 2, 15, 16, 17, 4, 3, 5, 2, 3)
        _priv_test_gradcheck_natten3dqkrpb(
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
            torch.float64,
            "cuda",
            1e-6,
            1e-5,
            1e-3,
            1e-8,
            False,
        )

    def test_natten3dqkrpb_gradcheck_cuda_fast(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        (
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
        ) = (1, 2, 15, 16, 17, 32, 3, 5, 2, 3)
        _priv_test_gradcheck_natten3dqkrpb(
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
            torch.float64,
            "cuda",
            1e-6,
            1e-5,
            1e-3,
            1e-8,
            True,
        )

    def test_natten3dav_gradcheck_cpu_fast(self):
        (
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
        ) = (1, 2, 15, 16, 17, 32, 3, 5, 2, 3)
        _priv_test_gradcheck_natten3dav(
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
            torch.float64,
            "cpu",
            1e-6,
            1e-5,
            1e-3,
            0,
            True,
        )

    def test_natten3dav_gradcheck_cuda_slow(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        (
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
        ) = (1, 2, 15, 16, 17, 4, 3, 5, 2, 3)
        _priv_test_gradcheck_natten3dav(
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
            torch.float64,
            "cuda",
            1e-6,
            1e-5,
            1e-3,
            1e-8,
            False,
        )

    def test_natten3dav_gradcheck_cuda_fast(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        (
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
        ) = (1, 2, 15, 16, 17, 32, 3, 5, 2, 3)
        _priv_test_gradcheck_natten3dav(
            batch,
            heads,
            depth,
            height,
            width,
            dim,
            kernel_d,
            kernel,
            dilation_d,
            dilation,
            torch.float64,
            "cuda",
            1e-6,
            1e-5,
            1e-3,
            1e-8,
            True,
        )

    def test_cpu_cuda_allclose(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        _priv_test_allclose_cpu_cuda(
            batch_size=2,
            depth=18,
            height=16,
            width=17,
        )


if __name__ == "__main__":
    unittest.main()
