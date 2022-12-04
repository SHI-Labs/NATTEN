"""
Neighborhood Attention 2D Unit Tests 

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
import os
import unittest

import torch
from torch.autograd import gradcheck
from torch.utils.cpp_extension import CUDA_HOME

from natten import NeighborhoodAttention2D
from natten.functional import natten2dav, natten2dqkrpb

HAS_CUDA = (
    torch.cuda.is_available()
    and (CUDA_HOME is not None)
    or os.getenv("FORCE_CUDA", "0") == "1"
)
logger = logging.getLogger(__name__)


def _priv_test_gradcheck_natten2dqk(
    batch_size,
    heads,
    height,
    width,
    dim,
    kernel_size,
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
    query = torch.randn((batch_size, heads, height, width, dim), **kwargs)
    key = torch.randn((batch_size, heads, height, width, dim), **kwargs)
    rpb = None
    variables = [query, key, rpb, kernel_size, dilation]

    assert gradcheck(
        natten2dqkrpb,
        variables,
        eps=eps,
        atol=atol,
        rtol=rtol,
        nondet_tol=ndtol,
        fast_mode=fast_mode,
    ), f"Gradcheck failed for `NATTEN2D` with {dtype} on {device}, {md} MODE."


def _priv_test_gradcheck_natten2dqkrpb(
    batch_size,
    heads,
    height,
    width,
    dim,
    kernel_size,
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
    query = torch.randn((batch_size, heads, height, width, dim), **kwargs)
    key = torch.randn((batch_size, heads, height, width, dim), **kwargs)
    rpb = torch.randn((heads, 2 * kernel_size - 1, 2 * kernel_size - 1), **kwargs)
    variables = [query, key, rpb, kernel_size, dilation]

    assert gradcheck(
        natten2dqkrpb,
        variables,
        eps=eps,
        atol=atol,
        rtol=rtol,
        nondet_tol=ndtol,
        fast_mode=fast_mode,
    ), f"Gradcheck failed for `NATTEN2D-QKRPB` with {dtype} on {device}, {md} MODE."


def _priv_test_gradcheck_natten2dav(
    batch_size,
    heads,
    height,
    width,
    dim,
    kernel_size,
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
    attn = torch.randn((batch_size, heads, height, width, kernel_size**2), **kwargs)
    value = torch.randn((batch_size, heads, height, width, dim), **kwargs)
    variables = [attn, value, dilation]

    assert gradcheck(
        natten2dav,
        variables,
        eps=eps,
        atol=atol,
        rtol=rtol,
        nondet_tol=ndtol,
        fast_mode=fast_mode,
    ), f"Gradcheck failed for `NATTEN2D-AV` with {dtype} on {device}, {md} MODE."


def _priv_test_allclose_cpu_cuda(
    batch_size,
    height,
    width,
    kernel_sizes=[3, 5, 7, 9],
    dims=[4, 8],
    heads=[1, 2, 3],
    tol=1e-8,
):
    for kernel_size in kernel_sizes:
        for dim in dims:
            for num_heads in heads:
                for rpb in [True, False]:
                    model_kwargs = {
                        "dim": dim * num_heads,
                        "kernel_size": kernel_size,
                        "num_heads": num_heads,
                        "qkv_bias": True,
                        "bias": rpb,
                    }

                    base_state_dict = NeighborhoodAttention2D(
                        **model_kwargs
                    ).state_dict()

                    x1 = torch.randn((batch_size, height, width, dim * num_heads))
                    x2 = x1.clone().detach().cuda(0)

                    nat1 = NeighborhoodAttention2D(**model_kwargs).eval()
                    nat1.load_state_dict(base_state_dict, strict=False)

                    nat2 = NeighborhoodAttention2D(**model_kwargs).cuda(0).eval()
                    nat2.load_state_dict(base_state_dict, strict=False)

                    y1 = nat1(x1)
                    y2 = nat2(x2)

                    forward_mse = ((y1.data - y2.cpu().data) ** 2).mean()

                    assert forward_mse < tol, (
                        f"FAIL: Forward MSE ({forward_mse}) was above the specified"
                        f" tolerance (tol) for heads={heads}, dim={dim},"
                        f" kernel_size={kernel_size}."
                    )

                    y1.sum().backward()
                    y2.sum().backward()

                    for name, n1w in nat1.named_modules():
                        if type(n1w) is not torch.nn.Linear:
                            continue
                        for name2, n2w in nat2.named_modules():
                            if name != name2:
                                continue
                            if n1w.weight.grad is None or n2w.weight.grad is None:
                                continue
                            mse = (
                                (n1w.weight.grad - n2w.weight.grad.cpu()) ** 2
                            ).mean()
                            if hasattr(n1w, "bias") and n1w.bias is not None:
                                if hasattr(n1w.bias, "grad") and hasattr(
                                    n2w.bias, "grad"
                                ):
                                    mse += (
                                        (n1w.bias.grad - n2w.bias.grad.cpu()) ** 2
                                    ).mean()

                            assert mse < tol, (
                                f"FAIL: {name} gradient MSE ({mse}) was above the"
                                f" specified tolerance ({tol}) for heads={heads},"
                                f" dim={dim}, kernel_size={kernel_size}."
                            )


class NA2DTest(unittest.TestCase):
    def test_natten2dqk_gradcheck_cpu_slow(self):
        b, h, li, lj, d, k, di = 1, 1, 14, 14, 4, 7, 2
        _priv_test_gradcheck_natten2dqk(
            b, h, li, lj, d, k, di, torch.float64, "cpu", 1e-6, 1e-5, 1e-3, 0, False
        )

    def test_natten2dqk_gradcheck_cpu_fast(self):
        b, h, li, lj, d, k, di = 1, 1, 14, 14, 32, 7, 2
        _priv_test_gradcheck_natten2dqk(
            b, h, li, lj, d, k, di, torch.float64, "cpu", 1e-6, 1e-5, 1e-3, 0, True
        )

    def test_natten2dqk_gradcheck_cuda_slow(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 14, 14, 4, 7, 2
        _priv_test_gradcheck_natten2dqk(
            b, h, li, lj, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, False
        )

    def test_natten2dqk_gradcheck_cuda_fast(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 14, 14, 32, 7, 2
        _priv_test_gradcheck_natten2dqk(
            b, h, li, lj, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, True
        )

    def test_natten2dqkrpb_gradcheck_cpu_slow(self):
        b, h, li, lj, d, k, di = 1, 1, 14, 14, 4, 7, 2
        _priv_test_gradcheck_natten2dqkrpb(
            b, h, li, lj, d, k, di, torch.float64, "cpu", 1e-6, 1e-5, 1e-3, 0, False
        )

    def test_natten2dqkrpb_gradcheck_cpu_fast(self):
        b, h, li, lj, d, k, di = 1, 1, 14, 14, 32, 7, 2
        _priv_test_gradcheck_natten2dqkrpb(
            b, h, li, lj, d, k, di, torch.float64, "cpu", 1e-6, 1e-5, 1e-3, 0, True
        )

    def test_natten2dqkrpb_gradcheck_cuda_slow(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 14, 14, 4, 7, 2
        _priv_test_gradcheck_natten2dqkrpb(
            b, h, li, lj, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, False
        )

    def test_natten2dqkrpb_gradcheck_cuda_fast(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 14, 14, 32, 7, 2
        _priv_test_gradcheck_natten2dqkrpb(
            b, h, li, lj, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, True
        )

    def test_natten2dav_gradcheck_cpu_slow(self):
        b, h, li, lj, d, k, di = 1, 1, 14, 14, 4, 7, 2
        _priv_test_gradcheck_natten2dav(
            b, h, li, lj, d, k, di, torch.float64, "cpu", 1e-6, 1e-5, 1e-3, 0, False
        )

    def test_natten2dav_gradcheck_cpu_fast(self):
        b, h, li, lj, d, k, di = 1, 1, 14, 14, 32, 7, 2
        _priv_test_gradcheck_natten2dav(
            b, h, li, lj, d, k, di, torch.float64, "cpu", 1e-6, 1e-5, 1e-3, 0, True
        )

    def test_natten2dav_gradcheck_cuda_slow(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 14, 14, 4, 7, 2
        _priv_test_gradcheck_natten2dav(
            b, h, li, lj, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, False
        )

    def test_natten2dav_gradcheck_cuda_fast(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 14, 14, 32, 7, 2
        _priv_test_gradcheck_natten2dav(
            b, h, li, lj, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, True
        )

    def test_cpu_cuda_allclose(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, li, lj = 4, 14, 16
        _priv_test_allclose_cpu_cuda(b, li, lj)

    def test_natten2dqkrpb_tiled3x3_gradcheck_cuda(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 6, 6, 32, 3, 2
        _priv_test_gradcheck_natten2dqkrpb(
            b, h, li, lj, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, False
        )

    def test_natten2dav_tiled3x3_gradcheck_cuda(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 6, 6, 32, 3, 2
        _priv_test_gradcheck_natten2dav(
            b, h, li, lj, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, False
        )

    def test_natten2dqkrpb_tiled5x5_gradcheck_cuda(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 10, 10, 32, 5, 2
        _priv_test_gradcheck_natten2dqkrpb(
            b, h, li, lj, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, False
        )

    def test_natten2dav_tiled5x5_gradcheck_cuda(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 10, 10, 32, 5, 2
        _priv_test_gradcheck_natten2dav(
            b, h, li, lj, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, False
        )

    def test_natten2dqkrpb_tiled7x7_gradcheck_cuda(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 14, 14, 32, 7, 2
        _priv_test_gradcheck_natten2dqkrpb(
            b, h, li, lj, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, False
        )

    def test_natten2dav_tiled7x7_gradcheck_cuda(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 14, 14, 32, 7, 2
        _priv_test_gradcheck_natten2dav(
            b, h, li, lj, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, False
        )

    def test_natten2dqkrpb_tiled9x9_gradcheck_cuda(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 18, 18, 32, 9, 2
        _priv_test_gradcheck_natten2dqkrpb(
            b, h, li, lj, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, False
        )

    def test_natten2dav_tiled9x9_gradcheck_cuda(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 18, 18, 32, 9, 2
        _priv_test_gradcheck_natten2dav(
            b, h, li, lj, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, False
        )

    def test_natten2dqkrpb_tiled11x11_gradcheck_cuda(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 22, 22, 32, 11, 2
        # TODO: Disable FAST MODE
        # Presently we do fast mode because otherwise this test will throw an OOM
        # Tested on an 80GB A100
        _priv_test_gradcheck_natten2dqkrpb(
            b,
            h,
            li,
            lj,
            d,
            k,
            di,
            torch.float64,
            "cuda",
            1e-6,
            1e-5,
            1e-3,
            1e-8,
            # False
            True,
        )

    def test_natten2dav_tiled11x11_gradcheck_cuda(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 22, 22, 32, 11, 2
        # TODO: Disable FAST MODE
        # Presently we do fast mode because otherwise this test will throw an OOM
        # Tested on an 80GB A100
        _priv_test_gradcheck_natten2dav(
            b,
            h,
            li,
            lj,
            d,
            k,
            di,
            torch.float64,
            "cuda",
            1e-6,
            1e-5,
            1e-3,
            1e-8,
            # False
            True,
        )

    def test_natten2dqkrpb_tiled13x13_gradcheck_cuda(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 26, 26, 32, 13, 2
        # TODO: Disable FAST MODE
        # Presently we do fast mode because otherwise this test will throw an OOM
        # Tested on an 80GB A100
        _priv_test_gradcheck_natten2dqkrpb(
            b,
            h,
            li,
            lj,
            d,
            k,
            di,
            torch.float64,
            "cuda",
            1e-6,
            1e-5,
            1e-3,
            1e-8,
            # False
            True,
        )

    def test_natten2dav_tiled13x13_gradcheck_cuda(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, li, lj, d, k, di = 1, 1, 26, 26, 32, 13, 2
        # TODO: Disable FAST MODE
        # Presently we do fast mode because otherwise this test will throw an OOM
        # Tested on an 80GB A100
        _priv_test_gradcheck_natten2dav(
            b,
            h,
            li,
            lj,
            d,
            k,
            di,
            torch.float64,
            "cuda",
            1e-6,
            1e-5,
            1e-3,
            1e-8,
            # False
            True,
        )


if __name__ == "__main__":
    unittest.main()
