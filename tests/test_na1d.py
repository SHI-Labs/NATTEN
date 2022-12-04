"""
Neighborhood Attention 1D Unit Tests 

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import logging
import os
import unittest

import torch
from torch.autograd import gradcheck
from torch.utils.cpp_extension import CUDA_HOME

from natten import NeighborhoodAttention1D
from natten.functional import natten1dav, natten1dqkrpb

HAS_CUDA = (
    torch.cuda.is_available()
    and (CUDA_HOME is not None)
    or os.getenv("FORCE_CUDA", "0") == "1"
)
logger = logging.getLogger(__name__)


def _priv_test_gradcheck_natten1dqk(
    batch_size,
    heads,
    length,
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
    query = torch.randn((batch_size, heads, length, dim), **kwargs)
    key = torch.randn((batch_size, heads, length, dim), **kwargs)
    rpb = None
    variables = [query, key, rpb, kernel_size, dilation]

    assert gradcheck(
        natten1dqkrpb,
        variables,
        eps=eps,
        atol=atol,
        rtol=rtol,
        nondet_tol=ndtol,
        fast_mode=fast_mode,
    ), f"Gradcheck failed for `NATTEN1D-QK` with {dtype} on {device}, {md} MODE."


def _priv_test_gradcheck_natten1dqkrpb(
    batch_size,
    heads,
    length,
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
    query = torch.randn((batch_size, heads, length, dim), **kwargs)
    key = torch.randn((batch_size, heads, length, dim), **kwargs)
    rpb = torch.randn((heads, 2 * kernel_size - 1), **kwargs)
    variables = [query, key, rpb, kernel_size, dilation]

    assert gradcheck(
        natten1dqkrpb,
        variables,
        eps=eps,
        atol=atol,
        rtol=rtol,
        nondet_tol=ndtol,
        fast_mode=fast_mode,
    ), f"Gradcheck failed for `NATTEN1D-QKRPB` with {dtype} on {device}, {md} MODE."


def _priv_test_gradcheck_natten1dav(
    batch_size,
    heads,
    length,
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
    attn = torch.randn((batch_size, heads, length, kernel_size), **kwargs)
    value = torch.randn((batch_size, heads, length, dim), **kwargs)
    variables = [attn, value, dilation]

    assert gradcheck(
        natten1dav,
        variables,
        eps=eps,
        atol=atol,
        rtol=rtol,
        nondet_tol=ndtol,
        fast_mode=fast_mode,
    ), f"Gradcheck failed for `NATTEN1D-AV` with {dtype} on {device}, {md} MODE."


def _priv_test_allclose_cpu_cuda(
    batch_size,
    length,
    kernel_sizes=[3, 5, 7, 9, 11, 13],
    dims=[4, 16, 32],
    heads=[1, 2, 3, 4],
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

                    base_state_dict = NeighborhoodAttention1D(
                        **model_kwargs
                    ).state_dict()

                    x1 = torch.randn((batch_size, length, dim * num_heads))
                    x2 = x1.clone().detach().cuda(0)

                    nat1 = NeighborhoodAttention1D(**model_kwargs).eval()
                    nat1.load_state_dict(base_state_dict, strict=False)

                    nat2 = NeighborhoodAttention1D(**model_kwargs).cuda(0).eval()
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


class NA1DTest(unittest.TestCase):
    def test_natten1dqk_gradcheck_cpu_slow(self):
        b, h, l, d, k, di = 1, 2, 14, 4, 7, 2
        _priv_test_gradcheck_natten1dqk(
            b, h, l, d, k, di, torch.float64, "cpu", 1e-6, 1e-5, 1e-3, 0, False
        )

    def test_natten1dqk_gradcheck_cpu_fast(self):
        b, h, l, d, k, di = 1, 2, 14, 32, 7, 2
        _priv_test_gradcheck_natten1dqk(
            b, h, l, d, k, di, torch.float64, "cpu", 1e-6, 1e-5, 1e-3, 0, True
        )

    def test_natten1dqk_gradcheck_cuda_slow(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, l, d, k, di = 1, 2, 14, 4, 7, 2
        _priv_test_gradcheck_natten1dqk(
            b, h, l, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, False
        )

    def test_natten1dqk_gradcheck_cuda_fast(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, l, d, k, di = 1, 2, 14, 32, 7, 2
        _priv_test_gradcheck_natten1dqk(
            b, h, l, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, True
        )

    def test_natten1dqkrpb_gradcheck_cpu_slow(self):
        b, h, l, d, k, di = 1, 2, 14, 4, 7, 2
        _priv_test_gradcheck_natten1dqkrpb(
            b, h, l, d, k, di, torch.float64, "cpu", 1e-6, 1e-5, 1e-3, 0, False
        )

    def test_natten1dqkrpb_gradcheck_cpu_fast(self):
        b, h, l, d, k, di = 1, 2, 14, 32, 7, 2
        _priv_test_gradcheck_natten1dqkrpb(
            b, h, l, d, k, di, torch.float64, "cpu", 1e-6, 1e-5, 1e-3, 0, True
        )

    def test_natten1dqkrpb_gradcheck_cuda_slow(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, l, d, k, di = 1, 2, 14, 4, 7, 2
        _priv_test_gradcheck_natten1dqkrpb(
            b, h, l, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, False
        )

    def test_natten1dqkrpb_gradcheck_cuda_fast(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, l, d, k, di = 1, 2, 14, 32, 7, 2
        _priv_test_gradcheck_natten1dqkrpb(
            b, h, l, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, True
        )

    def test_natten1dav_gradcheck_cpu_slow(self):
        b, h, l, d, k, di = 1, 2, 14, 4, 7, 2
        _priv_test_gradcheck_natten1dav(
            b, h, l, d, k, di, torch.float64, "cpu", 1e-6, 1e-5, 1e-3, 0, False
        )

    def test_natten1dav_gradcheck_cpu_fast(self):
        b, h, l, d, k, di = 1, 2, 14, 32, 7, 2
        _priv_test_gradcheck_natten1dav(
            b, h, l, d, k, di, torch.float64, "cpu", 1e-6, 1e-5, 1e-3, 0, True
        )

    def test_natten1dav_gradcheck_cuda_slow(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, l, d, k, di = 1, 2, 14, 4, 7, 2
        _priv_test_gradcheck_natten1dav(
            b, h, l, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, False
        )

    def test_natten1dav_gradcheck_cuda_fast(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, h, l, d, k, di = 1, 2, 14, 32, 7, 2
        _priv_test_gradcheck_natten1dav(
            b, h, l, d, k, di, torch.float64, "cuda", 1e-6, 1e-5, 1e-3, 1e-8, True
        )

    def test_cpu_cuda_allclose(self):
        if not HAS_CUDA:
            self.skipTest("CUDA not available.")
        b, l = 4, 16
        _priv_test_allclose_cpu_cuda(b, l)


if __name__ == "__main__":
    unittest.main()
