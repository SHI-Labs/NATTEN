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

import unittest
from typing import Optional

import torch
from natten._environment import _RUN_ADDITIONAL_KV_TESTS as ENABLE_ADDITIONAL_KV_TESTS
from natten.backends import get_compatible_backends, get_compatible_fmha_backends
from natten.functional import attention
from natten.modules import NeighborhoodAttentionGeneric
from natten.utils import log
from natten.utils.testing import skip_if_libnatten_is_not_supported
from natten.utils.varlen import generate_varlen_parameters
from torch import nn

from .utils import reset_torch_compile

logger = log.get_logger(__name__)


ADDITIONAL_KV_LENGTHS = [0, 64] if ENABLE_ADDITIONAL_KV_TESTS else [0]


def _reset_everything():
    reset_torch_compile(1024)
    torch.cuda.empty_cache()


class FMHABlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        qkv_bias: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.mlp_dim = int(self.embed_dim * self.mlp_ratio)
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q = nn.Linear(self.embed_dim, self.embed_dim, bias=qkv_bias)
        self.kv = nn.Linear(self.embed_dim, self.embed_dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.mlp_dim),
            nn.GELU(),
            nn.Linear(self.mlp_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, *args, **kwargs):
        B, sQ, D = x.shape
        B, sK, D = c.shape
        q = self.q(x).reshape(B, sQ, self.num_heads, self.head_dim)
        k, v = (
            self.kv(c)
            .reshape(B, sK, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 1, 3, 4)
        )

        x0 = attention(q, k, v, *args, **kwargs)
        assert isinstance(x0, torch.Tensor)
        x0 = x0.reshape(B, sQ, D)

        return self.mlp(x0)


class Block(nn.Module):
    def __init__(
        self,
        na_dim: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int,
        kernel_size,
        stride,
        dilation,
        is_causal,
        qkv_bias: bool = True,
    ):
        super().__init__()

        self.attn = NeighborhoodAttentionGeneric(
            na_dim=na_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            is_causal=is_causal,
            qkv_bias=qkv_bias,
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        additional_context: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        x0 = self.attn(x, additional_context=additional_context, *args, **kwargs)
        return self.mlp(x0)


class Model(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.layers = nn.ModuleList([*modules])

    def forward(
        self,
        x: torch.Tensor,
        additional_context: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ):
        for layer in self.layers:
            x = layer(x, additional_context=additional_context, *args, **kwargs)

        return x


class TorchCompileTests(unittest.TestCase):
    def _test_na_module(
        self,
        batch,
        token_layout_shape,
        num_heads,
        head_dim,
        kernel_size,
        dilation,
        stride,
        is_causal,
        atol,
        additional_tokens: int = 0,
    ):
        na_dim = len(token_layout_shape)

        embed_dim = num_heads * head_dim

        device = "cuda"
        dtype = torch.float16

        dummy = torch.randn(
            (batch, *token_layout_shape, num_heads, head_dim),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        # TODO: Flex FNA is also weirdly incompatible with torch compiling the model --
        # likely something to do with our mask...
        backends = [
            b
            for b in get_compatible_backends(dummy, dummy, dummy, torch_compile=False)
            if b != "flex-fna"
        ]
        for backend in backends:
            _reset_everything()

            logger.debug(
                f"Testing torch compile on {na_dim}-D module with input shapes: "
                f"{batch=}, {num_heads=}, {head_dim=}, {token_layout_shape=}, "
                f"{kernel_size=}, {dilation=}, {stride=}, {is_causal=}, "
                f"{additional_tokens=}, {dtype=}, {device=}, {backend=}."
            )

            model = (
                Model(
                    Block(
                        na_dim=na_dim,
                        embed_dim=embed_dim,
                        mlp_ratio=2,
                        num_heads=num_heads,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        stride=stride,
                        is_causal=is_causal,
                    ),
                    Block(
                        na_dim=na_dim,
                        embed_dim=embed_dim,
                        mlp_ratio=4,
                        num_heads=num_heads,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        stride=stride,
                        is_causal=is_causal,
                    ),
                    Block(
                        na_dim=na_dim,
                        embed_dim=embed_dim,
                        mlp_ratio=4,
                        num_heads=num_heads,
                        kernel_size=token_layout_shape,
                        dilation=1,
                        stride=1,
                        is_causal=False,
                    ),
                )
                .to(dtype)
                .to(device)
            )

            model_eager = model

            model_compiled = torch.compile(model, fullgraph=True, backend="inductor")

            x = torch.randn(
                (batch, *token_layout_shape, embed_dim), dtype=dtype, device=device
            )
            dy = (
                torch.randn(
                    (batch, *token_layout_shape, embed_dim), dtype=dtype, device=device
                )
                * 0.1
            )

            x_ref = x.clone().requires_grad_(True)
            dy_ref = dy.clone()

            additional_context, additional_context_ref = None, None
            if additional_tokens > 0:
                additional_context = torch.randn(
                    (batch, additional_tokens, embed_dim), dtype=dtype, device=device
                )
                additional_context_ref = additional_context.clone().requires_grad_(True)
                additional_context = additional_context.requires_grad_(True)

            # eager
            y_ref = model_eager(
                x_ref, additional_context_ref, backend=backend, torch_compile=True
            )
            y_ref.backward(dy_ref)
            dx_ref = x_ref.grad

            # compile on first attempt
            x = x.requires_grad_(True)
            y = model_compiled(
                x, additional_context, backend=backend, torch_compile=True
            )
            y.backward(dy)
            dx = x.grad

            torch.testing.assert_close(y, y_ref, atol=atol, rtol=0)
            torch.testing.assert_close(dx, dx_ref, atol=atol, rtol=0)
            if additional_tokens > 0:
                assert additional_context is not None
                assert additional_context_ref is not None
                assert additional_context.grad is not None
                assert additional_context_ref.grad is not None

                dc = additional_context.grad
                dc_ref = additional_context_ref.grad
                torch.testing.assert_close(dc, dc_ref, atol=atol, rtol=0)

            # Second run, just to make sure it doesn't crash
            y = model_compiled(
                x, additional_context, backend=backend, torch_compile=True
            )
            y.backward(dy)
            dx = x.grad

    def _test_fmha_module(
        self, batch, seqlens_Q, seqlens_KV, num_heads, head_dim, is_causal, atol
    ):
        device = "cuda"
        dtype = torch.float16

        embed_dim = num_heads * head_dim

        assert len(seqlens_Q) == len(seqlens_KV)
        assert len(seqlens_Q) >= 1
        assert len(seqlens_Q) == 1 or batch == len(seqlens_Q)

        seqlen_q = sum(seqlens_Q)
        seqlen_kv = sum(seqlens_KV)
        is_varlen = len(seqlens_Q) > 1

        batch_ = 1 if is_varlen else batch
        seqlens_Q_ = (
            torch.tensor(seqlens_Q, device=device, dtype=torch.int32)
            if is_varlen
            else None
        )
        seqlens_KV_ = (
            torch.tensor(seqlens_KV, device=device, dtype=torch.int32)
            if is_varlen
            else None
        )

        dummy_q = torch.randn(
            (batch_, seqlen_q, num_heads, head_dim),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        dummy_kv = torch.randn(
            (batch_, seqlen_kv, num_heads, head_dim),
            dtype=dtype,
            device=device,
            requires_grad=True,
        )

        # TODO: Flex FMHA also fails...
        backends = [
            b
            for b in get_compatible_fmha_backends(
                dummy_q,
                dummy_kv,
                dummy_kv,
                is_causal=is_causal,
                is_varlen=is_varlen,
                torch_compile=True,
            )
            if "b" != "flex-fmha"
        ]

        # seq maxes MUST be computed ahead of time when using torch compile
        # because need to be copied to host
        (
            cumulative_seqlen_Q,
            cumulative_seqlen_KV,
            max_seqlen_Q,
            max_seqlen_KV,
        ) = generate_varlen_parameters(
            query=dummy_q,
            key=dummy_kv,
            value=dummy_kv,
            seqlens_Q=seqlens_Q_,
            seqlens_KV=seqlens_KV_,
        )

        for backend in backends:
            _reset_everything()

            logger.debug(
                f"Testing torch compile on FMHA module with input shapes: "
                f"{batch=}, {num_heads=}, {head_dim=}, {seqlens_Q=}, {seqlens_KV=}, "
                f"{is_causal=}, {dtype=}, {device=}, {backend=}."
            )

            model_eager = (
                FMHABlock(
                    embed_dim=embed_dim,
                    mlp_ratio=2,
                    num_heads=num_heads,
                )
                .to(dtype)
                .to(device)
            )

            model_compiled = torch.compile(
                model_eager, fullgraph=True, backend="inductor"
            )

            x = torch.randn((batch_, seqlen_q, embed_dim), dtype=dtype, device=device)
            c = torch.randn((batch_, seqlen_kv, embed_dim), dtype=dtype, device=device)
            dy = (
                torch.randn((batch_, seqlen_q, embed_dim), dtype=dtype, device=device)
                * 0.1
            )

            x_ref = x.clone().requires_grad_(True)
            c_ref = c.clone().requires_grad_(True)
            dy_ref = dy.clone()

            # eager
            y_ref = model_eager(
                x_ref,
                c_ref,
                is_causal=is_causal,
                cumulative_seqlen_Q=cumulative_seqlen_Q,
                cumulative_seqlen_KV=cumulative_seqlen_KV,
                max_seqlen_Q=max_seqlen_Q,
                max_seqlen_KV=max_seqlen_KV,
                backend=backend,
                torch_compile=True,
            )
            y_ref.backward(dy_ref)
            dx_ref = x_ref.grad
            dc_ref = c_ref.grad

            # compile on first attempt
            x = x.requires_grad_(True)
            c = c.requires_grad_(True)
            y = model_compiled(
                x,
                c,
                is_causal=is_causal,
                cumulative_seqlen_Q=cumulative_seqlen_Q,
                cumulative_seqlen_KV=cumulative_seqlen_KV,
                max_seqlen_Q=max_seqlen_Q,
                max_seqlen_KV=max_seqlen_KV,
                backend=backend,
                torch_compile=True,
            )
            y.backward(dy)
            dx = x.grad
            dc = c.grad

            torch.testing.assert_close(y, y_ref, atol=atol, rtol=0)
            torch.testing.assert_close(dx, dx_ref, atol=atol, rtol=0)
            torch.testing.assert_close(dc, dc_ref, atol=atol, rtol=0)

            # Second run, just to make sure it doesn't crash
            y = model_compiled(
                x,
                c,
                is_causal=is_causal,
                cumulative_seqlen_Q=cumulative_seqlen_Q,
                cumulative_seqlen_KV=cumulative_seqlen_KV,
                max_seqlen_Q=max_seqlen_Q,
                max_seqlen_KV=max_seqlen_KV,
                backend=backend,
                torch_compile=True,
            )
            y.backward(dy)
            dx = x.grad
            dc = c.grad

    @skip_if_libnatten_is_not_supported()
    def test_compiled_na(self):
        problems = [
            (2, (672,), 4, 32, (5,), (1,), (1,), (False,)),
            (2, (688,), 4, 32, (5,), (1,), (1,), (False,)),
            (2, (9215,), 4, 32, (111,), (1,), (1,), (False,)),
            (2, (9215,), 4, 32, (9215,), (1,), (1,), (True,)),  # fmha case
            (2, (111,), 4, 32, (111,), (1,), (1,), (True,)),  # fmha case
            (2, (16384,), 4, 32, (128,), (16,), (8,), (False,)),
            (2, (16384,), 4, 32, (128,), (16,), (8,), (True,)),
            (1, (1024,), 2, 64, (64,), (2,), (2,), (False,)),
            (1, (1024,), 2, 64, (64,), (2,), (2,), (True,)),
            (8, (14, 14), 4, 32, (4, 4), (2, 2), (2, 3), (False, False)),
            (8, (14, 14), 4, 32, (4, 4), (2, 2), (2, 3), (False, False)),
            (2, (16, 24), 4, 32, (4, 6), (1, 1), (2, 1), (False, False)),
            (2, (9215,), 4, 32, (111,), (2,), (3,), (False,)),
            (2, (19, 27), 4, 32, (4, 2), (2, 7), (2, 1), (False, False)),
            (
                2,
                (16, 24, 24),
                4,
                32,
                (4, 6, 8),
                (2, 3, 2),
                (2, 1, 3),
                (True, False, False),
            ),
            (8, (14, 14), 4, 32, (14, 14), (1, 1), (1, 1), (False, False)),  # fmha case
            (1, (16384,), 2, 128, (16384,), (1,), (1,), (False,)),  # fmha case
            (1, (16384,), 2, 128, (16384,), (1,), (1,), (True,)),  # fmha case
            (1, (3287,), 4, 64, (3287,), (1,), (1,), (False,)),  # fmha case
            (1, (3287,), 4, 64, (3287,), (1,), (1,), (True,)),  # fmha case
        ]
        for problem in problems:
            (
                batch,
                token_layout_shape,
                num_heads,
                head_dim,
                kernel_size,
                dilation,
                stride,
                is_causal,
            ) = problem

            for additional_tokens in ADDITIONAL_KV_LENGTHS:
                self._test_na_module(
                    batch=batch,
                    token_layout_shape=token_layout_shape,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    stride=stride,
                    is_causal=is_causal,
                    atol=1e-3,
                    additional_tokens=additional_tokens,
                )

    @skip_if_libnatten_is_not_supported()
    def test_compiled_fmha(self):
        problem_sizes = [
            (1, 4, 128, [128], [128]),
            (1, 1, 128, [128], [1024]),
            (1, 1, 128, [128], [13568]),
            (1, 1, 128, [128], [13496]),
            (1, 1, 32, [128], [13496]),
            (1, 1, 32, [32], [13496]),
            (3, 1, 32, [77], [8504]),
            (1, 1, 32, [77], [8504]),
            (1, 1, 64, [40], [12296]),
            (1, 2, 64, [40], [12296]),
            (1, 2, 64, [40], [12296]),
            (1, 1, 128, [128], [128]),
            (6, 1, 128, [128, 128, 135, 121, 128, 128], [128, 128, 135, 121, 128, 128]),
            (5, 1, 128, [128, 128, 135, 128, 128], [128, 128, 135, 128, 128]),
            (2, 1, 128, [135, 200], [128, 768]),
            (2, 1, 128, [1024, 200], [128, 768]),
            (2, 1, 128, [135, 200], [135, 768]),
            (2, 1, 128, [1024, 200], [135, 768]),
            (2, 1, 128, [1024, 256], [128, 768]),
            (4, 1, 128, [1024, 8, 17, 2048], [10, 20, 512, 16]),
            (3, 2, 128, [268, 1584, 1571], [2448, 4088, 1925]),
            (2, 1, 128, [1024, 256], [512, 768]),
        ]
        for (
            batch,
            num_heads,
            head_dim,
            seqlens_Q,
            seqlens_KV,
        ) in problem_sizes:
            for is_causal in [False, True]:
                self._test_fmha_module(
                    batch=batch,
                    seqlens_Q=seqlens_Q,
                    seqlens_KV=seqlens_KV,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    is_causal=is_causal,
                    atol=1e-3,
                )


if __name__ == "__main__":
    torch.manual_seed(42)
    unittest.main()
