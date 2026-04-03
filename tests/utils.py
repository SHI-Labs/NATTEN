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

import time
from typing import Optional, Tuple, Union

import torch
from natten.backends.reference import reference_fna_generic
from natten.functional import neighborhood_attention_generic
from natten.types import CausalArgType, DimensionType, KernelSchedule
from natten.utils import log
from natten.utils.checks import check_all_args

logger = log.get_logger(__name__)


def reset_torch_compile(cache_size_limit):
    # Torch compile reset and sensible settings for unit testing
    logger.debug(
        f"Resetting torch compile cache. New cache size limit: {cache_size_limit}"
    )
    torch.compiler.reset()
    torch._dynamo.config.cache_size_limit = cache_size_limit
    torch._dynamo.config.accumulated_recompile_limit = cache_size_limit * 4
    torch._dynamo.config.fail_on_recompile_limit_hit = True


# Runs one backend once as a reference, and may run another backend multiple times
# with different configurations.
class NattenBackendTester:
    def __init__(
        self,
        batch: int,
        heads: int,
        head_dim: int,
        input_shape: DimensionType,
        kernel_size: DimensionType,
        stride: DimensionType,
        dilation: DimensionType,
        is_causal: CausalArgType,
        test_backprop: bool,
        reference_backend: str,
        reference_fmha_backend: str,
        dtype: torch.dtype,
        head_dim_v: Optional[int] = None,
        heads_kv: Optional[int] = None,
        reference_q_tile_shape: Optional[DimensionType] = None,
        reference_kv_tile_shape: Optional[DimensionType] = None,
        reference_backward_q_tile_shape: Optional[DimensionType] = None,
        reference_backward_kv_tile_shape: Optional[DimensionType] = None,
        reference_backward_kv_splits: Optional[DimensionType] = None,
        reference_backward_use_pt_reduction: bool = False,
    ):
        assert isinstance(input_shape, tuple)
        na_dim = len(input_shape)
        assert na_dim in [1, 2, 3], "Only supports NA1D, 2D, 3D."

        self.batch = batch
        self.heads = heads
        self.heads_kv = heads_kv or heads
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v or head_dim
        self.input_shape = input_shape
        self.kernel_size, self.stride, self.dilation, self.is_causal = check_all_args(
            na_dim, kernel_size, stride, dilation, is_causal
        )

        self.test_backprop = test_backprop
        self.reference_backend = reference_backend
        self.reference_fmha_backend = reference_fmha_backend

        with torch.no_grad():
            orig_dtype = dtype
            if dtype in [torch.float8_e5m2, torch.float8_e4m3fn]:
                dtype = torch.float16

            q_ref, k_ref, v_ref, d_out_ref = (
                torch.randn(
                    (self.batch, *self.input_shape, self.heads, self.head_dim),
                    device="cuda",
                    dtype=dtype,
                ),
                torch.randn(
                    (self.batch, *self.input_shape, self.heads_kv, self.head_dim),
                    device="cuda",
                    dtype=dtype,
                ),
                torch.randn(
                    (self.batch, *self.input_shape, self.heads_kv, self.head_dim_v),
                    device="cuda",
                    dtype=dtype,
                ),
                torch.randn(
                    (self.batch, *self.input_shape, self.heads, self.head_dim_v),
                    device="cuda",
                    dtype=dtype,
                )
                * 0.05,
            )

            if dtype != orig_dtype:
                q_ref = q_ref.to(orig_dtype)
                k_ref = k_ref.to(orig_dtype)
                v_ref = v_ref.to(orig_dtype)
                d_out_ref = d_out_ref.to(orig_dtype)

            self.q, self.k, self.v, self.d_out = (
                q_ref.clone(),
                k_ref.clone(),
                v_ref.clone(),
                d_out_ref.clone(),
            )

        # Reference
        torch.cuda.synchronize()
        start_time = time.time()

        q_ref.requires_grad_(True)
        k_ref.requires_grad_(True)
        v_ref.requires_grad_(True)
        d_out_ref.requires_grad_(True)
        if reference_backend is None or reference_backend == "reference":
            out_ref_ = reference_fna_generic(
                q_ref,
                k_ref,
                v_ref,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                return_lse=False,
            )

        else:
            # TODO: don't rely on `neighborhood_attention_generic` finding the right backend
            # and explicitly call the backend fns.
            out_ref_ = neighborhood_attention_generic(
                q_ref,
                k_ref,
                v_ref,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                is_causal=is_causal,
                backend=reference_backend,
                q_tile_shape=reference_q_tile_shape,
                kv_tile_shape=reference_kv_tile_shape,
                backward_q_tile_shape=reference_backward_q_tile_shape,
                backward_kv_tile_shape=reference_backward_kv_tile_shape,
                backward_kv_splits=reference_backward_kv_splits,
                backward_use_pt_reduction=reference_backward_use_pt_reduction,
                attention_kwargs={
                    "backend": reference_fmha_backend,
                },
            )

        self.out_ref = out_ref_.data.clone().float()  # type: ignore[union-attr]

        self.dq_ref, self.dk_ref, self.dv_ref = None, None, None
        if test_backprop:
            out_ref_.backward(d_out_ref)  # type: ignore[union-attr]
            with torch.no_grad():
                assert q_ref.grad is not None
                assert k_ref.grad is not None
                assert v_ref.grad is not None
                self.dq_ref, self.dk_ref, self.dv_ref = (
                    q_ref.grad.clone().float(),
                    k_ref.grad.clone().float(),
                    v_ref.grad.clone().float(),
                )

        torch.cuda.synchronize()
        reference_time = time.time() - start_time
        logger.debug(
            f"Reference ({reference_backend}/{reference_fmha_backend}) ran in {reference_time:.2f} seconds."
        )

    def test(
        self,
        eps: Union[
            float, Tuple[float, float], Tuple[float, Tuple[float, float, float]]
        ],
        dtype: torch.dtype,
        target_backend: str,
        target_fmha_backend: str = "cutlass-fmha",
        q_tile_shape: Optional[DimensionType] = None,
        kv_tile_shape: Optional[DimensionType] = None,
        backward_q_tile_shape: Optional[DimensionType] = None,
        backward_kv_tile_shape: Optional[DimensionType] = None,
        backward_kv_splits: Optional[DimensionType] = None,
        backward_use_pt_reduction: bool = False,
        run_persistent_kernel: bool = True,
        kernel_schedule: Optional[KernelSchedule] = None,
        torch_compile: bool = False,
        test_backprop: Optional[bool] = None,
    ):
        batch = self.batch
        heads = self.heads
        heads_kv = self.heads_kv
        head_dim = self.head_dim
        head_dim_v = self.head_dim_v
        input_shape = self.input_shape
        kernel_size = self.kernel_size
        stride = self.stride
        dilation = self.dilation
        is_causal = self.is_causal
        reference_backend = self.reference_backend
        test_backprop_safe: bool = (
            self.test_backprop if test_backprop is None else test_backprop
        )

        logger.debug(
            f"Testing {target_backend} against {reference_backend}:\n"
            f"{batch=}, {heads=}, {heads_kv=}, {head_dim=}, {head_dim_v=}, {input_shape=}, {dtype=}\n"
            f"{kernel_size=}, {stride=}, {dilation=}, {is_causal=},\n"
            f"{q_tile_shape=}, {kv_tile_shape=}, {run_persistent_kernel=}, {kernel_schedule=}, "
            f"{torch_compile=}"
            + (
                f"\n{backward_q_tile_shape=}, {backward_kv_tile_shape=}, "
                f"{backward_kv_splits=}, {backward_use_pt_reduction=}."
                if test_backprop_safe
                else "."
            )
        )

        q, k, v, d_out = (
            self.q.clone().to(dtype),
            self.k.clone().to(dtype),
            self.v.clone().to(dtype),
            self.d_out.clone().to(dtype),
        )
        q.requires_grad_(test_backprop_safe)
        k.requires_grad_(test_backprop_safe)
        v.requires_grad_(test_backprop_safe)
        d_out.requires_grad_(test_backprop_safe)

        torch.cuda.synchronize()
        start_time = time.time()

        out_: torch.Tensor = neighborhood_attention_generic(  # type: ignore[assignment]
            q,
            k,
            v,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            is_causal=is_causal,
            backend=target_backend,
            q_tile_shape=q_tile_shape,
            kv_tile_shape=kv_tile_shape,
            backward_q_tile_shape=backward_q_tile_shape,
            backward_kv_tile_shape=backward_kv_tile_shape,
            backward_kv_splits=backward_kv_splits,
            backward_use_pt_reduction=backward_use_pt_reduction,
            run_persistent_kernel=run_persistent_kernel,
            kernel_schedule=kernel_schedule,
            torch_compile=torch_compile,
            attention_kwargs={"backend": target_fmha_backend},
        )
        out = out_.data.clone().float()

        if test_backprop_safe:
            dq, dk, dv = None, None, None
            out_.backward(d_out)
            with torch.no_grad():
                assert q.grad is not None
                assert k.grad is not None
                assert v.grad is not None
                dq, dk, dv = (
                    q.grad.clone().float(),
                    k.grad.clone().float(),
                    v.grad.clone().float(),
                )

        if isinstance(eps, tuple):
            eps_forward, eps_backward = eps
        else:
            eps_forward, eps_backward = eps, eps

        torch.cuda.synchronize()
        runtime = time.time() - start_time
        logger.debug(
            f"Backend ({target_backend}/{target_fmha_backend}) ran in {runtime:.2f} seconds."
        )

        torch.testing.assert_close(out, self.out_ref, atol=eps_forward, rtol=0)

        if test_backprop_safe:
            if isinstance(eps_backward, tuple):
                assert len(eps_backward) == 3
                eps_dq, eps_dk, eps_dv = eps_backward
            else:
                assert isinstance(eps_backward, float)
                eps_dq, eps_dk, eps_dv = eps_backward, eps_backward, eps_backward

            torch.testing.assert_close(dq, self.dq_ref, atol=eps_dq, rtol=0)
            torch.testing.assert_close(dk, self.dk_ref, atol=eps_dk, rtol=0)
            torch.testing.assert_close(dv, self.dv_ref, atol=eps_dv, rtol=0)
