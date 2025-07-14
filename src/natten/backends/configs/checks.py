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
import functools
import math

import torch
from torch import Tensor

from ..._environment import _IS_TORCH_COMPILE_SUPPORTED, _TORCH_VERSION
from ..._libnatten import HAS_LIBNATTEN
from ...context import is_flex_compile_allowed, is_flex_compile_backprop_allowed
from ...utils.checks import fmha_tensor_checks, log_or_raise_error, na_tensor_checks
from ...utils.device import get_device_cc, is_cpu, is_cuda, is_rocm


### Blackwell FMHA/FNA


def can_run_cutlass_blackwell_fmha(
    query: Tensor, key: Tensor, value: Tensor, raise_error: bool = False
) -> bool:
    target_fn = functools.partial(log_or_raise_error, raise_error=raise_error)

    if not HAS_LIBNATTEN:
        target_fn("Can't run Blackwell FMHA; NATTEN was not built with libnatten.")
        return False

    if not fmha_tensor_checks(
        query,
        key,
        value,
        must_match_head_dims=True,
        raise_error=raise_error,
        backend_name="Blackwell FMHA",
    ):
        return False

    if query.dim() != 4:
        target_fn(
            f"Blackwell FMHA expects rank-4 input tensors, got {query.shape=}.",
            exception=ValueError,
        )
        return False

    if not is_cuda(query.device):
        target_fn("Can't run Blackwell FMHA; not a CUDA tensor.")
        return False

    device_cc = get_device_cc(query.device)

    if device_cc != 100:
        target_fn(
            "Can't run Blackwell FMHA; tensor was on CUDA device with "
            f"compute capability {device_cc}, expected 100."
        )
        return False

    if query.requires_grad and torch.are_deterministic_algorithms_enabled():
        target_fn(
            "Can't run Blackwell FMHA; its backprop does not have a deterministic mode, but "
            "PyTorch's deterministic mode was enabled.",
            exception=NotImplementedError,
        )
        return False

    head_dim = query.shape[-1]
    head_dim_v = value.shape[-1]

    if head_dim != head_dim_v:
        target_fn(
            "Can't run Blackwell FMHA; it does not support different head dims for QK and V, "
            f"got {head_dim=}, {head_dim_v=}.",
            exception=ValueError,
        )
        return False

    if query.dtype not in [torch.float16, torch.bfloat16]:
        target_fn(
            "Can't run Blackwell FMHA; it only supports FP16 and BF16 for now.",
            exception=ValueError,
        )
        return False

    if head_dim not in [32, 64, 128]:
        target_fn(
            "Can't run Blackwell FMHA; it only supports head dims 32, 64, and 128 for now.",
            exception=NotImplementedError,
        )
        return False

    return True


def can_run_cutlass_blackwell_fna(
    query: Tensor, key: Tensor, value: Tensor, raise_error: bool = False
) -> bool:
    target_fn = functools.partial(log_or_raise_error, raise_error=raise_error)

    if not HAS_LIBNATTEN:
        target_fn("Can't run Blackwell FNA; NATTEN was not built with libnatten.")
        return False

    if not na_tensor_checks(
        query,
        key,
        value,
        must_match_head_dims=True,
        raise_error=raise_error,
        backend_name="Blackwell FNA",
    ):
        return False

    if query.dim() not in [4, 5, 6]:
        target_fn(
            "Blackwell FNA expects 4-D, 5-D, or 6-D tensors as inputs (corresponding to NA1D, NA2D, and NA3D), "
            f"got {query.dim()=}.",
            exception=ValueError,
        )
        return False

    if not is_cuda(query.device):
        target_fn("Can't run Blackwell FNA; not a CUDA tensor.")
        return False

    device_cc = get_device_cc(query.device)

    if device_cc != 100:
        target_fn(
            "Can't run Blackwell FNA; tensor was on CUDA device with "
            f"compute capability {device_cc}, expected 100."
        )
        return False

    if query.requires_grad and torch.are_deterministic_algorithms_enabled():
        target_fn(
            "Can't run Blackwell FMHA; its backprop does not have a deterministic mode, but "
            "PyTorch's deterministic mode was enabled.",
            exception=NotImplementedError,
        )
        return False

    head_dim = query.shape[-1]
    head_dim_v = value.shape[-1]

    if head_dim != head_dim_v:
        target_fn(
            "Can't run Blackwell FNA; it does not support different head dims for QK and V, "
            f"got {head_dim=}, {head_dim_v=}.",
            exception=ValueError,
        )
        return False

    if query.dtype not in [torch.float16, torch.bfloat16]:
        target_fn(
            "Can't run Blackwell FNA; it only supports FP16 and BF16 for now.",
            exception=ValueError,
        )
        return False

    if head_dim not in [32, 64, 128]:
        target_fn(
            "Can't run Blackwell FNA; it only supports head dims 32, 64, and 128 for now.",
            exception=NotImplementedError,
        )
        return False

    return True


### Hopper FMHA/FNA


def can_run_cutlass_hopper_fmha(
    query: Tensor, key: Tensor, value: Tensor, raise_error: bool = False
) -> bool:
    target_fn = functools.partial(log_or_raise_error, raise_error=raise_error)

    if not HAS_LIBNATTEN:
        target_fn("Can't run Hopper FMHA; NATTEN was not built with libnatten.")
        return False

    if not fmha_tensor_checks(
        query,
        key,
        value,
        must_match_head_dims=True,
        raise_error=raise_error,
        backend_name="Hopper FMHA",
    ):
        return False

    if query.dim() != 4:
        target_fn(
            f"Hopper FMHA expects rank-4 input tensors, got {query.shape=}.",
            exception=ValueError,
        )
        return False

    if not is_cuda(query.device):
        target_fn("Can't run Hopper FMHA; not a CUDA tensor.")
        return False

    device_cc = get_device_cc(query.device)

    if device_cc != 90:
        target_fn(
            "Can't run Hopper FMHA; tensor was on CUDA device with "
            f"compute capability {device_cc}, expected 90."
        )
        return False

    head_dim = query.shape[-1]
    head_dim_v = value.shape[-1]

    if head_dim != head_dim_v:
        target_fn(
            "Can't run Hopper FMHA; it does not support different head dims for QK and V, "
            f"got {head_dim=}, {head_dim_v=}.",
            exception=ValueError,
        )
        return False

    if query.requires_grad and head_dim not in [32, 64, 128]:
        target_fn(
            f"Can't run Hopper FMHA; it does not support backpropagation for {head_dim=} yet; "
            "only head dims 32, 64, and 128 are allowed.",
            exception=NotImplementedError,
        )
        return False

    if query.requires_grad and torch.are_deterministic_algorithms_enabled():
        target_fn(
            "Can't run Hopper FMHA; its backprop does not have a deterministic mode, but "
            "PyTorch's deterministic mode was enabled.",
            exception=NotImplementedError,
        )
        return False

    if query.dtype not in [torch.float16, torch.bfloat16]:
        target_fn(
            "Can't run Hopper FMHA; it only supports FP16 and BF16 for now.",
            exception=ValueError,
        )
        return False

    if head_dim not in [32, 64, 128, 256]:
        target_fn(
            "Can't run Hopper FMHA; it only supports head dims 32, 64, 128, and 256 for now.",
            exception=NotImplementedError,
        )
        return False

    return True


def can_run_cutlass_hopper_fna(
    query: Tensor, key: Tensor, value: Tensor, raise_error: bool = False
) -> bool:
    target_fn = functools.partial(log_or_raise_error, raise_error=raise_error)

    if not HAS_LIBNATTEN:
        target_fn("Can't run Hopper FNA; NATTEN was not built with libnatten.")
        return False

    if not na_tensor_checks(
        query,
        key,
        value,
        must_match_head_dims=True,
        raise_error=raise_error,
        backend_name="Hopper FNA",
    ):
        return False

    if query.dim() not in [4, 5, 6]:
        target_fn(
            "Hopper FNA expects 4-D, 5-D, or 6-D tensors as inputs (corresponding to NA1D, NA2D, and NA3D), "
            f"got {query.dim()=}.",
            exception=ValueError,
        )
        return False

    if not is_cuda(query.device):
        target_fn("Can't run Hopper FNA; not a CUDA tensor.")
        return False

    device_cc = get_device_cc(query.device)

    if device_cc != 90:
        target_fn(
            "Can't run Hopper FNA; tensor was on CUDA device with "
            f"compute capability {device_cc}, expected 90."
        )
        return False

    head_dim = query.shape[-1]
    head_dim_v = value.shape[-1]

    if head_dim != head_dim_v:
        target_fn(
            "Can't run Hopper FNA; it does not support different head dims for QK and V, "
            f"got {head_dim=}, {head_dim_v=}.",
            exception=ValueError,
        )
        return False

    if query.requires_grad and head_dim not in [32, 64, 128]:
        target_fn(
            f"Can't run Hopper FNA; it does not support backpropagation for {head_dim=} yet; "
            "only head dims 32, 64, and 128 are allowed.",
            exception=NotImplementedError,
        )
        return False

    if query.requires_grad and torch.are_deterministic_algorithms_enabled():
        target_fn(
            "Can't run Hopper FNA; its backprop does not have a deterministic mode, but "
            "PyTorch's deterministic mode was enabled.",
            exception=NotImplementedError,
        )
        return False

    if query.dtype not in [torch.float16, torch.bfloat16]:
        target_fn(
            "Can't run Hopper FNA; it only supports FP16 and BF16 for now.",
            exception=ValueError,
        )
        return False

    if head_dim not in [32, 64, 128, 256]:
        target_fn(
            "Can't run Hopper FNA; it only supports head dims 32, 64, 128, and 256 for now.",
            exception=NotImplementedError,
        )
        return False

    return True


### CUTLASS FMHA/FNA


def can_run_cutlass_fmha(
    query: Tensor, key: Tensor, value: Tensor, raise_error: bool = False
) -> bool:
    target_fn = functools.partial(log_or_raise_error, raise_error=raise_error)

    if not HAS_LIBNATTEN:
        target_fn("Can't run CUTLASS FMHA; NATTEN was not built with libnatten.")
        return False

    if not fmha_tensor_checks(
        query,
        key,
        value,
        must_match_head_dims=False,
        raise_error=raise_error,
        backend_name="CUTLASS FMHA",
    ):
        return False

    if query.dim() != 4:
        target_fn(
            f"FMHA expects rank-4 input tensors, got {query.shape=}.",
            exception=ValueError,
        )
        return False

    if not is_cuda(query.device):
        target_fn("Can't run CUTLASS FMHA; not a CUDA tensor.")
        return False

    device_cc = get_device_cc(query.device)

    if device_cc < 60:
        target_fn(
            "CUTLASS FMHA only supports CUDA devices with compute capability 60 or higher, "
            f"got {device_cc}."
        )
        return False

    head_dim = query.shape[-1]
    head_dim_v = value.shape[-1]

    if query.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        target_fn(
            "Can't run CUTLASS FMHA; it only supports FP32, FP16, and BF16.",
            exception=ValueError,
        )
        return False

    if head_dim % 8 != 0:
        target_fn(
            "Can't run CUTLASS FMHA; it only supports head dims that are multiples of 8.",
            exception=ValueError,
        )
        return False

    if head_dim_v % 8 != 0:
        target_fn(
            "Can't run CUTLASS FMHA; it only supports head dims that are multiples of 8, "
            f"got {head_dim_v=}.",
            exception=ValueError,
        )
        return False

    if max(head_dim, head_dim_v) > 2**16:
        target_fn(
            f"Can't run CUTLASS FMHA; it supports max head dim of {2**16}, "
            f"got {head_dim=}, {head_dim_v=}.",
            exception=ValueError,
        )
        return False

    return True


def can_run_cutlass_fna(
    query: Tensor, key: Tensor, value: Tensor, raise_error: bool = False
) -> bool:
    target_fn = functools.partial(log_or_raise_error, raise_error=raise_error)

    if not HAS_LIBNATTEN:
        target_fn("Can't run CUTLASS FNA; NATTEN was not built with libnatten.")
        return False

    if not na_tensor_checks(
        query,
        key,
        value,
        must_match_head_dims=False,
        raise_error=raise_error,
        backend_name="CUTLASS FNA",
    ):
        return False

    if query.dim() not in [4, 5, 6]:
        target_fn(
            "CUTLASS FNA expects 4-D, 5-D, or 6-D tensors as inputs (corresponding to NA1D, NA2D, and NA3D), "
            f"got {query.dim()=}.",
            exception=ValueError,
        )
        return False

    if not is_cuda(query.device):
        target_fn("Can't run CUTLASS FNA; not a CUDA tensor.")
        return False

    device_cc = get_device_cc(query.device)

    if device_cc < 60:
        target_fn(
            "CUTLASS FNA only supports CUDA devices with compute capability 60 or higher, "
            f"got {device_cc}."
        )
        return False

    head_dim = query.shape[-1]
    head_dim_v = value.shape[-1]

    if query.dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        target_fn(
            "Can't run CUTLASS FNA; it only supports FP32, FP16, and BF16.",
            exception=ValueError,
        )
        return False

    if head_dim % 8 != 0:
        target_fn(
            "Can't run CUTLASS FNA; it only supports head dims that are multiples of 8, "
            f"got {head_dim=}.",
            exception=ValueError,
        )
        return False

    if head_dim_v % 8 != 0:
        target_fn(
            "Can't run CUTLASS FNA; it only supports head dims that are multiples of 8, "
            f"got {head_dim_v=}.",
            exception=ValueError,
        )
        return False

    if max(head_dim, head_dim_v) > 2**16:
        target_fn(
            f"Can't run CUTLASS FNA; it supports max head dim of {2**16}, "
            f"got {head_dim=}, {head_dim_v=}.",
            exception=ValueError,
        )
        return False

    return True


### Flex FMHA/FNA

_FLEX_SUPPORTED = _TORCH_VERSION >= [2, 7]
_FLEX_COMPILE_SUPPORTED = _TORCH_VERSION >= [2, 7] and _IS_TORCH_COMPILE_SUPPORTED


def can_run_flex_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    torch_compile: bool,
    raise_error: bool = False,
) -> bool:
    target_fn = functools.partial(log_or_raise_error, raise_error=raise_error)

    if not _FLEX_SUPPORTED:
        target_fn("Can't run NATTEN with Flex Attention with torch < 2.7.")
        return False

    if torch_compile and not _FLEX_COMPILE_SUPPORTED:
        target_fn("Can't run NATTEN with Flex Attention (compiled).)")
        return False

    if torch_compile and not is_flex_compile_allowed():
        target_fn(
            "NATTEN does not allow compiling Flex Attention. This is because we cannot verify "
            "Flex's correctness in all scenarios through NATTEN's tests. You can choose to override "
            "this, though it is discouraged, as it may affect your results significantly, "
            "by doing:\n"
            "  from natten import allow_flex_compile\n"
            "  allow_flex_compile()\n"
        )
        return False

    if torch_compile and query.requires_grad and not is_flex_compile_backprop_allowed():
        target_fn(
            "NATTEN does not allow compiling Flex Attention for backpropagation "
            "({q,k,v}.requires_grad=True). This is because we cannot verify Flex's correctness "
            "in all scenarios through NATTEN's tests. You can choose to override this, though "
            "it is HIGHLY discouraged, as it may affect the results of your training significantly, "
            "by doing:\n"
            "  from natten import allow_flex_compile_backprop\n"
            "  allow_flex_compile_backprop()\n"
        )
        return False

    # TODO: can we just have different checks for FMHA vs FNA, like the rest of the backends?
    if query.dim() == 4 and key.dim() == 4 and query.shape[1] != key.shape[1]:
        supported = fmha_tensor_checks(
            query,
            key,
            value,
            must_match_head_dims=True,
            raise_error=raise_error,
            backend_name="Flex FMHA",
        )
    else:
        supported = na_tensor_checks(
            query,
            key,
            value,
            must_match_head_dims=True,
            raise_error=raise_error,
            backend_name="Flex FMHA/FNA",
        )
    if not supported:
        return False

    if query.dim() not in [4, 5, 6]:
        target_fn(
            "Flex backend expects 4-D, 5-D, or 6-D tensors as inputs (corresponding to FMHA/NA1D, "
            f"NA2D, and NA3D), got {query.dim()=}.",
            exception=ValueError,
        )
        return False

    if not is_cuda(query.device):
        if not is_cpu(query.device) and not is_rocm(query.device):
            target_fn(
                "Can't run Flex Attention; tensor is not on a CUDA, ROCm, or CPU device: "
                f"{query.device.type}"
            )

            return False
        # TODO: check if ROCm device supports torch.compile/triton?

    else:
        device_cc = get_device_cc(query.device)

        if device_cc < 70:
            target_fn(
                "Flex Attention (compiled) only supports CUDA devices with compute capability "
                f"70 or higher, got {device_cc}."
            )
            return False

    head_dim = query.shape[-1]
    head_dim_v = value.shape[-1]

    if head_dim != head_dim_v:
        target_fn(
            "Can't run NATTEN with Flex Attention; we don't support different head dims for QK and "
            f"V in this backend yet, got {head_dim=}, {head_dim_v=}.",
            exception=ValueError,
        )
        return False

    if not torch_compile and query.dtype not in [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]:
        target_fn(
            "Can't run NATTEN with Flex Attention; we only support FP32, FP16, and BF16 for now.",
            exception=ValueError,
        )
        return False

    if torch_compile and query.dtype not in [torch.float16, torch.bfloat16]:
        target_fn(
            "Can't run NATTEN with Flex Attention (compiled); we only support FP32, FP16, and BF16 for now.",
            exception=ValueError,
        )
        return False

    if torch_compile and (
        head_dim < 32 or head_dim > 512 or not math.log2(head_dim).is_integer()
    ):
        target_fn(
            "Can't run NATTEN with Flex Attention (compiled); we only allow 32 <= head_dim <= 512 "
            f"and only powers of two, got {head_dim}.",
            exception=ValueError,
        )
        return False

    if not torch_compile and (
        head_dim < 8 or head_dim > 512 or not math.log2(head_dim).is_integer()
    ):
        target_fn(
            "Can't run NATTEN with Flex Attention (not compiled); we only allow 8 <= head_dim <= 512 "
            f"and only powers of two, got {head_dim}.",
            exception=ValueError,
        )
        return False

    return True
