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

from collections.abc import Sequence
from typing import Any, Optional, Tuple

import torch  # noqa: F401
from torch import Tensor

from ..types import CausalArgType, DimensionType, KernelSchedule
from ..utils.tuples import create_causal_arg_from_bool, create_dim_from_int

from . import log

logger = log.get_logger(__name__)


def _universal_tensor_checks(query: Tensor, key: Tensor, value: Tensor):
    if query.is_nested or key.is_nested or value.is_nested:
        raise NotImplementedError("NATTEN does not support nested tensors.")

    if query.device != key.device or query.device != value.device:
        raise ValueError(
            "Query, key, and value must be on the same device, "
            f"got {query.device=}, {key.device=}, {value.device=}."
        )

    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(
            "Query, key, and value must assume the same data type, "
            f"got {query.dtype=}, {key.dtype=}, {value.dtype=}."
        )


def na_tensor_checks(query: Tensor, key: Tensor, value: Tensor):
    _universal_tensor_checks(query, key, value)

    if query.dim() != key.dim() or query.dim() != value.dim():
        raise ValueError(
            "Query, key, and value must have the same rank, "
            f"got {query.dim()=}, {key.dim()=}, {value.dim()=}."
        )

    if query.dim() not in [4, 5, 6]:
        raise ValueError(
            "Expected 4-D, 5-D, or 6-D tensors as inputs (corresponding to NA1D, NA2D, and NA3D), "
            f"got {query.dim()=}."
        )

    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            "Neighborhood attention ops expect Q, K, and V to have the same shape, got "
            f"{query.shape=}, {key.shape=}, {value.shape=}."
        )


def fmha_tensor_checks(query: Tensor, key: Tensor, value: Tensor):
    _universal_tensor_checks(query, key, value)

    if query.dim() != key.dim() or query.dim() != value.dim():
        raise ValueError(
            "Query, key, and value must have the same rank, "
            f"got {query.dim()=}, {key.dim()=}, {value.dim()=}."
        )

    if query.dim() != 4:
        raise ValueError(
            "Expected 4-D tensors as inputs to FMHA, " f"got {query.dim()=}."
        )

    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            f"Q and K head dims must match, got {query.shape[-1]=}, {key.shape[-1]=}."
        )

    if query.shape[-1] != value.shape[-1]:
        raise ValueError(
            "This operation does not support different head dims for QK and V, got "
            f"{query.shape[-1]=}, {value.shape[-1]=}."
        )

    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError(
            "Q, K, and V must match in batch size, got "
            f"{query.shape[0]=}, {key.shape[0]=}, {value.shape[0]=}."
        )

    if key.shape[1] != value.shape[1]:
        raise ValueError(
            f"K and V must match in sequence length, got {key.shape[1]=}, {value.shape[1]=}."
        )

    if query.shape[2] != key.shape[2] or query.shape[2] != value.shape[2]:
        raise ValueError(
            "NATTEN operations do not support GQA/MQA, therefore number of heads in Q, K, and V "
            f"must match, got {query.shape[2]=}, {key.shape[2]=}, {value.shape[2]=}."
        )


def additional_kv_tensor_checks(
    query: Tensor, key: Optional[Tensor] = None, value: Optional[Tensor] = None
):

    if (key is not None) ^ (value is not None):
        raise ValueError(
            "`additional_keys` and `additional_values` must be either both Tensors or None."
        )

    if key is None:
        return

    assert key is not None and value is not None

    _universal_tensor_checks(query, key, value)

    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            f"Q and K head dims must match, got {query.shape[-1]=}, {key.shape[-1]=}."
        )

    if query.shape[-1] != value.shape[-1]:
        raise ValueError(
            "This operation does not support different head dims for QK and V, got "
            f"{query.shape[-1]=}, {value.shape[-1]=}."
        )

    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError(
            "Q, K, and V must match in batch size, got "
            f"{query.shape[0]=}, {key.shape[0]=}, {value.shape[0]=}."
        )

    if key.shape[1] != value.shape[1]:
        raise ValueError(
            f"K and V must match in sequence length, got {key.shape[1]=}, {value.shape[1]=}."
        )

    if query.shape[-2] != key.shape[-2] or query.shape[-2] != value.shape[-2]:
        raise ValueError(
            "NATTEN operations do not support GQA/MQA, therefore number of heads in Q, K, and V "
            f"must match, got {query.shape[-2]=}, {key.shape[-2]=}, {value.shape[-2]=}."
        )


def check_input_size_arg(na_dim: int, input_size: Any) -> DimensionType:
    assert na_dim > 0 and na_dim < 4
    if (
        isinstance(input_size, Sequence)
        and len(input_size) == na_dim
        and all(isinstance(x, int) and x > 1 for x in input_size)
    ):
        return tuple(x for x in input_size)

    if isinstance(input_size, int) and input_size > 1:
        return create_dim_from_int(na_dim, value=input_size)

    raise ValueError(
        "Invalid value for `input_size`; expected an integer or iterable of integers, all >= 2, "
        f"got {type(input_size)}"
    )


def check_kernel_size_arg(na_dim: int, kernel_size: Any) -> DimensionType:
    assert na_dim > 0 and na_dim < 4
    if (
        isinstance(kernel_size, Sequence)
        and len(kernel_size) == na_dim
        and all(isinstance(x, int) and x > 1 for x in kernel_size)
    ):
        return tuple(x for x in kernel_size)

    if isinstance(kernel_size, int) and kernel_size > 1:
        return create_dim_from_int(na_dim, value=kernel_size)

    raise ValueError(
        "Invalid value for `kernel_size`; expected an integer or iterable of integers, all >= 2, "
        f"got {type(kernel_size)}"
    )


def check_stride_arg(na_dim: int, stride: Any) -> DimensionType:
    assert na_dim > 0 and na_dim < 4
    if stride is None:
        return create_dim_from_int(na_dim, value=1)

    if (
        isinstance(stride, Sequence)
        and len(stride) == na_dim
        and all(isinstance(x, int) and x > 0 for x in stride)
    ):
        return tuple(x for x in stride)

    if isinstance(stride, int) and stride > 0:
        return create_dim_from_int(na_dim, value=stride)

    raise ValueError(
        "Invalid value for `stride`; expected an integer or tuple of positive integers, "
        f"got {type(stride)}"
    )


def check_dilation_arg(na_dim: int, dilation: Any) -> DimensionType:
    assert na_dim > 0 and na_dim < 4
    if dilation is None:
        return create_dim_from_int(na_dim, value=1)

    if (
        isinstance(dilation, Sequence)
        and len(dilation) == na_dim
        and all(isinstance(x, int) and x > 0 for x in dilation)
    ):
        return tuple(x for x in dilation)

    if isinstance(dilation, int) and dilation > 0:
        return create_dim_from_int(na_dim, value=dilation)

    raise ValueError(
        "Invalid value for `dilation`; expected an integer or tuple of positive integers, "
        f"got {type(dilation)}"
    )


def check_causal_arg(na_dim: int, is_causal: Any) -> CausalArgType:
    assert na_dim > 0 and na_dim < 4

    if is_causal is None:
        return create_causal_arg_from_bool(na_dim, value=False)

    if (
        isinstance(is_causal, Sequence)
        and len(is_causal) == na_dim
        and all(isinstance(c, bool) for c in is_causal)
    ):
        return tuple(c for c in is_causal)

    if isinstance(is_causal, bool):
        return create_causal_arg_from_bool(na_dim, value=is_causal)

    raise ValueError(
        "Invalid value for `is_causal`; expected a boolean or tuple of booleans, "
        f"got {type(is_causal)}"
    )


def check_all_args(
    na_dim: int, kernel_size: Any, stride: Any, dilation: Any, is_causal: Any
) -> Tuple[DimensionType, DimensionType, DimensionType, CausalArgType]:
    kernel_size_out, stride_out, dilation_out, is_causal_out = (
        check_kernel_size_arg(na_dim, kernel_size),
        check_stride_arg(na_dim, stride),
        check_dilation_arg(na_dim, dilation),
        check_causal_arg(na_dim, is_causal),
    )

    return kernel_size_out, stride_out, dilation_out, is_causal_out


def check_args_against_input(
    input_tensor: Tensor,
    kernel_size: DimensionType,
    stride: DimensionType,
    dilation: DimensionType,
    is_causal: CausalArgType,
):
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3
    input_size = input_tensor.shape[1 : 1 + na_dim]

    if any(k * d > x for x, k, d in zip(input_size, kernel_size, dilation)):
        raise ValueError(
            "The product of kernel size and dilation cannot be larger than input size "
            f"along any dimension, got {input_size=} ({input_tensor.shape=}), "
            f"{kernel_size=}, {dilation=}."
        )

    if any(s > k for k, s in zip(kernel_size, stride)):
        raise ValueError(
            "Stride cannot be larger than kernel size along any dimension, got "
            f"{kernel_size=}, {stride=}."
        )


def is_self_attention(
    input_tensor: Tensor, kernel_size: DimensionType, is_causal: CausalArgType
):
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3
    input_size = input_tensor.shape[1 : 1 + na_dim]

    return all(k == x and not c for x, k, c in zip(input_size, kernel_size, is_causal))


def check_tile_shape(
    tile_shape: Any,
) -> DimensionType:
    if (
        isinstance(tile_shape, Sequence)
        and len(tile_shape) <= 3
        and all(isinstance(x, int) for x in tile_shape)
    ):
        return tuple(x for x in tile_shape)

    raise ValueError(
        f"Unsupported value for tile shape; expected an iterable of at most 3 integers, "
        f"got {type(tile_shape)}: {tile_shape}"
    )


def log_or_raise_error(
    msg: str, raise_error: bool = False, exception: Any = RuntimeError
):
    if raise_error:
        raise exception(msg)
    else:
        logger.debug(msg)


def check_kernel_schedule(kernel_schedule: Any) -> Optional[KernelSchedule]:
    if kernel_schedule is None:
        return None

    if isinstance(kernel_schedule, KernelSchedule):
        return kernel_schedule

    if kernel_schedule == "non":
        return KernelSchedule.NonPersistent
    elif kernel_schedule == "coop":
        return KernelSchedule.WarpSpecializedCooperative
    elif kernel_schedule == "pp":
        return KernelSchedule.WarpSpecializedPingpong

    raise ValueError(
        f"Kernel schedule {kernel_schedule} is invalid; choices are: "
        "`non` (non-persistent), `coop` (warp-specialized cooperative), and "
        "`pp` (warp-specialized ping-ponging)."
    )
