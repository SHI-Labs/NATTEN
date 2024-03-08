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
from typing import Any, Tuple


def get_num_na_weights(
    kernel_size: Tuple[int] | Tuple[int, int] | Tuple[int, int, int]
) -> int:
    if not isinstance(kernel_size, tuple):
        raise ValueError(
            f"Expected `kernel_size` to be a tuple; got {type(kernel_size)}"
        )
    return math.prod(kernel_size)


def check_kernel_size_arg(na_dim: int, kernel_size: Any) -> Tuple:
    assert na_dim > 0 and na_dim < 4
    if (
        isinstance(kernel_size, tuple)
        and len(kernel_size) == na_dim
        and all(isinstance(d, int) for d in kernel_size)
    ):
        return kernel_size
    if (
        isinstance(kernel_size, list)
        and len(kernel_size) == na_dim
        and all(isinstance(d, int) for d in kernel_size)
    ):
        return tuple(kernel_size)
    if isinstance(kernel_size, int):
        return tuple(kernel_size for _ in range(na_dim))
    raise ValueError(
        "Invalid value for `kernel_size`; expected an integer or tuple of integers, "
        f"got {type(kernel_size)}"
    )


def check_dilation_arg(na_dim: int, dilation: Any) -> Tuple:
    assert na_dim > 0 and na_dim < 4
    if dilation is None:
        return tuple(1 for _ in range(na_dim))
    if (
        isinstance(dilation, tuple)
        and len(dilation) == na_dim
        and all(isinstance(d, int) for d in dilation)
    ):
        return dilation
    if (
        isinstance(dilation, list)
        and len(dilation) == na_dim
        and all(isinstance(d, int) for d in dilation)
    ):
        return tuple(dilation)
    if isinstance(dilation, int):
        return tuple(dilation for _ in range(na_dim))
    raise ValueError(
        "Invalid value for `dilation`; expected an integer or tuple of integers, "
        f"got {type(dilation)}"
    )


def check_causal_arg(na_dim: int, is_causal: Any) -> Tuple:
    assert na_dim > 0 and na_dim < 4
    if is_causal is None:
        return tuple(False for _ in range(na_dim))
    if (
        isinstance(is_causal, tuple)
        and len(is_causal) == na_dim
        and all(isinstance(c, bool) for c in is_causal)
    ):
        return is_causal
    if (
        isinstance(is_causal, list)
        and len(is_causal) == na_dim
        and all(isinstance(c, bool) for c in is_causal)
    ):
        return tuple(is_causal)
    if isinstance(is_causal, bool):
        return tuple(is_causal for _ in range(na_dim))
    raise ValueError(
        "Invalid value for `is_causal`; expected a boolean or tuple of booleans, "
        f"got {type(is_causal)}"
    )


def check_all_args(
    na_dim: int, kernel_size: Any, dilation: Any, is_causal: Any
) -> Tuple[Tuple, Tuple, Tuple]:
    return (
        check_kernel_size_arg(na_dim, kernel_size),
        check_dilation_arg(na_dim, dilation),
        check_causal_arg(na_dim, is_causal),
    )


def check_tiling_config(na_dim: int, tiling_config: Any) -> Tuple[Tuple, Tuple]:
    assert na_dim > 0 and na_dim < 4
    if (
        isinstance(tiling_config, tuple)
        and len(tiling_config) == 2
        and all(isinstance(x, tuple) and len(x) == na_dim for x in tiling_config)
    ):
        return tiling_config
    raise ValueError(
        "Invalid tiling config; expected a pair of integer tuples, "
        f"got {type(tiling_config)}: {tiling_config}"
    )
