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

from typing import List, Optional

import torch  # noqa: F401
from torch import Tensor

from ....types import (
    FlashFmhaForwardConfigType,
    FlashFmhaBackwardConfigType,
    FlashFnaForwardConfigType,
    FlashFnaBackwardConfigType,
    DimensionType
)
from ....utils.checks import check_tile_shape
from ....utils.device import get_device_cc


DTYPE_TO_BITS = {
    torch.float16: 16,
    torch.bfloat16: 16,
}

# {
#   NADim: {
#       Arch: {
#           Headdim: [
#               (q_tile_shape_tuple, kv_tile_shape_tuple)
#           ]
#       }
#   }
FLASH_FMHA_FORWARD_CONFIGS = {
    1: {
        80 : {
            32: [
                ((128,), (112,)),
            ],
            64: [
                ((128,), (112,)),
            ],
            96: [
                ((128,), (64,)),
            ],
            128: [
                ((128,), (128,)),
                ((128,), (64,)),
            ],
            192: [
                ((128,), (96,)),
            ],
            256: [
                ((128,), (96,)),
            ],
        },
        86 : {
            32: [
                ((128,), (112,)),
            ],
            64: [
                ((128,), (64,)),
            ],
            96: [
                ((128,), (64,)),
            ],
            128: [
                ((128,), (128,)),
            ],
            192: [
                ((128,), (96,)),
            ],
            256: [
                ((128,), (64,)),
            ],
        },
        89 : {
            32: [
                ((128,), (112,)),
            ],
            64: [
                ((128,), (64,)),
            ],
            96: [
                ((128,), (64,)),
            ],
            128: [
                ((128,), (128,)),
            ],
            192: [
                ((128,), (96,)),
            ],
            256: [
                ((128,), (64,)),
            ],
        },
    },

}

FLASH_FMHA_BACKWARD_CONFIGS = {
    1: {
        80: {
            32: [
                ((128,), (128,)),
            ],
            64: [
                ((128,), (128,)),
            ],
            96: [
                ((64,), (128,)),
            ],
            128: [
                ((64,), (128,)),
            ],
            192: [
                ((64,), (80,)),
            ],
            256: [
                ((64,), (64,)),
            ],
        },
        86: {
            32: [
                ((64,), (128,)),
            ],
            64: [
                ((64,), (128,)),
            ],
            96: [
                ((64,), (128,)),
            ],
            128: [
                ((64,), (96,)),
            ],
            192: [
                ((64,), (64,)),
            ],
            256: [
                ((32,), (64,)),
            ],
        },
        89: {
            32: [
                ((64,), (128,)),
            ],
            64: [
                ((64,), (128,)),
            ],
            96: [
                ((64,), (128,)),
            ],
            128: [
                ((64,), (96,)),
            ],
            192: [
                ((64,), (64,)),
            ],
            256: [
                ((32,), (64,)),
            ],
        },
    },
}


def get_all_forward_configs(
    input_tensor: Tensor,
) -> List[FlashFmhaForwardConfigType]:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim

    cc = get_device_cc(input_tensor.device)
    if cc not in [80, 86, 89]:
        return []

    dtype = input_tensor.dtype

    head_dim = input_tensor.shape[-1]

    # if dtype_bits not in FLASH_FMHA_FORWARD_CONFIGS[na_dim]:  # type: ignore
    if dtype not in [torch.float16, torch.bfloat16]:
        return []

    # if head_dim not in [32, 64, 128, 256]:
    if head_dim not in FLASH_FMHA_FORWARD_CONFIGS[na_dim][cc]:  # type: ignore
        return []

    return FLASH_FMHA_FORWARD_CONFIGS[na_dim][cc][head_dim]  # type: ignore


def get_all_backward_configs(
    input_tensor: Tensor,
) -> List[FlashFnaBackwardConfigType]:
    assert input_tensor.dim() in [4, 5, 6]
    na_dim = input_tensor.dim() - 3  # batch, heads, head_dim

    cc = get_device_cc(input_tensor.device)
    if cc not in [80, 86, 89]:
        return []

    dtype = input_tensor.dtype

    head_dim = input_tensor.shape[-1]

    # if dtype_bits not in FLASH_FMHA_FORWARD_CONFIGS[na_dim]:  # type: ignore
    if dtype not in [torch.float16, torch.bfloat16]:
        return []

    # if head_dim not in [32, 64, 128, 256]:
    if head_dim not in FLASH_FMHA_FORWARD_CONFIGS[na_dim][cc]:  # type: ignore
        return []


    return FLASH_FMHA_BACKWARD_CONFIGS[na_dim][cc][head_dim]  # type: ignore


# For FMHA
def get_all_fmha_forward_configs(
    input_tensor: Tensor,
) -> List[FlashFmhaForwardConfigType]:
    if input_tensor.dim() != 4:
        raise ValueError("Only 4-D tensors are supported in FMHA.")

    configs_multi_dim = get_all_forward_configs(input_tensor)
    assert all(len(q_t) == len(kv_t) == 1 for q_t, kv_t in configs_multi_dim)

    configs_fmha = [
        (q_t[0], kv_t[0]) for q_t, kv_t in configs_multi_dim
    ]

    return configs_fmha


def get_all_fmha_backward_configs(
    input_tensor: Tensor,
) -> List[FlashFmhaBackwardConfigType]:
    if input_tensor.dim() != 4:
        raise ValueError("Only 4-D tensors are supported in FMHA.")

    configs_multi_dim = get_all_backward_configs(input_tensor)
    assert all(len(q_t) == len(kv_t) == 1 for q_t, kv_t in configs_multi_dim)

    configs_fmha = [(q_t[0], kv_t[0]) for q_t, kv_t in configs_multi_dim]

    return configs_fmha


def get_default_forward_config(
    input_tensor: Tensor,
) -> FlashFnaForwardConfigType:
    all_configs = get_all_forward_configs(input_tensor)

    if len(all_configs) < 1:
        device_cc = get_device_cc(input_tensor.device)
        raise ValueError(
            "No configs exist for this use case; Flash FMHA/FNA does not support it: "
            f"{input_tensor.shape=}, {input_tensor.dtype=}, {device_cc=}."
        )

    return all_configs[0]

def get_default_backward_config(
    input_tensor: Tensor,
) -> FlashFnaBackwardConfigType:
    all_configs = get_all_backward_configs(input_tensor)

    if len(all_configs) < 1:
        device_cc = get_device_cc(input_tensor.device)
        raise ValueError(
            "No configs exist for this use case; Flash FMHA/FNA does not support it: "
            f"{input_tensor.shape=}, {input_tensor.dtype=}, {device_cc=}."
        )

    return all_configs[0]


def get_default_fmha_forward_config(
    input_tensor: Tensor,
) -> FlashFmhaForwardConfigType:
    if input_tensor.dim() != 4:
        raise ValueError("Only 4-D tensors are supported in FMHA.")

    q_t, kv_t = get_default_forward_config(input_tensor)
    assert len(q_t) == len(kv_t) == 1

    return q_t[0], kv_t[0]


def get_default_fmha_backward_config(
    input_tensor: Tensor,
) -> FlashFmhaBackwardConfigType:
    if input_tensor.dim() != 4:
        raise ValueError("Only 4-D tensors are supported in FMHA.")

    q_t, kv_t = get_default_backward_config(input_tensor)
    assert len(q_t) == len(kv_t) == 1

    return q_t[0], kv_t[0]


def check_cutlass_flash_fna_forward_config(
    input_tensor: Tensor,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
) -> FlashFnaForwardConfigType:
    raise NotImplementedError
    # assert input_tensor.dim() in [4, 5, 6]
    # na_dim = input_tensor.dim() - 3  # batch, heads, head_dim

    # if (q_tile_shape is None) ^ (kv_tile_shape is None):
    #     raise ValueError(
    #         "Please specify both q_tile_shape and kv_tile_shape, or neither one. "
    #         f"Got {q_tile_shape=}, {kv_tile_shape=}."
    #     )

    # default_q_tile_shape, default_kv_tile_shape = (
    #     get_default_forward_config(input_tensor=input_tensor)
    # )
    # if q_tile_shape is None and kv_tile_shape is None:
    #     return default_q_tile_shape, default_kv_tile_shape  # type: ignore[return-value]

    # elif q_tile_shape is None and kv_tile_shape is None:
    #     q_tile_shape = default_q_tile_shape
    #     kv_tile_shape = default_kv_tile_shape

    # q_tile_shape = check_tile_shape(q_tile_shape)
    # kv_tile_shape = check_tile_shape(kv_tile_shape)

    # configs = get_all_forward_configs(input_tensor=input_tensor)

    # for q_t, kv_t in configs:
    #     if (
    #         q_t == q_tile_shape
    #         and kv_t == kv_tile_shape
    #     ):
    #         return q_t, kv_t  # type: ignore

    # # Fail and make suggestions
    # MAX_EXAMPLES = 3
    # examples = ""
    # for i, (q_t, kv_t) in enumerate(configs):
    #     examples += f"\n  q_tile_shape={q_t}, kv_tile_shape={kv_t}"
    #     if i > MAX_EXAMPLES:
    #         break

    # raise ValueError(
    #     f"Invalid configuration for Flash FNA-{na_dim}D. "
    #     f"Q tile shape {q_tile_shape}, KV tile shape {kv_tile_shape} "
    #     f"are not among the {len(configs)} configurations implementable "
    #     f"with Flash FNA. "
    #     "Try selecting a combination from: \n"
    #     "  natten.get_configs_for_flash_fna(q, k, v)"
    #     "\n"
    #     "Here's a few examples of available combinations for your use case:\n"
    #     f"{examples}"
    # )


def check_cutlass_flash_fna_backward_config(
    input_tensor: Tensor,
    q_tile_shape: Optional[DimensionType] = None,
    kv_tile_shape: Optional[DimensionType] = None,
) -> FlashFmhaBackwardConfigType:
    raise NotImplementedError
    # assert input_tensor.dim() in [4, 5, 6]
    # na_dim = input_tensor.dim() - 3  # batch, heads, head_dim

    # if (q_tile_shape is None) ^ (kv_tile_shape is None):
    #     raise ValueError(
    #         "Please specify both q_tile_shape and kv_tile_shape, or neither one. "
    #         f"Got {q_tile_shape=}, {kv_tile_shape=}."
    #     )

    # default_q_tile_shape, default_kv_tile_shape = get_default_backward_config(
    #     input_tensor=input_tensor
    # )
    # if q_tile_shape is None and kv_tile_shape is None:
    #     return default_q_tile_shape, default_kv_tile_shape  # type: ignore[return-value]

    # elif q_tile_shape is None and kv_tile_shape is None:
    #     q_tile_shape = default_q_tile_shape
    #     kv_tile_shape = default_kv_tile_shape

    # q_tile_shape = check_tile_shape(q_tile_shape)
    # kv_tile_shape = check_tile_shape(kv_tile_shape)

    # configs = get_all_backward_configs(input_tensor=input_tensor)

    # for q_t, kv_t in configs:
    #     if q_t == q_tile_shape and kv_t == kv_tile_shape:
    #         return q_t, kv_t  # type: ignore

    # # Fail and make suggestions
    # MAX_EXAMPLES = 3
    # examples = ""
    # for i, (q_t, kv_t) in enumerate(configs):
    #     examples += f"\n  q_tile_shape={q_t}, kv_tile_shape={kv_t}"
    #     if i > MAX_EXAMPLES:
    #         break

    # raise ValueError(
    #     f"Invalid configuration for Flash FNA-{na_dim}D Backward. "
    #     f"Q tile shape {q_tile_shape} and KV tile shape {kv_tile_shape} "
    #     f"are not among the {len(configs)} configurations implementable "
    #     f"with CUTLASS Flash FNA Backward. "
    #     "Try selecting a combination from: \n"
    #     "  natten.get_bwd_configs_for_flash_fna(q, k, v)"
    #     "\n"
    #     "Here's a few examples of available combinations for your use case:\n"
    #     f"{examples}"
    # )


def check_flash_fmha_forward_config(
    input_tensor: Tensor,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
) -> FlashFmhaForwardConfigType:
    assert input_tensor.dim() == 4

    if (q_tile_size is None) ^ (kv_tile_size is None):
        raise ValueError(
            "Please specify both q_tile_size and kv_tile_size, or neither one. "
            f"Got {q_tile_size=}, {kv_tile_size=}."
        )

    default_q_tile_size, default_kv_tile_size = (
        get_default_fmha_forward_config(input_tensor=input_tensor)
    )
    if q_tile_size is None and kv_tile_size is None:
        return default_q_tile_size, default_kv_tile_size

    elif q_tile_size is None and kv_tile_size is None:
        q_tile_size = default_q_tile_size
        kv_tile_size = default_kv_tile_size

    configs = get_all_fmha_forward_configs(input_tensor=input_tensor)

    for q_t, kv_t in configs:
        if (
            q_t == q_tile_size
            and kv_t == kv_tile_size
        ):
            return q_t, kv_t  # type: ignore

    # Fail and make suggestions
    MAX_EXAMPLES = 3
    examples = ""
    for i, q_t, kv_t in enumerate(configs):
        examples += f"\n  q_tile_size={q_t}, kv_tile_size={kv_t}"
        if i > MAX_EXAMPLES:
            break

    raise ValueError(
        f"Invalid configuration for Flash FMHA. "
        f"Q tile size {q_tile_size}, KV tile size {kv_tile_size} "
        f"are not among the {len(configs)} configurations implementable "
        f"with Flash FMHA. "
        "Try selecting a combination from: \n"
        "  natten.get_configs_for_flash_fmha(q, k, v)"
        "\n"
        "Here's a few examples of available combinations for your use case:\n"
        f"{examples}"
    )


def check_flash_fmha_backward_config(
    input_tensor: Tensor,
    q_tile_size: Optional[int] = None,
    kv_tile_size: Optional[int] = None,
) -> FlashFmhaBackwardConfigType:
    assert input_tensor.dim() == 4

    if (q_tile_size is None) ^ (kv_tile_size is None):
        raise ValueError(
            "Please specify both q_tile_size and kv_tile_size, or neither one. "
            f"Got {q_tile_size=}, {kv_tile_size=}."
        )

    default_q_tile_size, default_kv_tile_size = get_default_fmha_backward_config(
        input_tensor=input_tensor
    )
    if q_tile_size is None and kv_tile_size is None:
        return default_q_tile_size, default_kv_tile_size

    elif q_tile_size is None and kv_tile_size is None:
        q_tile_size = default_q_tile_size
        kv_tile_size = default_kv_tile_size

    configs = get_all_fmha_backward_configs(input_tensor=input_tensor)

    for q_t, kv_t in configs:
        if q_t == q_tile_size and kv_t == kv_tile_size:
            return q_t, kv_t  # type: ignore

    # Fail and make suggestions
    MAX_EXAMPLES = 3
    examples = ""
    for i, (q_t, kv_t) in enumerate(configs):
        examples += f"\n  q_tile_size={q_t}, kv_tile_size={kv_t}"
        if i > MAX_EXAMPLES:
            break

    raise ValueError(
        f"Invalid configuration for Flash FMHA Backward. "
        f"Q tile size {q_tile_size} and KV tile size {kv_tile_size} "
        f"are not among the {len(configs)} configurations implementable "
        f"with Flash FMHA Backward. "
        "Try selecting a combination from: \n"
        "  natten.get_bwd_configs_for_flash_fmha(q, k, v)"
        "\n"
        "Here's a few examples of available combinations for your use case:\n"
        f"{examples}"
    )
