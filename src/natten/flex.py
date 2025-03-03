import functools
from typing import Any, Dict, Optional, Tuple

import torch
from torch import BoolTensor, IntTensor, Tensor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from .types import (
    CausalArg1DTypeOrDed,
    CausalArg2DTypeOrDed,
    CausalArg3DTypeOrDed,
    Dimension1DTypeOrDed,
    Dimension2DTypeOrDed,
    Dimension3DTypeOrDed,
)
from .utils import check_all_args


@functools.lru_cache(maxsize=1)
def get_flex_attention_compiled():
    return torch.compile(flex_attention, dynamic=False)


@functools.lru_cache(maxsize=None)
def get_block_mask(
    num_dimension: int,
    image_shape: Tuple[int],
    kernel_size: Tuple[int],
    dilation: Tuple[int],
    is_causal: Tuple[bool],
):

    def get_location_1d(idx: IntTensor) -> IntTensor:
        return (idx,)

    def get_location_2d(idx: IntTensor) -> Tuple[IntTensor, IntTensor]:
        return (idx // image_shape[1], idx % image_shape[1])

    def get_location_3d(idx: IntTensor) -> Tuple[IntTensor, IntTensor, IntTensor]:
        return (idx // image_shape[2] // image_shape[1], (idx // image_shape[2]) % image_shape[1], idx % image_shape[2])

    get_location = {
        1: get_location_1d,
        2: get_location_2d,
        3: get_location_3d,
    }[num_dimension]

    def natten_mask_mod(b: IntTensor, h: IntTensor, q_idx: IntTensor, kv_idx: IntTensor) -> BoolTensor:
        q_idx = get_location(q_idx)
        kv_idx = get_location(kv_idx)

        masks = []
        for i in range(num_dimension):
            dilate_kernel = kernel_size[i] * dilation[i]
            if is_causal[i]:
                mask = (
                    (q_idx[i] - kv_idx[i] >= 0)
                    & (q_idx[i] - kv_idx[i] < dilate_kernel)
                    & ((q_idx[i] % dilation[i]) == (kv_idx[i] % dilation[i]))
                )
            else:
                kernel_center_x = q_idx[i].clamp(
                    (dilate_kernel - 1) // 2, (image_shape[i] - 1) - (dilate_kernel - 1) // 2
                )
                mask = ((kernel_center_x - kv_idx[i]).abs() <= dilate_kernel // 2) & (
                    (q_idx[i] % dilation[i]) == (kv_idx[i] % dilation[i])
                )

            masks.append(mask)

        return functools.reduce(lambda x, y: x & y, masks)

    seq_length = functools.reduce(lambda x, y: x * y, image_shape)
    block_mask = create_block_mask(natten_mask_mod, 1, 1, seq_length, seq_length)
    return block_mask


def flex_na1d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension1DTypeOrDed,
    dilation: Dimension1DTypeOrDed = 1,
    is_causal: Optional[CausalArg1DTypeOrDed] = False,
    rpb: Optional[Tensor] = None,
    scale: Optional[float] = None,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    xformers_kwargs: Optional[Dict] = None,
) -> torch.Tensor:
    """
    Args:
        query: (batch_size, seq_length, num_head, head_dim)
        key: (batch_size, seq_length, num_head, head_dim)
        value: (batch_size, seq_length, num_head, head_dim)
        kernel_size: Union[int, Tuple[int]]
        dilation: Union[int, Tuple[int]]
        is_causal: Union[bool, Tuple[bool]]
    """

    kernel_size, dilation, is_causal = check_all_args(1, kernel_size, dilation, is_causal)
    assert rpb is None, "rpb is not supported"
    assert scale is None, "scale is not supported"
    assert additional_keys is None, "additional_keys is not supported"
    assert additional_values is None, "additional_values is not supported"
    assert xformers_kwargs is None, "xformers_kwargs is not supported"

    batch_size, seq_length, num_head, head_dim = query.shape
    image_shape = (seq_length,)

    _query = query.transpose(1, 2)
    _key = key.transpose(1, 2)
    _value = value.transpose(1, 2)

    block_mask = get_block_mask(1, image_shape, kernel_size, dilation, is_causal)
    flex_attention_compiled = get_flex_attention_compiled()
    out = flex_attention_compiled(_query, _key, _value, block_mask=block_mask)

    out = out.transpose(1, 2)

    return out


def flex_na2d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension2DTypeOrDed,
    dilation: Dimension2DTypeOrDed = 1,
    is_causal: Optional[CausalArg2DTypeOrDed] = False,
    rpb: Optional[Tensor] = None,
    scale: Optional[float] = None,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    xformers_kwargs: Optional[Dict] = None,
) -> torch.Tensor:
    """
    Args:
        query: (batch_size, image_height, image_width, num_head, head_dim)
        key: (batch_size, image_height, image_width, num_head, head_dim)
        value: (batch_size, image_height, image_width, num_head, head_dim)
        kernel_size: Union[int, Tuple[int, int]]
        dilation: Union[int, Tuple[int, int]]
        is_causal: Union[bool, Tuple[bool, bool]]
    """

    kernel_size, dilation, is_causal = check_all_args(2, kernel_size, dilation, is_causal)
    assert rpb is None, "rpb is not supported"
    assert scale is None, "scale is not supported"
    assert additional_keys is None, "additional_keys is not supported"
    assert additional_values is None, "additional_values is not supported"
    assert xformers_kwargs is None, "xformers_kwargs is not supported"

    batch_size, image_height, image_width, num_head, head_dim = query.shape
    seq_length = image_height * image_width
    image_shape = (image_height, image_width)

    _query = query.view(batch_size, seq_length, num_head, head_dim).transpose(1, 2)
    _key = key.view(batch_size, seq_length, num_head, head_dim).transpose(1, 2)
    _value = value.view(batch_size, seq_length, num_head, head_dim).transpose(1, 2)

    block_mask = get_block_mask(2, image_shape, kernel_size, dilation, is_causal)
    flex_attention_compiled = get_flex_attention_compiled()
    out = flex_attention_compiled(_query, _key, _value, block_mask=block_mask)

    out = out.transpose(1, 2).view(batch_size, image_height, image_width, num_head, head_dim)

    return out


def flex_na3d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension3DTypeOrDed,
    dilation: Dimension3DTypeOrDed = 1,
    is_causal: Optional[CausalArg3DTypeOrDed] = False,
    rpb: Optional[Tensor] = None,
    scale: Optional[float] = None,
    additional_keys: Optional[Tensor] = None,
    additional_values: Optional[Tensor] = None,
    xformers_kwargs: Optional[Dict] = None,
) -> torch.Tensor:
    """
    Args:
        query: (batch_size, image_height, image_width, num_head, head_dim)
        key: (batch_size, image_height, image_width, num_head, head_dim)
        value: (batch_size, image_height, image_width, num_head, head_dim)
        kernel_size: Union[int, Tuple[int, int]]
        dilation: Union[int, Tuple[int, int]]
        is_causal: Union[bool, Tuple[bool, bool]]
    """

    kernel_size, dilation, is_causal = check_all_args(3, kernel_size, dilation, is_causal)
    assert rpb is None, "rpb is not supported"
    assert scale is None, "scale is not supported"
    assert additional_keys is None, "additional_keys is not supported"
    assert additional_values is None, "additional_values is not supported"
    assert xformers_kwargs is None, "xformers_kwargs is not supported"

    batch_size, image_depth, image_height, image_width, num_head, head_dim = query.shape
    seq_length = image_depth * image_height * image_width
    image_shape = (image_depth, image_height, image_width)

    _query = query.view(batch_size, seq_length, num_head, head_dim).transpose(1, 2)
    _key = key.view(batch_size, seq_length, num_head, head_dim).transpose(1, 2)
    _value = value.view(batch_size, seq_length, num_head, head_dim).transpose(1, 2)

    block_mask = get_block_mask(3, image_shape, kernel_size, dilation, is_causal)
    flex_attention_compiled = get_flex_attention_compiled()
    out = flex_attention_compiled(_query, _key, _value, block_mask=block_mask)

    out = out.transpose(1, 2).view(batch_size, image_depth, image_height, image_width, num_head, head_dim)

    return out
