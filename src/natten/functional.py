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
from typing import Optional

from torch import Tensor

from .autotuner import autotune_fna
from .nested import (
    na1d_av_nested,
    na1d_qk_nested,
    na2d_av_nested,
    na2d_qk_nested,
    na3d_av_nested,
    na3d_qk_nested,
)
from .ops import (
    na1d_av_op,
    na1d_op,
    na1d_qk_op,
    na2d_av_op,
    na2d_op,
    na2d_qk_op,
    na3d_av_op,
    na3d_op,
    na3d_qk_op,
)
from .types import (
    CausalArg1DTypeOrDed,
    CausalArg2DTypeOrDed,
    CausalArg3DTypeOrDed,
    Dimension1DTypeOrDed,
    Dimension2DTypeOrDed,
    Dimension3DTypeOrDed,
)
from .utils import log

logger = log.get_logger(__name__)


#################################################################################################
############################## Unfused neighborhood attention ops. ##############################
#################################################################################################


def na1d_qk(
    query: Tensor,
    key: Tensor,
    kernel_size: Dimension1DTypeOrDed,
    dilation: Dimension1DTypeOrDed = 1,
    additional_keys: Optional[Tensor] = None,
    is_causal: Optional[CausalArg1DTypeOrDed] = False,
    rpb: Optional[Tensor] = None,
) -> Tensor:
    if query.is_nested or key.is_nested:
        return na1d_qk_nested(
            query,
            key,
            rpb,
            kernel_size,
            dilation,
            additional_keys=additional_keys,
            is_causal=is_causal,
        )

    return na1d_qk_op(
        query=query,
        key=key,
        bias=rpb,
        additional_keys=additional_keys,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
    )


def na1d_av(
    attn: Tensor,
    value: Tensor,
    kernel_size: Dimension1DTypeOrDed,
    dilation: Dimension1DTypeOrDed = 1,
    additional_values: Optional[Tensor] = None,
    is_causal: Optional[CausalArg1DTypeOrDed] = False,
) -> Tensor:
    if attn.is_nested or value.is_nested:
        return na1d_av_nested(
            attn,
            value,
            kernel_size,
            dilation,
            additional_values=additional_values,
            is_causal=is_causal,
        )

    return na1d_av_op(
        attn=attn,
        value=value,
        additional_values=additional_values,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
    )


def na2d_qk(
    query: Tensor,
    key: Tensor,
    kernel_size: Dimension2DTypeOrDed,
    dilation: Dimension2DTypeOrDed = 1,
    additional_keys: Optional[Tensor] = None,
    is_causal: Optional[CausalArg2DTypeOrDed] = False,
    rpb: Optional[Tensor] = None,
) -> Tensor:
    if query.is_nested or key.is_nested:
        return na2d_qk_nested(
            query,
            key,
            rpb,
            kernel_size,
            dilation,
            additional_keys=additional_keys,
            is_causal=is_causal,
        )

    return na2d_qk_op(
        query=query,
        key=key,
        bias=rpb,
        additional_keys=additional_keys,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
    )


def na2d_av(
    attn: Tensor,
    value: Tensor,
    kernel_size: Dimension2DTypeOrDed,
    dilation: Dimension2DTypeOrDed = 1,
    additional_values: Optional[Tensor] = None,
    is_causal: Optional[CausalArg2DTypeOrDed] = False,
) -> Tensor:
    if attn.is_nested or value.is_nested:
        return na2d_av_nested(
            attn,
            value,
            kernel_size,
            dilation,
            additional_values=additional_values,
            is_causal=is_causal,
        )

    return na2d_av_op(
        attn=attn,
        value=value,
        additional_values=additional_values,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
    )


def na3d_qk(
    query: Tensor,
    key: Tensor,
    kernel_size: Dimension3DTypeOrDed,
    dilation: Dimension3DTypeOrDed = 1,
    additional_keys: Optional[Tensor] = None,
    is_causal: Optional[CausalArg3DTypeOrDed] = False,
    rpb: Optional[Tensor] = None,
) -> Tensor:
    if query.is_nested or key.is_nested:
        return na3d_qk_nested(
            query,
            key,
            rpb,
            kernel_size,
            dilation,
            additional_keys=additional_keys,
            is_causal=is_causal,
        )

    return na3d_qk_op(
        query=query,
        key=key,
        bias=rpb,
        additional_keys=additional_keys,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
    )


def na3d_av(
    attn: Tensor,
    value: Tensor,
    kernel_size: Dimension3DTypeOrDed,
    dilation: Dimension3DTypeOrDed,
    additional_values: Optional[Tensor] = None,
    is_causal: Optional[CausalArg3DTypeOrDed] = False,
) -> Tensor:
    if attn.is_nested or value.is_nested:
        return na3d_av_nested(
            attn,
            value,
            kernel_size,
            dilation,
            additional_values=additional_values,
            is_causal=is_causal,
        )

    return na3d_av_op(
        attn=attn,
        value=value,
        additional_values=additional_values,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
    )


#################################################################################################
############################ Fused neighborhood attention (FNA) ops. ############################
#################################################################################################


def na1d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension1DTypeOrDed,
    dilation: Dimension1DTypeOrDed = 1,
    is_causal: Optional[CausalArg1DTypeOrDed] = False,
    rpb: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> Tensor:
    if query.is_nested or key.is_nested or value.is_nested:
        raise NotImplementedError(
            "Fused neighborhood attention does not support nested tensors yet."
        )

    tiling_config_forward, tiling_config_backward = autotune_fna(
        1, query, kernel_size, dilation, is_causal
    )
    scale = scale or query.shape[-1] ** -0.5

    return na1d_op(
        query=query,
        key=key,
        value=value,
        bias=rpb,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
        tiling_config_forward=tiling_config_forward,
        tiling_config_backward=tiling_config_backward,
    )


def na2d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension2DTypeOrDed,
    dilation: Dimension2DTypeOrDed = 1,
    is_causal: Optional[CausalArg2DTypeOrDed] = False,
    rpb: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> Tensor:
    if query.is_nested or key.is_nested or value.is_nested:
        raise NotImplementedError(
            "Fused neighborhood attention does not support nested tensors yet."
        )

    tiling_config_forward, tiling_config_backward = autotune_fna(
        2, query, kernel_size, dilation, is_causal
    )
    scale = scale or query.shape[-1] ** -0.5

    return na2d_op(
        query=query,
        key=key,
        value=value,
        bias=rpb,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
        tiling_config_forward=tiling_config_forward,
        tiling_config_backward=tiling_config_backward,
    )


def na3d(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    kernel_size: Dimension3DTypeOrDed,
    dilation: Dimension3DTypeOrDed = 1,
    is_causal: Optional[CausalArg3DTypeOrDed] = False,
    rpb: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> Tensor:
    if query.is_nested or key.is_nested or value.is_nested:
        raise NotImplementedError(
            "Fused neighborhood attention does not support nested tensors yet."
        )

    tiling_config_forward, tiling_config_backward = autotune_fna(
        3, query, kernel_size, dilation, is_causal
    )
    scale = scale or query.shape[-1] ** -0.5

    return na3d_op(
        query=query,
        key=key,
        value=value,
        bias=rpb,
        kernel_size=kernel_size,
        dilation=dilation,
        is_causal=is_causal,
        scale=scale,
        tiling_config_forward=tiling_config_forward,
        tiling_config_backward=tiling_config_backward,
    )


#################################################################################################
######################################### Deprecated ops ########################################
#################################################################################################


def natten1dqkrpb(
    query: Tensor,
    key: Tensor,
    rpb: Optional[Tensor],
    kernel_size: int,
    dilation: int,
) -> Tensor:
    logger.warning(
        "You're calling NATTEN op `natten.functional.natten1dqkrpb`, which is deprecated "
        "in favor of `natten.functional.na1d_qk`. Please consider switching, as this op "
        "will be removed soon."
    )
    return na1d_qk(
        query, key, kernel_size=(kernel_size,), dilation=(dilation,), rpb=rpb
    )


def natten2dqkrpb(
    query: Tensor,
    key: Tensor,
    rpb: Optional[Tensor],
    kernel_size: int,
    dilation: int,
) -> Tensor:
    logger.warning(
        "You're calling NATTEN op `natten.functional.natten2dqkrpb`, which is deprecated "
        "in favor of `natten.functional.na2d_qk`. Please consider switching, as this op "
        "will be removed soon."
    )
    return na2d_qk(
        query,
        key,
        kernel_size=(kernel_size, kernel_size),
        dilation=(dilation, dilation),
        rpb=rpb,
    )


def natten3dqkrpb(
    query: Tensor,
    key: Tensor,
    rpb: Optional[Tensor],
    kernel_size_d: int,
    kernel_size: int,
    dilation_d: int,
    dilation: int,
) -> Tensor:
    logger.warning(
        "You're calling NATTEN op `natten.functional.natten3dqkrpb`, which is deprecated "
        "in favor of `natten.functional.na3d_qk`. Please consider switching, as this op "
        "will be removed soon."
    )
    return na3d_qk(
        query,
        key,
        kernel_size=(kernel_size_d, kernel_size, kernel_size),
        dilation=(dilation_d, dilation, dilation),
        rpb=rpb,
    )


def natten1dqk(
    query: Tensor,
    key: Tensor,
    kernel_size: int,
    dilation: int,
) -> Tensor:
    logger.warning(
        "You're calling NATTEN op `natten.functional.natten1dqk`, which is deprecated "
        "in favor of `natten.functional.na1d_qk`. Please consider switching, as this op "
        "will be removed soon."
    )
    return na1d_qk(query, key, kernel_size=(kernel_size,), dilation=(dilation,))


def natten1dav(
    attn: Tensor,
    value: Tensor,
    kernel_size: int,
    dilation: int,
) -> Tensor:
    logger.warning(
        "You're calling NATTEN op `natten.functional.natten1dav`, which is deprecated "
        "in favor of `natten.functional.na1d_av`. Please consider switching, as this op "
        "will be removed soon."
    )
    return na1d_av(attn, value, kernel_size=(kernel_size,), dilation=(dilation,))


def natten2dqk(
    query: Tensor,
    key: Tensor,
    kernel_size: int,
    dilation: int,
) -> Tensor:
    logger.warning(
        "You're calling NATTEN op `natten.functional.natten2dqk`, which is deprecated "
        "in favor of `natten.functional.na2d_qk`. Please consider switching, as this op "
        "will be removed soon."
    )
    return na2d_qk(
        query,
        key,
        kernel_size=(kernel_size, kernel_size),
        dilation=(dilation, dilation),
    )


def natten2dav(
    attn: Tensor,
    value: Tensor,
    kernel_size: int,
    dilation: int,
) -> Tensor:
    logger.warning(
        "You're calling NATTEN op `natten.functional.natten2dav`, which is deprecated "
        "in favor of `natten.functional.na2d_av`. Please consider switching, as this op "
        "will be removed soon."
    )
    return na2d_av(
        attn,
        value,
        kernel_size=(kernel_size, kernel_size),
        dilation=(dilation, dilation),
    )


def natten3dqk(
    query: Tensor,
    key: Tensor,
    kernel_size_d: int,
    kernel_size: int,
    dilation_d: int,
    dilation: int,
) -> Tensor:
    logger.warning(
        "You're calling NATTEN op `natten.functional.natten3dqk`, which is deprecated "
        "in favor of `natten.functional.na3d_qk`. Please consider switching, as this op "
        "will be removed soon."
    )
    return na3d_qk(
        query,
        key,
        kernel_size=(kernel_size_d, kernel_size, kernel_size),
        dilation=(dilation_d, dilation, dilation),
    )


def natten3dav(
    attn: Tensor,
    value: Tensor,
    kernel_size_d: int,
    kernel_size: int,
    dilation_d: int,
    dilation: int,
) -> Tensor:
    logger.warning(
        "You're calling NATTEN op `natten.functional.natten3dav`, which is deprecated "
        "in favor of `natten.functional.na3d_av`. Please consider switching, as this op "
        "will be removed soon."
    )
    return na3d_av(
        attn,
        value,
        kernel_size=(kernel_size_d, kernel_size, kernel_size),
        dilation=(dilation_d, dilation, dilation),
    )
