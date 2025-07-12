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


from typing import List

from ..utils import log

logger = log.get_logger(__name__)

import torch  # noqa: F401
from torch import Tensor

from .blackwell_fmha import cutlass_blackwell_fmha
from .blackwell_fna import (
    cutlass_blackwell_fna_generic,
    na1d_cutlass_blackwell_fna,
    na2d_cutlass_blackwell_fna,
    na3d_cutlass_blackwell_fna,
)

from .configs import (
    get_bwd_configs_for_cutlass_blackwell_fmha,
    get_bwd_configs_for_cutlass_blackwell_fna,
    get_bwd_configs_for_cutlass_fmha,
    get_bwd_configs_for_cutlass_fna,
    get_bwd_configs_for_cutlass_hopper_fmha,
    get_bwd_configs_for_cutlass_hopper_fna,
    get_configs_for_cutlass_blackwell_fmha,
    get_configs_for_cutlass_blackwell_fna,
    get_configs_for_cutlass_fmha,
    get_configs_for_cutlass_fna,
    get_configs_for_cutlass_hopper_fmha,
    get_configs_for_cutlass_hopper_fna,
    get_configs_for_flex_fmha,
    get_configs_for_flex_fna,
)

from .configs.checks import (
    can_run_cutlass_blackwell_fmha,
    can_run_cutlass_blackwell_fna,
    can_run_cutlass_fna,
    can_run_cutlass_hopper_fmha,
    can_run_cutlass_hopper_fna,
    can_run_flex_attention,
)

from .flex import flex_fmha, flex_fna_generic, na1d_flex, na2d_flex, na3d_flex
from .fmha import can_run_cutlass_fmha, cutlass_fmha
from .fna import (
    cutlass_fna_generic,
    na1d_cutlass_fna,
    na2d_cutlass_fna,
    na3d_cutlass_fna,
)
from .hopper_fmha import cutlass_hopper_fmha
from .hopper_fna import (
    cutlass_hopper_fna_generic,
    na1d_cutlass_hopper_fna,
    na2d_cutlass_hopper_fna,
    na3d_cutlass_hopper_fna,
)


def choose_backend(
    query: Tensor, key: Tensor, value: Tensor, torch_compile: bool
) -> str:
    if can_run_cutlass_blackwell_fna(query, key, value):
        logger.debug("Backend not set; picked Blackwell FNA kernel.")
        return "blackwell-fna"

    if can_run_cutlass_hopper_fna(query, key, value):
        logger.debug("Backend not set; picked Hopper FNA kernel.")
        return "hopper-fna"

    if can_run_cutlass_fna(query, key, value):
        logger.debug("Backend not set; picked CUTLASS (2.X) FNA kernel.")
        return "cutlass-fna"

    if can_run_flex_attention(query, key, value, torch_compile=torch_compile):
        logger.debug("Backend not set; picked Flex Attention kernel.")
        return "flex-fna"

    raise NotImplementedError(
        "NATTEN could not find a suitable backend for this use case. "
        "Run with NATTEN_LOG_LEVEL=DEBUG to find out why."
    )


def choose_fmha_backend(
    query: Tensor, key: Tensor, value: Tensor, torch_compile: bool
) -> str:
    if can_run_cutlass_blackwell_fmha(query, key, value):
        logger.debug("Backend not set; picked Blackwell FMHA kernel.")
        return "blackwell-fmha"

    if can_run_cutlass_hopper_fmha(query, key, value):
        logger.debug("Backend not set; picked Hopper FMHA kernel.")
        return "hopper-fmha"

    if can_run_cutlass_fmha(query, key, value):
        logger.debug("Backend not set; picked CUTLASS (2.X) FMHA kernel.")
        return "cutlass-fmha"

    if can_run_flex_attention(query, key, value, torch_compile=torch_compile):
        logger.debug("Backend not set; picked Flex Attention kernel.")
        return "flex-fmha"

    raise NotImplementedError(
        "NATTEN could not find a suitable backend for this FMHA use case. "
        "Run with NATTEN_LOG_LEVEL=DEBUG to find out why."
    )


def get_compatible_backends(
    query: Tensor, key: Tensor, value: Tensor, torch_compile: bool
) -> List[str]:
    compatible_backends = []
    if can_run_cutlass_blackwell_fna(query, key, value):
        compatible_backends.append("blackwell-fna")

    if can_run_cutlass_hopper_fna(query, key, value):
        compatible_backends.append("hopper-fna")

    if can_run_cutlass_fna(query, key, value):
        compatible_backends.append("cutlass-fna")

    if can_run_flex_attention(query, key, value, torch_compile=torch_compile):
        compatible_backends.append("flex-fna")

    return compatible_backends


def get_compatible_fmha_backends(
    query: Tensor, key: Tensor, value: Tensor, torch_compile: bool
) -> List[str]:
    compatible_backends = []
    if can_run_cutlass_blackwell_fmha(query, key, value):
        compatible_backends.append("blackwell-fmha")

    if can_run_cutlass_hopper_fmha(query, key, value):
        compatible_backends.append("hopper-fmha")

    if can_run_cutlass_fmha(query, key, value):
        compatible_backends.append("cutlass-fmha")

    if can_run_flex_attention(query, key, value, torch_compile=torch_compile):
        compatible_backends.append("flex-fmha")

    return compatible_backends


__all__ = [
    "can_run_cutlass_fmha",
    "can_run_cutlass_fna",
    "can_run_cutlass_blackwell_fmha",
    "can_run_cutlass_blackwell_fna",
    "can_run_cutlass_hopper_fmha",
    "can_run_cutlass_hopper_fna",
    "can_run_flex_attention",
    "cutlass_fmha",
    "cutlass_fna_generic",
    "na1d_cutlass_fna",
    "na2d_cutlass_fna",
    "na3d_cutlass_fna",
    "cutlass_blackwell_fmha",
    "cutlass_blackwell_fna_generic",
    "cutlass_hopper_fmha",
    "cutlass_hopper_fna_generic",
    "na1d_cutlass_blackwell_fna",
    "na2d_cutlass_blackwell_fna",
    "na3d_cutlass_blackwell_fna",
    "flex_fmha",
    "flex_fna_generic",
    "na1d_flex",
    "na2d_flex",
    "na3d_flex",
    "na1d_cutlass_hopper_fna",
    "na2d_cutlass_hopper_fna",
    "na3d_cutlass_hopper_fna",
    "get_bwd_configs_for_cutlass_fmha",
    "get_bwd_configs_for_cutlass_fna",
    "get_bwd_configs_for_cutlass_blackwell_fmha",
    "get_bwd_configs_for_cutlass_blackwell_fna",
    "get_configs_for_cutlass_blackwell_fmha",
    "get_configs_for_cutlass_blackwell_fna",
    "get_configs_for_cutlass_fmha",
    "get_configs_for_cutlass_fna",
    "get_configs_for_cutlass_hopper_fmha",
    "get_bwd_configs_for_cutlass_hopper_fmha",
    "get_configs_for_cutlass_hopper_fna",
    "get_bwd_configs_for_cutlass_hopper_fna",
    "get_configs_for_flex_fmha",
    "get_configs_for_flex_fna",
]
