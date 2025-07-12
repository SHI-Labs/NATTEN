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

from ._environment import HAS_LIBNATTEN

from .backends import (
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

from .context import (
    allow_flex_compile,
    allow_flex_compile_backprop,
    are_deterministic_algorithms_enabled,
    disable_flex_compile,
    disable_flex_compile_backprop,
    get_memory_usage_preference,
    is_flex_compile_allowed,
    is_flex_compile_backprop_allowed,
    is_kv_parallelism_in_fused_na_enabled,
    is_memory_usage_default,
    is_memory_usage_strict,
    is_memory_usage_unrestricted,
    set_memory_usage_preference,
    use_deterministic_algorithms,
    use_kv_parallelism_in_fused_na,
)

from .functional import attention, merge_attentions, na1d, na2d, na3d

from .modules import (
    NeighborhoodAttention1D,
    NeighborhoodAttention2D,
    NeighborhoodAttention3D,
)

from .version import __version__

__all__ = [
    "__version__",
    "NeighborhoodAttention1D",
    "NeighborhoodAttention2D",
    "NeighborhoodAttention3D",
    "are_deterministic_algorithms_enabled",
    "use_deterministic_algorithms",
    "use_kv_parallelism_in_fused_na",
    "is_kv_parallelism_in_fused_na_enabled",
    "set_memory_usage_preference",
    "get_memory_usage_preference",
    "is_memory_usage_default",
    "is_memory_usage_strict",
    "is_memory_usage_unrestricted",
    "is_flex_compile_allowed",
    "is_flex_compile_backprop_allowed",
    "allow_flex_compile",
    "allow_flex_compile_backprop",
    "disable_flex_compile",
    "disable_flex_compile_backprop",
    "get_bwd_configs_for_cutlass_fmha",
    "get_bwd_configs_for_cutlass_fna",
    "get_configs_for_cutlass_fmha",
    "get_configs_for_cutlass_fna",
    "get_configs_for_cutlass_hopper_fmha",
    "get_bwd_configs_for_cutlass_hopper_fmha",
    "get_configs_for_cutlass_hopper_fna",
    "get_bwd_configs_for_cutlass_hopper_fna",
    "get_bwd_configs_for_cutlass_blackwell_fmha",
    "get_bwd_configs_for_cutlass_blackwell_fna",
    "get_configs_for_cutlass_blackwell_fmha",
    "get_configs_for_cutlass_blackwell_fna",
    "get_configs_for_flex_fmha",
    "get_configs_for_flex_fna",
    "HAS_LIBNATTEN",
    "na1d",
    "na2d",
    "na3d",
    "attention",
    "merge_attentions",
]
