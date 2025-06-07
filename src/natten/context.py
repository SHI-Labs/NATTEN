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
from enum import Enum

import torch

from .utils import log


logger = log.get_logger(__name__)


class MemoryUsagePreference(Enum):
    Default = 0
    Strict = 1
    Unrestricted = 2


class NattenContext:
    is_deterministic_mode_enabled: bool = False
    is_kv_parallelism_enabled: bool = True
    training_memory_preference: MemoryUsagePreference = MemoryUsagePreference.Default
    flex_compile_allowed: bool = False
    flex_compile_backprop_allowed: bool = False

    @staticmethod
    def reset():
        NattenContext.is_deterministic_mode_enabled = False
        NattenContext.is_kv_parallelism_enabled = True
        NattenContext.training_memory_preference = MemoryUsagePreference.Default
        NattenContext.flex_compile_allowed = False
        NattenContext.flex_compile_backprop_allowed = False


def set_memory_usage_preference(pref: str = "default"):
    """Sets memory usage preference for KV parallelism in `"cutlass-fna"` and `"cutlass-fmha"`
    backends.

    Args:
        pref: Choices are `"default"`, `"strict"`, and `"unrestricted"`.
    """
    if pref == "default":
        NattenContext.training_memory_preference = MemoryUsagePreference.Default
    elif pref == "strict":
        NattenContext.training_memory_preference = MemoryUsagePreference.Strict
    elif pref == "unrestricted":
        NattenContext.training_memory_preference = MemoryUsagePreference.Unrestricted
    else:
        raise ValueError(
            "natten.set_memory_usage_preference allows only one of three settings: "
            "`default`, `strict`, and `unrestricted`."
        )


def get_memory_usage_preference() -> MemoryUsagePreference:
    return NattenContext.training_memory_preference


def is_memory_usage_default() -> bool:
    """Returns whether memory usage preference for KV parallelism in `"cutlass-fna"` and
    `"cutlass-fmha"` backends is the default setting.
    """
    return get_memory_usage_preference() == MemoryUsagePreference.Default


def is_memory_usage_strict() -> bool:
    """Returns whether memory usage preference for KV parallelism in `"cutlass-fna"` and
    `"cutlass-fmha"` backends is the *restricted* setting.
    """
    return get_memory_usage_preference() == MemoryUsagePreference.Strict


def is_memory_usage_unrestricted() -> bool:
    """Returns whether memory usage preference for KV parallelism in `"cutlass-fna"` and
    `"cutlass-fmha"` backends is the *unrestricted* setting.
    """
    return get_memory_usage_preference() == MemoryUsagePreference.Unrestricted


def use_deterministic_algorithms(mode: bool = True):
    NattenContext.is_deterministic_mode_enabled = mode
    if mode:
        logger.warning(
            "You're enabling NATTEN's deterministic mode. This mode does not "
            "support auto-tuning, or training with positional biases. "
            "For more information please refer to https://github.com/SHI-Labs/NATTEN/tree/main/docs"
        )


def are_deterministic_algorithms_enabled() -> bool:
    return NattenContext.is_deterministic_mode_enabled


def use_kv_parallelism_in_fused_na(mode: bool = True):
    """Sets guards for using KV Parallelism in backpropagation in `"cutlass-fna"`/`"cutlass-fmha"`
    backends.

    Warning:
        Disabling KV parallelism can significantly slow down training, particularly in
        small-batch/head and large-token problems.

    Args:
        mode: If `True`, allows KV parallelism (default setting), and otherwise disables it.
    """
    if not mode:
        NattenContext.is_kv_parallelism_enabled = False
        return

    if torch.are_deterministic_algorithms_enabled():
        logger.warning(
            "Attempted to enable KV parallelism in FNA, which is non-deterministic, "
            "but PyTorch's deterministic flag has been enabled. Ignoring..."
        )
        return

    if are_deterministic_algorithms_enabled():
        raise RuntimeError(
            "You enabled NATTEN's deterministic mode, but attempted to "
            "enable KV parallelism, which results in non-determinism. "
        )

    NattenContext.is_kv_parallelism_enabled = True


def is_kv_parallelism_in_fused_na_enabled() -> bool:
    """Returns whether KV parallelism in `"cutlass-fna"` and `"cutlass-fmha"` backends is enabled."""
    return NattenContext.is_kv_parallelism_enabled


def is_flex_compile_allowed() -> bool:
    """Returns whether compilation is allowed in `"flex-fna"` and `"flex-fmha"` backends."""
    return NattenContext.flex_compile_allowed


def is_flex_compile_backprop_allowed() -> bool:
    """Returns whether compilation for backpropagation is allowed in `"flex-fna"` and `"flex-fmha"`
    backends.
    """
    return NattenContext.flex_compile_backprop_allowed


def allow_flex_compile(mode: bool = True, backprop: bool = False):
    """Sets guards for Flex Attention + `torch.compile`.

    Allows using our Flex FNA / Flex FMHA backends with `torch.compile`, meaning you can
    pass `torch_compile=True` to the `na{1,2,3}d` or `attention` operation, along with
    `backend="flex-fna"`/`backend="flex-fmha"`, and NATTEN will compile the block-sparse mask, as
    well as the attention operation using `torch.compile` for you.

    Warning:
        We have been *unable to verify the correctness* of this setting under all of our use
        cases. We are working on raising this issue with PyTorch directly, but until then we strongly
        recommend exercising caution when using this feature.

    Danger: backprop=True is strongly discouraged!
        Allowing `torch.compile` for backpropagation (detected by checking
        `tensor.requires_grad`) is guarded separately. We strongly recommend NOT using this setting, as
        it can impact your training results.

    Args:
        mode: If `True`, enable compilation for forward pass, otherwise disable.
        backprop: If `True`, assuming compilation for forward pass is allowed, enable compilation
            for backward pass, otherwise disable.
    """
    if not mode:
        NattenContext.flex_compile_allowed = False
        NattenContext.flex_compile_backprop_allowed = False

    if not NattenContext.flex_compile_allowed:
        logger.warning(
            "You are enabling Flex Attention compilation in NATTEN. "
            "NATTEN does not allow this by default, because we cannot verify Flex's correctness in all "
            "scenarios through NATTEN's tests. By choosing to override this, you acknowledge that your "
            "results may be affected significantly. If this was not intended, please call "
            "natten.disable_flex_compile()"
            ""
        )

    NattenContext.flex_compile_allowed = True

    if backprop:
        if not NattenContext.flex_compile_backprop_allowed:
            logger.warning(
                "You are enabling using compiled Flex Attention to backpropagate. "
                "NATTEN does not allow this by default, because we cannot verify Flex's correctness in all "
                "scenarios through NATTEN's tests, and it is HIGHLY discouraged. By choosing to override "
                "this, you acknowledge that your results may be heavily impacted significantly. "
                "If this was not intended, please call "
                "natten.disable_flex_compile_backprop()"
                ""
            )
        NattenContext.flex_compile_backprop_allowed = True


def allow_flex_compile_backprop(mode: bool = True):
    """Sets guards for Flex Attention + `torch.compile` for backpropagation only.

    Args:
        mode: If `True`, enable compilation for backprop (assuming forward compilation is already
            enabled), otherwise disable.
    """
    return allow_flex_compile(is_flex_compile_allowed(), mode)


def disable_flex_compile():
    """Disallow Flex Attention + `torch.compile` entirely."""
    return allow_flex_compile(False)


def disable_flex_compile_backprop():
    """Disallow Flex Attention + `torch.compile` for backpropagation entirely."""
    return allow_flex_compile(is_flex_compile_allowed(), False)
