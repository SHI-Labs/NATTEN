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


from typing import Optional, Tuple

import torch

from ..backends import (
    can_run_cutlass_blackwell_fmha,
    can_run_cutlass_blackwell_fna,
    can_run_cutlass_fmha,
    can_run_cutlass_fna,
    can_run_cutlass_hopper_fmha,
    can_run_cutlass_hopper_fna,
    can_run_flex_attention,
    choose_backend,
    choose_fmha_backend,
    get_bwd_configs_for_cutlass_blackwell_fmha,
    get_bwd_configs_for_cutlass_blackwell_fna,
    get_bwd_configs_for_cutlass_fmha,
    get_bwd_configs_for_cutlass_fna,
    get_bwd_configs_for_cutlass_hopper_fmha,
    get_bwd_configs_for_cutlass_hopper_fna,
    get_compatible_backends,
    get_compatible_fmha_backends,
    get_configs_for_cutlass_blackwell_fmha,
    get_configs_for_cutlass_blackwell_fna,
    get_configs_for_cutlass_fmha,
    get_configs_for_cutlass_fna,
    get_configs_for_cutlass_hopper_fmha,
    get_configs_for_cutlass_hopper_fna,
    get_configs_for_flex_fmha,
    get_configs_for_flex_fna,
)
from ..types import DimensionType, KernelSchedule
from ..utils import log

from . import print_table

from .pretty_printer import opt_progress_bar
from .problem import Problem
from .profiling import measure_natten_runtime

logger = log.get_logger(__name__)

# TODO: these are duplicated from profiler.py
NATTEN_BACKENDS = ["cutlass-fna", "blackwell-fna", "hopper-fna", "flex-fna"]
NATTEN_FMHA_BACKENDS = ["cutlass-fmha", "blackwell-fmha", "hopper-fmha", "flex-fmha"]


def dry_run_for_backend(
    problem: Problem,
    backend: Optional[str],
    fmha_backend: Optional[str],
    backprop: bool,
    torch_compile: bool,
    max_configs: int,
    should_print: bool = True,
) -> Tuple[bool, list, list]:
    if backend is None and fmha_backend is None:
        raise ValueError(
            "Either backend or fmha_backend must be specified. "
            "If you're not a developer, please open an issue."
        )

    if backend is None and not problem.is_self_attn:
        raise ValueError(
            "FNA backend must be specified for non-self-attention case. "
            "If you're not a developer, please open an issue."
        )

    if fmha_backend is None and problem.is_self_attn:
        raise ValueError(
            "FMHA backend must be specified for self-attention case. "
            "If you're not a developer, please open an issue."
        )

    if backend is not None and backend not in NATTEN_BACKENDS:
        raise ValueError("--dry-run is only supported for NATTEN backends.")

    is_fmha = problem.is_self_attn

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    q, k, v = problem.make_qkv_tensors(
        device=torch_device, requires_grad=backprop, heads_last=False, flatten=is_fmha
    )

    fwd_configs, bwd_configs = None, None
    fwd_config_keys, bwd_config_keys = None, None
    selected_backend = ""

    if is_fmha:
        assert fmha_backend is not None
        assert fmha_backend in NATTEN_FMHA_BACKENDS
        selected_backend = fmha_backend

        if fmha_backend == "blackwell-fmha":
            assert can_run_cutlass_blackwell_fmha(q, k, v, raise_error=True)
            fwd_configs = get_configs_for_cutlass_blackwell_fmha(q, k, v)  # type: ignore[assignment]
            bwd_configs = get_bwd_configs_for_cutlass_blackwell_fmha(q, k, v)  # type: ignore[assignment]
            fwd_config_keys = ("q_tile_size", "kv_tile_size")
            bwd_config_keys = (
                "backward_q_tile_size",
                "backward_kv_tile_size",
            )

        elif fmha_backend == "hopper-fmha":
            assert can_run_cutlass_hopper_fmha(q, k, v, raise_error=True)
            fwd_configs = get_configs_for_cutlass_hopper_fmha(q, k, v)  # type: ignore[assignment]
            bwd_configs = get_bwd_configs_for_cutlass_hopper_fmha(q, k, v)  # type: ignore[assignment]
            fwd_config_keys = (("q_tile_size", "kv_tile_size"), "kernel_schedule")  # type: ignore[assignment]
            bwd_config_keys = (
                "backward_q_tile_shape",
                "backward_kv_tile_shape",
            )

        elif fmha_backend == "cutlass-fmha":
            assert can_run_cutlass_fmha(q, k, v, raise_error=True)
            fwd_configs = get_configs_for_cutlass_fmha(q, k, v)  # type: ignore[assignment]
            bwd_configs = get_bwd_configs_for_cutlass_fmha(q, k, v)  # type: ignore[assignment]
            fwd_config_keys = ("q_tile_size", "kv_tile_size")
            bwd_config_keys = (
                "backward_q_tile_size",
                "backward_kv_tile_size",
            )

        elif fmha_backend == "flex-fmha":
            assert can_run_flex_attention(
                q, k, v, torch_compile=torch_compile, raise_error=True
            )
            fwd_configs = get_configs_for_flex_fmha(q, k, v)  # type: ignore[assignment]
            fwd_config_keys = ("q_tile_size", "kv_tile_size")

        else:
            raise NotImplementedError()

    else:
        assert backend is not None
        assert backend in NATTEN_BACKENDS
        selected_backend = backend

        if backend == "blackwell-fna":
            assert can_run_cutlass_blackwell_fna(q, k, v, raise_error=True)
            fwd_configs = get_configs_for_cutlass_blackwell_fna(q, k, v)  # type: ignore[assignment]
            bwd_configs = get_bwd_configs_for_cutlass_blackwell_fna(q, k, v)  # type: ignore[assignment]
            fwd_config_keys = ("q_tile_shape", "kv_tile_shape")  # type: ignore[assignment]
            bwd_config_keys = (
                "backward_q_tile_shape",
                "backward_kv_tile_shape",
            )

        elif backend == "hopper-fna":
            assert can_run_cutlass_hopper_fna(q, k, v, raise_error=True)
            fwd_configs = get_configs_for_cutlass_hopper_fna(q, k, v)  # type: ignore[assignment]
            bwd_configs = get_bwd_configs_for_cutlass_hopper_fna(q, k, v)  # type: ignore[assignment]
            fwd_config_keys = (("q_tile_shape", "kv_tile_shape"), "kernel_schedule")  # type: ignore[assignment]
            bwd_config_keys = (
                "backward_q_tile_shape",
                "backward_kv_tile_shape",
            )

        elif backend == "cutlass-fna":
            assert can_run_cutlass_fna(q, k, v, raise_error=True)
            fwd_configs = get_configs_for_cutlass_fna(q, k, v)  # type: ignore[assignment]
            bwd_configs = get_bwd_configs_for_cutlass_fna(q, k, v)  # type: ignore[assignment]
            fwd_config_keys = ("q_tile_shape", "kv_tile_shape")  # type: ignore[assignment]
            bwd_config_keys = (
                "backward_q_tile_shape",
                "backward_kv_tile_shape",
            )

        elif backend == "flex-fna":
            assert can_run_flex_attention(
                q, k, v, torch_compile=torch_compile, raise_error=True
            )
            fwd_configs = get_configs_for_flex_fna(q, k, v)  # type: ignore[assignment]
            fwd_config_keys = ("q_tile_shape", "kv_tile_shape")  # type: ignore[assignment]

        else:
            raise NotImplementedError()

    # TODO: refactor this
    def print_configs(title, configs, keys) -> list:
        filtered_configs = []
        for cfg in configs:
            assert len(cfg) == len(keys)
            cfg_annotated = {}
            for key, val in zip(keys, cfg):
                if isinstance(key, tuple):
                    assert isinstance(key, tuple)
                    for key_sub, val_sub in zip(key, val):
                        assert isinstance(key_sub, str)
                        cfg_annotated[key_sub] = val_sub
                else:
                    assert isinstance(key, str)
                    cfg_annotated[key] = val

            filtered_configs.append(cfg_annotated)

        assert len(filtered_configs) > 0

        headers = filtered_configs[0].keys()
        assert all(x.keys() == headers for x in filtered_configs)
        values = [[str(y) for y in x.values()] for x in filtered_configs]

        if max_configs > 0 and len(values) > max_configs:
            values = values[:max_configs]
            values.append(["..." for _ in headers])

        if should_print:
            print_table(title, headers, values, has_footer=False)

        return filtered_configs

    assert fwd_configs is not None and fwd_config_keys is not None
    if len(fwd_configs) < 1:
        return False, [], []

    fwd_configs_out = print_configs(
        title=f"Backend: {selected_backend}\nForward pass configurations",
        configs=fwd_configs,
        keys=fwd_config_keys,
    )

    bwd_configs_out = []
    if bwd_configs is not None:
        assert bwd_config_keys is not None
        bwd_configs_out = print_configs(
            title=f"Backend: {selected_backend}\nBackward pass configurations",
            configs=bwd_configs,
            keys=bwd_config_keys,
        )

    return True, fwd_configs_out, bwd_configs_out


def dry_run(
    problem: Problem,
    backend: Optional[str],
    fmha_backend: Optional[str],
    backprop: bool,
    torch_compile: bool,
    max_configs: int,
):
    if backend is not None and backend not in NATTEN_BACKENDS:
        raise ValueError("Dry run is only supported for NATTEN backends.")

    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    is_fmha = problem.is_self_attn

    backends = None
    fmha_backends = None
    if backend is None:
        q, k, v = problem.make_qkv_tensors(
            device=torch_device, requires_grad=backprop, heads_last=False, flatten=False
        )

        backends = get_compatible_backends(q, k, v, torch_compile=torch_compile)
    else:
        backends = [backend]

    if fmha_backend is None:
        q_flat, k_flat, v_flat = problem.make_qkv_tensors(
            device=torch_device, requires_grad=backprop, heads_last=False, flatten=True
        )

        fmha_backends = get_compatible_fmha_backends(
            q_flat, k_flat, v_flat, torch_compile=torch_compile
        )
    else:
        fmha_backends = [fmha_backend]

    compatible_backends = fmha_backends if is_fmha else backends

    ran_successfully = False
    for b in compatible_backends:
        print(f"Use case is compatible with backend {b}.")
        result, _, _ = dry_run_for_backend(
            problem=problem,
            backend=None if is_fmha else b,
            fmha_backend=b if is_fmha else None,
            backprop=backprop,
            torch_compile=torch_compile,
            max_configs=max_configs,
        )
        ran_successfully = ran_successfully or result

    if not ran_successfully:
        raise RuntimeError(
            "No backends are compatible with this use case, or you selected an incompatible backend. "
            "Run with NATTEN_LOG_LEVEL=DEBUG to find out why."
        )


def find_configs(
    problem: Problem,
    backend: Optional[str],
    fmha_backend: Optional[str],
    backprop: bool,
    torch_compile: bool,
) -> Tuple[str, str, list, list]:
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    q, k, v = problem.make_qkv_tensors(
        device=torch_device, requires_grad=backprop, heads_last=False, flatten=False
    )
    q_flat, k_flat, v_flat = problem.make_qkv_tensors(
        device=torch_device, requires_grad=backprop, heads_last=False, flatten=True
    )

    backend = backend or choose_backend(q, k, v, torch_compile=torch_compile)
    fmha_backend = fmha_backend or choose_fmha_backend(
        q_flat, k_flat, v_flat, torch_compile=torch_compile
    )

    result, fwd_configs, bwd_configs = dry_run_for_backend(
        problem=problem,
        backend=backend,
        fmha_backend=fmha_backend,
        backprop=backprop,
        torch_compile=torch_compile,
        max_configs=0,
        should_print=False,
    )

    if not result:
        raise RuntimeError(
            f"Selected backends ({backend=}, {fmha_backend=}) are not compatible "
            "with your use case."
        )

    return backend, fmha_backend, fwd_configs, bwd_configs


def optimize(
    problem: Problem,
    warmup_steps: int,
    backend: str,
    fmha_backend: str,
    backprop: bool,
    q_tile: Optional[DimensionType],
    kv_tile: Optional[DimensionType],
    backward_q_tile: Optional[DimensionType],
    backward_kv_tile: Optional[DimensionType],
    persistent: bool,
    schedule: Optional[str],
    torch_compile: bool,
    q_tile_size: Optional[int],
    kv_tile_size: Optional[int],
    backward_q_tile_size: Optional[int],
    backward_kv_tile_size: Optional[int],
) -> Tuple[
    str,
    str,
    Optional[DimensionType],
    Optional[DimensionType],
    Optional[DimensionType],
    Optional[DimensionType],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[int],
    Optional[KernelSchedule],
]:
    if backend is not None and backend not in NATTEN_BACKENDS:
        raise ValueError(
            f"Optimize is only available for NATTEN backends, got {backend=}."
        )

    opt_backend, opt_fmha_backend, fwd_configs, bwd_configs = find_configs(
        problem,
        backend=backend,
        fmha_backend=fmha_backend,
        backprop=backprop,
        torch_compile=torch_compile,
    )

    # Override with user inputs, if any
    if q_tile is not None:
        assert kv_tile is not None
        fwd_configs = (
            [
                {
                    "q_tile_shape": q_tile,
                    "kv_tile_shape": kv_tile,
                    "q_tile_size": q_tile_size,
                    "kv_tile_size": kv_tile_size,
                    "kernel_schedule": schedule,
                }
            ]
            if opt_backend == "hopper-fna"
            else [
                {
                    "q_tile_shape": q_tile,
                    "kv_tile_shape": kv_tile,
                    "q_tile_size": q_tile_size,
                    "kv_tile_size": kv_tile_size,
                }
            ]
        )

    if backward_q_tile is not None and backprop:
        assert backward_kv_tile is not None
        # assert opt_backend == "cutlass-fna"
        bwd_configs = [
            {
                "backward_q_tile_shape": backward_q_tile,
                "backward_kv_tile_shape": backward_kv_tile,
                "backward_q_tile_size": backward_q_tile_size,
                "backward_kv_tile_size": backward_kv_tile_size,
            }
        ]

    def run_optimize(configs, run_backprop: bool):
        best_time_str = None
        best_time = None
        best_config = None

        for i, cfg in opt_progress_bar(enumerate(configs), total=len(configs)):
            logger.debug(f"Testing config [{i+1}/{len(configs)}]")
            # Manually measure time using cuda events instead of profiling due to an
            # intermittent bug.

            runtime = measure_natten_runtime(
                problem=problem,
                backend=opt_backend,
                fmha_backend=opt_fmha_backend,
                run_persistent_kernel=persistent,
                torch_compile=torch_compile,
                warmup_steps=warmup_steps,
                disable_backward=not run_backprop,
                **cfg,
            )
            runtime_str = f"{runtime:.2f} ms"

            # logged_ops = do_profile(
            #     problem=problem,
            #     backend=opt_backend,
            #     fmha_backend=opt_fmha_backend,
            #     persistent=persistent,
            #     torch_compile=torch_compile,
            #     warmup_steps=warmup_steps,
            #     backprop=run_backprop,
            #     max_retries=20,
            #     fail_on_capture_error=False, # just skip the config if it doesn't get captured.
            #     **cfg,
            # )
            # if logged_ops is None or len(logged_ops) < 1:
            #     logger.debug(f"Could not profile configuration due to profiler bug, skipping...")
            #     continue
            # total = sum(logged_ops)
            # runtime = total.time
            # runtime_str = total.time_str

            logger.debug(f"Configuration: {cfg=}, runtime: {runtime_str}")
            if best_time is None or runtime < best_time:
                best_time_str = runtime_str
                best_time = runtime
                best_config = cfg

        assert best_time is not None
        assert best_config is not None

        return best_config, best_time_str

    best_cfg = {
        "backend": opt_backend,
        "fmha_backend": opt_fmha_backend,
    }

    print()
    print(f"Searching {len(fwd_configs)} forward pass configs")
    best_fwd_cfg, best_fwd_time = run_optimize(fwd_configs, run_backprop=False)
    logger.debug(
        f"Best forward configuration: {best_fwd_cfg} (runtime: {best_fwd_time})"
    )
    best_cfg.update(best_fwd_cfg)

    if backprop and len(bwd_configs) > 0:
        print()
        print(f"Searching {len(bwd_configs)} backward pass configs")
        best_bwd_cfg, best_bwd_time = run_optimize(bwd_configs, run_backprop=True)
        logger.debug(
            f"Best backward configuration: {best_bwd_cfg} (runtime: {best_bwd_time})"
        )
        best_cfg.update(best_bwd_cfg)

    print()
    print_table(
        "Best configuration",
        ["Parameter", "Value"],
        [[k, str(v)] for k, v in best_cfg.items()],
        has_footer=False,
    )
    print()

    assert ("q_tile_size" in best_cfg and "kv_tile_size" in best_cfg) or (
        "q_tile_shape" in best_cfg and "kv_tile_shape" in best_cfg
    )
    assert not ("backward_q_tile_size" in best_cfg) ^ (
        "backward_kv_tile_size" in best_cfg
    )
    assert not ("backward_q_tile_shape" in best_cfg) ^ (
        "backward_kv_tile_shape" in best_cfg
    )

    opt_q_tile: Optional[DimensionType] = None
    opt_kv_tile: Optional[DimensionType] = None
    opt_backward_q_tile: Optional[DimensionType] = None
    opt_backward_kv_tile: Optional[DimensionType] = None
    opt_q_tile_size: Optional[int] = None
    opt_kv_tile_size: Optional[int] = None
    opt_backward_q_tile_size: Optional[int] = None
    opt_backward_kv_tile_size: Optional[int] = None

    if "q_tile_shape" in best_cfg and "kv_tile_shape" in best_cfg:
        opt_q_tile = best_cfg["q_tile_shape"]  # type: ignore[assignment]
        opt_kv_tile = best_cfg["kv_tile_shape"]  # type: ignore[assignment]

    if "backward_q_tile_shape" in best_cfg and "backward_kv_tile_shape" in best_cfg:
        opt_backward_q_tile = best_cfg["backward_q_tile_shape"]  # type: ignore[assignment]
        opt_backward_kv_tile = best_cfg["backward_kv_tile_shape"]  # type: ignore[assignment]

    if "q_tile_size" in best_cfg and "kv_tile_size" in best_cfg:
        opt_q_tile_size = best_cfg["q_tile_size"]  # type: ignore[assignment]
        opt_kv_tile_size = best_cfg["kv_tile_size"]  # type: ignore[assignment]

    if "backward_q_tile_size" in best_cfg and "backward_kv_tile_size" in best_cfg:
        opt_backward_q_tile_size = best_cfg["backward_q_tile_size"]  # type: ignore[assignment]
        opt_backward_kv_tile_size = best_cfg["backward_kv_tile_size"]  # type: ignore[assignment]

    opt_kernel_schedule: Optional[KernelSchedule] = (
        None if "kernel_schedule" not in best_cfg else best_cfg["kernel_schedule"]  # type: ignore[assignment]
    )

    return (
        opt_backend,
        opt_fmha_backend,
        opt_q_tile,
        opt_kv_tile,
        opt_backward_q_tile,
        opt_backward_kv_tile,
        opt_q_tile_size,
        opt_kv_tile_size,
        opt_backward_q_tile_size,
        opt_backward_kv_tile_size,
        opt_kernel_schedule,
    )
