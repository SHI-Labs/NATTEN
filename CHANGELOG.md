# Changelog

## [Main branch]
* Bug in [modules](https://natten.org/modules) was fixed (see #228, #229).


## [0.20.0] - 2025-06-07
This release includes the new kernels and features discussed in
[Generalized Neighborhood Attention](https://arxiv.org/abs/2504.16922), and more.

* New documentations and website: [natten.org](https://natten.org/)
* New kernels: [Hopper FNA](https://natten.org/backends/#hopper-fna-fmha),
    [Blackwell FNA](https://natten.org/backends/#blackwell-fna-fmha), and
    [Flex FNA](https://natten.org/backends/#flex-fna-fmha) with multi-dimensional tiling.
    * Optional compilation of Flex Attention, and [guards](https://natten.org/context) against it
        due to instability. Read more
        [here](https://natten.org/context/#flex-attention-torchcompile).
* Add support for [Strided Neighborhood Attention](https://arxiv.org/abs/2504.16922):
  * You can now implement Neighborhood Attention with a delay step in the sliding window.
  * This feature can implement many sparse attention patterns, such as
        [HaloNet](https://arxiv.org/abs/2103.12731), Blocked Attention, and combinations in
        between.
  * All fused neighborhood attention APIs in NATTEN (`na1d`, `na2d`, `na3d`), and torch modules
        (`NeighborhoodAttention{1,2,3}d` now allow a `stride` argument, which has identical profile
        to `kernel_size` and `dilation`: either an integer, or a tuple with the same profile as the
        token layout.
  * This is feature supported by all of our existing and new backends.
  * When `stride == kernel_size`, the operation will implement Blocked Attention (a.k.a Window Self
        Attention in [Swin Transformer](https://arxiv.org/abs/2103.14030)).
* [Profiling toolkit](https://natten.org/profiler) now ships with NATTEN:
        `python -m natten.profiler`.
    * Removed Autotuner -- in favor of eventually replacing with profiling toolkit,
        [NATTEN Simulator](https://arxiv.org/abs/2504.16922), and:
    * Direct exposure of kernel configurations.
    * Interfaces for finding valid configurations for your use case
    * [Profiler dry runs](https://natten.org/profiler/#dry-run) can also help you navigate
        available backends, and their configurations that are suitable for your use case.
    * [Profiler optimize mode](https://natten.org/profiler/#optimize) can search through their
        configurations, and find the fastest one for your use case
* Dropped unfused / CPU backends in libnatten.
    * `na{1,2,3}d_{qk,qkrpb,av}` APIs and respective backends have been dropped. It was difficult
        to continue maintaining them, as they mostly ran with very outdated and naive kernels, and
        the exceptions to that were not at all flexible with respect to user parameters. Moving
        forward, we will only provide
        [Fused Neighborhood Attention](https://arxiv.org/abs/2403.04690) kernels, but unfused
        kernels may be revisited depending on demand and use case.
    * CPU implementations were all unfused, and were very limited as well, and are likewise removed.
    * Our new [Flex FNA](https://natten.org/backends/#flex-fna-fmha) backend will serve as the
        default option for non-NVIDIA GPU users.
* Dropped support for RPB.
* Dropped support for experimental torch ops.
* Massively improved error messages, type checking.
* Considerable refactor of libnatten, and reduced binary size.
* Unified interfaces for 1D/2D/3D forms, while still offering rank-specific interfaces.
* `torch < 2.7` is no longer officially supported.

## [0.17.5] - 2025-03-20
* Added support for even-sized kernels!
  * NATTEN now allows any kernel size greater than 1 in fused ops.
  * Only available in Fused NA (both the CUTLASS 2.X kernels and Flex) for now.
  * NOTE: any even sized kernel `2r` will force each token to attend to `r` tokens on the left,
    itself, and `r - 1` tokens on the right (in non-corner cases).
* Added Flex Attention as a backend.
    * Now you can use Flex Attention instead of FNA through NATTEN directly.
    * Just import `use_flex_attention()` from `natten`, call it, and enjoy potentially significant
     speedups on newer architectures.
    * With support for additional KV tokens.
    * NOTE: we've been observing some instabilities with Flex Attention when using torch 2.6. We'll
    try to raise the issue with the PyTorch team, but please proceed with caution.
* Better precision on fused ops with additional KV.
* Torch 2.6 support.
* Dropped support for CTK 11.X, and `torch < 2.5`
* Dropped deprecated ops (`natten.functional.natten*d{qk,qkrpb,av}`)

## [0.17.4] - 2025-01-28
* Support for additional KV tokens in FNA (requires xFormers)
  * Adds experimental support for additional KV tokens (attend to local neighborhood, and some
  additional context) to FNA interfaces, with training support.
  * The attention branch between Q and additional KVs runs with xFormers, which targets FAv2.
  Eventually, we'd want this branch to use PyTorch's SDPA directly, but as of now, there is no
  SDPA interface that returns logsumexp along with the output, which makes this impossible.
  * Reduction is done in pure torch, and if possible will be fused into a single op with torch
  compile.
  * In theory, any number of different attentions can be merged in this way, but the interface only
  allows one additional KV set for now.
* Bug fixes in FLOP counter (with fvcore)
  * Bugs introduced since FNA backwards were fixed.
  * Finally added unit tests.
* Better FLOP counting support
  * Rename instances of FLOP counting with fvcore to MACs, since that's what fvcore reports.
  * Add experimental support for torch's native flop counter
  * Better documentation

## [0.17.3] - 2024-11-01
* Bug fix for `torch < 2.4`
* 0.17.2 release will be directly replaced with 0.17.3.

## [0.17.2] - 2024-10-29 (REMOVED)
* Enable KV parallelism by default
  * No realistic use case will disable KV parallelism, because it virtually kills occupancy in any
  small-batch/few-head case. Most packages should be using this by default, the same way PyTorch's
  deterministic mode is disabled by default. Users will still get a warning if PyTorch or NATTEN's
  deterministic mode is enabled. (#167)
* Bug fixes
  * Fix rare DDP issue (#167).
  * Fix inconsistencies in docs. (#167)
* QoL
  * Switch from `torch.cuda.amp` to `torch.amp` since the former is deprecated (#168)
* Binaries for torch 2.5.

## [0.17.1] - 2024-05-19
* Fixed interface for python 3.8 and 3.9

## [0.17.0] - 2024-05-02
* [Fused neighborhood attention](https://github.com/SHI-Labs/NATTEN/tree/main/docs/fna) (FNA) kernels
  * 1D, 2D and 3D Neighborhood Attention are supported,
  * Causal neighborhood attention is implemented,
  * Window (kernel) size, dilation, and causality can be defined *per-axis*,
  * All GPU architectures since Maxwell (SM50) are supported,
    * SM50 up to SM70 are SIMT-only, but support both FP16 and FP32,
    * SM70 and SM75 target Tensor Cores in FP16, and SIMT-style in FP32,
    * SM80 and above target Tensor Cores in FP16, BF16, and FP32.
  * NATTEN [Auto-tuner](https://github.com/SHI-Labs/NATTEN/blob/main/docs/fna/autotuner.md),
  * Memory preferences and [KV parallelism](https://github.com/SHI-Labs/NATTEN/blob/main/docs/fna/kv-parallelism.md) modes,
  * Relative positional biases are only supported in forward pass (inference).
  * Memory layout in FNA is different from existing kernels (`[B, *, heads, dim]` instead of `[B, heads, *, dim]`.)
    * Eventually this layout can skip over the permute/explicit reshape step in the attention module following
    the QKV projection.
    * For more refer to [Fused vs unfused NA](docs/fna/fused-vs-unfused.md). 
* Naive kernels now implement and allow causal masking,
* Naive kernels (CPU and CUDA) now allow varying parameters (window size, dilation, causal) across axes,
* Major bug fix in Volta GEMM kernels
  * The epilogue was different for Volta, and it slipped through unit tests,
  * Tests are now more aggressive, and the issue has been fixed.
* Memory alignment bug in half RPB gradient kernels fixed
  * See [#97](https://github.com/SHI-Labs/NATTEN/issues/97).

## [0.15.1] - 2024-01-24
* Attention tensors can now be views, which allows combining neighborhood and any other attention pattern (i.e. registers,
  cross attention tokens, and the like) without extra copies. ([#85](https://github.com/SHI-Labs/NATTEN/pull/85) and [#87](https://github.com/SHI-Labs/NATTEN/pull/87)).
* Minor bug fixes ([#86](https://github.com/SHI-Labs/NATTEN/pull/86) and [#94](https://github.com/SHI-Labs/NATTEN/pull/94)).

## [0.15.0] - 2024-01-09
* Refactored kernels
  * The backend is messy, particularly the CUDA backend. A step in the right direction is at least factoring out duplicated.
  * Out of the 7 operations in NATTEN's backend, 6 have duplicates (really 3 underlying ops with different inputs.)
  * See [#26](https://github.com/SHI-Labs/NATTEN/pull/26) for more details.
* 3D Neighborhood Attention: naive CPU and CUDA kernels were added.
* Major refactoring of the C++ API ([#38](https://github.com/SHI-Labs/NATTEN/pull/38), [#47](https://github.com/SHI-Labs/NATTEN/pull/47), [#53](https://github.com/SHI-Labs/NATTEN/pull/53), and [#81](https://github.com/SHI-Labs/NATTEN/pull/81))
* GEMM kernels ([#38](https://github.com/SHI-Labs/NATTEN/pull/38) and [#47](https://github.com/SHI-Labs/NATTEN/pull/47))
* New build system with cmake ([#38](https://github.com/SHI-Labs/NATTEN/pull/38), [#53](https://github.com/SHI-Labs/NATTEN/pull/53), [#81](https://github.com/SHI-Labs/NATTEN/pull/81))
* Bfloat16 support ([#38](https://github.com/SHI-Labs/NATTEN/pull/38) and [#81](https://github.com/SHI-Labs/NATTEN/pull/81))
* Kepler and Maxwell support ([#81](https://github.com/SHI-Labs/NATTEN/pull/81))
* Forward mode automatic differentiation support ([#74](https://github.com/SHI-Labs/NATTEN/pull/74))
* Experimental support for Nested Tensors (inference only) ([#76](https://github.com/SHI-Labs/NATTEN/pull/76))
* Type checking, clang format, and other typesetting/formatting changes ([#80](https://github.com/SHI-Labs/NATTEN/pull/80))
* Added profiling scripts ([#81](https://github.com/SHI-Labs/NATTEN/pull/81))

## [0.14.6] - 2023-03-21
Just a really small update that syncs the changes to the private branch.
It's mostly about the changed signature in both QK and AV (both 1D and 2D), where we now take in both kernel size and dilation.
Up to 0.14.4 we only took in dilation because QK always took in RPB, and RPB's shape is a function of kernel size, and AV took 
in attention weights, whose last axis is of size kernel size (kernel size squared in 2D).
As of 0.14.5, we support optional RPB, which means now we have to take in kernel size in QK. To make the signatures consistent,
we added kernel size to AV as well, along with assertions that kernel sizes match up, but that was not synced from the private
branch when we pushed out 0.14.5 because we wanted to support PyTorch 2.0 as soon as possible.
0.14.6 is just mostly to keep the master branch consistent with the latest release, since the signature difference would create
an inconsistency between the pip package and the master branch.

## [0.14.5] - 2023-03-16
 
### Added
- Torch 2.0 support
- Optional RPB.

## [0.14.4] - 2022-10-31
 
### Added
- Python 3.10 and 3.11 wheels!
  - Only for supported torch versions.
- Support torch 1.13.
- Tiled NA2D for 3x3 kernels.

### Changed
- Minor changes to the setup script to fix `pip install natten`.

## [0.14.2] - 2022-10-15
 
### Added
- CPU support!
  - CPP backend for CPU computation.
  - CPU-only builds now supported.
  - Note we only have naive kernels for CPU at the moment. Feel free to open a PR!

### Changed
- Refactored the CPP/CUDA backend.
- Unit tests for NA1D and NA2D
  - Gradcheck tests in slow and fast mode
  - Gradcheck tests for CPU backend
  - Allclose tests between CPU and CUDA outputs and gradients

## [0.14.1] - 2022-10-08
 
### Added
- NATTEN is now available through PyPI and comes with pre-compiled wheels.
  - Wheel links and more information available on the project: [shi-labs.com/natten](https://www.shi-labs.com/natten/).
  - NATTEN now exists as a separate repository: [SHI-Labs/NATTEN](https://github.com/SHI-Labs/NATTEN).

### Changed
- Refactored the CPP/CUDA backend.
- Dropped LegacyNeighborhoodAttention (pure torch implementation) in favor of the upcoming CPU implementation.

 
## [0.13] - 2022-09-29
 
### Added
- Added dilation support to all kernels, including tiled NA kernels.
 
### Changed
- Renamed `NeighborhoodAttention` to `NeighborhoodAttention2D` in the python interface.
  - `NeighborhoodAttention` is now deprecated and will be removed in future versions.
- Renamed `NeighborhoodAttention` to `NeighborhoodAttention2D` in the python interface.
  - `NeighborhoodAttention` is now deprecated and will be removed in future versions.
 
## [0.12] - 2022-07-09
 
### Added
- Fixed the race condition in K-Backward and V-Backward kernels.
  - This was handled previously with atomic adds (non-deterministic).
  - Now the kernels compute inverse neighbors and compute gradients without a race condition.
- "Tiled" Neighborhood Attention kernels for 5x5, 7x7, 9x9, 11x11, and 13x13 window sizes.
  - Applies only to QKRPB-Forward and A-Backward.
  - Only supports dim per head = 32 for now.
    - Try to keep your channels a multiple of 32.
- Improved FP16 support.
  - Specific kernels for FP16 that use `half2` addressing.
  - Different threads per block for FP16.
- New 1D NA kernels
  - Slightly more efficient.
  - With FP16 support.
  - Arbitrary kernel sizes supported (there's no upper bound of 13x13 like the 2D version).
 
### Changed
- Window size templating, and some other common bits are factored out into a `commons` header file.
  - Makes extending support and editing easier.
- Gradchecks now run in fast mode (faster and less memory usage) unless instructed otherwise.
 
## [0.11a] - 2022-05-12
 
### Added
- 1D Neighborhood Attention
 
## [0.11] - 2022-04-30
 
### Changed
  
- Classification and downstream kernels combined.
- Refactored cuda extension to `natten/`.
- Minor speed improvements.
 
## [0.10]
Initial public release.
