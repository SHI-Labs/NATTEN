# Changelog

## [0.15.0] - 2024-01-09
* Refactored kernels
  * The backend is messy, particularly the CUDA backend. A step in the right direction is at least factoring out duplicated.
  * Out of the 7 operations in NATTEN's backend, 6 have duplicates (really 3 underlying ops with different inputs.)
  * See #26 for more details.
* 3D Neighborhood Attention: naive CPU and CUDA kernels were added.
* Major refactoring of the C++ API (#38, #47, #53, and #81)
* GEMM kernels (#38 and #47)
* New build system with cmake (#38, #53, #81)
* Bfloat16 support (#38 and #81)
* Kepler and Maxwell support (#81)
* Forward mode automatic differentiation support (#74)
* Experimental support for Nested Tensors (inference only) (#76)
* Type checking, clang format, and other typesetting/formatting changes (#80)
* Added profiling scripts (#81)

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
