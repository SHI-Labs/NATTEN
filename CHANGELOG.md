# Changelog

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
