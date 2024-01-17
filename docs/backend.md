## NATTEN Backend

### CPU

CPU backend is **very** limited. It only implements BMM-style NA, and not fused NA.

#### Naive

##### Description:
Very simple C++ implementations of BMM-style ops, which enable inference on non-CUDA devices and also serve as a reference for
CUDA in unit tests ([read more](tests.md)).
These implementations are NOT performance optimized.

##### Dependencies:
* `libtorch`: Torch API used for AVX.

### CUDA

#### Naive

##### Description:
Originally developed [back in 2022](history.md), slightly tuned in terms of launch parameters among other factors, but very
naive implementations.

*Tiled variants:* also developed [back in 2022](history.md), they implement only the PN-2D operation when dimensions per
attention head is 32.

##### Dependencies (excluding CUDA runtime):
* `libtorch`: half atomic add in the RPB backward kernel.

#### GEMM

##### Description:
TBD.

##### Dependencies (excluding CUDA runtime):
* CUTLASS

#### FNA

##### Description:
TBD.

##### Dependencies (excluding CUDA runtime):
* CUTLASS
