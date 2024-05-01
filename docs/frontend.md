## Front-end

This page describes the package's user-facing modules and operations in PyTorch.

### Modules
Using our modules are recommended if you intend to plug Neighborhood Attention into your existing architecture.
Simply import `NeighborhoodAttention1D`, `NeighborhoodAttention2D`, or `NeighborhoodAttention3D` from `natten`:
```python
from natten import NeighborhoodAttention1D
from natten import NeighborhoodAttention2D
from natten import NeighborhoodAttention3D

na1d = NeighborhoodAttention1D(dim=128, num_heads=4, kernel_size=7, dilation=2, is_causal=True)
na2d = NeighborhoodAttention2D(dim=128, num_heads=4, kernel_size=7, dilation=2)
na3d = NeighborhoodAttention3D(dim=128, num_heads=4, kernel_size=7, dilation=2)
```

You can also pass in tuples for `kernel_size`, `dilation`, and `is_causal`:
```python
na3d_video = NeighborhoodAttention3D(
  dim=128,
  kernel_size=(7, 9, 9),
  dilation=(1, 1, 1),
  is_causal=(True, False, False),
  num_heads=4)
```

Relative positional biases (RPB) is available in all modules as long as causal masking
is not enabled:
```python
na3d_with_bias = NeighborhoodAttention3D(
  dim=128,
  kernel_size=(7, 9, 9),
  dilation=(1, 1, 1),
  is_causal=False,
  rel_pos_bias=True,
  num_heads=4)
```

Modules expect inputs of shape `[batch_size, *, dim]`:

* NeighborhoodAttention1D: `[batch_size, sequence_length, dim]`
* NeighborhoodAttention2D: `[batch_size, height, width, dim]`
* NeighborhoodAttention3D: `[batch_size, depth, height, width, dim]`

#### Using Fused Neighborhood Attention
NATTEN can switch from the standard BMM-style backend implementation to our more recent Fused Neighborhood Attention (FNA), 
which operates similarly to Flash Attention in that attention weights are not stored to global memory.
You can expect improved performance, especially in half precision, and a potentially reduced memory footprint, especially if
you're dealing with large problem sizes.

Note that this feature is very new, and unlike the previous BMM-style implementations, offers a number
of different settings, which you may want to adjust to your use case.

To force NATTEN torch modules (`NeighborhoodAttention1D`, `NeighborhoodAttention2D`, and `NeighborhoodAttention3D`) to use FNA:

```python
from natten import enable_fused_na, disable_fused_na

enable_fused_na()
# Modules will start using fused neighborhood attention

disable_fused_na()
# Go back to BMM-style (default)
```

We highly recommend referring to [FNA quick start](fna/fna-quickstart.md) or 
the [fused vs unfused NA](fna/fused-vs-unfused.md) guide before
starting to use FNA, since the interface, memory layout, and feature set can differ from
all unfused ops in NATTEN.

#### Optimizing FNA 
Fused neighborhood attention can automatically tune its performance and provide additional gains in performance when
NATTEN's new autotuner is enabled.

Auto-tuner benchmarks every new problem (identified by problem size, data type, and NA parameters) once and caches
its optimal kernel configuration, which is reused throughout the lifetime of your program.

This means that when auto-tuner is activated, your first forward pass is expected to take longer than typical, 
but iterations that follow will likely be more performant than without auto-tuning.

Note that this feature only makes sense in static or mostly static runtimes (you don't expect your tensor shapes to change
very frequently after a certain point.)

This feature is still in early stages and relatively untested, and is not part of libnatten, and is therefore experimental and 
subject to change. Bug reports related to it and FNA in general are strongly appreciated.

NOTE: using auto-tuner during training, especially distributed training jobs, is not recommended at this time, as each
individual subprocess will have its own auto-tuner cache, and this can easily result in different processes using very
different kernel configurations, which can affect your reproducibility.

NOTE: auto-tuner cannot be used when
[PyTorch's deterministic mode](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)
is enabled.

```python
from natten import (
  use_fused_na,
  use_autotuner,
  disable_autotuner,
)

use_fused_na(True)
# FNA runs according to the default configuration for
# every problem size.

use_autotuner(forward_pass=True, backward_pass=False)
# FNA optimizes the configuration according
# to problem size. Enable for forward pass,
# disable for backward pass.

use_fused_na(False)
# Disable Fused NA (default)

disable_autotuner()
# Disable auto-tuner
# Or alternatively:
# use_autotuner(False, False)
```

For more information, refer to [autotuner guide](fna/autotuner.md).

#### Memory usage in FNA
Training with Fused Neighborhood Attention can be accelerated at the expense of using more global memory by using
KV parallelization. Depending on your use case (how big your memory footprint already is and what your memory cap is),
you can consider this option.

KV parallelism is disabled by default, and makes the backward pass non-deterministic, which means that it can't be used with
[PyTorch's deterministic mode](https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html).

To enable this feature:

```python
from natten import (
  use_kv_parallelism_in_fused_na,
  is_kv_parallelism_in_fused_na_enabled
)

use_kv_parallelism_in_fused_na(True)
# Enables using KV parallelism

use_kv_parallelism_in_fused_na(False)
# Go back to no KV parallelism (default)
```

If you're limited by memory capacity, but would still like to use KV parallelism, you can try adjusting
NATTEN's memory preference settings:

```python
from natten import set_memory_usage_preference

set_memory_usage_preference("strict")
# Strict: limits KV parallelism more than default.

set_memory_usage_preference("unrestricted")
# Unrestricted: no limit on KV parallelism other than
# maximum grid size.

set_memory_usage_preference("default")
# Default: limits KV parallelism, but not as aggressively
# as strict.
```

Future versions may offer more fine-grained control over this.

For more information, refer to [KV parallelism](fna/kv-parallelism.md).

### Operations
Operations are one level below our modules, and are intended to give you full control over the module-level
details, and only use the underlying neighborhood attention operators directly.
In other words, NATTEN operations are to NATTEN modules (above) what `torch.nn.functional.conv2d` is to `torch.nn.Conv2d`.

#### BMM-style neighborhood attention
Standard unfused implementation.
All unfused operations support:
* Additional KV tensors
  * Allows cross-attending to an additional key-value pair alongside neighborhood attention efficiently.
* Nested tensors (forward-pass only)
* Both forward mode and reverse automatic differentiation
* Relative positional biases (RPB)
  * RPB is not implemented for causal neighborhood attention
  * RPB is not supported in forward-mode autodiff

1-D example (with causal masking):
```python
from natten.functional import na1d_qk, na1d_av

attn_1d = na1d_qk(
  query_1d, key_1d,
  kernel_size=7, dilation=4, is_casual=True,
)
attn_1d = attn_1d.softmax(dim=-1)
output_1d = na1d_av(
  attn_1d, value_1d,
  kernel_size=7, dilation=4, is_casual=True,
)
```

2-D example (with RPB):
```python
from natten.functional import na2d_qk, na2d_av

attn_2d = na2d_qk(
  query_2d, key_2d,
  rpb=rpb_2d,
  kernel_size=7, dilation=4,
)
attn_2d = attn_2d.softmax(dim=-1)
output_2d = na2d_av(
  attn_2d, value_2d,
  kernel_size=7, dilation=4,
)
```

3-D example (with varying parameters):
```python
from natten.functional import na3d_qk, na3d_av

attn_3d = na3d_qk(
  query_3d, key_3d,
  kernel_size=(7, 9, 9),
  dilation=(1, 1, 1),
  is_causal=(True, False, False),
)
attn_3d = attn_3d.softmax(dim=-1)
output_3d = na3d_av(
  attn_3d, value_3d,
  kernel_size=(7, 9, 9),
  dilation=(1, 1, 1),
  is_causal=(True, False, False),
)
```

#### Fused neighborhood attention
These ops are a very recent addition, and their behavior and signatures may change in the future.

```python
from natten.functional import na1d, na2d, na3d

output_1d = na1d(
  query_1d, key_1d, value_1d,
  kernel_size=7, dilation=4, is_casual=True
)
output_2d = na2d(
  query_2d, key_2d, value_2d,
  kernel_size=7, dilation=4, is_casual=False
)
output_3d = na3d(
  query_3d, key_3d, value_3d,
  kernel_size=7, dilation=4, is_casual=False
)
```

Similarly to modules, you can tuples for `kernel_size`, `dilation`, and `is_causal`:
```python
output_2d = na2d(
  query_2d, key_2d, value_2d,
  kernel_size=(7, 3), dilation=(1, 3), is_casual=(True, False)
)
output_3d = na3d(
  query_3d, key_3d, value_3d,
  kernel_size=(7, 3, 5), dilation=(1, 1, 1), is_casual=(True, False, False)
)
```

:bangbang: **NOTE**: Fused ops **will** apply the default attention scale (square root of head dim in Q/K).
Unfused ops **won't** apply the default attention scale (square root of head dim in Q/K).
Some implementations scale the query, some scale both query and key, and this can result
in different outcomes when training.

```python
# Default option:
# Don't specify scale, default to qk_dim ** -0.5
o = na2d(
  q_1, k, v,
  kernel_size=(7, 3),
)

# Option 1:
# apply out of op, and to Q only
q_1 = q * attn_scale
o_1 = na2d(
  q_1, k, v,
  kernel_size=(7, 3),
  scale=1.0,
)

# Option 2:
# apply out of op, and to both Q and K
q_2 = q * attn_scale_sqrt
k_2 = k * attn_scale_sqrt
o_2 = na2d(
  q_2, k_2, v,
  kernel_size=(7, 3),
  scale=1.0,
)
```

### Static functions

#### Environment checks

*Check compute capability on CUDA device:*

```python
from natten.utils import get_device_cc
print(get_device_cc())  # Default torch cuda device
print(get_device_cc(0)) # cuda:0
```

*Check whether your NATTEN installation supports CUDA:*

Indicates whether your local NATTEN and PyTorch were compiled with CUDA, and
a compatible CUDA device is detected:

```python
from natten import has_cuda
print(has_cuda())
```

*Check whether your NATTEN installation supports FP16:*

```python
from natten import has_half
print(has_half())  # Default torch cuda device
print(has_half(0)) # cuda:0
```

*Check whether your NATTEN installation supports BFP16:*

```python
from natten import has_bfloat
print(has_bfloat())  # Default torch cuda device
print(has_bfloat(0)) # cuda:0
```

*Check whether your NATTEN installation has Fused Neighborhood Attention (FNA) kernels:*

Indicates whether your local NATTEN was compiled with our FNA kernels,
and whether your CUDA device is compatible with it.

```python
from natten import has_fna
print(has_fna())  # Default torch cuda device
print(has_fna(0)) # cuda:0
```

*Check whether your NATTEN installation has GEMM kernels:*

Indicates whether your local NATTEN was compiled with our GEMM kernels,
and whether your CUDA device is compatible with it.

```python
from natten.functional import has_gemm
print(has_gemm())  # Default torch cuda device
print(has_gemm(0)) # cuda:0
```

*Check whether your GEMM kernels support TF32/FP32 (full precision):*

Indicates whether your local NATTEN was compiled with our full precision 
GEMM kernels, and whether your CUDA device is compatible with it.

```python
from natten.functional import has_tf32_gemm, has_fp32_gemm
print(has_fp32_gemm())  # Default torch cuda device
print(has_tf32_gemm(0)) # cuda:0
```

*Check whether your GEMM kernels support FP64 (double precision):*

Indicates whether your local NATTEN was compiled with our double precision 
GEMM kernels, and whether your CUDA device is compatible with it.

```python
from natten.functional import has_fp64_gemm
print(has_fp64_gemm())  # Default torch cuda device
print(has_fp64_gemm(0)) # cuda:0
```

#### Dispatcher settings

Since our GEMM and naive backends use the same interface, we've designated
certain methods in Python that allow you to change the dispatcher settings.
This allows you to disable our GEMM kernels and force the NATTEN dispatcher
to only call naive kernels.
Read more about available backends [here](backend.md).

```python
from natten.functional import disable_gemm_na, enable_gemm_na

# Disables dispatching to GEMM kernels
# Calls to NATTEN ops that follow will only
# dispatch naive or tiled backends.
disable_gemm_na()

# Enables dispatching to GEMM kernels
# It is however NOT guaranteed.
# Refer to backend docs for more information.
enable_gemm_na()
```

In addition, you can disable our tiled kernels as well (2-D problems only, and
only under certain conditions, refer to [backend docs](backend.md) for more information.)

```python
from natten.functional import disable_tiled_na, enable_tiled_na

# Disables dispatching to tiled kernels
# Calls to NATTEN ops that follow will only
# dispatch naive or GEMM backends.
disable_tiled_na()

# Enables dispatching to GEMM kernels
# It is however NOT guaranteed.
# Refer to backend docs for more information.
enable_tiled_na()
```

### FLOP counting with FVCore

NATTEN can be paired with [fvcore](https://github.com/facebookresearch/fvcore)'s FLOP counter, as long as you correctly
import and add NATTEN's handles to the flop counter.

```python
from fvcore.nn import FlopCountAnalysis
from natten.flops import add_natten_handle

# ...

flop_ctr = FlopCountAnalysis(model, input)
flop_ctr = add_natten_handle(flop_ctr)

# ...
```

Alternatively, you can use our `get_flops` interface and count FLOPs for a model that may contain
NATTEN ops/modules:
```python
from natten.flops import get_flops

flops = get_flops(model, input)
```

#### Installing fvcore
fvcore is available through PyPI:

```shell
pip install fvcore
```
