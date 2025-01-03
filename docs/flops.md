## FLOP counting with NATTEN

NATTEN offers multiple solutions for counting FLOPs/MACs.

However, note that due to the variety of solutions available for this, it is important to have a
consistent definitions of FLOPs and how they are counted.

### Rules
Since NATTEN is limited to the dot product attention operator, here we note the general rules for
counting FLOPs/MACs in NATTEN:

Following standard practice, only Tensor Core (TC) FLOPs are counted.
Softmax, masking, attention scale, and positional biases are NOT included in the FLOP/MAC count.

For a GEMM problem with shape `(M, N, K)`, MACs (Multiply-and-Accumulate operations) is simply
`M * N * K`, while FLOPs (Floating Point Operations) considers multiply and accumulate separate
operations, and ends up being `2 * M * N * K`.

Causal masking also does **not** affect FLOPs/MACs today, but this behavior may change in the
future. It makes sense to measure in causal masking for various reasons, but the actual FLOP
decrease is not so granular in practice. More details will be added about this in the future.
Please refer to [#184](https://github.com/SHI-Labs/NATTEN/issues/184#issuecomment-2505022903) for
more information.

### Manual
All users can integrate NATTEN's FLOP/MAC counting logic into their own FLOP counting logic:

```python
from natten.flops import (
  fna_flop_count,
  na_qk_flop_count,
  na_av_flop_count
)
```

If you're using [Fused Neighborhood Attention (FNA)](fna/), or just prefer to compute FLOPs/MACs for
the entire attention operator all at once, use `fna_flop_count`:

```python
fna_flop_count(
    q_shape: torch.Size | Sequence[int],
    k_shape: torch.Size | Sequence[int],
    v_shape: torch.Size | Sequence[int],
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
    is_heads_last: bool,
    return_macs: bool = False
) -> int
```

Simply pass the shape of your QKV tensors, and NA-related arguments (`kernel_size`, `dilation`,
`is_causal` as tuples), and finally, two key arguments:
* `is_heads_last`: determines your QKV tensor layout. If `True`, your QKV is expected to assume the
  layout `[batch, *, heads, dim_per_head]`, and otherwise, `[batch, heads, *, dim_per_head`.
  NOTE: Fused ops in NATTEN (FNA) currently use the "heads last" layout, so be sure to pass `True`
  for them. Unfused ops in NATTEN use the "heads first" layout.
* `return_macs`: If `True`, returns MACs instead of FLOPs. Please be very careful when setting this.

If you're using unfused implementations, or prefer to compute FLOPs/MACs for the dot product and
weighting operations separately, use `na_qk_flop_count` and `na_av_flop_count`:

```python
na_qk_flop_count(
    q_shape: torch.Size | Sequence[int],
    k_shape: torch.Size | Sequence[int],
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
    is_heads_last: bool,
    return_macs: bool = False,
) -> int

na_av_flop_count(
    a_shape: torch.Size | Sequence[int],
    v_shape: torch.Size | Sequence[int],
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    is_causal: Sequence[bool],
    is_heads_last: bool,
    return_macs: bool = False,
) -> int
```

The APIs are mostly similar to the fused variant, and you need to pay extra attention to setting
`is_heads_last` and `return_macs`.

### PyTorch (>= 2.4.0)

PyTorch itself now supports counting model FLOPs, but obviously models with custom operators would
require extra work and `torch >= 2.4.0`.

**NOTE:** PyTorch reports FLOPs, not MACs!

PyTorch only counts FLOPs based on operations registered with torch.library. This excludes autograd
functions, which historically have been the only choice for binding custom operators to PyTorch and
getting autograd support.
NATTEN will eventually migrate to registering with torch.library, but that process is on hold until
certain features available in autograd functions are supported in the new API.
More details in #170.

In the meantime, we offer an experimental interface for NATTEN ops registered with torch.library,
which only supports inference, and only implements FNA operations.
Instead of using the `na*d` operations from `natten.functional` you will need to use the `na*d`
operations from `natten.experimental`:


```python
# from natten.functional import na1d, na2d, na3d
from natten.experimental import na1d, na2d, na3d
```

or if you're using NATTEN's torch modules (`NeighborhoodAttention*D`), you need to pass an
additional argument:

```python
from natten import (
  NeighborhoodAttention1D,
  NeighborhoodAttention2D
  NeighborhoodAttention3D
)

m_1d = NeighborhoodAttention1D(dim=128, num_heads=4, kernel_size=7, use_experimental_ops=True)
m_2d = NeighborhoodAttention2D(dim=128, num_heads=4, kernel_size=7, use_experimental_ops=True)
m_3d = NeighborhoodAttention3D(dim=128, num_heads=4, kernel_size=7, use_experimental_ops=True)
```

However, note that forward pass will fail if fused NA is disabled / not supported, but using
experimental ops is enabled.


As long as you use these experimental ops, you can:
* Successfully build NA-based models with `torch.compile` WITHOUT graph breaks,
* Count flops with `torch.utils.flops`

For more information on the `na*d` interface, please refer to our [frontend docs](frontend.md) or
[Fused Neighborhood Attention docs](fna/).

To count FLOPs with PyTorch's native FLOP counter:

```python
from torch.utils.flop_counter import FlopCounterMode

flop_counter = FlopCounterMode()

with flop_counter:
  y = model_with_natten_ops(x)

total_flops = flop_counter.get_total_flops()
```

**NOTE:** 

### FVCore

NATTEN can be paired with [fvcore](https://github.com/facebookresearch/fvcore)'s counter, as long as you correctly
import and add NATTEN's handles to the counter.

The only requirement is having fvcore installed.

**NOTE:** fvocre reports MACs, not FLOPs!

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
