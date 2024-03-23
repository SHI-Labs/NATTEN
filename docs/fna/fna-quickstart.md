# Fused Neighborhood Attention quick start guide

## If you use NATTEN modules
If you import our PyTorch modules (`NeighborhoodAttention1D`, `NeighborhoodAttention2D`, `NeighborhoodAttention3D`), then all
you really need is to enable FNA by importing and calling `natten.use_fused_na`:

```python
import natten

natten.use_fused_na(True)
```


## If you write your own modules and need just the op
If you have your own module and would like to insert neighborhood attention, or you have an existing
self attention module which you'd like to turn into a neighborhood attention module, then you're in the right place.

If you've already used NATTEN ops before, you probably imported two ops: QK and AV 
(depending on the version of NATTEN they might have been `natten2d` + `qkrpb`/`av`, or `na2d` + `qk`/`av`, or other such
variations.)

The typical use case for __unfused__ neighborhood attention was and still is:

```python
from natten.functional import na2d_qk, na2d_av

# Given Q, K and V;
# where q/k/v.shape is [batch, heads, height, width, head_dim]

# Self attn: attn = q @ k.transpose(-2, -1)
attn = na2d_qk(q, k, kernel_size=kernel_size, dilation=dilation)

attn = (attn * attn_scale).softmax(dim=-1)

# Self attn: output = attn @ v
output = na2d_av(attn, v, kernel_size=kernel_size, dilation=dilation)
```

Now let's look at the same example, but with __fused__ neighborhood attention:


```python
from natten.functional import na2d

# Given Q, K and V;
# where q/k/v.shape is [batch, height, width, heads, head_dim]
# NOTE: layout is different from unfused;
# it's batch, spatial extent, then heads, then head_dim.

# Self attn: output = sdpa(q, k, v, scale=attn_scale)
output = na2d(q, k, v, kernel_size=kernel_size, dilation=dilation)
```

And that's it!

## Recommendations

Here's a list of recommendations if you're just starting to use NATTEN or FNA:

1. Please review the [Fused vs unfused NA](fused-vs-unfused.md) guide to make
   sure the differences between the two interfaces is clear to you.
   Not doing so may result in unexpected behavior that the API/interface cannot check.

2. Consider supporting both fused and unfused NA in your code, simply by checking
   `natten.is_fused_na_enabled()`; certain GPU architectures may not support fused NA,
   and some applications may require unfused NA. Read more in [fused vs unfused NA](fused-vs-unfused.md).

3. Consider using [KV parallelism](kv-parallelism.md) to potentially gain in performance if you can afford
   additional global memory usage. This may slightly affect reproducibility, as KV parallelism
   makes the computation of `dQ` non-deterministic, but this should rarely affect your training
   stability. Note that KV parallelism is not guaranteed to improve performance in all cases, but
   it is still a setting worth configuring if you're not bound by memory capacity.

4. Consider using the [Autotuner](autotuner.md) during inference, and possibly during training.

5. Please open issues if you have any questions, or notice any issues with the code or documentation.
