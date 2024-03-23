## KV parallelism in Fused Neighborhood Attention

FNA backward pass does not parallelize across the input sequence by default, because it results in a race condition on the 
query gradient.  If we parallelize the outer loop over KV tiles, then multiple KV tiles will try to update the same tile in 
`dQ`.

This can be avoided with a mutex lock, but this also may increase the amount of scratch memory required by the kernel.
In addition, the race makes parallelizing across KV non-deterministic.

In this document, we outline how you can specify your preference for using KV parallelism.

### Enabling KV parallelism

```python
import natten

# Enable KV parallelism
natten.use_kv_parallelism_in_fused_na(True)

# Disable KV parallelism
natten.use_kv_parallelism_in_fused_na(False)
```

### Memory usage preference

In addition to a global context flag for whether or not KV parallelism is enabled, NATTEN also
offers "memory usage preferences", which controls the upper bound for parallelism.

Presently there are 3 modes, but we plan to improve this interface in the future by giving more fine-grained control
and improving the heuristic:

1. Default
2. Strict
3. Unrestricted

Default and strict limit the upper bound for KV parallelism by factoring in how much parallelism is already
gained across batch size and attention heads.

Unrestricted does not limit the upper bound of KV parallelism and defaults to as much as permitted.

To change memory preferences:

```python
import natten

natten.set_memory_usage_preference("default")

natten.set_memory_usage_preference("strict")

natten.set_memory_usage_preference("unrestricted")
```
