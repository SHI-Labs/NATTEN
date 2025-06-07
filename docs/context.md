# Global context

Certain features in NATTEN are guarded by a global context. Those features are:

* **KV parallelism in FNA backward**: necessary for speeding up training, but can introduce
    non-deterministic behavior, and increased memory footprint. This is standard in almost all
    fused attention implementations. [Read more](#kv-parallelism-in-fna-fmha).
* **Flex Attention + `torch.compile`**: Heavily experimental, and may lead to incorrect behavior,
    but users can choose to allow it at their own risk. [Read more](#flex-attention-torchcompile).


## KV parallelism in FNA / FMHA
FNA (as well as most FMHA) backward pass implementations need to parallelize across the KV sequence,
but this results in a race condition on the query gradient tiles. 
This is avoided with a mutex lock, which results in non-deterministic order of write, and therefore
makes the computation non-deterministic.
In addition, some additional scratch space may be required, which is a function of the parallelism
degree.

In this document, we outline how you can specify your preference for using KV parallelism.

### Controlling KV parallelism
KV parallelism is *enabled* by default, but you can choose to disable it explicitly using the
following operations, or just enable PyTorch's deterministic mode to disable KV parallelism.

::: natten
    options:
          heading_level: 4
          show_object_full_path: true
          members:
              - use_kv_parallelism_in_fused_na

You can additionally check the current setting:

::: natten
    options:
          heading_level: 4
          show_object_full_path: true
          members:
              - is_kv_parallelism_in_fused_na_enabled

### Memory usage preference

In addition to a global context flag for whether or not KV parallelism is enabled, NATTEN also
offers "memory usage preferences", which controls the upper bound for parallelism, as to control
the memory footprint.

Presently there are 3 modes, but we plan to improve this interface in the future by giving more
fine-grained control and improving the heuristic:

1. Default
2. Strict
3. Unrestricted

Default and strict limit the upper bound for KV parallelism by factoring in how much parallelism is
already gained across batch size and attention heads.
Unrestricted does not limit the upper bound of KV parallelism and uses the maximum parallelism
possible, and therefore gives the best performance, but also the highest memory footprint.

However, we note that in practice, we haven't seen any cases that run out of memory while using
the unrestricted setting, and therefore recommend trying that setting first to see if it fits your
use case, and downgrade only if not.

To change memory preferences, use the following function:

::: natten
    options:
          heading_level: 4
          show_object_full_path: true
          members:
              - set_memory_usage_preference


You can additionally check what the current setting is:

::: natten
    options:
          heading_level: 4
          show_object_full_path: true
          members:
              - is_memory_usage_default

::: natten
    options:
          heading_level: 4
          show_object_full_path: true
          members:
              - is_memory_usage_strict

::: natten
    options:
          heading_level: 4
          show_object_full_path: true
          members:
              - is_memory_usage_unrestricted

## Flex Attention + `torch.compile`

We have been *unable to verify the correctness* of our Flex backend when compilation is enabled
under all of our use cases. We believe this may be a PyTorch bug, because:

1. everything works as expected without `torch.compile`,
2. some cases are intermittent and changing the order of the tests fixes it,
3. in some cases, forward pass is correct, but backward pass fails, regardless of order.

We are working on raising this issue with PyTorch directly, but until it is resolved, we strongly
recommend exercising caution when using this feature.

Due to this, Flex + compilation is guarded with global context variables, which you can control
using the following functions.


::: natten
    options:
          heading_level: 4
          show_object_full_path: true
          members:
              - allow_flex_compile


::: natten
    options:
          heading_level: 4
          show_object_full_path: true
          members:
              - allow_flex_compile_backprop


::: natten
    options:
          heading_level: 4
          show_object_full_path: true
          members:
              - disable_flex_compile


::: natten
    options:
          heading_level: 4
          show_object_full_path: true
          members:
              - disable_flex_compile_backprop

You can additionally check the current settings:

::: natten
    options:
          heading_level: 4
          show_object_full_path: true
          members:
              - is_flex_compile_allowed

::: natten
    options:
          heading_level: 4
          show_object_full_path: true
          members:
              - is_flex_compile_backprop_allowed
