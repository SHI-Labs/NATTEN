# Operations

In this page we list our PyTorch Autograd-compatible operations.
These operations come with performance knobs (configurations), some of which are specific to
certain [backends](backends.md).

Changing those knobs is completely optional, and NATTEN will continue to be functionally correct in
all cases. However, to squeeze out the maximum performance achievable, we highly recommend looking
at [backends](backends.md), or just using our [profiler toolkit](profiler.md) and its
[dry run feature](profiler.md#dry-run) to navigate through
available backends and their valid configurations for your specific use case and GPU architecture.
You can also use the profiler's [optimize](profiler.md#optimize) feature to search and find the
best configuration.


## Neighborhood Attention

::: natten
    options:
          heading_level: 3
          show_object_full_path: true
          members:
              - na1d
              - na2d
              - na3d

## Standard Attention

::: natten
    options:
          heading_level: 3
          show_object_full_path: true
          members:
              - attention
              - merge_attentions
