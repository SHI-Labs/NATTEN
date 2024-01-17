## Fused neighborhood attention

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/fna_dark.png">
  <img alt="Simplified visualization of fused neighborhood attention." src="../asserts/fna_light.png" />
</picture>


Unlike [BMM-style implementations](bmm.md), fused implementations compute attention outputs directly by fusing
the two batched matrix multiplications into the same kernel. This is however more complicated than
fusing back-to-back GEMMs, primarily because of the softmax term in between, which would normally require
computing a full row of the attention weights matrix (multiply loaded queries by **all keys first**).
However, thanks to [online softmax](https://arxiv.org/abs/1805.02867), we can compute partial softmax,
which gets accumulated into exact softmax once the loop over key-value pairs is complete.
For more information on fused attention kernels, we highly recommend referring to
[Flash Attention](https://arxiv.org/abs/2205.14135).

For more information on how the operation is implemented, please refer to [backend docs](../backend.md).
or [our preprint](https://arxiv.org/abs/2403.04690).
