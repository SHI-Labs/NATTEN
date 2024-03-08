## BMM-style neighborhood attention

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/batched_gemm_na_dark.png">
  <img alt="Simplified visualization of GEMM-based neighborhood attention." src="../asserts/batched_gemm_na_light.png" />
</picture>


BMM-style implementations break the attention operation into three primary stages:

* A = QK^T
* P = Softmax(A)
* X = PV

BMM-style is typically the most straightforward way of implementing attention, and as a result of that
most of NATTEN's implementations are BMM-style.
Given that we can just use PyTorch's native softmax op, it is not independently implemented in NATTEN.

For more information on how the two operations (QK and PV) are implemented, please refer to [backend docs](../backend.md).

For more details, we highly recommend reading [our paper](https://arxiv.org/abs/2403.04690).
