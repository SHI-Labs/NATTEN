## Differences between fused and unfused ops

This document will highlight key differences between our fused and unfused implementations and interfaces, which users should
mind. The eventual goal is to minimize these differences and provide a single consistent interface, but given that unfused ops
have existed in NATTEN for far longer than fused ops, we plan to execute these changes in multiple phases.

### Features
Unfused ops have a few additional features that fused operations do not, some of which will be supported in the future.

* Relative positional bias:
  * Fused ops only support RPB in the forward pass, and only when there is no causal masking. 
  * RPB support will be dropped from NATTEN in the future, as attention biases and customized masking generally tend to limit
  the performance of attention kernels greatly.
  * Given that there are many better alternatives to using RPB, we currently do not have any plans in extending its support,
  and only offer it for inference to support running inference on legacy models trained with it.

* Forward mode automatic differentiation
  * This feature is not yet implemented. We encourage users to open an issue or pull request if you happen to figure out the 
    pass for all cases (with/without causal masking, etc.)

* Support for double precision
  * Fused kernels only support full and half precision; unfused kernels support double precision as well.
    * GEMM-based kernels only support full and double precision on Ampere and newer architectures.
    * Bfloat16 support in all implementations is only supported on Ampere and newer architectures.

* Attention dropout, masking, etc.
  * Unfused ops give users direct access to the attention weight tensor, which means custom biases, masking, and dropout can be
  implemented easily.
  * Fused ops do not offer any such fine-grained control over the attention weight tensor, as it is never fully stored to
  global memory.
  * Fused ops also do not offer dropout support. Given that neighborhood attention by definition "drops" attention weights, but
  in a deterministic manner, it was not a priority in the first release.
  * If you happen to have a use case for dropout, please open an issue.


### Memory layout

Unfused ops started out by forcing the "heads first" layout, meaning inputs are always `[batch, heads, *, head_dim]`.
This layout follows standard BMM-style implementations of multi-headed attention, which move the `heads` mode next to batch in
order to easily support all BMM operations.

The alternative, which, if supported by the underlying BMM op, could potentially be faster, is not moving the `heads` mode at
all, and just viewing the `[batch, *, embedding_dim]` tensor (where `embedding_dim = heads * heads_dim`) as a
`[batch, *, heads, head_dim]` tensor. This requires no data movement or additional kernel launches.

Fused operations typically make this change, as supporting the "heads last" layout is usually trivial, and the new
implementation could save the additional overhead and activation space from the permute op.

Users are encouraged to pay attention to this detail, because it means your attention module should prepare the `Q`, `K`, and
`V` tensors according to the operation being called.

A reference is NATTEN's native torch modules, i.e. [NeighborhoodAttention2D](../../src/natten/natten2d.py).
Below is an annotated and slightly modified version of the `forward` call in 2-D neighborhood attention.

```python
# NATTEN context flag indicating whether Fused NA is enabled
from natten import is_fna_enabled

# Fused Neighborhood Attention op
from natten.functional import na2d

# Standard unfused ops
from natten.functional import na2d_qk, na2d_av


def forward(self, x: Tensor) -> Tensor:
    B, H, W, C = x.shape
    assert C == self.heads * self.head_dim

    if is_fna_enabled():
        # [B, H, W, C] -> [B, H, W, 3C]
        qkv = self.qkv(x)
        # [B, H, W, 3C] -> [B, H, W, 3, heads, head_dim]
        qkv = qkv.reshape(B, H, W, 3, self.heads, self.head_dim)
        # [B, H, W, 3, heads, head_dim] -> [3, B, H, W, heads, head_dim]
        qkv = qkv.permute(3, 0, 1, 2, 4, 5)
        # 3 x [B, H, W, heads, head_dim] - > Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Call Fused Neighborhood Attention
        # Output: [B, H, W, heads, head_dim]
        x = na2d(
            q,
            k,
            v,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            scale=self.scale,
        )
        # [B, H, W, heads, head_dim] -> [B, H, W, C]
        x = x.reshape(B, H, W, C)

    else:
        # [B, H, W, C] -> [B, H, W, 3C]
        qkv = self.qkv(x)
        # [B, H, W, 3C] -> [B, H, W, 3, heads, head_dim]
        qkv = qkv.reshape(B, H, W, 3, self.heads, self.head_dim)
        # [B, H, W, 3, heads, head_dim] -> [3, B, heads, H, W, head_dim]
        qkv = qkv.permute(3, 0, 1, 2, 4, 5)
        # 3 x [B, heads, H, W, head_dim] - > Q, K, V
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Call unfused QK operation
        # Output: [B, heads, H, W, prod(self.kernel_size)]
        attn = na2d_qk(
            q,
            k,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
        )
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        # Call unfused AV/PV operation
        # Output: [B, heads, H, W, head_dim]
        x = na2d_av(
            attn,
            v,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            is_causal=self.is_causal,
        )
        # [B, heads, H, W, head_dim] -> [B, H, W, heads, head_dim]
        x = x.permute(0, 2, 3, 1, 4)
        # [B, H, W, heads, head_dim] -> [B, H, W, C]
        x = x.reshape(B, H, W, C)

    return self.proj(x)
```
