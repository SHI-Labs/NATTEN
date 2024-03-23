## C++ API

NOTE: this page is still a work in progress.

### PyTorch Bindings

#### FNA

```cpp
// Forward pass
void na*d_forward(
  Tensor output,
  Tensor query,
  Tensor key,
  Tensor value,
  optional<Tensor> rel_pos_bias,
  optional<Tensor> logsumexp,
  Tuple kernel_size,
  Tuple dilation,
  BooleanTuple is_causal,
  float attn_scale,
  Tuple query_tile_size,
  Tuple key_tile_size);
  
// Backward pass
void na*d_backward(
  Tensor grad_query,
  Tensor grad_key,
  Tensor grad_value,
  Tensor query,
  Tensor key,
  Tensor value,
  Tensor output,
  Tensor grad_output,
  Tensor logsumexp,
  Tuple kernel_size,
  Tuple dilation,
  BooleanTuple is_causal,
  float attn_scale,
  Tuple query_tile_size,
  Tuple key_tile_size,
  Tuple num_splits_key,
  bool compute_delta_with_torch);
```

Notes:
* `Tuple` and `BooleanTuple` instances must have the same size as the NA rank (i.e. `tuple<int, int>` and `tuple<bool, bool>` 
  for NA-2D.)

* `logsumexp` is optional for forward pass, and if empty, the kernel will skip writing logsumexp into global memory.
 
* Forward pass requires two kernel configuration arguments, `query_tile_size` and `key_tile_size`. These arguments indicate the
  kernel tiling shape, but they cannot be set to arbitrary shapes. The shape "area" (product of tuple elements) will have to
  multiply to exactly the GEMM tiling shape. For example, if the GEMM tiling shape is `<32, 128, *>`, and because the GEMM row
  mode corresponds to `query/output` rows and column mode corresponds to `key/value` rows, that would mean valid choices for
  `query_tile_size` are any tuples with the same size as the NA rank, the product of elements of which is 32 (i.e. `(32,)` for
  1-D, `(4, 8)` for 2-D, and `(2, 2, 8)` for 3-D.)

* Backward pass requires 2 additional configuration arguments in addition to `query_tile_size` and `key_tile_size`. Those are
  `num_splits_key` and `compute_delta_with_torch`. `num_splits_key` indicates KV parallelism across different axes. This value
  must be less than or equal to the upper bound number of KV tile iterations per each `batch x head x dilation_idx`.
  That upper bound is: `ceil_div(ceil_div(input_shape, dilation), key_tile_size)`.
  `compute_delta_with_torch` is just a boolean argument that specifies whether or not the backend should use native torch ops
  to compute `delta`, or use a specialized kernel.
  `delta` is a vector used in the backward pass, and is computed by casting `grad_output` and `output` into FP32,
  elementwise multiplying them, then summing over dim per head.
  The specialized kernel is a CUTLASS-based reduction kernel that fuses the type casting and multiplication into a single 
  kernel launch, and therefore may be faster in some settings.

* `query_tile_size`, `key_tile_size`, `num_splits_key`, and `compute_delta_with_torch` 
  will be REMOVED in the near future. Their sole purpose is to allow auto-tuner (written in Python) to pick the best 
  configuration. If feasible, auto-tuner should be moved into libnatten; otherwise configurations will switch to enums
  instead of exposing the literal tile sizes and other such settings to the users. If invalid tile sizes are input, the 
  final FNA dispatchers will fail to find a matching kernel, and raise an error.

#### BMM-style NA

##### QK operation

```cpp
// Forward pass
void na*d_qk_forward(
  Tensor attn,
  Tensor query,
  Tensor key,
  optional<Tensor> rel_pos_bias,
  Tuple kernel_size,
  Tuple dilation,
  BooleanTuple is_causal);
  
// Backward pass
void na*d_qk_backward(
  Tensor d_query,                   // Output
  Tensor d_key,                     // Output
  optional<Tensor> d_rel_pos_bias,  // Optional output
  Tensor d_attn,
  Tensor query,
  Tensor key,
  Tuple kernel_size,
  Tuple dilation,
  BooleanTuple is_causal);
```

Notes:
* `Tuple` and `BooleanTuple` instances must have the same size as the NA rank (i.e. `tuple<int, int>` and `tuple<bool, bool>` 
  for NA-2D.)

##### PV (AV) operation

```cpp
// Forward pass
void na*d_av_forward(
  Tensor output,
  Tensor attn,
  Tensor value,
  Tuple kernel_size,
  Tuple dilation,
  BooleanTuple is_causal);
  
// Backward pass
void na*d_av_backward(
  Tensor d_attn,           // Output
  Tensor d_value,          // Output
  Tensor d_out,
  Tensor attn,
  Tensor value,
  Tuple kernel_size,
  Tuple dilation,
  BooleanTuple is_causal);
```

Notes:
* `Tuple` and `BooleanTuple` instances must have the same size as the NA rank (i.e. `tuple<int, int>` and `tuple<bool, bool>` 
  for NA-2D.)

  
### Operators
TBD.
