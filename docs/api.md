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
  Tuple kernel_size,
  Tuple dilation,
  BooleanTuple is_causal,
  float attn_scale,
  Tuple query_tile_size,
  Tuple key_tile_size);
```

Notes:
* `Tuple` and `BooleanTuple` instances must have the same size as the NA rank (i.e. `tuple<int, int>` and `tuple<bool, bool>` 
  for NA-2D.)
* `query_tile_size` and `key_tile_size` will be REMOVED in the near future. Their sole purpose is to allow auto-tuner (written
  in Python) to pick the tiling configuration. If feasible, auto-tuner should be moved into libnatten; otherwise the tiling
  config must switch to enums instead of exposing the literal tile sizes to the users. If invalid tile sizes are input, the 
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
