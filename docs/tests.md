## Testing

Unit tests are separated based on problem rank (1D, 2D, 3D), and fused vs standard implementations.

Most tests are either gradchecks or elementwise-matching against a reference.

Gradchecks are useful when an implementation can perform both the forward pass and automatic differentiation, but they only
reveal whether or not the computed derivative tensors match the expected Jacobian. This means gradcheck will only require
both implementations and not a reference implementation (the reference is the Jacobian that torch computes through finite
differences.)

Elementwise-matching requires a reference implementation, which is why such unit tests are performed in the following manner:

1. CUDA backends are evaluated with respect to the CPU backend.
2. CUDA backends can be compared to each other.
3. CUDA backends can be compared to a reference SDPA implementation in torch, with `kernel_size` equal to input size, no
   dilation, no relative positional biases, and no causal masking for higher-rank problems (2D and 3D).

### To-do

* Test front-end modules and other elements in python,
* Test macros and static globals,
* Test against a reference implementation in pure PyTorch

The following require setting up a CI instance:
* Lint/style checks
* Automated testing (CI running CUDA),
* Performance regression tests (CI running CUDA),
* Automated builds,
* Automated wheel links on website.
