## NATTEN build system

NATTEN requires building libnatten, which holds all implementations (primarily CUDA kernels for now, and CPU references), as
well as the C++ API that connects them to PyTorch's API.

The current structure is a bit redundant, but not overly so, in that different kernel classes (naive, tiled, gemm, fna) can be
maintained somewhat independently. Because we're interested in reducing dependence on PyTorch beyond binding with torch API,
kernels and dispatchers expect problem metadata and pointers, instead of accepting torch objects (i.e. `at::Tensor`).

Originally, all kernel dispatchers were written using macros, packed into a few (originally just one) source files. The obvious
problem there is it makes it difficult to parallelize the build process.
Because of this, we never fully instantiate kernels in any of the source files (`csrc/src`) and instead auto-generate kernel
instantiations and dispatchers (see `scripts/`). Doing so allows us to adjust the number of generated source files arbitrarily
in order to maximize parallelization in the build process.

As the build process became more complicated, it no longer made sense to keep using Python or PyTorch's builders, which is why
we switched to using CMake.
We're still using Python's `setuptools` to build the python package, but the difference is that we hack around `setuptools`
compiling anything and force it to delegate building libnatten to CMake.
