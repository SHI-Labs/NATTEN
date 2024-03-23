## Installing NATTEN

If you are a Linux, your best option is installing via PyPI with wheels.
Just refer to our website, [shi-labs.com/natten](https://www.shi-labs.com/natten/), 
select your PyTorch version and backend (CUDA + version or CPU),
copy-paste the command and install in seconds (subject to your network bandwidth :D )!

For example, if you're on PyTorch 2.2 with CUDA, and it was tagged with `cu121`, the command would look like:

```bash
pip3 install natten==0.15.1+torch220cu121 -f https://shi-labs.com/natten/wheels/
```

### Mac
NATTEN can be installed via PyPI, but we're not able to build Mac wheels at this time.

You can simply do `pip3 install natten`, and the CPU build will be compiled and installed.

#### MPS Support
NATTEN does not come with any metal kernels for now, and therefore the only usable backend on Apple Silicon is still CPU.
We plan to port our naive kernels to metal soon, but we certainly welcome contributions!

### Windows

NATTEN does not support Windows builds yet.
This leaves Windows users with only one option: WSL.

Since [switching to cmake](https://github.com/SHI-Labs/NATTEN/tree/3036259bdc5c30b7b49fe8ba17d60a0ab0d780dd) for building
libnatten, we have not been able to support MSVC builds.
While we welcome contributions, we are not going to be able to fix MSVC builds any time in the near future,
so we recommend building and running NATTEN with either WSL or MinGW.

### Building from source
In order to build from source, please make sure that you have your preferred PyTorch build installed,
since the NATTEN setup script depends heavily on PyTorch.
Once set up, simply clone and build:

```bash
git clone https://github.com/SHI-Labs/NATTEN
cd NATTEN

pip install -r requirements.txt

make
```

NOTE: NATTEN will use the PyTorch API to detect your GPU architecture (if you have an NVIDIA GPU with CUDA set up).
If you want to specify your architecture(s) manually:
```bash
# Build targeting SM89 (Ada Lovelace)
make CUDA_ARCH="8.9"

# Build targeting SM90 (Hopper)
make CUDA_ARCH="9.0"
```

It will by default attempt to use 1/4th of the number of processor threads on your system, but if you can, we recommend using
up to 64 in order to maximize parallelization.
You can do so by passing in the following arguments:
```bash
# Build with 64 workers
make WORKERS=64
```

### NGC docker images
NATTEN supports PyTorch builds that are built from source, an example of which is the builds that ship with
NVIDIA's [NGC container images](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).

However, note that our PyPI wheels are only compatible with PyPI releases of PyTorch, which means you will have to build NATTEN
from source if you built PyTorch from source (assuming you're on linux.)

### FAQ

1. Why not just `pip install natten`?
  * When you don't specify a wheel link, pip only downloads NATTEN's packaged source, and attempts to compile it on your
  system. This means you'll not only have to wait longer, you also will be occupying your own processor for the duration of
  that time. Depending on what your hardware is, the build configuration and targets will vary, meaning the number of objects
  that need to be compiled is dependent on your hardware, and the more implementations and kernels available for your hardware,
  the longer your compile time.
  Because of this, we highly recommend installing using wheels instead, because wheels are simply a compressed file containing
  the python source, and a pre-built binary compatible with your hardware and software.
  
2. What if my CUDA version is different?
  * The CUDA version you choose should be the same as the version of CUDA your PyTorch install was built with.
  This version is not necessarily (and very rarely) identical to the version you have installed locally.
  It is however important to use the same major CUDA release (i.e. if your local CUDA is 11.X, make sure to install an 11.X
  pytorch build.)
  * If the version of NATTEN you install was not built for the your PyTorch build (mismatched major version or mismatched CUDA
  version), you will likely run into an ABI mismatch.

3. What's the difference between naive, GEMM, and fused neighborhood attention?
  These refer to the implementation of neighborhood attention that NATTEN can target. Generally, GEMM and fused kernels are
  expected to outperform naive implementations, but depending on your problem size, that may not be the case.
  Naive and GEMM have an identical memory footprint, but fused will use less memory than both in most cases.
  For more information, please refer to our [backend documentation](backend.md).

4. Can I use Fused Neighborhood Attention?
  * It depends on 1. whether you have an NVIDIA GPU, 2. whether it's SM50 or later. We'll trust users with figuring out 1. As
  for 2, you can either look up your GPU model and check the architecture or compute capability. Additionally, if you have
  PyTorch set up, you can run the following:
  ```python3
  import torch
  major, minor = torch.cuda.get_device_capability(device_index)
  sm = major * 10 + minor
  print(sm)
  ```
  If the number you see is equal to or greater than 50, you can use Fused Neighborhood Attention kernels with both FP16 and
  FP32 data types. If it's 80 or above, you can also do BF16.

5. Can I use GEMM-based Neighborhood Attention?
  Our GEMM kernels are very limited at the moment. We've only implemented it for 1D and 2D Neighborhood Attention, and it can
  only target Tensor Cores. This means that if you're running on Ampere (SM80) or later, you will be able to use our GEMM-based
  kernels with full precision (where they're the strongest). If you're on Volta (SM70) or Turing (SM75), you can only run those
  kernels in FP16, since their Tensor Cores can only do FP16 math.

6. When will Windows be supported?
  * As noted, we don't have a Windows machine with CUDA at our disposal, therefore we would greatly appreciate user feedback.
  Please refer to the [open issue](https://github.com/SHI-Labs/NATTEN/issues/18) if you're interested.
 
7. When will there be an MPS/ROCm backend?
  * Our top priority is feature completeness in our CUDA backend, and full compatibility with more recent PyTorch features such
  as nested tensors, torch.compile, and the like. Once the project is past those milestones, we will try to improve our CPU
  backend and possibly attempt to port some of our kernels to Metal and ROCm.
  
8. When should I build from source?
  * Only when you're interested in the latest available features! If there haven't been any commits on the main branch since
  the last release, building from source will be the same as building via pip (as long as you don't use wheels.)
  * Installing via wheels is almost identical pip installing without wheels. The only exception is that the latter builds
  libnatten for your GPU architecture only, whereas wheels are multi-architecture (a.k.a. fat binaries).

9. What are NATTEN's dependencies?
   NATTEN requires to be linked with libtorch in order to use torch's C++ API since inputs to the C++ functions are torch
   tensors. In addition, NATTEN currently binds to PyTorch using torch's pybind headers, and some of our naive kernels still
   depend on torch routines, but we expect to eliminate that dependency in order to allow linking with other frameworks.
   If you use either GEMM or Fused Neighborhood Attention kernels, NATTEN will also depend on 
   [CUTLASS](https://github.com/NVIDIA/cutlass/).
