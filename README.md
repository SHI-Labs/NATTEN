![NATTENLogo](assets/natten_dark.png#gh-dark-mode-only) ![NATTENLogo](assets/natten_light.png#gh-light-mode-only)

<a href="https://www.shi-labs.com/natten/"><img src="https://img.shields.io/badge/pip%20install%20natten-read%20more-%23C209C1" /></a>

*Neighborhood Attention Extension*

Bringing attention to a neighborhood near you!

NATTEN is an open-source project aimed at providing an interface to neighborhood attention, and more generally sliding window
attention.
If you're not familiar with neighborhood attention, we recommend referring to 
[our papers](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer), or watching our 
presentation on [YouTube](https://www.youtube.com/watch?v=Ya4BfioxIHA).

NATTEN currently works as an extension to PyTorch, but we plan to reduce dependency on the torch API and possibly support other
deep learning frameworks in the future.
NATTEN provides <a href="https://arxiv.org/abs/2204.07143">Neighborhood Attention</a> (local attention)
and <a href="https://arxiv.org/abs/2209.15001">Dilated Neighborhood Attention</a> 
(sparse global attention, a.k.a. dilated local attention) as PyTorch modules for both 1D and 2D data. 

## NEW: GEMM-based CUDA kernels
We are finally releasing our new GEMM-based CUDA kernels, which depend on and are modeled after 
[CUTLASS](https://github.com/NVIDIA/cutlass/)'s [Implicit GEMM](https://github.com/NVIDIA/cutlass/blob/main/media/docs/implicit_gemm_convolution.md) 
kernels for convolution.

Note that these kernels were developed before the CUTLASS 3.0 release, and are therefore still following the CUTLASS 2.X
structure. We plan to write new kernels based on CUTLASS 3.X and CUTE in the near future.

### What does this mean?
It means that if you're running CUDA 11 and above on SM70 or higher (Volta, Turing, Ampere, Ada Lovelace, Hopper), you can start 
using our GEMM based kernels and see up to 10X improvement in latency. However, do note that their current float16/bfloat16 
implementations do not typically result in improved latency, due to a memory alignment issue, which will be resolved in future releases.

![GEMMvsNaive](assets/gemm_vs_naive.png)

NOTE: the table presents the average improvement in latency over different problem sizes with full precision (tfloat32).

The new NATTEN is also heavily refactored to both continue to support older architectures with our naive kernels, and to
accommodate our new kernels which only target SM70 (Volta) and above.

### How do I tell if I'm on SM70 or above?
Simple, just Google your GPU model, and check its compute capability.
If you've already set up PyTorch, you could also run:
```python
import torch

cuda_device = torch.cuda.get_device_properties(torch.cuda.current_device())
sm = cuda_device.major * 10 + cuda_device.minor

print(f"Your main GPU is SM{sm}")
```

Note: SM70 and SM75 Tensor Cores only support FP16 math, which means you only observe the speedup when you're using mixed precision,
or manually casting to half precision. Full and double precision fall back to naive kernels.

### How do I use the new kernels if I'm on SM70 or above?
We're still in the process of deciding the best way to roll out the new kernels via PyPi, which means you can't get these new
kernels via pip.
However, you can build NATTEN from source! Just look at the [instructions below on building from source](#build-from-source).

### How do I know if I'm using the new kernels?
The new NATTEN library sets up constants that are binded to the python interface, which will allow you to
check whether you've compiled with: a. CUDA, b. Float16 (half) support, c. Bfloat16 support, d. New GEMM kernels.

```python
import natten

# Whether NATTEN was built with CUDA
print(natten.has_cuda())

# Whether NATTEN with CUDA was built with support for float16
print(natten.has_half())

# Whether NATTEN with CUDA was built with support for bfloat16
print(natten.has_bfloat())

# Whether NATTEN with CUDA was built with the new GEMM kernels
print(natten.has_gemm())
```

If `natten.has_gemm()` returns true, by default NATTEN will call the faster GEMM kernels instead of the original naive kernels
for both NA1D and NA2D. 3D Neighborhood attention is not supported at this time, but you can still use the naive kernels.

In addition, we will be adding scripts that allow you to profile and observe latency from the kernels with those options
available.

### What else is new?
With the latest code refactor, naive kernels now support arbitrary kernel sizes, and support for bfloat16 (BF16) was also added.

## About NATTEN
Sliding window self attention mechanisms have been relatively overlooked, in part due to implementation difficulties.
For example, in a paper proposing one of the earliest examples of such methods, 
[SASA](https://proceedings.neurips.cc/paper/2019/file/3416a75f4cea9109507cacd8e2f2aefc-Paper.pdf), 
it was noted that
although such methods are theoretically efficient, they're relatively slow in practice, compared to convolutions, 
which have been implemented in most well-known deep learning libraries.

That is why we started developing NATTEN, an extension to existing libraries with efficient implementations of sliding window
attention mechanisms, which will enable research in this direction including building powerful hierarchical vision
transformers.

For more information, we highly recommend reading our preprints [NAT](https://arxiv.org/abs/2204.07143) and
[DiNAT](https://arxiv.org/abs/2209.15001), and check out their [repository](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer).

### How fast is NATTEN?
The latest version of NATTEN runs pretty fast on Ampere with the latest torch and CUDA versions.

![TimePlot](assets/cudatime_dark.png#gh-dark-mode-only) ![TimePlot](assets/cudatime_light.png#gh-light-mode-only)
![MemPlot](assets/cudamemory_dark.png#gh-dark-mode-only) ![MemPlot](assets/cudamemory_light.png#gh-light-mode-only)


## Requirements

* python >= 3.7
* torch >= 1.8
* cmake >= 3.20
 
NATTEN supports PyTorch version 1.8 and later, and Python versions 3.7, 3.8, 3.9, 3.10(only torch >= 1.11), and 3.11 (only torch >= 1.13).

**NOTE:** The current version of NATTEN comes with Linux-only wheels, and supports Pascal and above (`SM >= 60`, i.e. Tesla P100).
Make sure your GPU is supported by referring to 
[this webpage](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).
Future versions will extend support to older GPUs.

## Getting started

### Linux
Just refer to our website, [shi-labs.com/natten](https://www.shi-labs.com/natten/), select your PyTorch version and the CUDA
version it was compiled with, copy-paste the command and install in seconds!

For example, if you're on `torch==2.0.0+cu118`, you should install NATTEN using the following wheel:
```bash
pip3 install natten -f https://shi-labs.com/natten/wheels/cu118/torch2.0.0/index.html
```

More generally:
```bash
pip3 install natten -f https://shi-labs.com/natten/wheels/{cu_version}/torch{torch_version}/index.html
```

**NOTE:** If you do not specify a wheel URL, pip will collect NATTEN and try to compile on locally, which depending
on your system might take up to 30 minutes.
We strongly recommend using our website if you're a Linux user.

### Mac
Unfortunately we are not yet able to build Mac wheels, but you can compile on install, so just run:

```bash
pip3 install natten
```

### Windows
NATTEN now supports Windows devices with CUDA, but does not yet have Windows wheels.
This means you need to clone this repository, and build NATTEN from source, as instructed below.

### Build from source
Once you've set up your Python environment and installed PyTorch with CUDA, simply clone and build:

```bash
git clone https://github.com/SHI-Labs/NATTEN
cd NATTEN

pip install -r requirements.txt

make
```

NOTE: NATTEN will use the PyTorch API to detect your GPU architecture, and will by default attempt to use 1/4th of the number
of processes your system allows to build. You can override them by passing in the following arguments:
```bash
# Build with 2 workers/processes
make WORKERS=2

# Build targeting SM89 (Ada Lovelace)
make CUDA_ARCH="8.9"
```

Please also note that building with the latest GEMM kernels can be a bit time consuming, which means at least 10 - 20 minutes
given that you use enough workers. It is technically possible to improve build time by generating more source files and using
more workers (at the expense of generating a larger binary), but that option will be made available in the future.

#### Optional: run unit tests
You can optionally run unit tests to verify building from source finished successfully:

```bash
make test
```


## Catalog
- [x] Neighborhood Attention 1D (CPU, naive)
- [x] Neighborhood Attention 2D (CPU, naive)
- [x] Neighborhood Attention 3D (CPU, naive)
- [x] Neighborhood Attention 1D (CUDA, naive)
- [x] Neighborhood Attention 2D (CUDA, naive)
- [x] Neighborhood Attention 3D (CUDA, naive)
- [x] Neighborhood Attention 1D (CUDA, gemm-based, SM80 and above)
- [x] Neighborhood Attention 2D (CUDA, gemm-based, SM80 and above)
- [x] Dilation support
- [x] Float16 support and utilization
- [x] BFloat16 support
- [x] Windows builds
- [ ] Neighborhood Attention 1D (CUDA, fused kernels)
- [ ] Neighborhood Attention 2D (CUDA, fused kernels)
- [ ] Kepler and Maxwell (30<=SM<60) support

## Usage
Simply import `NeighborhoodAttention1D`, `NeighborhoodAttention2D`, or `NeighborhoodAttention3D` from `natten`:
```python
from natten import NeighborhoodAttention1D
from natten import NeighborhoodAttention2D
from natten import NeighborhoodAttention3D

na1d = NeighborhoodAttention1D(dim=128, kernel_size=7, dilation=2, num_heads=4)
na2d = NeighborhoodAttention2D(dim=128, kernel_size=7, dilation=2, num_heads=4)
na3d = NeighborhoodAttention3D(dim=128, kernel_size=7, dilation=2, num_heads=4)
```

NA3D also supports different kernel size and dilation values for depth:
```python
na3d = NeighborhoodAttention3D(
	dim=128,
	kernel_size=7,
	kernel_size_d=5,
	dilation=2,
	dilation_d=3,
	num_heads=4)
```

Modules expect inputs of shape `[batch_size, *, dim]`:
* NA1D: `[batch_size, sequence_length, dim]`
* NA2D: `[batch_size, height, width, dim]`
* NA3D: `[batch_size, depth, height, width, dim]`


### FLOPs
We recommend counting flops through [fvcore](https://github.com/facebookresearch/fvcore).

```shell
pip install fvcore
```

Once you have fvcore installed, you can directly use our dedicated FLOP counter:
```python
from natten.flops import get_flops

flops = get_flops(model, input)
```

Alternatively, if you are using fvcore's `FlopCountAnalysis` directly, be sure to add our op handles:
```python
from fvcore.nn import FlopCountAnalysis
from natten.flops import add_natten_handle

# ...

flop_ctr = FlopCountAnalysis(model, input)
flop_ctr = add_natten_handle(flop_ctr)

# ...
```

## License
NATTEN is released under the [MIT License](LICENSE).

## Citation
```bibtex
@inproceedings{hassani2023neighborhood,
	title        = {Neighborhood Attention Transformer},
	author       = {Ali Hassani and Steven Walton and Jiachen Li and Shen Li and Humphrey Shi},
	year         = 2023,
        booktitle    = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}
}
@article{hassani2022dilated,
	title        = {Dilated Neighborhood Attention Transformer},
	author       = {Ali Hassani and Humphrey Shi},
	year         = 2022,
	url          = {https://arxiv.org/abs/2209.15001},
	eprint       = {2209.15001},
	archiveprefix = {arXiv},
	primaryclass = {cs.CV}
}
```

## Acknowledgements
We would like to thank NVIDIA, and the [CUTLASS project](https://github.com/NVIDIA/cutlass/) and team for their efforts in
creating and open-sourcing CUTLASS. We would also like to thank Haicheng Wu for his valuable feedback and comments which led to
the creation of Implicit GEMM NA.
We also thank Meta, and the [PyTorch](https://github.com/pytorch/pytorch/) project and team.
