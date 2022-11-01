![NATTENLogo](assets/natten_dark.png#gh-dark-mode-only) ![NATTENLogo](assets/natten_light.png#gh-light-mode-only)

<a href="https://www.shi-labs.com/natten/"><img src="https://img.shields.io/badge/pip%20install%20natten-read%20more-%23C209C1" /></a>

*Neighborhood Attention Extension*

Bringing attention to a neighborhood near you!

NATTEN is an extension to PyTorch, which provides the first fast sliding window attention with efficient CUDA kernels. 
It provides <a href="https://arxiv.org/abs/2204.07143">Neighborhood Attention</a> (local attention)
and <a href="https://arxiv.org/abs/2209.15001">Dilated Neighborhood Attention</a> 
(sparse global attention, a.k.a. dilated local attention) as PyTorch modules for both 1D and 2D data. 

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
NATTEN supports PyTorch version 1.8 and later, and Python versions 3.7, 3.8, 3.9, 3.10(only torch >= 1.11), and 3.11 (only torch >= 1.13).

**NOTE:** The current version of NATTEN comes with Linux-only wheels, and supports Pascal and above (`SM >= 60`, i.e. Tesla P100).
Make sure your GPU is supported by referring to 
[this webpage](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).
Future versions will extend support to older GPUs.

You may try and build from source on Windows, but do so at your own risk.
We also welcome contributions in all forms.

## Getting started

### Linux
Just refer to our website, [shi-labs.com/natten](https://www.shi-labs.com/natten/), select your PyTorch version and the CUDA
version it was compiled with, copy-paste the command and install in seconds!

For example, if you're on `torch==1.12.1+cu116`, you should install NATTEN using the following wheel:
```bash
pip3 install natten -f https://shi-labs.com/natten/wheels/cu116/torch1.12.1/index.html
```

More generally:
```bash
pip3 install natten -f https://shi-labs.com/natten/wheels/{cu_version}/torch{torch_version}/index.html
```

**NOTE:** If you do not specify a wheel URL, you will install a "placeholder" version of NATTEN, which is not usable.
We strongly recommend using our website, or building from source.

### Windows
NATTEN should support Windows devices with CUDA, but does not yet have Windows wheels.
You can try and build NATTEN from source (see below).

### Build from source
Once you've set up your Python environment and installed PyTorch with CUDA, simply clone and build:

```bash
pip install ninja # Recommended, not required
git clone https://github.com/SHI-Labs/NATTEN
cd NATTEN
pip install -e .
```

#### Optional: unit tests
You can optionally run unit tests to verify building from source finished successfully:
```bash
python -m unittest discover -v -s ./tests
```


## Catalog
- [x] Neighborhood Attention 1D (CUDA)
- [x] Neighborhood Attention 2D (CUDA)
- [ ] Neighborhood Attention 3D (CUDA)
- [x] Neighborhood Attention 1D (CPU)
- [x] Neighborhood Attention 2D (CPU)
- [ ] Neighborhood Attention 3D (CPU)
- [x] Dilation support
- [x] Float16 support and utilization
- [ ] BFloat16 support (awaiting CUDA 11.8/12 builds of torch)
- [ ] Kepler and Maxwell (30<=SM<60) support
- [ ] Windows builds

## Usage
Simply import `NeighborhoodAttention1D` or `NeighborhoodAttention2D` from `natten`:
```python
from natten import NeighborhoodAttention1D
from natten import NeighborhoodAttention2D

na1d = NeighborhoodAttention1D(dim=128, kernel_size=7, dilation=2, num_heads=4)
na2d = NeighborhoodAttention2D(dim=128, kernel_size=7, dilation=2, num_heads=4)
```

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
@article{hassani2022neighborhood,
	title        = {Neighborhood Attention Transformer},
	author       = {Ali Hassani and Steven Walton and Jiachen Li and Shen Li and Humphrey Shi},
	year         = 2022,
	url          = {https://arxiv.org/abs/2204.07143},
	eprint       = {2204.07143},
	archiveprefix = {arXiv},
	primaryclass = {cs.CV}
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
