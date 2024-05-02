<img src="https://www.shi-labs.com/natten/assets/img/natten_light.png" width="384" />

*Neighborhood Attention Extension*

Bringing attention to a neighborhood near you!

<a href="https://www.shi-labs.com/natten/">Website / Releases</a>
| <a href="https://github.com/SHI-Labs/NATTEN/tree/main/docs/">Documentation</a>

<div align="center">
  <img alt="Visualization of neighborhood attention in 2D." src="https://shi-labs.com/natten/pypi-assets/docs/assets/neighborhood_attn_2d_vis_light.png" width="384" />
  <img alt="Visualization of dilated neighborhood attention in 2D." src="https://shi-labs.com/natten/pypi-assets/docs/assets/dilated_neighborhood_attn_2d_vis_light.png" width="384" />
</div>

NATTEN is an open-source project dedicated to providing fast implementations for
[Neighborhood Attention](https://scholar.google.com/citations?view_op=view_citation&citation_for_view=Ndu0dUcAAAAJ:b0M2c_1WBrUC),
a sliding window self-attention mechanism.

If you're not familiar with neighborhood attention, please refer to 
[our papers](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer), or watch our 
[YouTube video](https://www.youtube.com/watch?v=Ya4BfioxIHA) from CVPR 2023.

To read more about our GEMM-based and fused neighborhood attention kernels, please refer to
our new preprint, [Faster Neighborhood Attention](https://arxiv.org/abs/2403.04690).

## New: Fused Neighborhood Attention now supports backpropagation!

We've released the Fused Neighborhood Attention (FNA) backward kernel and interface, which means you can now
train models based on neighborhood attention faster and more efficiently.

FNA can be seen as a generalization of methods such as [Flash Attention](https://github.com/Dao-AILab/flash-attention/) and
[FMHA](https://github.com/facebookresearch/xformers/) from back-to-back matrix multiplication to
back-to-back tensor-tensor contraction, and comes with neighborhood attention masking built in.
This accelerates accelerates neighborhood attention, a multi-dimensional sliding window attention pattern,
by never storing the attention tensor to global memory, which aside from reducing global memory footprint also reduces
the memory bandwidth bottleneck.

<div align="center">
  <img alt="Op-level average speedup." src="https://shi-labs.com/natten/pypi-assets/assets/fna-chart-light.png" />
</div>

We highly recommend referring to [FNA quick start](https://github.com/SHI-Labs/NATTEN/tree/main/docs/fna/fna-quickstart.md) or 
the [Fused vs unfused NA](https://github.com/SHI-Labs/NATTEN/tree/main/docs/fna/fused-vs-unfused.md) guide before
starting to use FNA, since the interface, memory layout, and feature set can differ from
all unfused ops in NATTEN.

## Getting started
 
NATTEN supports PyTorch version 2.0 and later, and Python versions 3.8 and above. 
Python 3.12 is only supported with torch >= 2.2.0.

Older NATTEN releases supported python >= 3.7 and torch >= 1.8.

Please refer to [install instructions](https://github.com/SHI-Labs/NATTEN/tree/main/docs/install.md) to find out whether your operating system and hardware accelerator is
compatible with NATTEN.

## Feature availability

| Problem space | CPU backend | CUDA backend     |
| -----------   | ----------- | ---------------- |
| 1D            | naive       | naive, gemm, fna |
| 2D            | naive       | naive, gemm, fna |
| 3D            | naive       | naive, fna       |

### CPU

| Problem space | CPU Backend | Causal masking     | Varying parameters | Relative positional bias | Autograd support         |
| -----------   | ----------- | ------------------ | ------------------ | ------------------------ | ------------------------ |
| 1D            | naive       | &#10003;           | &#10003;           | &#10003;                 | Forward and reverse mode |
| 2D            | naive       | &#10003;           | &#10003;           | &#10003;                 | Forward and reverse mode |
| 3D            | naive       | &#10003;           | &#10003;           | &#10003;                 | Forward and reverse mode |

Notes:
* Forward mode autograd does not support relative positional biases and causal masking yet.
* Relative positional biases are not yet supported when any axis has causal masking enabled.

### CUDA

| Problem space | CUDA Backend | Causal masking     | Varying parameters | Relative positional bias | Autograd support         | Min. Arch |
| -----------   | -----------  | ------------------ | ------------------ | ------------------------ | ------------------------ | --------- |
| 1D            | naive        | &#10003;           | &#10003;           | &#10003;                 | Forward and reverse mode | SM35      |
| 2D            | naive        | &#10003;           | &#10003;           | &#10003;                 | Forward and reverse mode | SM35      |
| 3D            | naive        | &#10003;           | &#10003;           | &#10003;                 | Forward and reverse mode | SM35      |
| 1D            | gemm         | -                  | -                  | &#10003;                 | Forward and reverse mode | SM70      |
| 2D            | gemm         | -                  | -                  | &#10003;                 | Forward and reverse mode | SM70      |
| 1D            | fna          | &#10003;           | &#10003;           | &#10003;                 | Reverse mode             | SM50      |
| 2D            | fna          | &#10003;           | &#10003;           | &#10003;                 | Reverse mode             | SM50      |
| 3D            | fna          | &#10003;           | &#10003;           | &#10003;                 | Reverse mode             | SM50      |

Notes: 
* FP16 kernels are only available on SM50 and above*, and BF16 requires SM80 and above.
  * Naive FP16 kernels are only available on **SM60** and above.
  * FNA FP16 kernels are only available on SM50 and above.
* GEMM backend on SM70 and SM75 can only do FP16.
* Tiled only implements 1/3 of the ops, is only implemented for 2D problems, and requires head dim = 32.
* Forward mode autograd does not support relative positional biases and causal masking yet.
* Relative positional biases are not yet supported when any axis has causal masking enabled.
* Relative positional biases are not supported in FNA during backward pass.

Features that will likely no longer be worked on or improved:
* Relative positional biases
  * There's just better alternatives that don't involve explicitly biasing the attention weight matrix, and they will be more
  performant on top of providing similar or better accuracy levels.
* GEMM-based kernels
  * Since FNA covers more features than our unfused GEMM-based kernels, and we know it to be a better solution
    (please refer to Faster Neighborhood Attention for details), we do not plan to extend or improve these kernels.
  * This includes support for varying parameters, causal masking, and 3-D problems.

## License
NATTEN is released under the [MIT License](LICENSE).

## Citation
```bibtex
@misc{hassani2024faster,
  title        = {Faster Neighborhood Attention: Reducing the O(n^2) Cost of Self Attention at the Threadblock Level},
  author       = {Ali Hassani and Wen-Mei Hwu and Humphrey Shi},
  year         = 2024,
  url          = {https://arxiv.org/abs/2403.04690},
  eprint       = {2403.04690},
  archiveprefix = {arXiv},
  primaryclass = {cs.CV}
}
@inproceedings{hassani2023neighborhood,
  title        = {Neighborhood Attention Transformer},
  author       = {Ali Hassani and Steven Walton and Jiachen Li and Shen Li and Humphrey Shi},
  year         = 2023,
  booktitle    = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}
}
@misc{hassani2022dilated,
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
We thank NVIDIA, and the [CUTLASS project](https://github.com/NVIDIA/cutlass/) and team for their efforts in
creating and open-sourcing CUTLASS. We would also like to thank Haicheng Wu for his valuable feedback and comments which led to
the creation of GEMM-based NA.
We also thank Meta and the [xFormers](https://github.com/facebookresearch/xformers/) team
for their FMHA kernel, which is what our Fused Neighborhood Attention kernel is based on.
We thank the [PyTorch](https://github.com/pytorch/pytorch/) project and team.
