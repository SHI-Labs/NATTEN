![NATTENLogo](https://www.shi-labs.com/natten/assets/img/natten_light.png) 

<a href="https://www.shi-labs.com/natten/"><img src="https://img.shields.io/badge/pip%20install%20natten-read%20more-%23C209C1" /></a>

*Neighborhood Attention Extension*

Bringing attention to a neighborhood near you!

NATTEN is an open-source project dedicated to providing fast implementations for
[Neighborhood Attention](https://scholar.google.com/citations?view_op=view_citation&citation_for_view=Ndu0dUcAAAAJ:b0M2c_1WBrUC),
a sliding window self-attention mechanism.

If you're not familiar with neighborhood attention, please refer to 
[our papers](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer), or watch our 
[YouTube video](https://www.youtube.com/watch?v=Ya4BfioxIHA) from CVPR 2023.

## Getting started
 
NATTEN supports PyTorch version 2.0 and later, and Python versions 3.8 and above. 
Python 3.12 is only supported with torch >= 2.2.0.

Older NATTEN releases supported python >= 3.7 and torch >= 1.8.

Please refer to [install instructions](docs/install.md) to find out whether your operating system and hardware accelerator is
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
| 1D            | naive       | :white_check_mark: | :white_check_mark: | :white_check_mark:       | Forward and reverse mode |
| 2D            | naive       | :white_check_mark: | :white_check_mark: | :white_check_mark:       | Forward and reverse mode |
| 3D            | naive       | :white_check_mark: | :white_check_mark: | :white_check_mark:       | Forward and reverse mode |

Notes:
* Forward mode autograd does not support relative positional biases and causal masking yet.
* Relative positional biases are not yet supported when any axis has causal masking enabled.

### CUDA

| Problem space | CUDA Backend | Causal masking     | Varying parameters | Relative positional bias | Autograd support         | Min. Arch |
| -----------   | -----------  | ------------------ | ------------------ | ------------------------ | ------------------------ | --------- |
| 1D            | naive        | :white_check_mark: | :white_check_mark: | :white_check_mark:       | Forward and reverse mode | SM35      |
| 2D            | naive        | :white_check_mark: | :white_check_mark: | :white_check_mark:       | Forward and reverse mode | SM35      |
| 3D            | naive        | :white_check_mark: | :white_check_mark: | :white_check_mark:       | Forward and reverse mode | SM35      |
| 1D            | gemm         |                    |                    | :white_check_mark:       | Forward and reverse mode | SM70      |
| 2D            | gemm         |                    |                    | :white_check_mark:       | Forward and reverse mode | SM70      |
| 1D            | fna          | :white_check_mark: | :white_check_mark: | :white_check_mark:       | Coming soon              | SM50      |
| 2D            | fna          | :white_check_mark: | :white_check_mark: | :white_check_mark:       | Coming soon              | SM50      |
| 3D            | fna          | :white_check_mark: | :white_check_mark: | :white_check_mark:       | Coming soon              | SM50      |

Notes: 
* FP16 kernels are only available on SM50 and above, and BF16 requires SM80 and above.
* GEMM backend on SM70 and SM75 can only do FP16.
* Tiled only implements 1/3 of the ops, is only implemented for 2D problems, and requires head dim = 32.
* Forward mode autograd does not support relative positional biases and causal masking yet.
* Relative positional biases are not yet supported when any axis has causal masking enabled.
* Naive backend allows FP16 for SM50 and above only. FP32/FP64 are available for SM35 and above.

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
We thank NVIDIA, and the [CUTLASS project](https://github.com/NVIDIA/cutlass/) and team for their efforts in
creating and open-sourcing CUTLASS. We would also like to thank Haicheng Wu for his valuable feedback and comments which led to
the creation of GEMM-based NA.
We also thank Meta and the [xFormers](https://github.com/facebookresearch/xformers/) team
for their FMHA kernel, which is what our Fused Neighborhood Attention kernel is based on.
We thank the [PyTorch](https://github.com/pytorch/pytorch/) project and team.
