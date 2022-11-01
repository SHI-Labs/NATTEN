#!/usr/bin/env python
"""
Neighborhood Attention Extension (NATTEN) 

Setup file

Heavily borrowed from detectron2 setup:
github.com/facebookresearch/detectron2

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import warnings
import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

from pathlib import Path
this_directory = Path(__file__).parent
try:
    long_description = (this_directory / "assets/README_pypi.md").read_text()
except:
    long_description = "Neighborhood Attention Extension."

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "NATTEN requires PyTorch >= 1.8"
AVX_INT = torch_ver >= [1, 10]
TORCH_113 = torch_ver >= [1, 13]
HAS_CUDA = (torch.cuda.is_available() and (CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1")
NATTEN_VERSION_SUFFIX = os.getenv("NATTEN_VERSION_SUFFIX", "")


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "natten", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    if NATTEN_VERSION_SUFFIX != "1":
        return f'{version}{NATTEN_VERSION_SUFFIX}'
    PYTORCH_VERSION = ''.join(torch.__version__.split('+')[0].split('.'))

    if HAS_CUDA:
        CUDA_VERSION = ''.join(torch.version.cuda.split('.')[:2])
        CU = f'cu{CUDA_VERSION}'
    else:
        CU = 'cpu'
    version = f'{version}+torch{PYTORCH_VERSION}{CU}'

    return version


def get_extension():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "natten", "src")

    main_source = path.join(extensions_dir, "natten.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    from torch.utils.cpp_extension import ROCM_HOME

    is_rocm_pytorch = (
        True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    )
    assert not is_rocm_pytorch, "Unfortunately NATTEN does not support ROCM."

    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        path.join(extensions_dir, "*.cu")
    )
    sources = [main_source] + sources

    extension = CppExtension
    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = []
    if TORCH_113:
        define_macros += [("TORCH_113", 1)]
    if AVX_INT:
        define_macros += [("AVX_INT", 1)]
    else:
        warnings.warn('Compiling CPU kernels without AVX vectorization, because you are using PyTorch < 1.10.', UserWarning, stacklevel=2)

    if HAS_CUDA:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", 1)]
        extra_compile_args["nvcc"] = ["-O3"]

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "natten._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="natten",
    version=get_version(),
    author="Ali Hassani",
    url="https://github.com/SHI-Labs/NATTEN",
    description="Neighborhood Attention Extension.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['natten/'],
    python_requires=">=3.7",
    install_requires=[
        "packaging",
    ],
    extras_require={
        # optional dependencies, required by some features
        "all": [
            "fvcore>=0.1.5,<0.1.6",  # required like this to make it pip installable
        ],
        # dev dependencies. Install them by `pip install 'natten[dev]'`
        "dev": [
        ],
    },
    ext_modules=get_extension(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
