#!/usr/bin/env python
"""
Neighborhood Attention Extension (NATTEN) 

Setup file

Heavily borrowed from detectron2 setup:
github.com/facebookresearch/detectron2

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import glob
import os
import shutil
from os import path
from setuptools import find_packages, setup
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "NATTEN requires PyTorch >= 1.8"
HAS_CUDA = (torch.cuda.is_available() and (CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1")


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "natten", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    if not HAS_CUDA:
        # Non-CUDA build
        return version
    PYTORCH_VERSION = ''.join(torch.__version__.split('.')[:2])

    assert HAS_CUDA, 'Unfortunately NATTEN does not support cpu only build presently. Please make sure you have CUDA.'
    CUDA_VERSION = ''.join(torch.version.cuda.split('.')[:2])
    version = f'{version}+torch{PYTORCH_VERSION}cu{CUDA_VERSION}'

    return version


def get_extension():
    this_dir = path.dirname(path.abspath(__file__))
    extensions_dir = path.join(this_dir, "natten", "src")

    if not HAS_CUDA:
        # Non-CUDA build
        # TODO: implement NATTEN for CPU
        cpp_source = path.join(extensions_dir, "placeholder.cpp")
        return [CppExtension("natten._C", [cpp_source])]

    main_source = path.join(extensions_dir, "natten.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    from torch.utils.cpp_extension import ROCM_HOME

    is_rocm_pytorch = (
        True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    )
    assert not is_rocm_pytorch, "Unfortunately NATTEN does not support ROCM."

    # common code between cuda and rocm platforms, for hipify version [1,0,0] and later.
    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        path.join(extensions_dir, "*.cu")
    )
    sources = [main_source] + sources

    extra_compile_args = {"cxx": []}
    define_macros = []

    extension = CUDAExtension
    sources += source_cuda

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "natten._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
        )
    ]

    return ext_modules


setup(
    name="natten",
    version=get_version(),
    author="Ali Hassani",
    url="https://github.com/SHI-Labs/Neighborhood-Attention-Transformer",
    description="Neighborhood Attention Extension.",
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
