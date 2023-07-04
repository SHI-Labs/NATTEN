#!/usr/bin/env python
#################################################################################################
# Copyright (c) 2023 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################

import warnings
import glob
import sys
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
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "src/natten", "__init__.py")
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
    extensions_dir = path.join(this_dir, "src", "natten", "csrc")

    main_source = path.join(extensions_dir, "natten.cpp")
    sources_cpu  = glob.glob(path.join(extensions_dir, "cpu", "*.cpp"))
    sources_cuda = glob.glob(path.join(extensions_dir, "cuda", "*.cu"))
    sources_base = [main_source] + sources_cpu
    sources = sources_base.copy()

    from torch.utils.cpp_extension import ROCM_HOME

    is_rocm_pytorch = (
        True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    )
    assert not is_rocm_pytorch, "NATTEN does not support ROCM."

    extension = CppExtension
    extra_compile_args = {"cxx": ["-O3"]}
    define_macros = []
    if TORCH_113:
        # Torch 1.13 and above have a new dispatcher template
        define_macros += [("TORCH_113", 1)]
    if AVX_INT:
        define_macros += [("AVX_INT", 1)]
    else:
        warnings.warn('Compiling CPU kernels without AVX vectorization, because you are using PyTorch < 1.10.', UserWarning, stacklevel=2)

    if HAS_CUDA:
        extension = CUDAExtension
        sources += sources_cuda
        define_macros += [("WITH_CUDA", 1)]
        extra_compile_args["nvcc"] = ["-O3"]

        if sys.platform == "win32":
            # Inspired by  xFormers setup script
            extra_compile_args["nvcc"] += [
                "-std=c++17",
                "-Xcompiler",
                "/Zc:lambda",
                "-Xcompiler",
                "/Zc:preprocessor",
            ]

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
    package_dir={"": "src"},
    packages=['natten/'],
    package_data={
            '': ['csrc/*', 'csrc/cpu/*', 'csrc/cuda/*'],
            },
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
            "fvcore>=0.1.5,<0.1.6",  # required like this to make it pip installable
        ],
    },
    ext_modules=get_extension(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)
