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
import multiprocessing
import subprocess
from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension, _nt_quote_args, SHARED_FLAG, LIB_EXT
from pathlib import Path

IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform.startswith('darwin')
IS_LINUX = sys.platform.startswith('linux')

this_directory = Path(__file__).parent
try:
    long_description = (this_directory / "assets/README_pypi.md").read_text()
except:
    long_description = "Neighborhood Attention Extension."

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 8], "NATTEN requires PyTorch >= 1.8"
AVX_INT = torch_ver >= [1, 10]
HAS_CUDA = (torch.cuda.is_available() and (CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1")
NATTEN_VERSION_SUFFIX = os.getenv("NATTEN_VERSION_SUFFIX", "")
DEFAULT_N_WORKERS = max(1, (multiprocessing.cpu_count() // 4))
DEFAULT_CUDA_ARCH_LIST=""
if HAS_CUDA:
    cuda_device = torch.cuda.get_device_properties(torch.cuda.current_device())
    sm = cuda_device.major + cuda_device.minor * 0.1
    DEFAULT_CUDA_ARCH_LIST = f"{sm}"

    # TODO: raise an error or at least a warning when torch cuda doesn't match
    # system.
    TORCH_CUDA_VERSION = [x for x in torch.version.cuda.split(".")[:2]]
    CUDA_TAG = ''.join([x for x in TORCH_CUDA_VERSION])
    CUDA_VERSION = [int(x) for x in TORCH_CUDA_VERSION]

    assert CUDA_VERSION >= [10, 2], "NATTEN only supports CUDA 10.2 and above."

cuda_arch = os.environ.get("NATTEN_CUDA_ARCH", DEFAULT_CUDA_ARCH_LIST)
# In case the env variable is set, but to an empty string
if cuda_arch == "":
    cuda_arch = DEFAULT_CUDA_ARCH_LIST

n_workers = os.environ.get("NATTEN_N_WORKERS", DEFAULT_N_WORKERS)
# In case the env variable is set, but to an empty string
if n_workers == "":
    n_workers = DEFAULT_N_WORKERS

if HAS_CUDA:
    print(f"Building NATTEN with CUDA {CUDA_TAG}")
    print(f"Building NATTEN for SM: {cuda_arch}")
else:
    print(f"Building NATTEN for CPU ONLY.")

print(f"Number of workers: {n_workers}")

# TODO: pretty sure this means nothing with the latest CMake config.
# We should be getting all of torch's additional compiler flags.
torch_ext_fn = CUDAExtension if HAS_CUDA else CppExtension


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "src/natten", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    if NATTEN_VERSION_SUFFIX != "1":
        return f'{version}{NATTEN_VERSION_SUFFIX}'
    PYTORCH_VERSION = ''.join(torch.__version__.split('+')[0].split('.'))

    if HAS_CUDA:
        CU = f'cu{CUDA_TAG}'
    else:
        CU = 'cpu'
    version = f'{version}+torch{PYTORCH_VERSION}{CU}'

    return version


def _check_cuda_arch(arch):
    try:
       arch = float(arch)
       arch = arch * 10 # 8.6 => 86
       assert arch >= 30 and arch < 100, f"Only SM30 and above are supported at this time, got {arch}."
       return str(int(arch))
    except ValueError:
        raise ValueError(f"Invalid architecture list {cuda_arch_list}.")

def get_cuda_arch_list(cuda_arch_list: str):
    sep = ";" # expects semicolon separated list
    if sep not in cuda_arch_list:
        # Probably a single arch
        return _check_cuda_arch(cuda_arch_list), 1
    arch_list = cuda_arch_list.split(sep)
    output_arch_list = []
    for arch in arch_list:
        output_arch_list.append(_check_cuda_arch(arch))
    return ";".join(output_arch_list), len(output_arch_list)


class BuildExtension(build_ext):
    def build_extension(self, ext):
        this_dir = path.dirname(path.abspath(__file__))
        cmake_lists_dir = path.join(this_dir, "csrc")
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError('Cannot find CMake executable')

        # TODO: fix this!!!
        # This is where it gets super hacky
        # We're cutting off whatever compiler setuptools picks right here.
        # What's worse is that we're taking the path where it --would-- dump the final
        # library, and passing it down to CMake so that it dumps it there.

        output_so_name = self.get_ext_filename(ext.name) # i.e. _C.cpython-VERSION-ARCH-OS.so
        output_so_name = output_so_name.replace(LIB_EXT, '')

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        n_arch = 0
        cuda_arch_list=""
        torch_cuda_arch_list=""
        if HAS_CUDA:
            torch_cuda_arch_list = cuda_arch # Expects x.x -- i.e. 8.6
            cuda_arch_list, n_arch = get_cuda_arch_list(cuda_arch) # Expects xx -- i.e. 86

            # TODO: this assertion prevents building binaries that target multiple architectures,
            # a.k.a. wheels. The reason for it is that we need to separate builds by architecture now
            # with the bfloat16 support, and the GEMM kernels which target SM80.
            # This should be resolved before our next release, and the assertion will be removed or
            # made conditional.
            assert n_arch == 1, f"This commit does not yet support building NATTEN for multiple architectures. " + \
                f"You selected {n_arch} architectures: {cuda_arch_list}. " + \
                f"Please select only one architecture."
            current_arch = int(cuda_arch_list)
            print(f"Current arch: {current_arch}")

        cmake_args = [
            f"-DOUTPUT_FILE_NAME={output_so_name}",
            f"-DNATTEN_CUDA_ARCH_LIST={cuda_arch_list}",
            f"-DNATTEN_TORCH_CUDA_ARCH_LIST={torch_cuda_arch_list}",
            f"-DNATTEN_IS_WINDOWS={int(IS_WINDOWS)}",
            f"-DNATTEN_IS_MAC={int(IS_MACOS)}",
        ]

        if AVX_INT:
            cmake_args.append("-DNATTEN_WITH_AVX=1")

        if HAS_CUDA:
            cmake_args.append("-DNATTEN_WITH_CUDA=1")
            # TODO: this is connected to the assertion above; this is wrong, but temporary.
            if current_arch >= 60:
                cmake_args.append("-DNATTEN_WITH_CUDA_FP16=1")
            if current_arch >= 80 and CUDA_VERSION >= [11, 0]:
                cmake_args.append("-DNATTEN_WITH_CUDA_BF16=1")
            if current_arch >= 70 and CUDA_VERSION >= [11, 0]:
                cmake_args.append("-DNATTEN_WITH_CUTLASS=1")
                if current_arch >= 80:
                    cmake_args.append("-DNATTEN_CUTLASS_TARGET_SM=80")
                elif current_arch >= 75:
                    cmake_args.append("-DNATTEN_CUTLASS_TARGET_SM=75")
                elif current_arch >= 70:
                    cmake_args.append("-DNATTEN_CUTLASS_TARGET_SM=70")
                else:
                    raise ValueError(f"This should not have happened. "
                                     "NATTEN can only be built with CUTLASS on SM70 and above, "
                                     f"got {current_arch}.")

        if not os.path.exists(self.build_lib):
            os.makedirs(self.build_lib)

        so_dir = os.path.join(self.build_lib, os.path.dirname(output_so_name))
        if not os.path.exists(so_dir):
            os.makedirs(so_dir)

        # Config and build the extension
        subprocess.check_call(['cmake', cmake_lists_dir] + cmake_args, cwd=self.build_lib)
        subprocess.check_call(['make', f"-j{n_workers}"], cwd=self.build_lib)


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
    ext_modules=[torch_ext_fn("natten._C", [])],
    cmdclass={'build_ext': BuildExtension},
)
