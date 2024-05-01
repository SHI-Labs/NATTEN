#!/usr/bin/env python
#################################################################################################
# Copyright (c) 2022-2024 Ali Hassani.
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

import multiprocessing
import os
import shutil
import subprocess
import sys
from os import path
from pathlib import Path
from typing import Any, List

import torch
from setuptools import Extension, setup  # type: ignore
from setuptools.command.build_ext import build_ext  # type: ignore
from torch.utils.cpp_extension import CUDA_HOME, LIB_EXT

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform.startswith("darwin")
IS_LINUX = sys.platform.startswith("linux")
IS_LIBTORCH_BUILT_WITH_CXX11_ABI = torch._C._GLIBCXX_USE_CXX11_ABI

this_directory = Path(__file__).parent
try:
    long_description = (this_directory / "assets/README_pypi.md").read_text()
except:
    long_description = "Neighborhood Attention Extension."

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [2, 0], "NATTEN requires PyTorch >= 2.0"
AVX_INT = torch_ver >= [1, 10]
FORCE_CUDA = (
    os.getenv("FORCE_CUDA", "0") == "1" or os.getenv("NATTEN_WITH_CUDA", "0") == "1"
)
HAS_CUDA = FORCE_CUDA or (torch.cuda.is_available() and (CUDA_HOME is not None))
NATTEN_IS_BUILDING_DIST = bool(os.getenv("NATTEN_IS_BUILDING_DIST", 0))
DEFAULT_N_WORKERS = max(1, (multiprocessing.cpu_count() // 4))

cuda_arch = os.getenv("NATTEN_CUDA_ARCH", "")
if FORCE_CUDA and not cuda_arch:
    raise RuntimeError(
        "Target architecture tags must be specified when "
        "forcing CUDA builds; but environment variable "
        f"NATTEN_CUDA_ARCH={cuda_arch} . "
        "If you're using NATTEN's Makefile, you can "
        "pass archtags with the `CUDA_ARCH` flag; i.e. "
        'make WITH_CUDA=1 CUDA_ARCH="8.0;8.6"'
    )

if HAS_CUDA:
    if not cuda_arch:
        cuda_device = torch.cuda.get_device_properties(torch.cuda.current_device())
        sm = cuda_device.major + cuda_device.minor * 0.1
        cuda_arch = f"{sm}"

    # TODO: raise an error or at least a warning when torch cuda doesn't match
    # system.
    assert torch.version.cuda is not None
    TORCH_CUDA_VERSION = [x for x in torch.version.cuda.split(".")[:2]]
    CUDA_TAG = "".join([x for x in TORCH_CUDA_VERSION])
    CUDA_VERSION = [int(x) for x in TORCH_CUDA_VERSION]

    assert CUDA_VERSION >= [11, 0], "NATTEN only supports CUDA 11.0 and above."
    if CUDA_VERSION >= [12, 0] and IS_WINDOWS:
        print(
            "WARNING: Torch cmake will likely fail on Windows with CUDA 12.X. "
            "Please refer to NATTEN documentation to read more about the issue "
            "and how to get around it until the issue is fixed in torch."
        )


n_workers = str(os.environ.get("NATTEN_N_WORKERS", DEFAULT_N_WORKERS)).strip()
# In case the env variable is set, but to an empty string
if n_workers == "":
    n_workers = str(DEFAULT_N_WORKERS)

if HAS_CUDA:
    print(f"Building NATTEN with CUDA {CUDA_TAG}")
    print(f"Building NATTEN for SM: {cuda_arch}")
else:
    print("Building NATTEN for CPU ONLY.")

print(f"Number of workers: {n_workers}")

verbose = os.environ.get("NATTEN_VERBOSE", 0)


def get_version() -> str:
    init_py_path = path.join(
        path.abspath(path.dirname(__file__)), "src/natten", "__init__.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [ln.strip() for ln in init_py if ln.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    if not NATTEN_IS_BUILDING_DIST:
        return version
    PYTORCH_VERSION = "".join(torch.__version__.split("+")[0].split("."))

    if HAS_CUDA:
        CU = f"cu{CUDA_TAG}"
    else:
        CU = "cpu"
    version = f"{version}+torch{PYTORCH_VERSION}{CU}"

    return version


def _check_cuda_arch(arch: Any) -> int:
    try:
        arch = float(arch)
        arch = arch * 10  # 8.6 => 86
        assert (
            arch >= 30 and arch < 100
        ), f"Only SM30 and above are supported at this time, got {arch}."
        return int(arch)
    except ValueError:
        raise ValueError(f"Invalid architecture list {arch}.")


def get_cuda_arch_list(cuda_arch_list: str) -> List[int]:
    sep = ";"  # expects semicolon separated list
    if sep not in cuda_arch_list:
        # Probably a single arch
        return [_check_cuda_arch(cuda_arch_list)]
    arch_list = cuda_arch_list.split(sep)
    output_arch_list = []
    for arch in arch_list:
        output_arch_list.append(_check_cuda_arch(arch))
    return output_arch_list


def arch_list_to_cmake_tags(arch_list: List[int]) -> str:
    return "-real;".join([str(x) for x in arch_list]) + "-real"


class BuildExtension(build_ext):
    def build_extension(self, ext):
        this_dir = path.dirname(path.abspath(__file__))
        cmake_lists_dir = path.join(this_dir, "csrc")
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("Cannot find CMake executable")

        output_so_name = self.get_ext_filename(
            ext.name
        )  # i.e. libnatten.cpython-VERSION-ARCH-OS.so
        output_so_name = output_so_name.replace(LIB_EXT, "")

        # extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        max_sm = 0
        cuda_arch_list = []
        cuda_arch_list_str = ""
        if HAS_CUDA:
            cuda_arch_list = get_cuda_arch_list(cuda_arch)  # Expects xx -- i.e. 86
            cuda_arch_list_str = arch_list_to_cmake_tags(cuda_arch_list)
            max_sm = max(cuda_arch_list)

            print(f"Current arch list: {cuda_arch_list} (max: {max_sm})")

        cmake_args = [
            f"-DPYTHON_PATH={sys.executable}",
            f"-DOUTPUT_FILE_NAME={output_so_name}",
            f"-DNATTEN_CUDA_ARCH_LIST={cuda_arch_list_str}",
            f"-DNATTEN_IS_WINDOWS={int(IS_WINDOWS)}",
            f"-DNATTEN_IS_MAC={int(IS_MACOS)}",
            f"-DIS_LIBTORCH_BUILT_WITH_CXX11_ABI={int(IS_LIBTORCH_BUILT_WITH_CXX11_ABI)}",
        ]

        if AVX_INT:
            cmake_args.append("-DNATTEN_WITH_AVX=1")

        if HAS_CUDA:
            assert max_sm >= 30
            cmake_args.append("-DNATTEN_WITH_CUDA=1")
            if max_sm >= 50:
                cmake_args.append("-DNATTEN_WITH_CUTLASS=1")

        if IS_WINDOWS:
            python_path = sys.executable
            assert (
                "python.exe" in python_path
            ), f"Expected the python executable path to end with python.exe, got {python_path}"
            python_lib_dir = python_path.replace("python.exe", "libs").strip()
            cmake_args.append(f"-DPY_LIB_DIR={python_lib_dir}")
            cmake_args.append("-G Ninja")
            cmake_args.append("-DCMAKE_BUILD_TYPE=Release")

        if not os.path.exists(self.build_lib):
            os.makedirs(self.build_lib)

        so_dir = os.path.join(self.build_lib, os.path.dirname(output_so_name))
        if not os.path.exists(so_dir):
            os.makedirs(so_dir)

        # Config and build the extension
        subprocess.check_call(
            ["cmake", cmake_lists_dir] + cmake_args, cwd=self.build_lib
        )
        cmake_build_args = [
            "--build",
            self.build_lib,
            "-j",
            n_workers,
        ]
        if verbose:
            cmake_build_args.append("--verbose")
        subprocess.check_call(["cmake", *cmake_build_args])

        # Clean up cmake files when building dist package;
        # otherwise they will get packed into the wheel.
        if NATTEN_IS_BUILDING_DIST:
            for file in os.listdir(self.build_lib):
                fn = os.path.join(self.build_lib, file)
                if os.path.isfile(fn):
                    os.remove(fn)
                elif file != "natten":
                    shutil.rmtree(fn)


setup(
    name="natten",
    version=get_version(),
    author="Ali Hassani",
    url="https://github.com/SHI-Labs/NATTEN",
    description="Neighborhood Attention Extension.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=["natten", "natten/utils", "natten/autotuner", "natten/autotuner/configs"],
    package_data={
        "": ["csrc/**/*"],
    },
    python_requires=">=3.8",
    install_requires=[
        "packaging",
        "torch>=2.0.0",
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
    ext_modules=[Extension("natten.libnatten", [])],
    cmdclass={"build_ext": BuildExtension},
)
