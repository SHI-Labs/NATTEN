#!/usr/bin/env python
#################################################################################################
# Copyright (c) 2022-2025 Ali Hassani.
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
import tempfile
from os import path
from pathlib import Path
from typing import Any, List

import torch
from setuptools import Extension, setup  # type: ignore
from setuptools.command.build_ext import build_ext  # type: ignore
from torch.utils.cpp_extension import LIB_EXT

IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")
IS_LIBTORCH_BUILT_WITH_CXX11_ABI = torch._C._GLIBCXX_USE_CXX11_ABI

this_directory = Path(__file__).parent
try:
    long_description = (this_directory / "docs/README_pypi.md").read_text()
except:
    long_description = "Neighborhood Attention Extension."

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]

assert torch_ver >= [2, 5], "NATTEN only supports PyTorch >= 2.5"

CUDA_ARCH = os.getenv("NATTEN_CUDA_ARCH", "")
HAS_CUDA_ARCH = CUDA_ARCH != ""

NATTEN_IS_BUILDING_DIST = bool(os.getenv("NATTEN_IS_BUILDING_DIST", "0") == "1")

VERBOSE = bool(os.getenv("NATTEN_VERBOSE", "0") == "1")

tmp_dir = tempfile.TemporaryDirectory()
NATTEN_BUILD_DIR = os.getenv("NATTEN_BUILD_DIR", tmp_dir.name)
if not os.path.isdir(NATTEN_BUILD_DIR):
    NATTEN_BUILD_DIR = tmp_dir.name

DEFAULT_N_WORKERS = max(1, (multiprocessing.cpu_count() // 4))
try:
    N_WORKERS = int(os.getenv("NATTEN_N_WORKERS", DEFAULT_N_WORKERS))
except:
    N_WORKERS = int(DEFAULT_N_WORKERS)

if not HAS_CUDA_ARCH:
    HAS_CUDA = torch.cuda.is_available()

    if HAS_CUDA:
        cuda_device = torch.cuda.get_device_properties(torch.cuda.current_device())
        sm = cuda_device.major + cuda_device.minor * 0.1
        CUDA_ARCH = f"{sm}"
        print(
            "`NATTEN_CUDA_ARCH` not set, but detected CUDA driver with PyTorch. "
            f"Building for {CUDA_ARCH=}."
        )

        assert torch.version.cuda is not None
        TORCH_CUDA_VERSION = [x for x in torch.version.cuda.split(".")[:2]]
        CUDA_TAG = "".join([x for x in TORCH_CUDA_VERSION])
        CUDA_VERSION = [int(x) for x in TORCH_CUDA_VERSION]

        assert CUDA_VERSION >= [12, 0], "NATTEN only supports CUDA 12.0 and above."
        if CUDA_VERSION >= [12, 0] and IS_WINDOWS:
            print(
                "WARNING: Torch cmake will likely fail on Windows with CUDA 12.X. "
                "Please refer to NATTEN documentation to read more about the issue "
                "and how to get around it until the issue is fixed in torch."
            )

        print(f"PyTorch was built with CUDA Toolkit {CUDA_TAG}")
        print(f"Building NATTEN for the following architecture(s): {CUDA_ARCH}")

        print(f"Number of workers: {N_WORKERS}")

    else:
        print(
            "Building WITHOUT libnatten. `NATTEN_CUDA_ARCH` is not set, and did not detect CUDA "
            "driver with PyTorch."
        )


BUILD_WITH_CUDA = CUDA_ARCH != ""


def get_version() -> str:
    init_py_path = path.join(
        path.abspath(path.dirname(__file__)), "src/natten", "version.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [ln.strip() for ln in init_py if ln.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    if not NATTEN_IS_BUILDING_DIST:
        return version
    PYTORCH_VERSION = "".join(torch.__version__.split("+")[0].split("."))

    if BUILD_WITH_CUDA:
        if torch.version.cuda is None:
            raise ValueError(
                "Attempted to build NATTEN with libnatten, which requires PyTorch be "
                f"built with CUDA, but {torch.version.cuda=}."
            )

        TORCH_CUDA_VERSION = [x for x in torch.version.cuda.split(".")[:2]]
        CUDA_TAG = "".join([x for x in TORCH_CUDA_VERSION])
        CU = f"cu{CUDA_TAG}"
        version = f"{version}+torch{PYTORCH_VERSION}{CU}"

    return version


def _check_cuda_arch(arch: Any) -> int:
    try:
        arch = float(arch)
        arch = arch * 10  # 8.6 => 86
        assert (
            arch >= 30
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
    return (
        "-real;".join([str(x) if x not in [90, 100] else f"{x}a" for x in arch_list])
        + "-real"
    )


class BuildExtension(build_ext):
    def build_extension(self, ext):
        if BUILD_WITH_CUDA:
            # Hack so that we can build somewhere other than /tmp in development mode.
            # Also because we want CMake to build everything elsewhere, otherwise pypi will package
            # build files.
            build_dir = self.build_lib if NATTEN_BUILD_DIR is None else NATTEN_BUILD_DIR

            this_dir = path.dirname(path.abspath(__file__))
            cmake_lists_dir = path.join(this_dir, "csrc")
            try:
                subprocess.check_output(["cmake", "--version"])
            except OSError:
                raise RuntimeError("Cannot find CMake executable")

            output_binary_name = self.get_ext_filename(
                ext.name
            )  # i.e. libnatten.cpython-VERSION-ARCH-OS.so
            output_so_name = output_binary_name.replace(LIB_EXT, "")

            # extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

            max_sm = 0
            cuda_arch_list = []
            cuda_arch_list_str = ""

            cuda_arch_list = get_cuda_arch_list(CUDA_ARCH)  # Expects xx -- i.e. 86
            cuda_arch_list_str = arch_list_to_cmake_tags(cuda_arch_list)
            max_sm = max(cuda_arch_list)

            print(
                f"Building NATTEN for the following archs: {cuda_arch_list} (max: {max_sm})"
            )
            print(f"Building with {N_WORKERS} workers.")
            print(f"Build directory: {build_dir}")
            print(f"{IS_LIBTORCH_BUILT_WITH_CXX11_ABI=}")

            cmake_args = [
                f"-DPYTHON_PATH={sys.executable}",
                f"-DOUTPUT_FILE_NAME={output_so_name}",
                f"-DNATTEN_CUDA_ARCH_LIST={cuda_arch_list_str}",
                f"-DNATTEN_IS_WINDOWS={int(IS_WINDOWS)}",
                f"-DIS_LIBTORCH_BUILT_WITH_CXX11_ABI={int(IS_LIBTORCH_BUILT_WITH_CXX11_ABI)}",
            ]

            if max_sm < 50:
                raise RuntimeError(
                    "NATTEN's CUDA backend only supports SM50 and above, "
                    f"saw SM{max_sm} in {CUDA_ARCH}."
                )

            if 90 in cuda_arch_list:
                cmake_args.append("-DNATTEN_WITH_HOPPER_FNA=1")

            if 100 in cuda_arch_list:
                cmake_args.append("-DNATTEN_WITH_BLACKWELL_FNA=1")

            if IS_WINDOWS:
                python_path = sys.executable
                assert (
                    "python.exe" in python_path
                ), f"Expected the python executable path to end with python.exe, got {python_path}"
                python_lib_dir = python_path.replace("python.exe", "libs").strip()
                cmake_args.append(f"-DPY_LIB_DIR={python_lib_dir}")
                cmake_args.append("-G Ninja")
                cmake_args.append("-DCMAKE_BUILD_TYPE=Release")

            if not os.path.exists(build_dir):
                os.makedirs(build_dir)

            if not os.path.exists(self.build_lib):
                os.makedirs(self.build_lib)

            so_dir_local = os.path.join(build_dir, os.path.dirname(output_binary_name))
            so_path_local = f"{build_dir}/{output_binary_name}"
            if not os.path.exists(so_dir_local):
                os.makedirs(so_dir_local)

            so_dir = os.path.join(self.build_lib, os.path.dirname(output_binary_name))
            so_path_final = f"{self.build_lib}/{output_binary_name}"
            if not os.path.exists(so_dir):
                os.makedirs(so_dir)

            # Config and build the extension
            subprocess.check_call(
                ["cmake", cmake_lists_dir] + cmake_args, cwd=build_dir
            )
            cmake_build_args = [
                "--build",
                build_dir,
                "-j",
                str(N_WORKERS),
            ]
            if VERBOSE:
                cmake_build_args.append("--verbose")
            subprocess.check_call(["cmake", *cmake_build_args])

            assert os.path.isfile(
                so_path_local
            ), f"Expected libnatten binary in {so_path_local}."
            if build_dir != self.build_lib:
                shutil.copy(so_path_local, so_path_final)
            assert os.path.isfile(so_path_final)

            # Clean up cmake files when building dist package;
            # otherwise they will get packed into the wheel.
            if NATTEN_IS_BUILDING_DIST:
                for file in os.listdir(build_dir):
                    fn = os.path.join(build_dir, file)
                    if os.path.isfile(fn):
                        os.remove(fn)
                    elif file != "natten":
                        shutil.rmtree(fn)
        else:
            # Libnatten is CUDA only now.
            pass


setup(
    name="natten",
    version=get_version(),
    author="Ali Hassani",
    url="https://natten.org",
    description="Neighborhood Attention Extension.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=[
        "natten",
        "natten/profiling_utils",
        "natten/utils",
        "natten/backends",
        "natten/backends/configs",
        "natten/backends/configs/cutlass",
        "natten/backends/configs/cutlass_blackwell",
        "natten/backends/configs/cutlass_hopper",
        "natten/backends/configs/flex",
    ],
    package_data={
        "": ["csrc/**/*"],
    },
    python_requires=">=3.9",
    install_requires=[
        "torch",
    ],
    extras_require={},
    ext_modules=[Extension("natten.libnatten", [])] if BUILD_WITH_CUDA else [],
    cmdclass={"build_ext": BuildExtension} if BUILD_WITH_CUDA else {},
)
