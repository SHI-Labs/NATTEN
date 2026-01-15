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

import filecmp
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

AUTOGEN_POLICY = os.getenv("NATTEN_AUTOGEN_POLICY", "default")
AUTOGEN_POLICY = AUTOGEN_POLICY if AUTOGEN_POLICY != "" else "default"

tmp_dir = tempfile.TemporaryDirectory()
print(f"***************** {tmp_dir=}")
NATTEN_BUILD_DIR = os.getenv("NATTEN_BUILD_DIR", tmp_dir.name)
if not os.path.isdir(NATTEN_BUILD_DIR):
    NATTEN_BUILD_DIR = tmp_dir.name
print(f"***************** {NATTEN_BUILD_DIR=}")

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
        "-real;".join(
            [str(x) if x not in [90, 100, 103] else f"{x}a" for x in arch_list]
        )
        + "-real"
    )


# determines the number of build targets for each category of kernels under different
# policies. More build targets means fewer kernels in each build target, and more room for
# build parallelism.

NUM_SPLITS = {
    "default": {
        "reference": 2,
        "fna": 64,
        "fmha": 6,
        "flash-fna": 64,
        "flash-fna-bwd": 64,
        "flash-fmha": 6,
        "flash-fmha-bwd": 6,
        "hopper-fna": 8,
        "hopper-fna-bwd": 4,
        "hopper-fmha": 1,
        "hopper-fmha-bwd": 1,
        "blackwell-fna": 28,
        "blackwell-fna-bwd": 14,
        "blackwell-fmha": 4,
        "blackwell-fmha-bwd": 4,
    },
    "fine": {
        "reference": 4,
        "fna": 128,
        "fmha": 12,
        "flash-fna": 128,
        "flash-fna-bwd": 128,
        "flash-fmha": 12,
        "flash-fmha-bwd": 12,
        "hopper-fna": 16,
        "hopper-fna-bwd": 8,
        "hopper-fmha": 1,
        "hopper-fmha-bwd": 1,
        "blackwell-fna": 56,
        "blackwell-fna-bwd": 28,
        "blackwell-fmha": 2,
        "blackwell-fmha-bwd": 2,
    },
    "coarse": {
        "reference": 1,
        "fna": 32,
        "fmha": 3,
        "flash-fna": 32,
        "flash-fna-bwd": 32,
        "flash-fmha": 3,
        "flash-fmha-bwd": 3,
        "hopper-fna": 4,
        "hopper-fna-bwd": 2,
        "hopper-fmha": 1,
        "hopper-fmha-bwd": 1,
        "blackwell-fna": 14,
        "blackwell-fna-bwd": 7,
        "blackwell-fmha": 1,
        "blackwell-fmha-bwd": 1,
    },
}


def autogen_directories_match(dir1, dir2):
    """
    Compare two directories recursively, ignoring timestamps and hidden files.
    Only compares files with .cu, .cpp, .h, .hpp, .cuh extensions.

    Used for skipping redundant autogen writes that can mislead cmake into recompiling things.
    """

    extensions = {".cu", ".cpp", ".h", ".hpp", ".cuh"}

    def should_include(filename):
        return not filename.startswith(".") and Path(filename).suffix in extensions

    def get_matching_files(directory):
        files = {}
        for root, dirs, filenames in os.walk(directory):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for filename in filenames:
                if should_include(filename):
                    filepath = os.path.join(root, filename)
                    rel_path = os.path.relpath(filepath, directory)
                    files[rel_path] = filepath
        return files

    files1 = get_matching_files(dir1)
    files2 = get_matching_files(dir2)

    # Check if same files exist in both directories
    if set(files1.keys()) != set(files2.keys()):
        return False

    # Compare file contents
    for rel_path in files1:
        if not filecmp.cmp(files1[rel_path], files2[rel_path]):
            return False

    return True


def autogen_kernel_instantitations(
    this_dir: str,
    autogen_dir: str,
    scripts_dir: str,
    policy: str,
    cuda_arch_list: List[int],
):
    if policy not in NUM_SPLITS.keys():
        raise ValueError(
            f"Unrecognized autogen policy {policy}; supported policies: {list(NUM_SPLITS.keys())}."
        )

    NUM_SPLITS_POLICY = NUM_SPLITS[policy]

    # if path.isdir(autogen_dir):
    #     shutil.rmtree(autogen_dir)

    categories = {
        "reference": ("autogen_reference_fna.py", "reference"),
        "fna": ("autogen_fna.py", "fna"),
        "fmha": ("autogen_fmha.py", "fmha"),
        "flash-fna": ("autogen_flash_fna.py", "flash_fna"),
        "flash-fna-bwd": ("autogen_flash_fna_bwd.py", "flash_fna_bwd"),
        "flash-fmha": ("autogen_flash_fmha.py", "flash_fmha"),
        "flash-fmha-bwd": ("autogen_flash_fmha_bwd.py", "flash_fmha_bwd"),
    }
    categories_sm90 = {
        "hopper-fna": ("autogen_hopper_fna.py", "hopper_fna"),
        "hopper-fmha": ("autogen_hopper_fmha.py", "hopper_fmha"),
        "hopper-fna-bwd": ("autogen_hopper_fna_bwd.py", "hopper_fna_bwd"),
        "hopper-fmha-bwd": ("autogen_hopper_fmha_bwd.py", "hopper_fmha_bwd"),
    }
    categories_sm100 = {
        "blackwell-fna": ("autogen_blackwell_fna.py", "blackwell_fna"),
        "blackwell-fmha": ("autogen_blackwell_fmha.py", "blackwell_fmha"),
        "blackwell-fna-bwd": ("autogen_blackwell_fna_bwd.py", "blackwell_fna_bwd"),
        "blackwell-fmha-bwd": ("autogen_blackwell_fmha_bwd.py", "blackwell_fmha_bwd"),
    }

    if 90 in cuda_arch_list:
        categories = categories | categories_sm90

    if 100 in cuda_arch_list or 103 in cuda_arch_list:
        categories = categories | categories_sm100

    tmp_dir_autogen = tempfile.TemporaryDirectory()
    tmp_output_dir = path.join(tmp_dir_autogen.name, "csrc")

    for cat, (script, out_dir) in categories.items():
        assert cat in NUM_SPLITS_POLICY
        script_path = path.join(scripts_dir, script)
        if not path.isfile(script_path):
            raise RuntimeError(
                f"Expected to find autogen script {script} under {scripts_dir}, but "
                f"{script_path} does not exist. Raise an issue if you didn't change anything."
            )

        print(f"Stamping out {cat} kernels")

        subprocess.check_call(
            [
                "python",
                script_path,
                "--num-splits",
                str(NUM_SPLITS_POLICY[cat]),
                "-o",
                tmp_output_dir,
            ],
            cwd=this_dir,
        )

        target_include_dir = path.join(
            tmp_output_dir, "autogen", "include", "natten_autogen", "cuda", out_dir
        )
        target_src_dir = path.join(tmp_output_dir, "autogen", "src", "cuda", out_dir)

        current_include_dir = path.join(
            autogen_dir, "include", "natten_autogen", "cuda", out_dir
        )
        current_src_dir = path.join(autogen_dir, "src", "cuda", out_dir)

        for current_dir, target_dir in zip(
            [current_include_dir, current_src_dir], [target_include_dir, target_src_dir]
        ):
            if not path.isdir(target_dir):
                raise RuntimeError(
                    f"Autogen for {cat} failed; {target_dir} is not a directory."
                )

            if not path.isdir(current_dir):
                print(
                    f" -- {cat} did not have any previously generated targets; direct copy."
                )
                shutil.move(target_dir, current_dir)
                continue

            if autogen_directories_match(current_dir, target_dir):
                print(f" -- autogen targets for {cat} are unchanged; skipping...")
                continue

            print(
                f" -- autogen targets for {cat} are different, replacing with new ones."
            )
            shutil.rmtree(current_dir)
            shutil.move(target_dir, current_dir)


class BuildExtension(build_ext):
    def build_extension(self, ext):
        if BUILD_WITH_CUDA:

            print("Preparing to build LIBNATTEN")
            this_dir = path.dirname(path.abspath(__file__))
            cmake_lists_dir = path.join(this_dir, "csrc")
            autogen_dir = path.join(cmake_lists_dir, "autogen")
            scripts_dir = path.join(this_dir, "scripts")

            # Hack so that we can build somewhere other than /tmp in development mode.
            # Also because we want CMake to build everything elsewhere, otherwise pypi will package
            # build files.
            build_dir = self.build_lib if NATTEN_BUILD_DIR is None else NATTEN_BUILD_DIR

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

            # Auto-gen kernel instantiations
            print("Auto-generating kernel instantiations")
            autogen_kernel_instantitations(
                this_dir=this_dir,
                autogen_dir=autogen_dir,
                scripts_dir=scripts_dir,
                policy=AUTOGEN_POLICY,
                cuda_arch_list=cuda_arch_list,
            )

            # Set up cmake
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

            if 100 in cuda_arch_list or 103 in cuda_arch_list:
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
    ext_modules=[Extension("natten.libnatten", [])] if BUILD_WITH_CUDA else [],
    cmdclass={"build_ext": BuildExtension} if BUILD_WITH_CUDA else {},
)
