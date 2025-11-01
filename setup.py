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
import glob
import multiprocessing
import os
import shutil
import subprocess
import sys
from os import path
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, List

import torch
from setuptools import Extension, setup  # type: ignore
from setuptools.command.build_ext import build_ext  # type: ignore
from torch.utils.cpp_extension import LIB_EXT

################################################################################
################################## Constants  ##################################
################################################################################
MIN_TORCH_VERSION : float = 2.6
MIN_CUDA_VERSION : float = 12.0
MIN_SM : int = 30
SUPPORTED_GPU_ARCH : list[int] = [90, 100, 103]
# Key is the sm number and the value is the associated name.
CAT2ARCH : dict = {
    90: "hopper",
    100: "blackwell",
    103: "blackwell",
}
# If we're going to warn Windows users if they have >= MIN_CUDA_VERSION, then
#   let's just warn Windows users (marked with TODO)
WINDOWS_EXPERIMENTAL : bool = True
#MIN_WINDOWS_CUDA_VERSION : float = 12.0
################################################################################
DEFAULT_N_WORKERS : int = max(1, (multiprocessing.cpu_count() // 4))

# Temp building dir. We could clean this up with a context manager.
# Not called until BuildExtension
tmp_dir = TemporaryDirectory()
env = {
    "AUTOGEN_POLICY": os.getenv("NATTEN_AUTOGEN_POLICY", "default"),
    "BUILD_DIR": os.getenv("NATTEN_BUILD_DIR", tmp_dir.name),
    "CMAKE_CUDA_FLAGS": os.getenv("CMAKE_CUDA_FLAGS", None),
    "CUDA_ARCH": os.getenv("NATTEN_CUDA_ARCH", None),
    "DISABLE_LIBNATTEN": os.environ.__contains__("DISABLE_LIBNATTEN"),
    "HAS_CUDA_ARCH": False,
    "IS_LIBTORCH_BUILT_WITH_CXX11_ABI": torch._C._GLIBCXX_USE_CXX11_ABI,
    "NATTEN_IS_BUILDING_DIST": int(os.getenv("NATTEN_IS_BUILDING_DIST", 0)),
    "N_WORKERS": int(os.getenv("NATTEN_N_WORKERS", DEFAULT_N_WORKERS)),
    "LONG_DESC": None,
    "CWD": Path(__file__).parent,
    "OS_TYPE": sys.platform,
    "TORCH_VER": float(".".join(torch.__version__.split(".")[:2])),
    "TORCH_HAS_CUDA": torch.cuda.is_available(),
    "TORCH_CUDA_TAG": None,
    "VERBOSE": int(os.getenv("NATTEN_VERBOSE", 0)),
}

def verify_env(env : dict) -> dict:
    if env['AUTOGEN_POLICY'] == "":
        env['AUTOGEN_POLICY'] = "default"

    if not os.path.isdir(env['BUILD_DIR']):
        os.makedirs(env['BUILD_DIR'])

    try:
        env['LONG_DESC'] = env['CWD'].joinpath("docs/README_pypi.md").read_text()
    except:
        env['LONG_DESC'] = "Neighborhood Attention Extension."

    #if env['OS_TYPE'] == "Windows" and _CUDA_VERSION >= MIN_WINDOWS_CUDA_VERSION:
    if env['OS_TYPE'] == 'win32' and WINDOWS_EXPERIMENTAL:
        #TODO: If becomes version specific then change print statement
        print(
            "WARNING: NATTEN builds for Windows are currently experimental. "
            "Please refer to the NATTEN documentation at "
            "https://github.com/SHI-Labs/NATTEN/blob/main/docs/install.md "
            "for more details."
        )

    if env['TORCH_VER'] < MIN_TORCH_VERSION:
        raise ValueError(
            f"NATTEN only supports PyTorch >= {MIN_TORCH_VERSION}, "
            f"detected {env['TORCH_VER']}"
        )

    if env['TORCH_HAS_CUDA']:
        env['TORCH_CUDA_TAG'] = torch.__version__.split("+")[1]

    return env

env = verify_env(env)

################################################################################
############################### Autogen Policies ###############################
################################################################################
# Let's increase flexibility and readability here and not hard code.
#
# Current policies are just scales of a subset of keys in the default policy.
# Writing this way we'll make clear which keys are being tuned as well as make
# this easier to provide different scaling factors.
# We may wish to extend _tune_ag_policy to incorporate excluding specific keys.
# Worst case we can hard code the union.
# Note: Union operator means last key wins.
def _tune_ag_policy(policy: dict, scale : float) -> dict:
    for key in policy:
        policy[key] = int(policy[key] * scale)
    return policy

_AG_POLICY_TUNABLES = {
    "reference": 2,
    "fna": 64,
    "fmha": 6,
    "hopper-fna": 8,
    "hopper-fna-bwd": 4,
    "blackwell-fna": 28,
    "blackwell-fna-bwd": 14,
}

_AG_POLICIES_CONSTS = {
    "hopper-fmha": 1,
    "hopper-fmha-bwd": 1,
    "blackwell-fmha": 1,
    "blackwell-fmha-bwd": 1,
}

AG_POLICY_DEFAULT = _AG_POLICIES_CONSTS | _AG_POLICY_TUNABLES

# Now this is more explicit
AG_POLICY_FINE = AG_POLICY_DEFAULT | _tune_ag_policy(_AG_POLICY_TUNABLES, 2)
AG_POLICY_COARSE = AG_POLICY_DEFAULT | _tune_ag_policy(_AG_POLICY_TUNABLES, 0.5)

AG_POLICIES = {
    "default": AG_POLICY_DEFAULT,
    "fine": AG_POLICY_FINE,
    "coarse": AG_POLICY_COARSE,
}
# Apply similar flexibility to category logic.
# Will make adaptation to new architectures easier.

AG_CATEGORIES = {
    "reference": ("autogen_reference_fna.py", "reference"),
    "fna": ("autogen_fna.py", "fna"),
    "fmha": ("autogen_fmha.py", "fmha"),
}

def _category_generator(arch: str) -> dict:
    _arch_dict = {}
    for x in ["fna", "fmha"]:
        _arch_dict |= {
            f"{arch}-{x}": (f"autogen_{arch}_{x}.py", f"{arch}_{x}"),
            f"{arch}-{x}-bwd": (f"autogen_{arch}_{x}_bwd.py", f"{arch}_{x}_bwd"),
        }
    return _arch_dict



##################
# Helper functions
##################
def _get_torch_cuda_version() -> float:
    _version = None
    try:
        _version = float(".".join(torch.version.cuda.split(".")[:2]))
    except exception as e:
        raise ValueError from e(
            "Could not get valid CUDA version from PyTorch "
            f"({torch.version.cuda=}"
        )
    return _version

def autodetect_cuda() -> str:
    def check_submodules() -> None:
        if len(os.listdir("third_party/cutlass/")) == 0:
            raise OSError(
                "The directory 'third_party/cutlass' is empty but we have "
                "detected a PyTorch build with CUDA."
                "This likely means you did not build the submodules correctly. "
                "Please run `git submodule update --init --recursive` and try "
                "again. "
                "If you wish to continue without CUDA support then please set "
                "environment variable DISABLE_LIBNATTEN before trying again."
            )

    _device = torch.cuda.get_device_properties(torch.cuda.current_device())
    _CUDA_ARCH = _device.major + 0.1 * _device.minor
    print(
        "Environment variable NATTEN_CUDA_ARCH unset but PyTorch version is "
        f"CUDA capable. Attempting to build with {_CUDA_ARCH}"
    )
    # Fail if user doesn't have submodules
    check_submodules()
    # Check for major and minor versions only
    #_TORCH_CUDA_VERSION = float(".".join(torch.version.cuda.split(".")[:2]))
    _TORCH_CUDA_VERSION = _get_torch_cuda_version()
    if _TORCH_CUDA_VERSION < MIN_CUDA_VERSION:
        raise ValueError(
            f"NATTEN only supports CUDA {MIN_CUDA_VERSION} and above "
            f"(you have {_TORCH_CUDA_VERSION})"
        )
    print(f"PyTorch built with CUDA Toolkit {torch.version.cuda}")
    # Detection via torch doesn't allow for multiple architectures
    print(f"Building NATTEN for the following architecture: {_CUDA_ARCH}")

    return str(_CUDA_ARCH)


def set_natten_version(
        is_building_dist: bool,
        torch_has_cuda: bool,
        torch_version : float,
        torch_cu_tag: str,
        cwd: os.Path,
    ) -> str:

    version_file = path.join(
        cwd, "src/natten/version.py"
    )

    with open(version_file, 'r') as _vfile:
        for line in _vfile:
            if line.startswith("__version__"):
                _version = line.split("=")[1].strip().strip('"')
                break

    if is_building_dist:
        if not torch_has_cuda:
            raise ValueError(
                "Cannot build libnatten without a valid CUDA version within "
                "PyTorch."
            )
        _version = f"{_version}+torch{torch_version}{torch_cu_tag}"

    return _version


def _check_cuda_arch(arch: str|int|float) -> int:
    arch = float(arch)
    arch = arch * 10
    if arch < MIN_SM:
        raise ValueError(
            f"Only SM{MIN_SM} and above are currently supported. Detected SM{arch}"
        )
    return int(arch)


def get_cuda_arch_list(cuda_arch_list : str,
    ) -> List[int]:

    delimiter = ";"
    # Assume one value
    if delimiter not in cuda_arch_list:
        return[_check_cuda_arch(cuda_arch_list)]
    arch_list = cuda_arch_list.split(delimiter)
    for i, arch in enumerate(arch_list):
        arch_list[i] = _check_cuda_arch(arch)
    return arch_list

def _arch_list_to_cmake_tags(arch_list: List[int]) -> str:
    return (
        "-real;".join(
            [str(x) if x not in SUPPORTED_GPU_ARCH else f"{x}a" for x in arch_list]
        ) + "-real"
    )

def autogen_directories_match(dir1 : str, dir2: str) -> bool:
    """
    Compare two directories recursively, ignoring hidden files.
    Only compares files with .cu, .cpp, .h, .hpp, .cuh extensions.

    Used for skipping redundant autogen writes that can mislead cmake into recompiling things.
    """
    extns = (".cpp", ".h", ".hpp", ".cu", ".cuh")
    # Get files with extension but remove leading directory name
    def strip_glob(_dir, extn):
        return [str(Path(p).relative_to(_dir))
                for p in glob.glob(f"{_dir}/**/*.{extn}",
                                   recursive=True
                          )
                ]
    # Iterate over all extensions
    for extn in extns:
        # Find files in dir1 and dir2 with extensions
        d1_files = strip_glob(dir1, extn)
        d2_files = strip_glob(dir2, extn)
        # If the list of files are different then we'll rebuild
        if d1_files != d2_files:
            return False
        # If list of files are the same, check if they are identical
        for f1 in d1_files:
            if not filecmp.cmp(f"{dir1}/{f1}", f"{dir2}/{f1}"):
                return False
    # Directories are identical
    return True

def autogen_kernel_instantiations(
        this_dir: str,
        autogen_dir : str,
        script_dir : str,
        policy : str,
        cuda_arch_list : List[int],
        _categories : dict = AG_CATEGORIES,
    ) -> None:

    if policy not in AG_POLICIES.keys():
        raise ValueError(
            "The requested autogen policy cannot be found. "
            f"Got '{policy}' but expected one of {list(AG_POLICIES.keys())}"
        )

    if not os.path.isdir(script_dir):
        raise NotADirectoryError(
            f"Script directory {script_dir} does not exist or is not a directory."
        )

    _split_policies = AG_POLICIES[policy]
    # Generate the categories for each of the architectures
    for arch in cuda_arch_list:
        if arch in CAT2ARCH.keys():
            # Update dict by union. Won't create dupes
            _categories |= _category_generator(CAT2ARCH[arch])

    # Build into a temporary directory then we'll move over
    with TemporaryDirectory() as _autogen_dir:
        _tmp_dir = path.join(_autogen_dir, "csrc")

        for cat, (script, out_dir) in _categories.items():
            if cat not in _split_policies:
                raise ValueError(
                    f"Category {cat} is not an allowed category"
                )
            script_path = path.join(script_dir, script)
            if not path.isfile(script_path):
                raise FileNotFoundError(
                    f"Expected to find autogen script {script} under "
                    f"{script_dir}, but {script_path} does not exist. "
                    "Please raise an issue if you didn't change anything."
                )

            subprocess.check_call(
                [
                    "python",
                    script_path,
                    "--num-splits",
                    str(_split_policies[cat]),
                    "-o",
                    _tmp_dir,
                ],
                #cwd=path.dirname(path.abspath(__file__)),
                cwd=this_dir,
            )

            # Temporary directories
            tgt_inc_dir = path.join(
                _tmp_dir, "autogen/include/natten_autogen/cuda", out_dir
            )
            tgt_src_dir = path.join(
                _tmp_dir, "autogen/src/cuda", out_dir
            )

            # Final destination
            cur_src_dir = path.join(
                autogen_dir, "src/cuda", out_dir
            )
            cur_inc_dir = path.join(
                autogen_dir, "include/natten_autogen/cuda", out_dir
            )

            for cur, tgt in zip([cur_inc_dir, cur_src_dir], [tgt_inc_dir, tgt_src_dir]):
                if not path.isdir(tgt):
                    raise NotADirectoryError(
                        f"Autogen for {cat} failed: {tgt} is not a directory"
                    )

                if not path.isdir(cur):
                    print(
                        f" -- {cat} did not have previously generated targets; "
                        "directly copying"
                    )
                    shutil.move(tgt, cur)
                    continue

                if autogen_directories_match(cur, tgt):
                    print(
                        f"\n\t\t -- autogen targets for {cat} are unchanged; "
                        "skipping..."
                    )
                    continue

                print(
                    f" -- autogen targets for {cat} are different; "
                    "replacing with new ones."
                )

                shutil.rmtree(cur)
                shutil.move(tgt, cur)

# Try to autodetect CUDA if user hasn't specified.
_autodetect_cuda = \
        env['CUDA_ARCH'] is None \
        and env['TORCH_HAS_CUDA'] \
        and not env['DISABLE_LIBNATTEN']

if _autodetect_cuda:
    env['CUDA_ARCH'] = autodetect_cuda()
else:
    print(
        "Building without CUDA support (libnatten). Either DISABLE_LIBNATTEN "
        f"was set (DISABLE_LIBNATTEN={env['DISABLE_LIBNATTEN']}) or we could not "
        f"detect a PyTorch build with CUDA ({torch.cuda.is_available()=}"
    )
    env['DISABLE_LIBNATTEN'] = True

natten_version = set_natten_version(
    *list(map(env.get, ["NATTEN_IS_BUILDING_DIST",
                        "TORCH_HAS_CUDA",
                        "TORCH_VER",
                        "TORCH_CUDA_TAG",
                        "CWD",
                       ]
             )
         )
    )


class BuildExtension(build_ext):
    def build_extension(self, ext):
        if not env['DISABLE_LIBNATTEN']:
            this_dir = path.dirname(path.abspath(__file__))
            cmake_dir = path.join(this_dir, "csrc")
            autogen_dir = path.join(cmake_dir, "autogen")
            script_dir = path.join(this_dir, "scripts")
            os.makedirs(cmake_dir, exist_ok=True)
            os.makedirs(autogen_dir, exist_ok=True)

            if env['BUILD_DIR'] is not None:
                build_dir = env['BUILD_DIR']
            else:
                build_dir = self.build_lib

            os.makedirs(build_dir, exist_ok=True)

            # Why do these exist?
            os.makedirs(self.build_lib, exist_ok=True)

            cmv = os.system('cmake --version | head -n1')
            if cmv != 0:
                raise RuntimeError(
                    "Could not find a CMake executable. Possibly you do not "
                    "have cmake installed or it is not part of the current "
                    "PATH environment. "
                    f"When checking cmake version we received error:\n{e}"
                )

            # i.e. libnatten.cpython-VERSION-ARCH-OS.so
            #   output_so_name = output_bin_name[:-(len(LIB_EXT))]
            output_bin_name = self.get_ext_filename(ext.name)

            max_sm = 0
            cuda_arch_list = []
            cuda_arch_list_str = ""

            cuda_arch_list = get_cuda_arch_list(env['CUDA_ARCH'])
            cuda_arch_list_str = _arch_list_to_cmake_tags(cuda_arch_list)
            max_sm = max(cuda_arch_list)
            # We shouldn't need to check max version again because this should
            # have been done through get_cuda_arch_list

            # Autogen kernel inistantiations
            autogen_kernel_instantiations(
                this_dir = this_dir,
                autogen_dir = autogen_dir,
                script_dir = script_dir,
                policy = env['AUTOGEN_POLICY'],
                cuda_arch_list = cuda_arch_list,
                _categories = AG_CATEGORIES
            )

            # Cmake
            cmake_args = [
                f"-DPYTHON_PATH={sys.executable}",
                f"-DOUTPUT_FILE_NAME={output_bin_name[:-(len(LIB_EXT))]}",
                f"-DNATTEN_CUDA_ARCH_LIST={cuda_arch_list_str}",
                f"-DNATTEN_IS_WINDOWS={int(env['OS_TYPE'] == 'win32')}",
                f"-DIS_LIBTORCH_BUILT_WITH_CXX11_ABI={int(env['IS_LIBTORCH_BUILT_WITH_CXX11_ABI'])}",
                #f"-DCMAKE_CUDA_FLAGS='-Wno-deprecated-declarations'",
            ]
            if env['CMAKE_CUDA_FLAGS'] is not None:
                cmake_args.append(f"-DCMAKE_CUDA_FLAGS='{env['CMAKE_CUDA_FLAGS']}'")

            # If we have GPU specific optimizations let's enable them
            for arch in cuda_arch_list:
                if arch in CAT2ARCH.keys():
                    print(f"Found {arch=} so -DNATTEN_WITH_{CAT2ARCH[arch].upper()}_FNA=1")
                    cmake_args.append(f"-DNATTEN_WITH_{CAT2ARCH[arch].upper()}_FNA=1")

            if env['OS_TYPE'] == 'win32':
                python_path = sys.executable
                if "python.exe" not in python_path:
                    raise FileNotFoundError(
                        "Expected the Python executable to end with python.ext "
                        f"but got {python_path}"
                    )
                python_lib_dir = python_path.replace("python.exe", "libs").strip()
                cmake_args.append(f"-DPY_LIB_DIR={python_lib_dir}")
                cmake_args.append("-G Ninja")
                cmake_args.append("-DCMAKE_BUILD_TYPE=Release")

            so_dir_local = os.path.join(build_dir,
                                        os.path.dirname(output_bin_name)
            )
            so_path_local = f"{build_dir}/{output_bin_name}"
            os.makedirs(so_dir_local, exist_ok=True)

            so_dir_final = os.path.join(self.build_lib,
                                        os.path.dirname(output_bin_name)
            )
            so_path_final = f"{self.build_lib}/{output_bin_name}"
            os.makedirs(so_dir_final, exist_ok=True)

            # Configure and build extension
            subprocess.check_call(
                ["cmake", cmake_dir] + cmake_args, cwd=build_dir
            )
            cmake_build_args = [
                "--build", build_dir,
                "-j", str(env['N_WORKERS']),
            ]

            if env['VERBOSE']:
                cmake_build_args.append("--verbose")
            subprocess.check_call(["cmake", *cmake_build_args])

            if not os.path.isfile(so_path_local):
                raise FileNotFoundError(
                    f"Expected libnatten binary to be located in {so_path_local}"
                    f" ({os.listdir(path.dirname(so_path_local))=})"
                )

            if build_dir != self.build_lib:
                shutil.copy(so_path_local, so_path_final)

            if not os.path.isfile(so_path_final):
                raise FileNotFoundError(
                    f"Cannot find {so_path_final}"
                )

            # Cleanup cmake when building distribution
            #   Files packaged within wheel
            if env['NATTEN_IS_BUILDING_DIST']:
                for file in os.listdir(build_dir):
                    fn = os.path.join(build_dir, file)
                    if os.path.isfile(fn):
                        os.remove(fn)
                    elif file != "natten":
                        shutil.rmtree(fn)


setup(
    name="natten",
    version=natten_version,
    author="NATTEN Project",
    url="https://natten.org",
    description="Neighborhood Attention Extension.",
    ext_modules=[Extension("natten.libnatten", [])] if not env['DISABLE_LIBNATTEN'] else [],
    cmdclass={"build_ext": BuildExtension} if not env['DISABLE_LIBNATTEN'] else {},
)
