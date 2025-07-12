# Install NATTEN

Newest release: `0.21.0`

Starting version `0.20.0`, NATTEN only supports PyTorch 2.7 and newer.
However, you can still attempt to install NATTEN with PyTorch >= 2.5 at your own risk.
For earlier NATTEN releases, please refer to [Older NATTEN builds](#older-natten-builds).

We strongly recommend NVIDIA GPU users to install NATTEN with our CUDA kernel library, `libnatten`.
`libnatten` packs our FNA kernels built with [CUTLASS](https://github.com/NVIDIA/cutlass/), which
are specifically designed for delivering all the functionality available in NATTEN, and the best
possible performance.

If you're a Linux user (or WSL), and you have NVIDIA GPUs, refer to
[NATTEN + `libnatten` via PyPI](#natten-libnatten-via-pypi) for instructions on installing NATTEN
with wheels (fastest option).

If you're using a PyTorch version we don't support, or you need to use NATTEN with
non-official PyTorch builds (nightly builds, NGC images, etc), you have to
[build NATTEN + `libnatten`](#build-natten-libnatten).

If you're a Windows user with NVIDIA GPUs, you can try to
[build NATTEN from source with MSVC](#build-with-msvc). However, we note that we are unable to
regularly test and support Windows builds. Pull requests and patches are strongly encouraged.

All non-NVIDIA GPU users can only use our PyTorch (Flex Attention) backend, and not `libnatten`.
Refer to [NATTEN via PyPI](#natten-via-pypi) for more information.


## NATTEN + `libnatten` via PyPI
We offer pre-built wheels (binaries) for the most recent official PyTorch builds.
To install NATTEN using wheels, please first check your PyTorch version, and select it below.

???+ pip-install "`torch==2.7.0+cu128`"

    ```python
    pip install natten==0.21.0+torch270cu128 -f https://whl.natten.org
    ```

??? pip-install "`torch==2.7.0+cu126`"
    ```python
    pip install natten==0.21.0+torch270cu126 -f https://whl.natten.org
    ```

    !!! warning 
        Blackwell FNA/FMHA kernels are not available in this build. Blackwell support was
        introduced in CUDA Toolkit 12.8.

??? question "Don't know your PyTorch build? Other questions?"
    If you don't know your PyTorch build, simply run this command:

    ```shell
    python -c "import torch; print(torch.__version__)"
    ```
    <div class="result" markdown>

        2.7.0+cu126

    </div>

    In the above example, we're seeing PyTorch 2.7.0, built with CUDA Toolkit 12.6.

    If you see a different version tag pattern, you might be using a nightly build, or your
    environment/container built PyTorch from source.
    If you see an older PyTorch version, sadly we don't support it and therefore don't
    ship wheels for it.

    In either case, your only option is to
    [build NATTEN + `libnatten`](#build-natten-libnatten) locally.

    If you see a CUDA Toolkit build older than 12.0, we unfortunately don't support those moving
    forward.

    If you see something else, or have further questions, feel free to
    [open an issue](https://github.com/SHI-Labs/NATTEN/issues).

## NATTEN via PyPI

If you can't install NATTEN with `libnatten`, you can still use our
[Flex Attention](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html#module-torch.nn.attention.flex_attention)
backend.
However, note that you can only use this backend as long as Flex Attention is supported on your
device.
Also note that Flex Attention with `torch.compile` is not allowed by default for safety.
For more information refer to
[Flex Attention + `torch.compile`](context.md#flex-attention-torchcompile).

???+ pip-install "`torch>=2.7.0`"

    ```python
    pip install natten==0.21.0
    ```

## Build NATTEN + `libnatten`

First make sure that you have your preferred PyTorch build installed, since NATTEN's setup
script requires PyTorch to even run.

??? question "Dependencies & Requirements"
    Obviously, Python and [PyTorch](https://pytorch.org/).

    Builds with `libnatten` require `cmake` (1), CUDA runtime (2), and CUDA Toolkit (3).
    { .annotate }
    
    1. Ideally install system-wide, but you can also install via PyPI:

        ```shell
        pip install cmake==3.20.3
        ```

    2. Usually ships with the CUDA driver. As long as you can run SMI successfully, you should be
        good:

        ```shell
        nvidia-smi
        ```
    
    3. Mainly for the CUDA compiler. Confirm you have it by running:

        ```shell
        which nvcc
        ```
        <div class="result" markdown>

            /usr/bin/nvcc

        </div>

        to make sure it exists, and then confirming the CUDA toolkit version reported is 12.0 or
        higher:

        ```
        nvcc --version
        ```
        <div class="result" markdown>

            nvcc: NVIDIA (R) Cuda compiler driver
            Copyright (c) 2005-2023 NVIDIA Corporation
            Built on Tue_Aug_15_22:02:13_PDT_2023
            Cuda compilation tools, release 12.2, V12.2.140
            Build cuda_12.2.r12.2/compiler.33191640_0

        </div>

    `libnatten` depends heavily on [CUTLASS](https://github.com/NVIDIA/cutlass/), which is a header
    only open source library that gets fetched as a submodule, so you don't need to install anything
    for that.


To build NATTEN with `libnatten`, you can still use PyPI, or [build from source](#build-from-source).

???+ pip-install "Build NATTEN with `libnatten` on CUDA-supported devices"
    ```python
    pip install natten==0.21.0
    ```

By default, NATTEN will detect your GPU architecture and build `libnatten` specifically for that
architecture.
However, note that this **will not work** in container image builders (i.e. Dockerfile), as they do
not typically expose the CUDA driver.
If the setup does not detect a CUDA device, it will **skip building `libnatten`**.

If you want to specify your target GPU architecture(s) manually, which will allow building without
the CUDA driver present (still requires the compiler in CUDA Toolkit), set the `NATTEN_CUDA_ARCH`
environment variable to a semicolon-separated list of the compute capabilities corresponding to
your desired architectures.

```python
NATTEN_CUDA_ARCH="8.9" pip install natten==0.21.0 # (1)!

NATTEN_CUDA_ARCH="9.0" pip install natten==0.21.0 # (2)!

NATTEN_CUDA_ARCH="10.0" pip install natten==0.21.0 # (3)!

NATTEN_CUDA_ARCH="8.0;8.6;9.0;10.0" pip install natten==0.21.0 # (4)!
```

1. Build targeting SM89 (Ada Lovelace)
2. Build targeting SM90 (Hopper)
3. Build targeting SM100 (Blackwell)
4. Multi-arch build targeting SM80, SM86, SM90 and SM100

By default the setup will attempt to use 1/4th of the number of processor threads available in your
system. If possible, we recommend using more to maximize build parallelism, and minimize build time.
Using more than 64 workers (if available) should generally not make a difference, since we usually
generate around 60 build targets.

You can customize the number of workers by setting the `NATTEN_N_WORKERS` environment variable:

```python
NATTEN_N_WORKERS=16 pip install natten==0.21.0 # (1)!

NATTEN_N_WORKERS=64 pip install natten==0.21.0 # (2)!
```

1. Build with 16 parallel workers
2. Build with 64 parallel workers

You can also use the `--verbose` option in `pip` to monitor the compilation, and set the
`NATTEN_VERBOSE` environment variable to toggle `cmake`'s verbose mode, which prints out all
compilation targets and compile flags.

### Build from source
Just clone, optionally checkout to your desired version or commit, and run `make`:

```bash
git clone --recursive https://github.com/SHI-Labs/NATTEN
cd NATTEN

make
```

The setup script is identical to the one we ship [with PyPI](#build-natten-libnatten), therefore
you have the same options and the same behavior.

If you want to specify your target GPU architecture(s) manually, instead of letting them get
detected via PyTorch talking to the CUDA driver:

```python
make CUDA_ARCH="8.9" # (1)!

make CUDA_ARCH="9.0" # (2)!

make CUDA_ARCH="10.0" # (3)!

make CUDA_ARCH="8.0;8.6;9.0;10.0" # (4)!
```

1. Build targeting SM89 (Ada Lovelace)
2. Build targeting SM90 (Hopper)
3. Build targeting SM100 (Blackwell)
4. Multi-arch build targeting SM80, SM86, SM90 and SM100

Customizing the number of workers again works similarly.
Using more than 64 workers should generally not make a difference, unless your use case demands it,
and you use our autogen scripts (under `scripts/autogen*.py` in
[our repository](https://github.com/SHI-Labs/NATTEN/tree/main/scripts)) and increase the number of
targets.

```python
make WORKERS=16 # (1)!

make WORKERS=64 # (2)!
```

1. Build with 16 parallel workers
2. Build with 64 parallel workers

You can enable verbose mode to view more details, and compilation progress:

```python
make VERBOSE=1
```

It is recommended to run all unit tests when you build from source. It may take up to 30 minutes:

```python
make test
```

If you want to do an editable (development) install, use `make dev` instead of `make`.

??? question "Build from source on Windows"
    ### Build with MSVC
    **NOTE: Windows builds are experimental and not regularly tested.**


    First clone NATTEN, and make sure to fetch all submodules. If you're cloning with Visual Studio,
    it might clone submodules by default. If you're using a command line tool like git bash (MinGW) or
    WSL, it should be the same as linux.

    To build with MSVC, please open the "Native Tools Command Prompt for Visual Studio".
    The exact name may depend on your version of Windows, Visual Studio, and cpu architecture (in our
    case it was "x64 Native Tools Command Prompt for VS".)

    Once in the command prompt, make sure your correct Python environment is in the system path. If
    you're using anaconda, you should be able to do `conda activate $YOUR_ENV_NAME`.

    Then simply confirm you have PyTorch installed, and use our Windows batch script to build:

    ```python
    WindowsBuilder.bat install

    WindowsBuilder.bat install WORKERS=8 # (1)!

    WindowsBuilder.bat install CUDA_ARCH=8.9 # (2)!
    ```

    1. Build with 8 parallel workers
    2. Build targeting SM89 (Ada)

    Note that depending on how many workers you end up using, build time may vary, and the MSVC
    compiler tends to throw plenty of warnings at you, but as long as it does not fail and give you
    back the command prompt, just let it keep building.

    Once it's done building, it is highly recommended to run the unit tests to make sure everything
    went as expected:
    ```
    WindowsBuilder.bat test
    ```

    ??? warning "PyTorch issue: nvToolsExt not found"

        Windows users may come across this issue when building NATTEN from source with CUDA 12.0
        and newer. The build process fails with an error indicating "nvtoolsext" cannot be found on
        your system.  This is because nvtoolsext binaries are no longer part of the CUDA toolkit
        for Windows starting CUDA 12.0, but the PyTorch cmake still looks for it
        (as of torch==2.2.1).

        The only workaround is to modify the following files:
        ```
        $PATH_TO_YOUR_PYTHON_ENV\Lib\site-packages\torch\share\cmake\Torch\TorchConfig.cmake
        $PATH_TO_YOUR_PYTHON_ENV\Lib\site-packages\torch\share\cmake\Caffe2\Caffe2Targets.cmake
        $PATH_TO_YOUR_PYTHON_ENV\Lib\site-packages\torch\share\cmake\Caffe2\public\cuda.cmake
        ```

        find all mentions of nvtoolsext (or nvToolsExt), and comment them out, to get past it.

        More information on the issue:
        [pytorch/pytorch#116926](https://github.com/pytorch/pytorch/pull/116926)

## Checking whether you have `libnatten`

If you installed NATTEN, and want to check whether you have `libnatten`, or your device supports it,
run the following:

```shell
python -c "import natten; print(natten.HAS_LIBNATTEN)"
```

If `True`, you're good to go. If `False`, it could mean either you didn't install NATTEN with
`libnatten`, or that your current device doesn't support it. Feel free to
[open an issue](https://github.com/SHI-Labs/NATTEN/issues) if you have questions.

## Older NATTEN builds
We highly recommend using the latest NATTEN builds, but if you need to install older NATTEN
versions, you can install them via [PyPI](https://pypi.org/project/natten/).
We only offer wheels for >=`0.20.0` and `0.17.5` releases. Earlier releases will have to be
[compiled locally](#natten-via-pypi).

#### `0.21.1`
Released on 2025-06-14.
[Changelog](https://github.com/SHI-Labs/NATTEN/blob/main/CHANGELOG.md#0201---2025-06-14).

??? pip-install "`torch==2.7.0+cu128`"

    ```python
    pip install natten==0.20.1+torch270cu128 -f https://whl.natten.org
    ```

??? pip-install "`torch==2.7.0+cu126`"
    ```python
    pip install natten==0.20.1+torch270cu126 -f https://whl.natten.org
    ```

    !!! warning 
        Blackwell FNA/FMHA kernels are not available in this build. Blackwell support was
        introduced in CUDA Toolkit 12.8.

??? pip-install "Compile locally (custom torch build)"

    ```python
    pip install natten==0.20.1
    ```

    Refer to [NATTEN via PyPI](#natten-via-pypi) for more details.


#### `0.20.0`
Released on 2025-06-07.
[Changelog](https://github.com/SHI-Labs/NATTEN/blob/main/CHANGELOG.md#0200---2025-06-07).

??? pip-install "`torch==2.7.0+cu128`"

    ```python
    pip install natten==0.20.0+torch270cu128 -f https://whl.natten.org
    ```

??? pip-install "`torch==2.7.0+cu126`"
    ```python
    pip install natten==0.20.0+torch270cu126 -f https://whl.natten.org
    ```

    !!! warning 
        Blackwell FNA/FMHA kernels are not available in this build. Blackwell support was
        introduced in CUDA Toolkit 12.8.

??? pip-install "Compile locally (custom torch build)"

    ```python
    pip install natten==0.20.0
    ```

    Refer to [NATTEN via PyPI](#natten-via-pypi) for more details.


#### `0.17.5`
Released on 2025-03-20.
[Changelog](https://github.com/SHI-Labs/NATTEN/blob/main/CHANGELOG.md#0175---2025-03-20).

!!! warning 
    Wheels will be phased out and removed in the coming months. We strongly recommend upgrading to
    NATTEN `0.21.0`.

??? pip-install "`torch==2.6.0+cu126`"

    ```python
    pip install natten==0.17.5+torch260cu126 -f https://whl.natten.org
    ```

??? pip-install "`torch==2.6.0+cu124`"

    ```python
    pip install natten==0.17.5+torch260cu124 -f https://whl.natten.org
    ```

??? pip-install "`torch==2.6.0+cpu`"

    ```python
    pip install natten==0.17.5+torch260cpu -f https://whl.natten.org
    ```

??? pip-install "`torch==2.5.0+cu124`"

    ```python
    pip install natten==0.17.5+torch260cu124 -f https://whl.natten.org
    ```

??? pip-install "`torch==2.5.0+cu121`"

    ```python
    pip install natten==0.17.5+torch260cu121 -f https://whl.natten.org
    ```

??? pip-install "`torch==2.5.0+cpu`"

    ```python
    pip install natten==0.17.5+torch260cpu -f https://whl.natten.org
    ```

??? pip-install "Compile locally (custom torch build)"

    ```python
    pip install natten==0.17.5
    ```

    Refer to [NATTEN via PyPI](#natten-via-pypi) for more details.

#### < `0.17.5`
Refer to our release index on [PyPI](https://pypi.org/project/natten/#history).
