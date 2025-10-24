#!/bin/bash

set -ex

ldconfig  # https://github.com/NVIDIA/nvidia-docker/issues/854

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

setup_cuda
setup_wheel_python

export NATTEN_IS_BUILDING_DIST=1
export NATTEN_VERBOSE=1
export NATTEN_CUDA_ARCH=$TORCH_CUDA_ARCH_LIST

# Allow number of workers and autogen policy to be overridden.
if [ -z "${NATTEN_N_WORKERS}" ]; then
  # Up to 3 parallel builds
  export NATTEN_N_WORKERS=$(($(nproc) / 3))
  export NATTEN_N_WORKERS=$((NATTEN_N_WORKERS < 1 ? 1 : NATTEN_N_WORKERS))
  echo "NATTEN_N_WORKERS not set; setting to $NATTEN_N_WORKERS"
fi

if [ -z "${NATTEN_AUTOGEN_POLICY}" ]; then
  # choices are: 'default', 'coarse', 'fine'
  # Warning: 'fine' has a linker error with cu129/cu130 
  export NATTEN_AUTOGEN_POLICY="default"
  echo "NATTEN_AUTOGEN_POLICY not set; setting to $NATTEN_AUTOGEN_POLICY"
fi

echo "Build Settings:\n"
echo "CU_VERSION: $CU_VERSION"          
echo "PYTHON_VERSION: $PYTHON_VERSION"  
echo "PYTORCH_VERSION: $PYTORCH_VERSION"
echo "NATTEN_CUDA_ARCH: $NATTEN_CUDA_ARCH"
echo "NATTEN_N_WORKERS: $NATTEN_N_WORKERS"
echo "NATTEN_AUTOGEN_POLICY: $NATTEN_AUTOGEN_POLICY"

pip_install pip numpy setuptools -U
pip_install \
  -U "torch==${PYTORCH_VERSION}+${CU_VERSION}" \
  -f https://download.pytorch.org/whl/torch/

pip_install "cmake==4.1.0"

# Print torch version and support acrhs
python -c "import torch;\
print(f\"Current torch version: {torch.__version__}\");\
print(f\"Current torch CUDA support (build and runtime): {torch.cuda.is_available()=}\");\
print(f\"Architectures current torch was built for: {torch._C._cuda_getArchFlags()}\");"

echo "Building NATTEN ..."

python -m build \
  --no-isolation \
  --wheel \
  -o "$OUTPUT_DIR/wheels/$CU_VERSION/torch$PYTORCH_VERSION"
