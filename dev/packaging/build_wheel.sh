#!/bin/bash
# Based on detectron2's builder:
# github.com/facebookresearch/detectron2
set -ex

ldconfig  # https://github.com/NVIDIA/nvidia-docker/issues/854

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

echo "Build Settings:"
echo "CU_VERSION: $CU_VERSION"                 # e.g. cu101
echo "PYTHON_VERSION: $PYTHON_VERSION"         # e.g. 3.7
echo "PYTORCH_VERSION: $PYTORCH_VERSION"       # e.g. 1.4

setup_cuda
setup_wheel_python
      
export NATTEN_IS_BUILDING_DIST=1
export NATTEN_CUDA_ARCH=$TORCH_CUDA_ARCH_LIST
export NATTEN_N_WORKERS=64
export NATTEN_VERBOSE=0

pip_install pip numpy -U
pip_install -U "torch==$PYTORCH_VERSION" \
  -f https://download.pytorch.org/whl/"$CU_VERSION"/torch_stable.html
pip install cmake==3.20.3

python setup.py \
  bdist_wheel -d "$OUTPUT_DIR/wheels/$CU_VERSION/torch$PYTORCH_VERSION"
