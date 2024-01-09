#!/bin/bash -e
# Based on detectron2's builder:
# github.com/facebookresearch/detectron2

# Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}
# Install with pip a bit more robustly than the default
pip_install() {
  retry pip install --progress-bar off "$@"
}


setup_cuda() {
  # Like other torch domain libraries, we choose common GPU architectures only.
  # See https://github.com/pytorch/pytorch/blob/master/torch/utils/cpp_extension.py
  # and https://github.com/pytorch/vision/blob/main/packaging/pkg_helpers.bash for reference.
  export FORCE_CUDA=1
  case "$CU_VERSION" in
    cu121)
      cuda_path=/usr/local/cuda-12.1/
      export PATH=${cuda_path}/bin${PATH:+:${PATH}}
      export LD_LIBRARY_PATH=${cuda_path}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
      export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
      ;;
    cu118)
      cuda_path=/usr/local/cuda-11.8/
      export PATH=${cuda_path}/bin${PATH:+:${PATH}}
      export LD_LIBRARY_PATH=${cuda_path}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
      export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
      ;;
    cu117)
      cuda_path=/usr/local/cuda-11.7/
      export PATH=${cuda_path}/bin${PATH:+:${PATH}}
      export LD_LIBRARY_PATH=${cuda_path}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
      export TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6"
      ;;
    cpu)
      unset FORCE_CUDA
      export CUDA_VISIBLE_DEVICES=
      ;;
    *)
      echo "Unrecognized CU_VERSION=$CU_VERSION"
      exit 1
      ;;
  esac
}

setup_wheel_python() {
  case "$PYTHON_VERSION" in
    3.7) python_abi=cp37-cp37m ;;
    3.8) python_abi=cp38-cp38 ;;
    3.9) python_abi=cp39-cp39 ;;
    3.10) python_abi=cp310-cp310 ;;
    3.11) python_abi=cp311-cp311 ;;
    *)
      echo "Unrecognized PYTHON_VERSION=$PYTHON_VERSION"
      exit 1
      ;;
  esac
  export PATH="/opt/python/$python_abi/bin:$PATH"
}
