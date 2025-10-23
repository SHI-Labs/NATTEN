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
    cu130)
      cuda_path=/usr/local/cuda-13.0/
      export PATH=${cuda_path}/bin${PATH:+:${PATH}}
      export LD_LIBRARY_PATH=${cuda_path}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
      export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0"
      ;;
    cu129)
      cuda_path=/usr/local/cuda-12.9/
      export PATH=${cuda_path}/bin${PATH:+:${PATH}}
      export LD_LIBRARY_PATH=${cuda_path}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
      export TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0"
      ;;
    cu128)
      cuda_path=/usr/local/cuda-12.8/
      export PATH=${cuda_path}/bin${PATH:+:${PATH}}
      export LD_LIBRARY_PATH=${cuda_path}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
      export TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0"
      ;;
    cu126)
      cuda_path=/usr/local/cuda-12.6/
      export PATH=${cuda_path}/bin${PATH:+:${PATH}}
      export LD_LIBRARY_PATH=${cuda_path}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
      export TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
      ;;
    cu124)
      cuda_path=/usr/local/cuda-12.4/
      export PATH=${cuda_path}/bin${PATH:+:${PATH}}
      export LD_LIBRARY_PATH=${cuda_path}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
      export TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
      ;;
    cu121)
      cuda_path=/usr/local/cuda-12.1/
      export PATH=${cuda_path}/bin${PATH:+:${PATH}}
      export LD_LIBRARY_PATH=${cuda_path}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
      export TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
      ;;
    *)
      echo "Unrecognized CU_VERSION=$CU_VERSION"
      exit 1
      ;;
  esac
}

setup_wheel_python() {
  case "$PYTHON_VERSION" in
    3.9) python_abi=cp39-cp39 ;;
    3.10) python_abi=cp310-cp310 ;;
    3.11) python_abi=cp311-cp311 ;;
    3.12) python_abi=cp312-cp312 ;;
    3.13) python_abi=cp313-cp313 ;;
    3.13t) python_abi=cp313-cp313t ;;
    *)
      echo "Unrecognized PYTHON_VERSION=$PYTHON_VERSION"
      exit 1
      ;;
  esac
  export PATH="/opt/python/$python_abi/bin:$PATH"
}
