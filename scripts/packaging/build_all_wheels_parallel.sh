#!/bin/bash -e
# Copyright (c) 2022-2025 Ali Hassani.

LOGDIR="wheel-logs/"
mkdir -p $LOGDIR

# Always ensure cutlass/other submods are cloned
# before attempting to build.
git submodule update --init --recursive

[[ -d "scripts/packaging" ]] || {
  echo "Please run this script from natten root!"
  exit 1
}

##################################################
##################################################

build_target () {
  ctk_full=$1                                                  # cuda13.0 (container tag)
  pytorch_ver=$2                                               # 2.9.0 (pytorch version)
  pytorch_ctk_tag=${ctk_full/cuda/cu}                          # cu13.0
  pytorch_ctk_tag=${pytorch_ctk_tag/./}                        # cu130 (pytorch tag)
  ctk_int=${pytorch_ctk_tag/cu/}                               # 130
  torch_major=$(echo $pytorch_ver | awk -F"." '{print $1$2}')  # 29 (pytorch major version)

  if [[ $torch_major -lt 27 ]]; then
    echo "Only torch 2.7 and later are supported from now on."
    exit 1
  fi

  if [[ $ctk_int -lt 126 ]]; then
    echo "Support for CTK < 12.6 builds has been dropped."
    exit 1
  fi

  # Torch started supporting python 3.13 since ~2.5
  # We are building wheels for 3.13 starting 0.21.1
  py_versions=(3.10 3.11 3.12 3.13 3.13t)

  # Torch 2.10 started supporting python 3.14
  if [[ $torch_major -ge 210 ]]; then
    py_versions+=(3.14 3.14t)
  fi

  # Torch 2.9 no longer ships python 3.9 wheels.
  if [[ $torch_major -lt 29 ]]; then
    py_versions+=(3.9)
  fi

  SUPPORTED_ARCHES=("linux/amd64:pytorch/manylinux2_28-builder")

  if [[ $torch_major -gt 28 ]]; then
    SUPPORTED_ARCHES+=("linux/arm64:pytorch/manylinuxaarch64-builder")
  fi

  for ARCH_CONTAINER in "${SUPPORTED_ARCHES[@]}";do
    arr=(${ARCH_CONTAINER//:/ });

    arch_tag="${arr[0]}"                                       # linux/arm64
    arch_tag_f=${arch_tag/\//-}                                # linux-arm64

    image_name="${arr[1]}:${ctk_full}-main"                    # manylinuxaarch64-builder:cuda13.0-main

    for py in "${py_versions[@]}"; do
      container_name="natten_${ctk_full}_${pytorch_ver}_${arch_tag_f}_${py}"

      # On interrupt or exit, kill dangling containers (if any)
      teardown() {
        docker kill $container_name 2>/dev/null || true
      }
      trap teardown INT
      trap teardown EXIT

      echo "Building target $pytorch_ver+$pytorch_ctk_tag, cpython $py, $arch_tag"
      docker run \
        --platform $arch_tag \
        --name "$container_name" \
        --rm \
        --detach \
        --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -e NATTEN_N_WORKERS="$NATTEN_N_WORKERS" \
        -e NATTEN_AUTOGEN_POLICY="$NATTEN_AUTOGEN_POLICY" \
        -e CU_VERSION="$pytorch_ctk_tag" \
        -e PYTHON_VERSION="$py" \
        -e PYTORCH_VERSION="$pytorch_ver" \
        -e OUTPUT_DIR="/natten" \
        --mount type=bind,source="$(pwd)",target=/natten \
        $image_name \
        bash -c "
          cp -r /natten /natten-build
          cd /natten-build && 
          make clean &&
          ./scripts/packaging/build_wheel.sh" >/dev/null

      docker container logs --follow "$container_name" &> $LOGDIR/$container_name.txt
    done
  done
}

##################################################
##################################################

build_target cuda13.0 2.10.0 & \
  build_target cuda12.8 2.10.0 & \
  build_target cuda12.6 2.10.0

build_target cuda13.0 2.9.0 & \
  build_target cuda12.8 2.9.0 & \
  build_target cuda12.6 2.9.0

wait

