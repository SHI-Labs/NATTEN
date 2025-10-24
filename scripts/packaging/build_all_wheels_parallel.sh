#!/bin/bash -e

# Always ensure cutlass/other submods are cloned
# before attempting to build.
git submodule update --init --recursive

[[ -d "scripts/packaging" ]] || {
  echo "Please run this script from natten root!"
  exit 1
}

build_one() {
  target=$1
  pytorch_ver=$2
  cu=${target/cuda/cu}
  cu=${cu/./}
  ctk_ver=${cu/cu/}

  SUPPORTED_ARCHES=(
    "linux/amd64:pytorch/manylinux2_28-builder"
    "linux/arm64:pytorch/manylinuxaarch64-builder"
  )

  if [[ $ctk_ver -lt 126 ]]; then
    echo "Support for CTK < 12.6 builds has been dropped."
    exit 1
  fi

  for ARCH_CONTAINER in "${SUPPORTED_ARCHES[@]}";do
    arr=(${ARCH_CONTAINER//:/ });

    arch_tag="${arr[0]}"
    arch_tag_f=${arch_tag/\//-}

    image_name="${arr[1]}:${target}-main"

    echo "Launching container with $image_name to build for torch $pytorch_ver + $cu..."
    container_name="natten_build_"_"$cu"_"$pytorch_ver_${arch_tag_f}"

    # Torch started supporting python 3.13 since ~2.5
    # We are building wheels for 3.13 starting 0.21.1
    py_versions=(3.10 3.11 3.12 3.13 3.13t)

    # NOTE: I can't suppress the warning from sub
    # when --output-delimiter is "", and I'm not
    # spending more time on this.
    torch_major=$(echo $pytorch_ver | cut -d "." -f 1,2  --output-delimiter="")

    if [[ $torch_major -lt 27 ]]; then
      echo "Only torch 2.7 and later are supported from now on."
      exit 1
    fi

    # Torch 2.9 no longer ships python 3.9 wheels.
    if [[ $torch_major -lt 29 ]]; then
      py_versions+=(3.9)
    fi

    for py in "${py_versions[@]}"; do
      container_name="${container_name}_${py}"

        #--gpus=all \
      docker run -itd --rm \
        --platform $arch_tag \
        --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        --name "$container_name" \
        --mount type=bind,source="$(pwd)",target=/natten \
        $image_name
      cat <<EOF | docker exec -i $container_name sh
        export CU_VERSION=$cu PYTHON_VERSION=$py
        export PYTORCH_VERSION=$pytorch_ver
        export OUTPUT_DIR=/natten
        cp -r /natten /natten-build
        cd /natten-build && 
        make clean &&
        ./scripts/packaging/build_wheel.sh
EOF
      docker container stop $container_name
    done
  done
}

build_one_and_capture_output ()
{
  echo "Building for torch $2, $1"
  build_one $1 $2 &> log_${1}_${2}.txt
}


if [[ -n "$1" ]] && [[ -n "$2" ]]; then
  build_one "$1" "$2"
else
  # We don't need to build for every minor torch release; they're usually
  # compatible in their python API and ABIs.

  build_one_and_capture_output cuda13.0 2.9.0 & \
    build_one_and_capture_output cuda12.8 2.9.0 & \
    build_one_and_capture_output cuda12.6 2.9.0

  build_one_and_capture_output cuda12.9 2.8.0 & \
    build_one_and_capture_output cuda12.8 2.8.0 & \
    build_one_and_capture_output cuda12.6 2.8.0

  wait

fi
