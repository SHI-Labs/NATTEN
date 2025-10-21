#!/bin/bash -e
# Based on detectron2's builder:
# github.com/facebookresearch/detectron2

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

  if [[ $ctk_ver -ge 126 ]]; then
    # (ahassani) what in god's name is manylinux2_28?
    image_name=pytorch/manylinux2_28-builder:${target}-main
  else
    image_name=pytorch/manylinux-builder:${target}-main
  fi

  echo "Launching container with $image_name to build for torch $pytorch_ver + $cu..."
  container_name="natten_build_"_"$cu"_"$pytorch_ver"

  # Torch started supporting python 3.13 since ~2.5
  # We are building wheels for 3.13 starting 0.21.1
  py_versions=(3.9 3.10 3.11 3.12 3.13 3.13t)

  # NOTE: I can't surpress the warning from sub
  # when --output-delimiter is "", and I'm not
  # spending more time on this.
  torch_major=$(echo $pytorch_ver | cut -d "." -f 1,2  --output-delimiter="")

  if [[ $torch_major -lt 27 ]]; then
    echo "Only torch 2.7 and later are supported from now on."
    exit 1
  fi

  for py in "${py_versions[@]}"; do
    container_name_="${container_name}_${py}"

    docker run -itd --rm \
      --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      --name "$container_name_" \
      --mount type=bind,source="$(pwd)",target=/natten \
      $image_name
    cat <<EOF | docker exec -i $container_name_ sh
      export CU_VERSION=$cu PYTHON_VERSION=$py
      export PYTORCH_VERSION=$pytorch_ver
      export OUTPUT_DIR=/natten
      cp -r /natten /natten-build
      cd /natten-build && 
      make clean &&
      ./scripts/packaging/build_wheel.sh
EOF
    docker container stop $container_name_
  done
}

build_one_and_capture_output ()
{
  build_one $1 $2 2>&1 > log_${1}_${2}.txt
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
