#!/bin/bash -e
# Based on detectron2's builder:
# github.com/facebookresearch/detectron2

[[ -d "dev/packaging" ]] || {
  echo "Please run this script at natten root!"
  exit 1
}

build_one() {
  cu=$1
  pytorch_ver=$2

  case "$cu" in
    cu*)
      image_name=manylinux-cuda${cu/cu/}
      ;;
    cpu)
      image_name=manylinux-cpu
      ;;
    *)
      echo "Unrecognized cu=$cu"
      exit 1
      ;;
  esac

  echo "Launching container with $image_name ..."
  container_name="$image_name"_"$cu"_"$pytorch_ver"

  py_versions=(3.8 3.9 3.10 3.11)

  for py in "${py_versions[@]}"; do
    container_name_="${container_name}_${py}"

    docker run -itd --rm \
      --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      --name "$container_name_" \
      --mount type=bind,source="$(pwd)",target=/natten \
      pytorch/$image_name
    cat <<EOF | docker exec -i $container_name_ sh
      export CU_VERSION=$cu PYTHON_VERSION=$py
      export PYTORCH_VERSION=$pytorch_ver
      export OUTPUT_DIR=/natten
      cp -r /natten /natten-build
      cd /natten-build && 
      make clean &&
      ./dev/packaging/build_wheel.sh
EOF
    docker container stop $container_name_
  done
}


if [[ -n "$1" ]] && [[ -n "$2" ]]; then
  build_one "$1" "$2"
else
  # We don't need to build for every minor torch release; they're usually
  # compatible in their python API and ABIs.

  build_one cu121 2.1.0 & build_one cu118 2.1.0

  build_one cu118 2.0.0 & build_one cu117 2.0.0 

  build_one cpu 2.1.0 & build_one cpu 2.0.0
fi
