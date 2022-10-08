#!/bin/bash -e
# Based on detectron2's builder:
# github.com/facebookresearch/detectron2
# Copyright (c) Facebook, Inc. and its affiliates.

[[ -d "dev/packaging" ]] || {
  echo "Please run this script at natten root!"
  exit 1
}

pytorch_ver="1.8"
container_name=manylinux-cuda101
cu="cpu"
py_versions=(3.7 3.8 3.9)

echo "Launching container $container_name ..."
container_id="$container_name"_"$pytorch_ver"

for py in "${py_versions[@]}"; do
    docker run -itd \
      --name "$container_id" \
      --mount type=bind,source="$(pwd)",target=/natten \
      pytorch/$container_name

    cat <<EOF | docker exec -i $container_id sh
      export CU_VERSION=$cu PYTHON_VERSION=$py
      export PYTORCH_VERSION=$pytorch_ver
      cd /natten && ./dev/packaging/build_cpu_wheel.sh
EOF

    docker container stop $container_id
    docker container rm $container_id
done
