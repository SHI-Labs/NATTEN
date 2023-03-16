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
  cp310=${3:-0}

  case "$cu" in
    cu*)
      container_name=manylinux-cuda${cu/cu/}
      ;;
    cpu)
      container_name=manylinux-cpu
      ;;
    *)
      echo "Unrecognized cu=$cu"
      exit 1
      ;;
  esac

  echo "Launching container $container_name ..."
  container_id="$container_name"_"$cu"_"$pytorch_ver"

  if [ $cp310 -eq 3 ]; then
    py_versions=(3.8 3.9 3.10 3.11)
  elif [ $cp310 -eq 2 ]; then
    py_versions=(3.7 3.8 3.9 3.10 3.11)
  elif [ $cp310 -eq 1 ]; then
    py_versions=(3.7 3.8 3.9 3.10)
  else
    py_versions=(3.7 3.8 3.9)
  fi

  for py in "${py_versions[@]}"; do
    docker run -itd \
      --name "$container_id" \
      --mount type=bind,source="$(pwd)",target=/natten \
      pytorch/$container_name

    cat <<EOF | docker exec -i $container_id sh
      export CU_VERSION=$cu PYTHON_VERSION=$py
      export PYTORCH_VERSION=$pytorch_ver
      cd /natten && ./dev/packaging/build_wheel.sh
EOF

    docker container stop $container_id
    docker container rm $container_id
  done
}


if [[ -n "$1" ]] && [[ -n "$2" ]]; then
  build_one "$1" "$2"
else
  # 2.0 and newer -- build 3.8 <= python <= 3.11 wheels
  build_one cu118 2.0.0 3 & build_one cu117 2.0.0 3 &  build_one cpu 2.0.0 3

  # 1.13 and newer -- build python 3.11 wheels
  build_one cu117 1.13 2 & build_one cu116 1.13 2 &  build_one cpu 1.13 2

  # 1.11 and newer -- build python 3.10 wheels
  build_one cu116 1.12.1 1 & build_one cu113 1.12.1 1 &  build_one cu102 1.12.1 1 &  build_one cpu 1.12.1 1

  build_one cu116 1.12 1 & build_one cu113 1.12 1 &  build_one cu102 1.12 1 &  build_one cpu 1.12 1

  build_one cu115 1.11 1 &  build_one cu113 1.11 1 & build_one cu102 1.11 1 & build_one cpu 1.11 1

  # 1.10 and older

  build_one cu113 1.10.1 & build_one cu111 1.10.1 & build_one cu102 1.10.1 & build_one cpu 1.10.1

  build_one cu113 1.10 & build_one cu111 1.10 & build_one cu102 1.10 & build_one cpu 1.10

  build_one cu111 1.9 & build_one cu102 1.9 & build_one cpu 1.9

  build_one cu111 1.8 & build_one cu102 1.8 & build_one cu101 1.8 & build_one cpu 1.8
fi
