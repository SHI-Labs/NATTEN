#!/bin/bash -e

NATTEN_VERSION="0.17.5"
WHEELS_FOUND=0
WHEELS_MISSING=0

[[ -d "dev/packaging" ]] || {
  echo "Please run this script at natten root!"
  exit 1
}

check_one() {
  cu=$1
  pytorch_ver=$2
  torch_build="torch${pytorch_ver//./}${cu}"

  py_versions=(3.9 3.10 3.11)

  torch_major=$(echo $pytorch_ver | cut -d "." -f 1,2  --output-delimiter=";")
  torch_major=${torch_major/;/}
  if [[ $torch_major -ge 22 ]]; then
    py_versions+=(3.12)
  fi
  # Torch 2.5 dropped support for python 3.8.
  if [[ $torch_major -le 24 ]]; then
    py_versions+=(3.8)
  fi

  for py in "${py_versions[@]}"; do
    python_tag="cp${py//./}-cp${py//./}"
    WHEEL_FILE="wheels/${cu}/torch${pytorch_ver}/natten-${NATTEN_VERSION}+${torch_build}-${python_tag}-linux_x86_64.whl"
    if [ -f $WHEEL_FILE ]; then
      echo "[x] Wheel found for v${NATTEN_VERSION} with ${torch_build} for Python $py."
      WHEELS_FOUND=$((WHEELS_FOUND+1))
    else
      echo "[ ] Wheel MISSING for v${NATTEN_VERSION} with ${torch_build} for Python $py."
      WHEELS_MISSING=$((WHEELS_MISSING+1))
    fi
  done
}

check_one cu126 2.6.0
check_one cu124 2.6.0
check_one cpu 2.6.0

check_one cu124 2.5.0
check_one cu121 2.5.0
check_one cpu 2.5.0

WHEELS_TOTAL=$((WHEELS_FOUND+WHEELS_MISSING))

echo ""
echo ""
echo "Wheels found: $WHEELS_FOUND"
echo "Wheels missing: $WHEELS_MISSING"
echo "Wheels total: $WHEELS_TOTAL"
