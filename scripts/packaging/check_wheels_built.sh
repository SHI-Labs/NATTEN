#!/bin/bash -e

NATTEN_VERSION="0.20.0"
WHEELS_FOUND=0
WHEELS_MISSING=0

[[ -d "scripts/packaging" ]] || {
  echo "Please run this script from natten root!"
  exit 1
}

check_one() {
  cu=$1
  pytorch_ver=$2
  torch_build="torch${pytorch_ver//./}${cu}"

  py_versions=(3.9 3.10 3.11 3.12)

  torch_major=$(echo $pytorch_ver | cut -d "." -f 1,2  --output-delimiter=";")
  torch_major=${torch_major/;/}

  if [[ $torch_major -lt 27 ]]; then
    echo "Only torch 2.7 and later are supported from now on."
    exit 1
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

check_one cu128 2.7.0
check_one cu126 2.7.0
# check_one cu118 2.7.0

WHEELS_TOTAL=$((WHEELS_FOUND+WHEELS_MISSING))

echo ""
echo ""
echo "Wheels found: $WHEELS_FOUND"
echo "Wheels missing: $WHEELS_MISSING"
echo "Wheels total: $WHEELS_TOTAL"
