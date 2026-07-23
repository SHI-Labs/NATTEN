#!/bin/bash -e
# Copyright (c) 2022 - 2026 Ali Hassani.

NATTEN_VERSION="0.21.7"
WHEELS_FOUND=0
WHEELS_MISSING=0
WHEELS_SKIPPED=0

[[ -d "scripts/packaging" ]] || {
  echo "Please run this script from natten root!"
  exit 1
}

# Wheels that are expected to be missing. A wheel listed here is reported as
# skipped instead of missing. Anything else missing is a real miss.
declare -A ALLOWED_MISSING_WHEELS

check_one() {
  cu=$1
  pytorch_ver=$2
  torch_build="torch${pytorch_ver//./}${cu}"
  torch_major=$(echo $pytorch_ver | awk -F"." '{print $1$2}')  # 29 (pytorch major version)

  # Torch started supporting python 3.13 since ~2.5
  # We are building wheels for 3.13 starting 0.21.1
  # Starting 0.21.7 we no longer ship the free-threaded ("t") python variants.
  py_versions=(3.10 3.11 3.12 3.13)

  if [[ $torch_major -lt 27 ]]; then
    echo "Only torch 2.7 and later are supported from now on."
    exit 1
  fi

  # Torch 2.10 started supporting python 3.14
  if [[ $torch_major -ge 210 ]]; then
    py_versions+=(3.14)
  fi

  # Torch 2.13 started supporting python 3.15
  if [[ $torch_major -ge 213 ]]; then
    py_versions+=(3.15)
  fi

  # Torch 2.9 no longer ships for python 3.9.
  if [[ $torch_major -lt 29 ]]; then
    py_versions+=(3.9)
  fi

  # Torch also started shipping arm builds since 2.8.
  SUPPORTED_ARCHES=("x86_64")

  if [[ $torch_major -gt 28 ]]; then
    SUPPORTED_ARCHES+=("aarch64")
  fi

  for py in "${py_versions[@]}"; do
    pytag_a=${py//./}
    python_tag="cp${pytag_a/t/}-cp${py//./}"
    for arch_tag in "${SUPPORTED_ARCHES[@]}";do
      WHEEL_NAME="natten-${NATTEN_VERSION}+${torch_build}-${python_tag}-linux_${arch_tag}.whl"
      WHEEL_FILE="wheels/${cu}/torch${pytorch_ver}/${WHEEL_NAME}"
      if [ -f $WHEEL_FILE ]; then
        echo "[x] Wheel found for v${NATTEN_VERSION} with ${torch_build} for Python $py, arch $arch_tag."
        WHEELS_FOUND=$((WHEELS_FOUND+1))
      elif [[ -v ALLOWED_MISSING_WHEELS["$WHEEL_NAME"] ]]; then
        echo "[-] Wheel SKIPPED (known missing) for v${NATTEN_VERSION} with ${torch_build} for Python $py, arch $arch_tag."
        WHEELS_SKIPPED=$((WHEELS_SKIPPED+1))
      else
        echo "[ ] Wheel MISSING for v${NATTEN_VERSION} with ${torch_build} for Python $py, arch $arch_tag."
        WHEELS_MISSING=$((WHEELS_MISSING+1))
      fi
    done
  done
}

# Torch 2.13.X
check_one cu132 2.13.0
check_one cu130 2.13.0

WHEELS_TOTAL=$((WHEELS_FOUND+WHEELS_MISSING+WHEELS_SKIPPED))

echo ""
echo ""
echo "Wheels found: $WHEELS_FOUND"
echo "Wheels missing: $WHEELS_MISSING"
echo "Wheels skipped (known missing): $WHEELS_SKIPPED"
echo "Wheels total: $WHEELS_TOTAL"
