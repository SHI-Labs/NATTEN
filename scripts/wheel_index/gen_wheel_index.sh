#!/usr/bin/env bash

set -e


[[ -d "scripts/wheel_index" ]] || {
  echo "Please run this script from natten root!"
  exit 1
}

source scripts/wheel_index/utils.sh

TARGET_DIRECTORY="wheel-index/"

ROOT_WHEEL_LIST=""
declare -A CTK_VER_WHEEL_DICT
declare -A TORCH_VER_WHEEL_DICT

check_one() {
  NATTEN_VERSION=$1
  cu=$2
  pytorch_ver=$3
  URL_PREFIX=$4
  torch_build="torch${pytorch_ver//./}${cu}"
  torch_major=$(echo $pytorch_ver | awk -F"." '{print $1$2}')      # 29 (pytorch major version)
  natten_minor=$(echo $NATTEN_VERSION | awk -F"." '{print $2$3}')  # 211 (natten minor version)

  # Map stuff
  TORCH_VER_WHEEL_DICT_KEY="${cu}/torch${pytorch_ver}"

  if [[ ! -v CTK_VER_WHEEL_DICT["$cu"] ]]; then
    CTK_VER_WHEEL_DICT["$cu"]=""
  fi

  if [[ ! -v TORCH_VER_WHEEL_DICT["$TORCH_VER_WHEEL_DICT_KEY"] ]]; then
    TORCH_VER_WHEEL_DICT["$TORCH_VER_WHEEL_DICT_KEY"]=""
  fi

  py_versions=(3.10 3.11 3.12)
  SUPPORTED_ARCHES=("x86_64")

  if [[ $torch_major -lt 25 ]]; then
    echo "Only torch 2.5 and later are supported from now on."
    exit 1
  fi

  # Torch 2.9 no longer ships for python 3.9.
  if [[ $torch_major -lt 29 ]]; then
    py_versions+=(3.9)
  fi

  if [[ $natten_minor -ge 211 ]]; then
    # Torch started supporting python 3.13 since ~2.5
    # We are building wheels for 3.13 starting 0.21.1
    py_versions+=(3.13 3.13t)

    # Torch also started shipping arm CUDA builds since 2.9.
    # NATTEN started since 0.21.1
    if [[ $torch_major -gt 28 ]]; then
      SUPPORTED_ARCHES+=("aarch64")
    fi
  fi

  for py in "${py_versions[@]}"; do
    pytag_a=${py//./}
    python_tag="cp${pytag_a/t/}-cp${py//./}"

    NATTEN_VERSION_WITH_TORCH_TAG="${NATTEN_VERSION}+${torch_build}"

    for arch_tag in "${SUPPORTED_ARCHES[@]}";do
      EXPECTED_WHL_LINK=$(gen_whl_link $URL_PREFIX $NATTEN_VERSION $NATTEN_VERSION_WITH_TORCH_TAG $python_tag $arch_tag)

      WHL_FILENAME=$(gen_whl_filename $NATTEN_VERSION_WITH_TORCH_TAG $python_tag $arch_tag)
      ROOT_WHL_FILENAME="${cu}/torch${pytorch_ver}/$WHL_FILENAME"
      CTK_WHL_FILENAME="torch${pytorch_ver}/$WHL_FILENAME"

      if curl --head --silent --fail $EXPECTED_WHL_LINK 2> /dev/null 1> /dev/null; then
        echo "[CHECK] $EXPECTED_WHL_LINK"
      else
        echo "[NOT FOUND] $EXPECTED_WHL_LINK"
        exit 1
      fi

      # Add link to HTML
      TORCH_LINK="<a href=\"${EXPECTED_WHL_LINK/+/%2B}\">$WHL_FILENAME</a><br>"
      CTK_LINK="<a href=\"${EXPECTED_WHL_LINK/+/%2B}\">$CTK_WHL_FILENAME</a><br>"
      ROOT_LINK="<a href=\"${EXPECTED_WHL_LINK/+/%2B}\">$ROOT_WHL_FILENAME</a><br>"

      TORCH_VER_WHEEL_DICT[$TORCH_VER_WHEEL_DICT_KEY]+="$TORCH_LINK"
      CTK_VER_WHEEL_DICT[$cu]+="$CTK_LINK"
      ROOT_WHEEL_LIST+="$ROOT_LINK"
    done

  done
}

##################################
# ADD IN SUPPORTED BUILDS HERE
##################################

URL_PREFIX="https://github.com/SHI-Labs/NATTEN/releases/download/"

# v0.17.5
check_one 0.17.5 cu126 2.6.0 $URL_PREFIX
check_one 0.17.5 cu124 2.6.0 $URL_PREFIX
check_one 0.17.5 cpu   2.6.0 $URL_PREFIX

check_one 0.17.5 cu124 2.5.0 $URL_PREFIX
check_one 0.17.5 cu121 2.5.0 $URL_PREFIX
check_one 0.17.5 cpu   2.5.0 $URL_PREFIX

# v0.20.0
check_one 0.20.0 cu128 2.7.0 $URL_PREFIX
check_one 0.20.0 cu126 2.7.0 $URL_PREFIX

# v0.20.1
check_one 0.20.1 cu128 2.7.0 $URL_PREFIX
check_one 0.20.1 cu126 2.7.0 $URL_PREFIX

# v0.21.0
check_one 0.21.0 cu128 2.7.0 $URL_PREFIX
check_one 0.21.0 cu126 2.7.0 $URL_PREFIX

# v0.21.1
check_one 0.21.1 cu129 2.8.0 $URL_PREFIX
check_one 0.21.1 cu128 2.8.0 $URL_PREFIX
check_one 0.21.1 cu126 2.8.0 $URL_PREFIX

check_one 0.21.1 cu130 2.9.0 $URL_PREFIX
check_one 0.21.1 cu128 2.9.0 $URL_PREFIX
check_one 0.21.1 cu126 2.9.0 $URL_PREFIX

##################################
##################################
##################################

echo "Generating index files..."

#rm -rf $TARGET_DIRECTORY

for key in "${!TORCH_VER_WHEEL_DICT[@]}"; do
  TARGET_DIR="${TARGET_DIRECTORY}/$key/"
  TARGET_IDX="${TARGET_DIR}/index.html"

  echo "Generating index for torch version $key"
  mkdir -p $TARGET_DIR

  rm -f $TARGET_IDX
  echo $(gen_html_header) >> $TARGET_IDX

  # Add links to HTML
  WHEEL_LINKS="${TORCH_VER_WHEEL_DICT[$key]}"
  echo $WHEEL_LINKS >> $TARGET_IDX

  echo $(gen_html_footer) >> $TARGET_IDX
done

for key in "${!CTK_VER_WHEEL_DICT[@]}"; do
  TARGET_DIR="${TARGET_DIRECTORY}/$key/"
  TARGET_IDX="${TARGET_DIR}/index.html"

  echo "Generating index for CTK version $key"

  rm -f $TARGET_IDX
  echo $(gen_html_header) >> $TARGET_IDX

  # Add links to HTML
  WHEEL_LINKS="${CTK_VER_WHEEL_DICT[$key]}"
  echo $WHEEL_LINKS >> $TARGET_IDX

  echo $(gen_html_footer) >> $TARGET_IDX
done

# Generate root index
echo "Generating root index"
TARGET_IDX="${TARGET_DIRECTORY}/index.html"

rm -f $TARGET_IDX
echo $(gen_html_header "NATTEN Wheel Index") >> $TARGET_IDX

# Add links to HTML
echo $ROOT_WHEEL_LIST >> $TARGET_IDX

echo $(gen_html_footer) >> $TARGET_IDX
