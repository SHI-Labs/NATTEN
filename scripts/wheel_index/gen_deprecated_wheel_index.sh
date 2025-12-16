#!/usr/bin/env bash

set -e

# Wheel index for old wheels


[[ -d "scripts/wheel_index" ]] || {
  echo "Please run this script from natten root!"
  exit 1
}

source scripts/wheel_index/utils.sh

TARGET_DIRECTORY="wheel-index/old/"

ROOT_WHEEL_LIST=""
declare -A CTK_VER_WHEEL_DICT
declare -A TORCH_VER_WHEEL_DICT

check_one() {
  NATTEN_VERSION=$1
  cu=$2
  pytorch_ver=$3
  URL_PREFIX=$4
  torch_build="torch${pytorch_ver//./}${cu}"
  torch_base=$(echo $pytorch_ver | awk -F"." '{print $1}')         # 2 (pytorch base version)
  torch_major=$(echo $pytorch_ver | awk -F"." '{print $2}')        # 9 (pytorch major version)
  torch_minor=$(echo $pytorch_ver | awk -F"." '{print $3}')        # 0 (pytorch minor version)
  natten_minor=$(echo $NATTEN_VERSION | awk -F"." '{print $2$3}')  # 211 (natten version)

  # Map stuff
  TORCH_VER_WHEEL_DICT_KEY="${cu}/torch${pytorch_ver}"

  if [[ ! -v CTK_VER_WHEEL_DICT["$cu"] ]]; then
    CTK_VER_WHEEL_DICT["$cu"]=""
  fi

  if [[ ! -v TORCH_VER_WHEEL_DICT["$TORCH_VER_WHEEL_DICT_KEY"] ]]; then
    TORCH_VER_WHEEL_DICT["$TORCH_VER_WHEEL_DICT_KEY"]=""
  fi

  py_versions=(3.9)
  SUPPORTED_ARCHES=("x86_64")

  if [[ $torch_base -eq 2 && $torch_major -ge 2 ]] || [[ $torch_base -gt 2 ]]; then
    py_versions+=(3.12)
  fi

  # Torch 2.5 dropped support for python 3.8.
  if [[ $torch_base -eq 2 && $torch_major -le 4 ]] || [[ $torch_base -lt 2 ]]; then
    py_versions+=(3.8)
  fi

  # Torch 2.0 dropped support for python 3.7.
  if [[ $torch_base -lt 2 ]]; then
    py_versions+=(3.7m)
  fi

  # Torch 1.13 added support for python 3.11.
  if [[ $torch_base -eq 1 && $torch_major -ge 13 ]] || [[ $torch_base -gt 1 ]]; then
    py_versions+=(3.11)
  fi

  # Torch 1.12 added support for python 3.10.
  # With the exception of NATTEN 0.14.1 and 0.14.2 releases, which didn't ship for 3.10
  if [[ $torch_base -eq 1 && $torch_major -ge 12 ]] || [[ $torch_base -gt 1 ]] && [[ $natten_minor -gt 142 ]]; then
    py_versions+=(3.10)
  fi

  for py in "${py_versions[@]}"; do
    pytag_a=${py//./}
    python_tag="cp${pytag_a/m/}-cp${py//./}"

    NATTEN_VERSION_WITH_TORCH_TAG="${NATTEN_VERSION}+${torch_build}"

    for arch_tag in "${SUPPORTED_ARCHES[@]}";do
      EXPECTED_WHL_LINK=$(gen_whl_link $URL_PREFIX $NATTEN_VERSION $NATTEN_VERSION_WITH_TORCH_TAG $python_tag $arch_tag)

      WHL_FILENAME=$(gen_whl_filename $NATTEN_VERSION_WITH_TORCH_TAG $python_tag $arch_tag)
      ROOT_WHL_FILENAME="${cu}/torch${pytorch_ver}/$WHL_FILENAME"
      CTK_WHL_FILENAME="torch${pytorch_ver}/$WHL_FILENAME"

      if curl --head --silent --fail $EXPECTED_WHL_LINK 2> /dev/null 1> /dev/null; then
        echo "[CHECK] $EXPECTED_WHL_LINK"
      else
        # Wheels missing since release
        # Will investigate only if issues are raised
        if [[ $natten_minor -eq 142 && $torch_base -eq 1 && $torch_major -eq 12 && $torch_minor -eq 0 && $cu == "cu116" && $py == "3.9" ]] || 
           [[ $natten_minor -eq 142 && $torch_base -eq 1 && $torch_major -eq 11 && $torch_minor -eq 0 && $cu == "cu115" && $py == "3.9" ]]; then
          echo "[WARNING] Skipping wheel missing since release $EXPECTED_WHL_LINK"
          continue
        else
          echo "[NOT FOUND] $EXPECTED_WHL_LINK"
          exit 1
        fi
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

# v0.17.4
check_one 0.17.4 cu124 2.4.0 $URL_PREFIX
check_one 0.17.4 cu121 2.4.0 $URL_PREFIX
check_one 0.17.4 cu118 2.4.0 $URL_PREFIX
check_one 0.17.4 cpu   2.4.0 $URL_PREFIX

check_one 0.17.4 cu121 2.3.0 $URL_PREFIX
check_one 0.17.4 cu118 2.3.0 $URL_PREFIX
check_one 0.17.4 cpu   2.3.0 $URL_PREFIX

check_one 0.17.4 cu121 2.2.0 $URL_PREFIX
check_one 0.17.4 cu118 2.2.0 $URL_PREFIX
check_one 0.17.4 cpu   2.2.0 $URL_PREFIX

check_one 0.17.4 cu121 2.1.0 $URL_PREFIX
check_one 0.17.4 cu118 2.1.0 $URL_PREFIX
check_one 0.17.4 cpu   2.1.0 $URL_PREFIX

check_one 0.17.4 cu118 2.0.0 $URL_PREFIX
check_one 0.17.4 cu117 2.0.0 $URL_PREFIX
check_one 0.17.4 cpu   2.0.0 $URL_PREFIX

# v0.17.3
check_one 0.17.3 cu124 2.4.0 $URL_PREFIX
check_one 0.17.3 cu121 2.4.0 $URL_PREFIX
check_one 0.17.3 cu118 2.4.0 $URL_PREFIX
check_one 0.17.3 cpu   2.4.0 $URL_PREFIX

check_one 0.17.3 cu121 2.3.0 $URL_PREFIX
check_one 0.17.3 cu118 2.3.0 $URL_PREFIX
check_one 0.17.3 cpu   2.3.0 $URL_PREFIX

check_one 0.17.3 cu121 2.2.0 $URL_PREFIX
check_one 0.17.3 cu118 2.2.0 $URL_PREFIX
check_one 0.17.3 cpu   2.2.0 $URL_PREFIX

check_one 0.17.3 cu121 2.1.0 $URL_PREFIX
check_one 0.17.3 cu118 2.1.0 $URL_PREFIX
check_one 0.17.3 cpu   2.1.0 $URL_PREFIX

check_one 0.17.3 cu118 2.0.0 $URL_PREFIX
check_one 0.17.3 cu117 2.0.0 $URL_PREFIX
check_one 0.17.3 cpu   2.0.0 $URL_PREFIX

# v0.17.1
check_one 0.17.1 cu124 2.4.0 $URL_PREFIX
check_one 0.17.1 cu121 2.4.0 $URL_PREFIX
check_one 0.17.1 cu118 2.4.0 $URL_PREFIX
check_one 0.17.1 cpu   2.4.0 $URL_PREFIX

check_one 0.17.1 cu121 2.3.0 $URL_PREFIX
check_one 0.17.1 cu118 2.3.0 $URL_PREFIX
check_one 0.17.1 cpu   2.3.0 $URL_PREFIX

check_one 0.17.1 cu121 2.2.0 $URL_PREFIX
check_one 0.17.1 cu118 2.2.0 $URL_PREFIX
check_one 0.17.1 cpu   2.2.0 $URL_PREFIX

check_one 0.17.1 cu121 2.1.0 $URL_PREFIX
check_one 0.17.1 cu118 2.1.0 $URL_PREFIX
check_one 0.17.1 cpu   2.1.0 $URL_PREFIX

check_one 0.17.1 cu118 2.0.0 $URL_PREFIX
check_one 0.17.1 cu117 2.0.0 $URL_PREFIX
check_one 0.17.1 cpu   2.0.0 $URL_PREFIX

# v0.17.0
check_one 0.17.0 cu121 2.3.0 $URL_PREFIX
check_one 0.17.0 cu118 2.3.0 $URL_PREFIX
check_one 0.17.0 cpu   2.3.0 $URL_PREFIX

check_one 0.17.0 cu121 2.2.0 $URL_PREFIX
check_one 0.17.0 cu118 2.2.0 $URL_PREFIX
check_one 0.17.0 cpu   2.2.0 $URL_PREFIX

check_one 0.17.0 cu121 2.1.0 $URL_PREFIX
check_one 0.17.0 cu118 2.1.0 $URL_PREFIX
check_one 0.17.0 cpu   2.1.0 $URL_PREFIX

check_one 0.17.0 cu118 2.0.0 $URL_PREFIX
check_one 0.17.0 cu117 2.0.0 $URL_PREFIX
check_one 0.17.0 cpu   2.0.0 $URL_PREFIX

# v0.15.1
check_one 0.15.1 cu121 2.2.0 $URL_PREFIX
check_one 0.15.1 cu118 2.2.0 $URL_PREFIX
check_one 0.15.1 cpu   2.2.0 $URL_PREFIX

check_one 0.15.1 cu121 2.1.0 $URL_PREFIX
check_one 0.15.1 cu118 2.1.0 $URL_PREFIX
check_one 0.15.1 cpu   2.1.0 $URL_PREFIX

check_one 0.15.1 cu118 2.0.0 $URL_PREFIX
check_one 0.15.1 cu117 2.0.0 $URL_PREFIX
check_one 0.15.1 cpu   2.0.0 $URL_PREFIX

# v0.15.0
check_one 0.15.0 cu121 2.1.0 $URL_PREFIX
check_one 0.15.0 cu118 2.1.0 $URL_PREFIX
check_one 0.15.0 cpu   2.1.0 $URL_PREFIX

check_one 0.15.0 cu118 2.0.0 $URL_PREFIX
check_one 0.15.0 cu117 2.0.0 $URL_PREFIX
check_one 0.15.0 cpu   2.0.0 $URL_PREFIX

# v0.14.6
check_one 0.14.6 cu118 2.0.0 $URL_PREFIX
check_one 0.14.6 cu117 2.0.0 $URL_PREFIX
check_one 0.14.6 cpu   2.0.0 $URL_PREFIX

check_one 0.14.6 cu117 1.13.0 $URL_PREFIX
check_one 0.14.6 cu116 1.13.0 $URL_PREFIX
check_one 0.14.6 cpu   1.13.0 $URL_PREFIX

check_one 0.14.6 cu116 1.12.1 $URL_PREFIX
check_one 0.14.6 cu113 1.12.1 $URL_PREFIX
check_one 0.14.6 cu102 1.12.1 $URL_PREFIX
check_one 0.14.6 cpu   1.12.1 $URL_PREFIX

check_one 0.14.6 cu116 1.12.0 $URL_PREFIX
check_one 0.14.6 cu113 1.12.0 $URL_PREFIX
check_one 0.14.6 cu102 1.12.0 $URL_PREFIX
check_one 0.14.6 cpu   1.12.0 $URL_PREFIX

check_one 0.14.6 cu115 1.11.0 $URL_PREFIX
check_one 0.14.6 cu113 1.11.0 $URL_PREFIX
check_one 0.14.6 cu102 1.11.0 $URL_PREFIX
check_one 0.14.6 cpu   1.11.0 $URL_PREFIX

check_one 0.14.6 cu113 1.10.1 $URL_PREFIX
check_one 0.14.6 cu111 1.10.1 $URL_PREFIX
check_one 0.14.6 cu102 1.10.1 $URL_PREFIX
check_one 0.14.6 cpu   1.10.1 $URL_PREFIX

check_one 0.14.6 cu113 1.10.0 $URL_PREFIX
check_one 0.14.6 cu111 1.10.0 $URL_PREFIX
check_one 0.14.6 cu102 1.10.0 $URL_PREFIX
check_one 0.14.6 cpu   1.10.0 $URL_PREFIX

check_one 0.14.6 cu111 1.9.0 $URL_PREFIX
check_one 0.14.6 cu102 1.9.0 $URL_PREFIX
check_one 0.14.6 cpu   1.9.0 $URL_PREFIX

check_one 0.14.6 cu111 1.8.0 $URL_PREFIX
check_one 0.14.6 cu102 1.8.0 $URL_PREFIX
check_one 0.14.6 cu101 1.8.0 $URL_PREFIX
check_one 0.14.6 cpu   1.8.0 $URL_PREFIX

# v0.14.5
check_one 0.14.5 cu118 2.0.0 $URL_PREFIX
check_one 0.14.5 cu117 2.0.0 $URL_PREFIX
check_one 0.14.5 cpu   2.0.0 $URL_PREFIX

check_one 0.14.5 cu117 1.13.0 $URL_PREFIX
check_one 0.14.5 cu116 1.13.0 $URL_PREFIX
check_one 0.14.5 cpu   1.13.0 $URL_PREFIX

check_one 0.14.5 cu116 1.12.1 $URL_PREFIX
check_one 0.14.5 cu113 1.12.1 $URL_PREFIX
check_one 0.14.5 cu102 1.12.1 $URL_PREFIX
check_one 0.14.5 cpu   1.12.1 $URL_PREFIX

check_one 0.14.5 cu116 1.12.0 $URL_PREFIX
check_one 0.14.5 cu113 1.12.0 $URL_PREFIX
check_one 0.14.5 cu102 1.12.0 $URL_PREFIX
check_one 0.14.5 cpu   1.12.0 $URL_PREFIX

check_one 0.14.5 cu115 1.11.0 $URL_PREFIX
check_one 0.14.5 cu113 1.11.0 $URL_PREFIX
check_one 0.14.5 cu102 1.11.0 $URL_PREFIX
check_one 0.14.5 cpu   1.11.0 $URL_PREFIX

check_one 0.14.5 cu113 1.10.1 $URL_PREFIX
check_one 0.14.5 cu111 1.10.1 $URL_PREFIX
check_one 0.14.5 cu102 1.10.1 $URL_PREFIX
check_one 0.14.5 cpu   1.10.1 $URL_PREFIX

check_one 0.14.5 cu113 1.10.0 $URL_PREFIX
check_one 0.14.5 cu111 1.10.0 $URL_PREFIX
check_one 0.14.5 cu102 1.10.0 $URL_PREFIX
check_one 0.14.5 cpu   1.10.0 $URL_PREFIX

check_one 0.14.5 cu111 1.9.0 $URL_PREFIX
check_one 0.14.5 cu102 1.9.0 $URL_PREFIX
check_one 0.14.5 cpu   1.9.0 $URL_PREFIX

check_one 0.14.5 cu111 1.8.0 $URL_PREFIX
check_one 0.14.5 cu102 1.8.0 $URL_PREFIX
check_one 0.14.5 cu101 1.8.0 $URL_PREFIX
check_one 0.14.5 cpu   1.8.0 $URL_PREFIX

# v0.14.4
check_one 0.14.4 cu117 1.13.0 $URL_PREFIX
check_one 0.14.4 cu116 1.13.0 $URL_PREFIX
check_one 0.14.4 cpu   1.13.0 $URL_PREFIX

check_one 0.14.4 cu116 1.12.1 $URL_PREFIX
check_one 0.14.4 cu113 1.12.1 $URL_PREFIX
check_one 0.14.4 cu102 1.12.1 $URL_PREFIX
check_one 0.14.4 cpu   1.12.1 $URL_PREFIX

check_one 0.14.4 cu116 1.12.0 $URL_PREFIX
check_one 0.14.4 cu113 1.12.0 $URL_PREFIX
check_one 0.14.4 cu102 1.12.0 $URL_PREFIX
check_one 0.14.4 cpu   1.12.0 $URL_PREFIX

check_one 0.14.4 cu115 1.11.0 $URL_PREFIX
check_one 0.14.4 cu113 1.11.0 $URL_PREFIX
check_one 0.14.4 cu102 1.11.0 $URL_PREFIX
check_one 0.14.4 cpu   1.11.0 $URL_PREFIX

check_one 0.14.4 cu113 1.10.1 $URL_PREFIX
check_one 0.14.4 cu111 1.10.1 $URL_PREFIX
check_one 0.14.4 cu102 1.10.1 $URL_PREFIX
check_one 0.14.4 cpu   1.10.1 $URL_PREFIX

check_one 0.14.4 cu113 1.10.0 $URL_PREFIX
check_one 0.14.4 cu111 1.10.0 $URL_PREFIX
check_one 0.14.4 cu102 1.10.0 $URL_PREFIX
check_one 0.14.4 cpu   1.10.0 $URL_PREFIX

check_one 0.14.4 cu111 1.9.0 $URL_PREFIX
check_one 0.14.4 cu102 1.9.0 $URL_PREFIX
check_one 0.14.4 cpu   1.9.0 $URL_PREFIX

check_one 0.14.4 cu111 1.8.0 $URL_PREFIX
check_one 0.14.4 cu102 1.8.0 $URL_PREFIX
check_one 0.14.4 cu101 1.8.0 $URL_PREFIX
check_one 0.14.4 cpu   1.8.0 $URL_PREFIX

# v0.14.2
check_one 0.14.2 cu116 1.12.1 $URL_PREFIX
check_one 0.14.2 cu113 1.12.1 $URL_PREFIX
check_one 0.14.2 cu102 1.12.1 $URL_PREFIX
check_one 0.14.2 cpu   1.12.1 $URL_PREFIX

check_one 0.14.2 cu116 1.12.0 $URL_PREFIX
check_one 0.14.2 cu113 1.12.0 $URL_PREFIX
check_one 0.14.2 cu102 1.12.0 $URL_PREFIX
check_one 0.14.2 cpu   1.12.0 $URL_PREFIX

check_one 0.14.2 cu115 1.11.0 $URL_PREFIX
check_one 0.14.2 cu113 1.11.0 $URL_PREFIX
check_one 0.14.2 cu102 1.11.0 $URL_PREFIX
check_one 0.14.2 cpu   1.11.0 $URL_PREFIX

check_one 0.14.2 cu113 1.10.1 $URL_PREFIX
check_one 0.14.2 cu111 1.10.1 $URL_PREFIX
check_one 0.14.2 cu102 1.10.1 $URL_PREFIX
check_one 0.14.2 cpu   1.10.1 $URL_PREFIX

check_one 0.14.2 cu113 1.10.0 $URL_PREFIX
check_one 0.14.2 cu111 1.10.0 $URL_PREFIX
check_one 0.14.2 cu102 1.10.0 $URL_PREFIX
check_one 0.14.2 cpu   1.10.0 $URL_PREFIX

check_one 0.14.2 cu111 1.9.0 $URL_PREFIX
check_one 0.14.2 cu102 1.9.0 $URL_PREFIX
check_one 0.14.2 cpu   1.9.0 $URL_PREFIX

check_one 0.14.2 cu111 1.8.0 $URL_PREFIX
check_one 0.14.2 cu102 1.8.0 $URL_PREFIX
check_one 0.14.2 cu101 1.8.0 $URL_PREFIX
check_one 0.14.2 cpu   1.8.0 $URL_PREFIX

# v0.14.1 (no cpu builds)
check_one 0.14.1 cu116 1.12.1 $URL_PREFIX
check_one 0.14.1 cu113 1.12.1 $URL_PREFIX
check_one 0.14.1 cu102 1.12.1 $URL_PREFIX

check_one 0.14.1 cu116 1.12.0 $URL_PREFIX
check_one 0.14.1 cu113 1.12.0 $URL_PREFIX
check_one 0.14.1 cu102 1.12.0 $URL_PREFIX

check_one 0.14.1 cu115 1.11.0 $URL_PREFIX
check_one 0.14.1 cu113 1.11.0 $URL_PREFIX
check_one 0.14.1 cu102 1.11.0 $URL_PREFIX

check_one 0.14.1 cu113 1.10.1 $URL_PREFIX
check_one 0.14.1 cu111 1.10.1 $URL_PREFIX
check_one 0.14.1 cu102 1.10.1 $URL_PREFIX

check_one 0.14.1 cu113 1.10.0 $URL_PREFIX
check_one 0.14.1 cu111 1.10.0 $URL_PREFIX
check_one 0.14.1 cu102 1.10.0 $URL_PREFIX

check_one 0.14.1 cu111 1.9.0 $URL_PREFIX
check_one 0.14.1 cu102 1.9.0 $URL_PREFIX

check_one 0.14.1 cu111 1.8.0 $URL_PREFIX
check_one 0.14.1 cu102 1.8.0 $URL_PREFIX
check_one 0.14.1 cu101 1.8.0 $URL_PREFIX

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
