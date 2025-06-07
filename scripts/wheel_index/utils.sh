#!/usr/bin/env bash -e


gen_whl_filename() {
  NATTEN_VERSION_WITH_TORCH_TAG=$1
  PY=$2

  echo "natten-${NATTEN_VERSION_WITH_TORCH_TAG}-${PY}-linux_x86_64.whl"
}

gen_whl_link() {
  PREFIX=$1
  NATTEN_VERSION=$2
  NATTEN_VERSION_WITH_TORCH_TAG=$3
  PY=$4

  WHL_PATH="v${NATTEN_VERSION}/"
  WHL_FILENAME=$(gen_whl_filename $NATTEN_VERSION_WITH_TORCH_TAG $PY)

  echo "${PREFIX}${WHL_PATH}${WHL_FILENAME}"
}

gen_html_header() {
  TITLE=${1:-"Index of packages"}

  echo "<!DOCTYPE html><html lang=\"en\"><head>"
  echo "<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
  echo "<title>${TITLE}</title></head><body>"
}

gen_html_footer() {
  echo "</body></html>"
}

