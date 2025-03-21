#!/bin/bash -e
# Based on detectron2 
# github.com/facebookresearch/detectron2


root=$(readlink -f $1)
if [[ -z "$root" ]]; then
  echo "Usage: ./gen_wheel_index.sh /absolute/path/to/wheels"
  exit
fi

export LC_ALL=C  # reproducible sort
# NOTE: all sort in this script might not work when xx.10 is released

index=$root/index.html

cd "$root"
for cu in cpu cu101 cu102 cu111 cu113 cu115 cu116 cu117 cu118 cu121 cu124 cu126; do
  mkdir -p "$root/$cu"
  cd "$root/$cu"
  echo "Creating $PWD/index.html ..."
  # First sort by torch version, then stable sort by d2 version with unique.
  # As a result, the latest torch version for each d2 version is kept.
  rm -f index.html
  echo "<!DOCTYPE html><html lang=\"en\"><head>" >> index.html
  echo "<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">" >> index.html
  echo "<title>Index of packages</title></head><body>" >> index.html
  for whl in $(find -type f -name '*.whl' -printf '%P\n' \
    | sort -k 1 -r  | sort -t '/' -k 2 --stable -r --unique); do
    echo "<a href=\"${whl/+/%2B}\">$whl</a><br>"
  done >> index.html
  echo "</body></html>" >> index.html


  for torch in torch*; do
    cd "$root/$cu/$torch"

    rm -f index.html
    echo "<!DOCTYPE html><html lang=\"en\"><head>" >> index.html
    echo "<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">" >> index.html
    echo "<title>Index of packages</title></head><body>" >> index.html

    # list all whl for each cuda,torch version
    echo "Creating $PWD/index.html ..."
    for whl in $(find . -type f -name '*.whl' -printf '%P\n' | sort -r); do
      echo "<a href=\"${whl/+/%2B}\">$whl</a><br>"
    done >> index.html
    echo "</body></html>" >> index.html
  done
done

cd "$root"
# Just list everything:
rm -f index.html
echo "<!DOCTYPE html><html lang=\"en\"><head>" >> $index
echo "<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">" >> $index
echo "<title>Index of packages</title></head><body>" >> $index

echo "Creating $index ..."
for whl in $(find . -type f -name '*.whl' -printf '%P\n' | sort -r); do
  echo "<a href=\"${whl/+/%2B}\">$whl</a><br>"
done >> "$index"
echo "</body></html>" >> $index

