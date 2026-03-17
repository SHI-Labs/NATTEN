#!/bin/bash

set -e

find src/ tests/ setup.py -type f \
  -name '*.py' \
  -exec sed -i 's/\(Copyright (c) 2022\) - 2025 \(Ali Hassani.\)/\1 - 2026 \2/g' {} \;

find csrc/ -type f \
  \( -name '*.cu' -o \
  -name '*.cpp' -o \
  -name '*.h' -o \
  -name '*.cuh' -o \
  -name '*.hpp' -o \
  -name '*.txt' -o \
  -name '*.cc' \) \
  -exec sed -i 's/\(Copyright (c) 2022 -\) 2025 \(Ali Hassani.\)/\1 2026 \2/g' {} \;

find scripts/ -type f \
  -exec sed -i 's/\(Copyright (c) 2022 -\) 2025 \(Ali Hassani.\)/\1 2026 \2/g' {} \;

find LICENSE NOTICE Dockerfile Makefile -type f \
  -exec sed -i 's/\(Copyright (c) 2022 -\) 2025 \(Ali Hassani.\)/\1 2026 \2/g' {} \;

find mkdocs.yml -type f \
  -exec sed -i 's/\(2022 -\) 2025 \(Ali Hassani\)/\1 2026 \2/g' {} \;

# nvidia copyrights
find csrc/ -type f \
  \( -name '*.cu' -o \
  -name '*.cpp' -o \
  -name '*.h' -o \
  -name '*.cuh' -o \
  -name '*.hpp' -o \
  -name '*.txt' -o \
  -name '*.cc' \) \
  -exec sed -i 's/\(Copyright (c) 20[0-9][0-9] -\) 20[0-9][0-9] \(NVIDIA CORPORATION\)/\1 2026 \2/g' {} \;
