#!/bin/bash


find src/ tests/ setup.py -type f \
  -name '*.py' \
  -exec sed -i 's/Copyright (c) 2022 - 2026 Ali Hassani./Copyright (c) 2022 - 2026 Ali Hassani./g' {} \;

find csrc/ -type f \
  \( -name '*.cu' -o \
  -name '*.cpp' -o \
  -name '*.h' -o \
  -name '*.cuh' -o \
  -name '*.hpp' -o \
  -name '*.txt' -o \
  -name '*.cc' \) \
  -exec sed -i 's/Copyright (c) 2022 - 2026 Ali Hassani./Copyright (c) 2022 - 2026 Ali Hassani./g' {} \;

find scripts/ -type f \
  -exec sed -i 's/Copyright (c) 2022 - 2026 Ali Hassani./Copyright (c) 2022 - 2026 Ali Hassani./g' {} \;

find LICENSE NOTICE Dockerfile Makefile -type f \
  -exec sed -i 's/Copyright (c) 2022 - 2025 Ali Hassani./Copyright (c) 2022 - 2026 Ali Hassani./g' {} \;

find mkdocs.yml -type f \
  -exec sed -i 's/2022 - 2025 Ali Hassani/2022 - 2026 Ali Hassani/g' {} \;
