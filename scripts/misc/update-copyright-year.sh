#!/bin/bash


find src/ tests/ setup.py -type f \
  -name '*.py' \
  -exec sed -i 's/Copyright (c) 2022-2024 Ali Hassani./Copyright (c) 2022-2025 Ali Hassani./g' {} \;

find csrc/ -type f \
  \( -name '*.cu' -o \
  -name '*.cpp' -o \
  -name '*.h' -o \
  -name '*.cuh' -o \
  -name '*.hpp' -o \
  -name '*.txt' -o \
  -name '*.cc' \) \
  -exec sed -i 's/Copyright (c) 2022-2024 Ali Hassani./Copyright (c) 2022-2025 Ali Hassani./g' {} \;

find scripts/ -type f \
  -exec sed -i 's/Copyright (c) 2022-2024 Ali Hassani./Copyright (c) 2022-2025 Ali Hassani./g' {} \;

find LICENSE -type f \
  -exec sed -i 's/Copyright (c) 2022 - 2024 Ali Hassani./Copyright (c) 2022 - 2025 Ali Hassani./g' {} \;
