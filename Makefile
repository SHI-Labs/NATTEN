# Copyright (c) 2022-2025 Ali Hassani.

.PHONY: fetch-submodules build-wheels build-dist deep-clean clean uninstall install install-dev test format serve-docs build-docs

# build flags
CUDA_ARCH=${NATTEN_CUDA_ARCH}
WORKERS=${NATTEN_N_WORKERS}
VERBOSE=${NATTEN_VERBOSE}
AUTOGEN_POLICY=${NATTEN_AUTOGEN_POLICY}

# test flags
RUN_EXTENDED_TESTS=${NATTEN_RUN_EXTENDED_TESTS}
RUN_ADDITIONAL_KV_TESTS=${NATTEN_RUN_ADDITIONAL_KV_TESTS}
RUN_FLEX_TESTS=${NATTEN_RUN_FLEX_TESTS}
NUM_RAND_SWEEP_TESTS=${NATTEN_RAND_SWEEP_TESTS}

# env
PYTHON=python
PIP=pip
GIT=git
TWINE=twine
PYTEST=pytest
UFMT=ufmt
MYPY=mypy
FLAKE8=flake8
MKDOCS=mkdocs

check_dirs := src/natten tests scripts setup.py

all: clean uninstall fetch-submodules install

dev: clean uninstall fetch-submodules install-dev

fetch-submodules:
	@echo "Fetching all third party submodules"
	$(GIT) submodule update --init --recursive

build-wheels:
	$(MAKE) fetch-submodules
	@echo "Building release wheels"
	NATTEN_AUTOGEN_POLICY="${AUTOGEN_POLICY}" \
	NATTEN_N_WORKERS="${WORKERS}" \
	./scripts/packaging/build_all_wheels_parallel.sh

build-wheel-index:
	@echo "Building wheel index"
	./scripts/wheel_index/gen_wheel_index.sh

build-dist:
	$(MAKE) fetch-submodules
	@echo "Generating source dist"
	NATTEN_AUTOGEN_POLICY="${AUTOGEN_POLICY}" \
	$(PYTHON) -m build --sdist --no-isolation

release:
	$(TWINE) upload --repository natten dist/*

deep-clean: 
	@echo "Cleaning up (deep clean)"
	rm -rf build/ 
	rm -rf build_dir/ 
	rm -rf dist/ 
	rm -rf csrc/autogen/ 
	rm -rf natten.egg-info/ 
	rm -rf src/natten/_C.* 
	rm -rf src/natten/libnatten.* 
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf src/__pycache__
	rm -rf src/natten/__pycache__
	rm -rf src/natten.egg*
	rm -rf install.out

clean:
	@echo "Cleaning up"
	rm -rf dist/
	rm -rf natten.egg-info/
	rm -rf src/natten/_C.*
	rm -rf src/natten/libnatten.*
	rm -rf __pycache__
	rm -rf tests/__pycache__
	rm -rf src/__pycache__
	rm -rf src/natten/__pycache__
	rm -rf src/natten.egg*
	rm -rf install.out

uninstall: 
	@echo "Uninstalling NATTEN"
	$(PIP) uninstall -y natten

install-dev: 
	$(MAKE) fetch-submodules
	@echo "Installing NATTEN from source - development mode (editable)"
	mkdir -p $(PWD)/build_dir/
	NATTEN_CUDA_ARCH="${CUDA_ARCH}" \
	NATTEN_N_WORKERS="${WORKERS}" \
	NATTEN_AUTOGEN_POLICY="${AUTOGEN_POLICY}" \
	NATTEN_VERBOSE="${VERBOSE}" \
	NATTEN_BUILD_DIR="$(PWD)/build_dir/" \
	$(PIP) install --verbose --no-build-isolation -e . 2>&1 | tee install.out

install: 
	$(MAKE) fetch-submodules
	@echo "Installing NATTEN from source"
	mkdir -p $(PWD)/build_dir/
	NATTEN_CUDA_ARCH="${CUDA_ARCH}" \
	NATTEN_N_WORKERS="${WORKERS}" \
	NATTEN_AUTOGEN_POLICY="${AUTOGEN_POLICY}" \
	NATTEN_VERBOSE="${VERBOSE}" \
	NATTEN_BUILD_DIR="$(PWD)/build_dir/" \
	$(PIP) install --verbose --no-build-isolation . 2>&1 | tee install.out

test:
	NATTEN_LOG_LEVEL="CRITICAL" \
	NATTEN_RUN_EXTENDED_TESTS="${RUN_EXTENDED_TESTS}" \
	NATTEN_RUN_ADDITIONAL_KV_TESTS="${RUN_ADDITIONAL_KV_TESTS}" \
	NATTEN_RUN_FLEX_TESTS="${RUN_FLEX_TESTS}" \
	NATTEN_RAND_SWEEP_TESTS="${NUM_RAND_SWEEP_TESTS}" \
	PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
	CUBLAS_WORKSPACE_CONFIG=":4096:8" \
	$(PYTEST) -v -x ./tests

format:
	$(UFMT) format $(check_dirs)
	$(FLAKE8) $(check_dirs)
	$(MYPY) $(check_dirs)
	find csrc/include/ \
		-iname \*.h -o \
		-iname \*.cpp -o \
		-iname \*.cuh -o \
		-iname \*.cu -o \
		-iname \*.hpp -o \
		-iname \*.c -o \
		-iname \*.cxx | xargs \
		clang-format -i
	find csrc/src/ \
		-iname \*.h -o \
		-iname \*.cpp -o \
		-iname \*.cuh -o \
		-iname \*.cu -o \
		-iname \*.hpp -o \
		-iname \*.c -o \
		-iname \*.cxx | xargs \
		clang-format -i

serve-docs:
	$(MKDOCS) serve

build-docs:
	$(MKDOCS) build
