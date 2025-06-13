.PHONY: sdist clean uninstall install-deps install test style quality

DEV=
CUDA_ARCH=${NATTEN_CUDA_ARCH}
WORKERS=${NATTEN_N_WORKERS}
VERBOSE=${NATTEN_VERBOSE}
RUN_EXTENDED_TESTS=${NATTEN_RUN_EXTENDED_TESTS}

RELEASE=

check_dirs := src/natten tests scripts setup.py

all: clean uninstall fetch-submodules install

full: clean uninstall install-deps fetch-submodules install

dev: clean uninstall fetch-submodules install-dev

install-deps:
	@echo "Recognized python bin:"
	@which python3
	pip install -r requirements.txt

install-release-deps:
	pip3 install twine

fetch-submodules:
	@echo "Fetching all third party submodules"
	git submodule update --init --recursive

build-wheels:
	./dev/packaging/build_all_wheels_parallel.sh

build-dist:
	@echo "Generating source dist"
	python3 setup.py sdist

release:
	twine upload --repository ${RELEASE} dist/*

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
	pip uninstall -y natten

install-dev: 
	@echo "Installing NATTEN from source; development mode (editable)"
	NATTEN_CUDA_ARCH="${CUDA_ARCH}" \
	NATTEN_N_WORKERS="${WORKERS}" \
	NATTEN_VERBOSE="${VERBOSE}" \
	NATTEN_BUILD_DIR="$(PWD)/build_dir/" \
	pip3 install --verbose --no-build-isolation -e . 2>&1 | tee install.out

install: 
	@echo "Installing NATTEN from source"
	NATTEN_CUDA_ARCH="${CUDA_ARCH}" \
	NATTEN_N_WORKERS="${WORKERS}" \
	NATTEN_VERBOSE="${VERBOSE}" \
	NATTEN_BUILD_DIR="$(PWD)/build_dir/" \
	pip3 install --verbose --no-build-isolation . 2>&1 | tee install.out

test:
	NATTEN_LOG_LEVEL="CRITICAL" \
	NATTEN_RUN_EXTENDED_TESTS="${RUN_EXTENDED_TESTS}" \
	PYTORCH_NO_CUDA_MEMORY_CACHING=1 \
	CUBLAS_WORKSPACE_CONFIG=":4096:8" \
	pytest -v -x ./tests

style:
	ufmt format $(check_dirs)
	flake8 $(check_dirs)
	mypy $(check_dirs)
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
	mkdocs serve

build-docs:
	mkdocs build
