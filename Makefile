.PHONY: sdist clean uninstall install-deps install test style quality

CUDA_ARCH=
WORKERS=
VERBOSE=

RELEASE=

check_dirs := src/natten tests tools scripts setup.py

all: clean uninstall fetch-submodules install

full: clean uninstall install-deps fetch-submodules install

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
	rm -rf build/ 
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

install: 
	@echo "Installing NATTEN from source"
	NATTEN_CUDA_ARCH="${CUDA_ARCH}" NATTEN_N_WORKERS="${WORKERS}" NATTEN_VERBOSE="${VERBOSE}" pip install -v -e . 2>&1 | tee install.out

test:
	pytest -v -x ./tests

style:
	ufmt format $(check_dirs)
	flake8 $(check_dirs)
	mypy $(check_dirs)
	clang-format -i csrc/include/**/*.*
	clang-format -i csrc/src/**/*.*
