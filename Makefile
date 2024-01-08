.PHONY: sdist clean uninstall install-deps install test style quality

CUDA_ARCH=
WORKERS=
VERBOSE=

check_dirs := src/natten tests tools

all: clean uninstall fetch-submodules install

full: clean uninstall install-deps fetch-submodules install

fetch-submodules:
	@echo "Fetching all third party submodules"
	git submodule update --init --recursive

sdist:
	@echo "Generating source dist"
	python3 setup.py sdist

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

install-deps:
	@echo "Recognized python bin:"
	@which python3
	pip install -r requirements.txt

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
