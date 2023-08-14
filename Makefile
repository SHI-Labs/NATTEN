.PHONY: sdist clean uninstall install-deps install test style quality

CUDA_ARCH=
WORKERS=

check_dirs := src tests

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
	NATTEN_CUDA_ARCH="${CUDA_ARCH}" NATTEN_N_WORKERS="${WORKERS}" pip install -v -e . 2>&1 | tee install.out

test:
	@echo "Running unit tests"
	python -m unittest discover -v -s ./tests

quality:
	@echo "Quality check"
	black --check --preview $(check_dirs)
	isort --check-only $(check_dirs)
	flake8 $(check_dirs)

style:
	black --preview $(check_dirs)
	isort $(check_dirs)
