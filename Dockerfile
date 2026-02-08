# Copyright (c) 2022 - 2026 Ali Hassani.
#
# By default, this dockerfile builds on top of the September 2025
# NGC PyTorch image (CTK 13.0), and builds NATTEN for SM75-SM120
# with 4 workers.
#
# You can customize those by using build args:
#
#     docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3 ...
#     docker build --build-arg NATTEN_N_WORKERS=128 --build-arg NATTEN_CUDA_ARCH="9.0;10.0" ...

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3

FROM $BASE_IMAGE

ARG NATTEN_VERSION=0.21.5
ARG NATTEN_VERBOSE=1
ARG NATTEN_CUDA_ARCH="7.5;8.0;8.6;8.9;9.0;10.0;10.3;12.0"

# choices are: 'default', 'coarse', 'fine'
# using coarse by default since the default number of
# workers is only 4.
ARG NATTEN_AUTOGEN_POLICY="coarse"
ARG NATTEN_N_WORKERS=4

RUN echo "Building NATTEN docker image, version $NATTEN_VERSION"
RUN echo "Number of workers: $NATTEN_N_WORKERS"
RUN echo "Verbose: $NATTEN_VERBOSE"
RUN echo "Target architectures: $NATTEN_CUDA_ARCH"

# Build from distribution
RUN pip3 install \
      --verbose \
      --no-deps \
      --no-build-isolation \
      natten==${NATTEN_VERSION}

# Build from source (not recommended unless you want to try unreleased features or custom branches)
# RUN mkdir /natten
#
# RUN cd /natten && \
#       git clone --recursive https://github.com/SHI-Labs/NATTEN
#
# RUN cd /natten/NATTEN && make
#
# RUN rmdir /natten
