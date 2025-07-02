# Copyright (c) 2022-2025 Ali Hassani.
#
# By default, this dockerfile builds on top of the May 2025
# NGC PyTorch image, and builds NATTEN for SM50-SM100 with
# 4 workers.
#
# You can customize those by using build args:
#
#     docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:25.04-py3 ...
#     docker build --build-arg NATTEN_N_WORKERS=128 --build-arg NATTEN_CUDA_ARCH="9.0;10.0" ...

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.05-py3

FROM $BASE_IMAGE

ARG NATTEN_VERSION=0.20.1
ARG NATTEN_N_WORKERS=4
ARG NATTEN_VERBOSE=1
ARG NATTEN_CUDA_ARCH="5.0;6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0"

RUN echo "Building NATTEN docker image, version $NATTEN_VERSION"
RUN echo "Number of workers: $NATTEN_N_WORKERS"
RUN echo "Verbose: $NATTEN_VERBOSE"
RUN echo "Target architectures: $NATTEN_CUDA_ARCH"

# Build from distribution
RUN pip3 install --verbose natten==${NATTEN_VERSION}

# Build from source (not recommended unless intentional)
# RUN mkdir /natten
#
# RUN cd /natten && \
#       git clone --recursive https://github.com/SHI-Labs/NATTEN
#
# RUN cd /natten/NATTEN && make
#
# RUN rmdir /natten
