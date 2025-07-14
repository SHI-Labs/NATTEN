# Copyright (c) 2022-2025 Ali Hassani.

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(natten LANGUAGES CXX CUDA)

find_package(CUDAToolkit 12.0 REQUIRED)
set(CXX_STD "17" CACHE STRING "C++ standard")


if(NOT DEFINED PYTHON_PATH)
    # Find Python
    message("Python path not specified; looking up local python.")
    find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
    set(PYTHON_PATH "python" CACHE STRING "Python path")
endif()
message("Python: " ${PYTHON_PATH})

## Python includes 
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; from distutils import sysconfig; print(sysconfig.get_python_inc());"
                RESULT_VARIABLE _PYTHON_SUCCESS
                OUTPUT_VARIABLE PY_INCLUDE_DIR)
if (NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Python launch failed.")
endif()
list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})


# Find torch
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import torch; print(torch.__version__,end='');"
                RESULT_VARIABLE _PYTHON_SUCCESS
                OUTPUT_VARIABLE TORCH_VERSION)
if (NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Python launch failed.")
endif()
## Check torch version
if (TORCH_VERSION VERSION_LESS "1.8.0")
    message(FATAL_ERROR "PyTorch >= 1.8.0 is required.")
endif()

## Torch CMake
execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
                RESULT_VARIABLE _PYTHON_SUCCESS
                OUTPUT_VARIABLE TORCH_DIR)
if (NOT _PYTHON_SUCCESS MATCHES 0)
    message(FATAL_ERROR "Torch config Error.")
endif()

# I'm going to stop finding torch through cmake because of caffe.
# Sorry -- but I'm done wasting time on a dependency that shouldn't be shipped
# with CPU binaries!! Even on CUDA it breaks everything so frequently, and all
# I really need to find is the include paths, and let the linker take it from there.
#list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
#find_package(Torch REQUIRED)
set(TORCH_INCLUDE_DIRS "${TORCH_DIR}/include" "${TORCH_DIR}/include/torch/csrc/api/include")
message("Torch dir: " ${TORCH_DIR})
message("Torch include dir: " ${TORCH_INCLUDE_DIRS})

# CUTLASS
list(APPEND COMMON_HEADER_DIRS ../third_party/cutlass/include)

# Compiler flags
if(${NATTEN_IS_WINDOWS})
  message("Building NATTEN on Windows.")
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".pyd")
  set(CMAKE_SHARED_LIBRARY_SUFFIX_C ".pyd")
  set(CMAKE_SHARED_LIBRARY_SUFFIX_CXX ".pyd")
  add_definitions("-DNATTEN_WINDOWS")
endif()

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17")
set(CMAKE_CUDA_FLAGS "-Xcompiler -Wall -ldl")

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "-O0 -G -Xcompiler -Wall")

set(CMAKE_CXX_STANDARD "${CXX_STD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wconversion")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fno-strict-aliasing")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ftemplate-backtrace-limit=0")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
# Huuuuuge regression from device-side asserts (funny it happened to Ampere FNA and now Blackwell
# FNA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNDEBUG")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++${CXX_STD}")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler -O3")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math")


if(${NATTEN_IS_WINDOWS})
  # Copied from xformers
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /MP /Zc:lambda /Zc:preprocessor")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /Zc:lambda -Xcompiler /Zc:preprocessor")
  set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /MP /Zc:lambda /Zc:preprocessor")
  set(CMAKE_CUDA_FLAGS_DEBUG "-Xcompiler /Zc:lambda -Xcompiler /Zc:preprocessor")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MP /Zc:lambda /Zc:preprocessor")
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler /Zc:lambda -Xcompiler /Zc:preprocessor")

  # TODO: MSVC can't build without /bigobj since FNA-backward
  # Those lambda expressions we use for handling memory planning
  # through torch appear to push the object size past its limit
  # on Windows. See csrc/src/pytorch/cuda/na1d.cu for more
  # (under na1d_forward/na1d_backward).
  set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -Xcompiler /bigobj")
endif()

message("CUDA compiler: " ${CMAKE_CUDA_COMPILER})
message("NVCC executable: " ${CUDA_NVCC_EXECUTABLE})

if(${IS_LIBTORCH_BUILT_WITH_CXX11_ABI})
  message("Building with -D_GLIBCXX_USE_CXX11_ABI=1")
  add_definitions("-D_GLIBCXX_USE_CXX11_ABI=1")
else()
  message("Building with -D_GLIBCXX_USE_CXX11_ABI=0")
  add_definitions("-D_GLIBCXX_USE_CXX11_ABI=0")
endif()

# CUDA flags
#add_definitions("-D__CUDA_NO_HALF_OPERATORS__")
#add_definitions("-D__CUDA_NO_HALF_CONVERSIONS__")
#add_definitions("-D__CUDA_NO_BFLOAT16_CONVERSIONS__")
#add_definitions("-D__CUDA_NO_HALF2_OPERATORS__")

# Torch/pybind flags
# Pybind specifically needs these to recognize the module being initialized.
add_definitions("-DTORCH_API_INCLUDE_EXTENSION_H")
add_definitions("-DTORCH_EXTENSION_NAME=libnatten")


# TODO: do we even use `WITH_CUDA` anymore?
add_definitions("-DWITH_CUDA")
# TODO: does it even make sense to have this anymore? Everything in libnatten is with CUTLASS.
add_definitions("-DNATTEN_WITH_CUTLASS")
add_definitions("-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1")

if(${NATTEN_WITH_HOPPER_FNA})
  add_definitions("-DNATTEN_WITH_HOPPER_FNA=1")
endif()

if(${NATTEN_WITH_BLACKWELL_FNA})
  add_definitions("-DNATTEN_WITH_BLACKWELL_FNA=1")
endif()

# Add local headers
list(APPEND COMMON_HEADER_DIRS ./include)
list(APPEND COMMON_HEADER_DIRS ./autogen/include)

# Add source files
file(GLOB MAIN_SOURCE  ./natten.cpp)
file(GLOB TORCH_APIS  ./src/*.cpp ./src/*.cu)
file(GLOB AUTOGEN_REFERENCE ./autogen/src/cuda/reference/*.cu)
file(GLOB AUTOGEN_FNA ./autogen/src/cuda/fna/*.cu ./src/cuda/fna/*.cu)
file(GLOB AUTOGEN_FMHA ./autogen/src/cuda/fmha/*.cu ./src/cuda/fmha/*.cu)
if(${NATTEN_WITH_HOPPER_FNA})
  file(GLOB AUTOGEN_HOPPER_FNA ./autogen/src/cuda/hopper_fna/*.cu ./autogen/src/cuda/hopper_fna_bwd/*.cu)
  file(GLOB AUTOGEN_HOPPER_FMHA ./autogen/src/cuda/hopper_fmha/*.cu ./autogen/src/cuda/hopper_fmha_bwd/*.cu)
endif()
if(${NATTEN_WITH_BLACKWELL_FNA})
  file(GLOB AUTOGEN_BLACKWELL_FNA ./autogen/src/cuda/blackwell_fna/*.cu ./autogen/src/cuda/blackwell_fna_bwd/*.cu)
  file(GLOB AUTOGEN_BLACKWELL_FMHA ./autogen/src/cuda/blackwell_fmha/*.cu ./autogen/src/cuda/blackwell_fmha_bwd/*.cu)
endif()

file(GLOB ALL_SOURCES 
  ${AUTOGEN_REFERENCE} 
  ${AUTOGEN_FNA} 
  ${AUTOGEN_FMHA} 
  ${AUTOGEN_BLACKWELL_FNA} 
  ${AUTOGEN_BLACKWELL_FMHA} 
  ${AUTOGEN_HOPPER_FNA} 
  ${AUTOGEN_HOPPER_FMHA} 
  ${TORCH_APIS} 
  ${MAIN_SOURCE} 
  )

# Add headers
include_directories(${COMMON_HEADER_DIRS})

# Find torch lib dir so we can link with libtorch
link_directories("${TORCH_DIR}/lib/")

if(NATTEN_IS_WINDOWS)
  # Windows builds require linking with python*.lib
  link_directories("${PY_LIB_DIR}")
endif()

add_library(natten SHARED ${ALL_SOURCES})

set_target_properties(natten PROPERTIES PREFIX "" OUTPUT_NAME ${OUTPUT_FILE_NAME})
set_target_properties(natten PROPERTIES LINKER_LANGUAGE CXX)
message("Building NATTEN for the following architectures: ${NATTEN_CUDA_ARCH_LIST}")
set_target_properties(natten PROPERTIES CUDA_ARCHITECTURES "${NATTEN_CUDA_ARCH_LIST}")

target_include_directories(natten SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(natten PUBLIC c10 torch torch_cpu torch_python cudart c10_cuda torch_cuda)
