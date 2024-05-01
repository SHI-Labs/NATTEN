# Copyright (c) 2022-2024 Ali Hassani.

cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
if(${NATTEN_WITH_CUDA})
project(natten LANGUAGES CXX CUDA)
else()
project(natten LANGUAGES CXX)
endif()

if(${NATTEN_WITH_CUDA})
find_package(CUDA 11.0 REQUIRED)
message("Building NATTEN with CUDA.")
endif()
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
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)

## Torch includes
list(APPEND COMMON_HEADER_DIRS ${TORCH_INCLUDE_DIRS})

# CUTLASS
if(${NATTEN_WITH_CUDA})
  if(${NATTEN_WITH_CUTLASS})
    message("Building NATTEN with CUTLASS-based kernels.")
    list(APPEND COMMON_HEADER_DIRS ../third_party/cutlass/include)
  endif()
endif()

# Compiler flags
if(${NATTEN_IS_MAC})
  message("Building NATTEN on MacOS.")
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
  set(CMAKE_SHARED_LIBRARY_SUFFIX_C ".so")
  set(CMAKE_SHARED_LIBRARY_SUFFIX_CXX ".so")
elseif(${NATTEN_IS_WINDOWS})
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
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-allow-expensive-optimizations=true")
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

if(${NATTEN_WITH_CUDA})
  message("CUDA compiler: " ${CMAKE_CUDA_COMPILER})
  message("NVCC executable: " ${CUDA_NVCC_EXECUTABLE})
endif()

if(${IS_LIBTORCH_BUILT_WITH_CXX11_ABI})
  message("Building with -D_GLIBCXX_USE_CXX11_ABI=1")
  add_definitions("-D_GLIBCXX_USE_CXX11_ABI=1")
else()
  message("Building with -D_GLIBCXX_USE_CXX11_ABI=0")
  add_definitions("-D_GLIBCXX_USE_CXX11_ABI=0")
endif()

if(${NATTEN_WITH_CUDA})
  # CUDA flags
  add_definitions("-D__CUDA_NO_HALF_OPERATORS__")
  add_definitions("-D__CUDA_NO_HALF_CONVERSIONS__")
  add_definitions("-D__CUDA_NO_BFLOAT16_CONVERSIONS__")
  add_definitions("-D__CUDA_NO_HALF2_OPERATORS__")
endif()

# Torch/pybind flags
# Pybind specifically needs these to recognize the module being initialized.
add_definitions("-DTORCH_API_INCLUDE_EXTENSION_H")
add_definitions("-DTORCH_EXTENSION_NAME=libnatten")

if(${NATTEN_WITH_AVX})
  add_definitions("-DAVX_INT")
endif()

if(${NATTEN_WITH_CUDA})
  add_definitions("-DWITH_CUDA")
  add_definitions("-DNATTEN_WITH_CUDA")
  if(${NATTEN_WITH_CUTLASS})
    add_definitions("-DNATTEN_WITH_CUTLASS")
  endif()
endif()

# Add local headers
list(APPEND COMMON_HEADER_DIRS ./include)
list(APPEND COMMON_HEADER_DIRS ./autogen/include)

# Add source files
file(GLOB MAIN_SOURCE  ./natten.cpp)
file(GLOB MISC_SOURCES  ./src/*.cpp ./src/*.cu)
file(GLOB TORCH_INTERFACE ./src/pytorch/*.cpp)
file(GLOB TORCH_CPU_SOURCES  ./src/pytorch/cpu/*.cpp)
file(GLOB AUTOGEN_CPU  ./autogen/src/cpu/naive/*.cpp)
if(${NATTEN_WITH_CUDA})
  file(GLOB MISC_SOURCES_CUDA  ./src/*.cpp ./src/*.cu)
  file(GLOB AUTOGEN_CUDA ./autogen/src/cuda/naive/*.cu)
  file(GLOB TORCH_CUDA_SOURCES ./src/pytorch/cuda/*.cu)
  if(${NATTEN_WITH_CUTLASS})
    file(GLOB AUTOGEN_CUDA_GEMM_SM70 ./autogen/src/cuda/gemm/1d/sm70/*.cu ./autogen/src/cuda/gemm/2d/sm70/*.cu)
    file(GLOB AUTOGEN_CUDA_GEMM_SM75 ./autogen/src/cuda/gemm/1d/sm75/*.cu ./autogen/src/cuda/gemm/2d/sm75/*.cu)
    file(GLOB AUTOGEN_CUDA_GEMM_SM80 ./autogen/src/cuda/gemm/1d/sm80/*.cu ./autogen/src/cuda/gemm/2d/sm80/*.cu)
    file(GLOB AUTOGEN_FNA ./autogen/src/cuda/fna/*.cu ./src/cuda/fna/*.cu)
  endif()
endif()

file(GLOB ALL_SOURCES 
  ${TORCH_INTERFACE} 
  ${TORCH_CPU_SOURCES} 
  ${TORCH_CUDA_SOURCES} 
  ${AUTOGEN_CPU} 
  ${AUTOGEN_CUDA} 
  ${AUTOGEN_CUDA_GEMM_SM70} 
  ${AUTOGEN_CUDA_GEMM_SM75} 
  ${AUTOGEN_CUDA_GEMM_SM80} 
  ${AUTOGEN_FNA} 
  ${MISC_SOURCES} 
  ${MISC_SOURCES_CUDA} 
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
if(${NATTEN_WITH_CUDA})
  message("Building NATTEN for the following architectures: ${NATTEN_CUDA_ARCH_LIST}")
  set_target_properties(natten PROPERTIES CUDA_ARCHITECTURES "${NATTEN_CUDA_ARCH_LIST}")
endif()

if(${NATTEN_IS_MAC})
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-undefined,dynamic_lookup")
endif()

if(${NATTEN_WITH_CUDA})
  target_link_libraries(natten PUBLIC c10 torch torch_cpu torch_python cudart c10_cuda torch_cuda)
else()
  target_link_libraries(natten PUBLIC c10 torch torch_cpu torch_python)
endif()
