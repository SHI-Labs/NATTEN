/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdlib.h>
namespace natten {
namespace cuda {
namespace flash {

#define FLASH_CHECK_CUDA(call)                        \
    do {                                                                                                  \
        cudaError_t status_ = call;                                                                       \
        if (status_ != cudaSuccess) {                                                                     \
            fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
            exit(1);                                                                                      \
        }                                                                                                 \
    } while(0)

#define FLASH_CHECK_CUDA_KERNEL_LAUNCH() FLASH_CHECK_CUDA(cudaGetLastError())

} // namespace flash
} // namespace cuda
} // namespace natten
