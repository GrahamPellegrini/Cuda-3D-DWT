// cuda_utils.h
#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdexcept>
#include <cstdio>

#define CHECK_CUDA_ERROR(call)                                        \
    {                                                                 \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            throw std::runtime_error("CUDA error detected!");         \
        }                                                             \
    }

#endif // CUDA_UTILS_H
