#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <iostream>

// Define the max filter size (currently db4)
const int MAX_FILTER_SIZE = 8;
// Contant Memory declaration for coefficients
__constant__ float d_coeff[2*MAX_FILTER_SIZE];

__global__ void row_kernel(float* data, float* temp, size_t filter_size, size_t depth, size_t rows, size_t cols) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t d = blockIdx.z * blockDim.z + threadIdx.z;
    size_t c = blockIdx.y * blockDim.y + threadIdx.y;


    if (d < depth && c < cols && i < rows / 2) {
        float sum_low = 0.0f;
        float sum_high = 0.0f;

        for (size_t j = 0; j < filter_size; ++j) {
            size_t idx = (2 * i + j) % rows;
            size_t d_idx = d * rows * cols + idx * cols + c;

            sum_low += d_coeff[j] * data[d_idx];
            sum_high += d_coeff[j + filter_size] * data[d_idx];
        }

        size_t low_idx = d * rows * cols + i * cols + c;
        size_t high_idx = d * rows * cols + (i + rows / 2) * cols + c;

        temp[low_idx] = sum_low;
        temp[high_idx] = sum_high;
    }
}

__global__ void col_kernel(float* data, float* temp, size_t filter_size, size_t depth, size_t rows, size_t cols) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t d = blockIdx.z * blockDim.z + threadIdx.z;
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;


    if (d < depth && r < rows && i < cols / 2) {
        float sum_low = 0.0f;
        float sum_high = 0.0f;

        for (size_t j = 0; j < filter_size; ++j) {
            size_t idx = (2 * i + j) % cols;
            size_t d_idx = d * rows * cols + r * cols + idx;

            sum_low += d_coeff[j] * data[d_idx];
            sum_high += d_coeff[j + filter_size] * data[d_idx];
        }

        size_t low_idx = d * rows * cols + r * cols + i;
        size_t high_idx = d * rows * cols + r * cols + (i + cols / 2);

        temp[low_idx] = sum_low;
        temp[high_idx] = sum_high;
    }
}

__global__ void depth_kernel(float* data, float* temp, size_t filter_size, size_t depth, size_t rows, size_t cols) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t r = blockIdx.z * blockDim.z + threadIdx.z;
    size_t c = blockIdx.y * blockDim.y + threadIdx.y;


    if (r < rows && c < cols && i < depth / 2) {
        float sum_low = 0.0f;
        float sum_high = 0.0f;

        for (size_t j = 0; j < filter_size; ++j) {
            size_t idx = (2 * i + j) % depth;
            size_t d_idx = idx * rows * cols + r * cols + c;

            sum_low += d_coeff[j] * data[d_idx];
            sum_high += d_coeff[j + filter_size] * data[d_idx];
        }

        size_t low_idx = i * rows * cols + r * cols + c;
        size_t high_idx = (i + depth / 2) * rows * cols + r * cols + c;

        temp[low_idx] = sum_low;
        temp[high_idx] = sum_high;
    }
}

__global__ void map_kernel(float* d_volume, float* d_transformed, size_t depth, size_t rows, size_t cols) {
    int d = blockIdx.z * blockDim.z + threadIdx.z;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (d < depth && r < rows && c < cols) {
        d_volume[(d * rows * cols) + (r * cols) + c] = d_transformed[(d * rows * cols) + (r * cols) + c];
    }
}

#endif // KERNELS_CUH