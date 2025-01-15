#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <iostream>

__global__ void row_kernel(float* data, float* temp, float* d_coeff, size_t filter_size, size_t depth_limit, size_t row_limit, size_t col_limit) {
    extern __shared__ float shared_mem[];
    float* shared_coeff = shared_mem;

    // Load filters into shared memory
    if (threadIdx.x < filter_size) {
        shared_coeff[threadIdx.x] = d_coeff[threadIdx.x];
    }
    __syncthreads();

    size_t d = blockIdx.z * blockDim.z + threadIdx.z;
    size_t c = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (d < depth_limit && c < col_limit && i < row_limit / 2) {
        float sum_low = 0.0f;
        float sum_high = 0.0f;

        // Perform convolution
        for (size_t j = 0; j < filter_size; ++j) {
            size_t index = (2 * i + j) % row_limit;
            size_t data_index = d * row_limit * col_limit + index * col_limit + c;

            float input_val = data[data_index];
            sum_low += shared_coeff[j] * input_val;
            sum_high += shared_coeff[j + filter_size] * input_val;
        }

        size_t low_index = d * row_limit * col_limit + i * col_limit + c;
        size_t high_index = d * row_limit * col_limit + (i + row_limit / 2) * col_limit + c;

        temp[low_index] = sum_low;
        temp[high_index] = sum_high;
    }
}

__global__ void col_kernel(float* data, float* temp, float* d_coeff, size_t filter_size, size_t depth_limit, size_t row_limit, size_t col_limit) {
    extern __shared__ float shared_mem[];
    float* shared_coeff = shared_mem;

    // Load filters into shared memory
    if (threadIdx.x < filter_size) {
        shared_coeff[threadIdx.x] = d_coeff[threadIdx.x];
    }
    __syncthreads();

    size_t d = blockIdx.z * blockDim.z + threadIdx.z;
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (d < depth_limit && r < row_limit && i < col_limit / 2) {
        float sum_low = 0.0f;
        float sum_high = 0.0f;

        // Perform convolution
        for (size_t j = 0; j < filter_size; ++j) {
            size_t index = (2 * i + j) % col_limit;
            size_t data_index = d * row_limit * col_limit + r * col_limit + index;

            float input_val = data[data_index];
            sum_low += shared_coeff[j] * input_val;
            sum_high += shared_coeff[j + filter_size] * input_val;
        }

        size_t low_index = d * row_limit * col_limit + r * col_limit + i;
        size_t high_index = d * row_limit * col_limit + r * col_limit + (i + col_limit / 2);

        temp[low_index] = sum_low;
        temp[high_index] = sum_high;
    }
}

__global__ void depth_kernel(float* data, float* temp, float* d_coeff, size_t filter_size, size_t depth_limit, size_t row_limit, size_t col_limit) {
    extern __shared__ float shared_mem[];
    float* shared_coeff = shared_mem;

    // Load filters into shared memory
    if (threadIdx.x < filter_size) {
        shared_coeff[threadIdx.x] = d_coeff[threadIdx.x];
    }
    __syncthreads();

    size_t r = blockIdx.z * blockDim.z + threadIdx.z;
    size_t c = blockIdx.y * blockDim.y + threadIdx.y;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < row_limit && c < col_limit && i < depth_limit / 2) {
        float sum_low = 0.0f;
        float sum_high = 0.0f;

        // Perform convolution
        for (size_t j = 0; j < filter_size; ++j) {
            size_t index = (2 * i + j) % depth_limit;
            size_t data_index = index * row_limit * col_limit + r * col_limit + c;

            float input_val = data[data_index];
            sum_low += shared_coeff[j] * input_val;
            sum_high += shared_coeff[j + filter_size] * input_val;
        }

        size_t low_index = i * row_limit * col_limit + r * col_limit + c;
        size_t high_index = (i + depth_limit / 2) * row_limit * col_limit + r * col_limit + c;

        temp[low_index] = sum_low;
        temp[high_index] = sum_high;
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