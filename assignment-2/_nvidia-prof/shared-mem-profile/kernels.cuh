#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <iostream>

__global__ void dwt_col(float* d_volume, float* transformed, float* low_coeff, float* high_coeff, size_t filter_size, size_t depth, size_t row, size_t col) {
    size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    size_t c = blockIdx.z * blockDim.z + threadIdx.z;

    if (d < depth && r < row && c < (col / 2)) {
        float low_sum = 0.0f;
        float high_sum = 0.0f;

        for (size_t i = 0; i < filter_size; i++) {
            size_t data_index = d * row * col + r * col + (c * 2 + i);
            low_sum += d_volume[data_index] * low_coeff[i];
            high_sum += d_volume[data_index] * high_coeff[i];
        }
        size_t low_index = d * row * col + r * col + c;
        size_t high_index = d * row * col + r * col + (c + (col / 2));

        transformed[low_index] = low_sum;
        transformed[high_index] = high_sum;
    }
}

__global__ void dwt_row(float* d_volume, float* transformed, float* low_coeff, float* high_coeff, size_t filter_size, size_t depth, size_t row, size_t col) {
    size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    size_t c = blockIdx.z * blockDim.z + threadIdx.z;

    if (d < depth && r < (row / 2) && c < col) {
        float low_sum = 0.0f;
        float high_sum = 0.0f;

        for (size_t i = 0; i < filter_size; i++) {
            size_t data_index = d * row * col + (r * 2 + i) * col + c;
            low_sum += d_volume[data_index] * low_coeff[i];
            high_sum += d_volume[data_index] * high_coeff[i];
        }
        size_t low_index = d * row * col + r * col + c;
        size_t high_index = d * row * col + (r + (row / 2)) * col + c;

        transformed[low_index] = low_sum;
        transformed[high_index] = high_sum;
    }
}

__global__ void dwt_depth(float* d_volume, float* transformed, float* low_coeff, float* high_coeff, size_t filter_size, size_t depth, size_t row, size_t col) {
    size_t d = blockIdx.x * blockDim.x + threadIdx.x;
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    size_t c = blockIdx.z * blockDim.z + threadIdx.z;

    if (d < (depth / 2) && r < row && c < col) {
        float low_sum = 0.0f;
        float high_sum = 0.0f;

        for (size_t i = 0; i < filter_size; i++) {
            size_t data_index = (d * 2 + i) * row * col + r * col + c;
            low_sum += d_volume[data_index] * low_coeff[i];
            high_sum += d_volume[data_index] * high_coeff[i];
        }
        size_t low_index = d * row * col + r * col + c;
        size_t high_index = (d + (depth / 2)) * row * col + r * col + c;

        transformed[low_index] = low_sum;
        transformed[high_index] = high_sum;
    }
}



__global__ void row_kernel(float* data, float* temp, const float* lpf, const float* hpf, size_t filter_size, size_t depth_limit, size_t row_limit, size_t col_limit) {
    extern __shared__ float shared_mem[];
    float* shared_lpf = shared_mem;
    float* shared_hpf = shared_mem + filter_size;

    // Load filters into shared memory
    if (threadIdx.x < filter_size) {
        shared_lpf[threadIdx.x] = lpf[threadIdx.x];
        shared_hpf[threadIdx.x] = hpf[threadIdx.x];
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
            sum_low += shared_lpf[j] * input_val;
            sum_high += shared_hpf[j] * input_val;
        }

        size_t low_index = d * row_limit * col_limit + i * col_limit + c;
        size_t high_index = d * row_limit * col_limit + (i + row_limit / 2) * col_limit + c;

        temp[low_index] = sum_low;
        temp[high_index] = sum_high;
    }
}

__global__ void col_kernel(float* data, float* temp, const float* lpf, const float* hpf, size_t filter_size, size_t depth_limit, size_t row_limit, size_t col_limit) {
    extern __shared__ float shared_mem[];
    float* shared_lpf = shared_mem;
    float* shared_hpf = shared_mem + filter_size;

    // Load filters into shared memory
    if (threadIdx.x < filter_size) {
        shared_lpf[threadIdx.x] = lpf[threadIdx.x];
        shared_hpf[threadIdx.x] = hpf[threadIdx.x];
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
            sum_low += shared_lpf[j] * input_val;
            sum_high += shared_hpf[j] * input_val;
        }

        size_t low_index = d * row_limit * col_limit + r * col_limit + i;
        size_t high_index = d * row_limit * col_limit + r * col_limit + (i + col_limit / 2);

        temp[low_index] = sum_low;
        temp[high_index] = sum_high;
    }
}

__global__ void depth_kernel(float* data, float* temp, const float* lpf, const float* hpf, size_t filter_size, size_t depth_limit, size_t row_limit, size_t col_limit) {
    extern __shared__ float shared_mem[];
    float* shared_lpf = shared_mem;
    float* shared_hpf = shared_mem + filter_size;

    // Load filters into shared memory
    if (threadIdx.x < filter_size) {
        shared_lpf[threadIdx.x] = lpf[threadIdx.x];
        shared_hpf[threadIdx.x] = hpf[threadIdx.x];
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
            sum_low += shared_lpf[j] * input_val;
            sum_high += shared_hpf[j] * input_val;
        }

        size_t low_index = i * row_limit * col_limit + r * col_limit + c;
        size_t high_index = (i + depth_limit / 2) * row_limit * col_limit + r * col_limit + c;

        temp[low_index] = sum_low;
        temp[high_index] = sum_high;
    }
}

__global__ void map_kernel(float* d_transformed, float* d_final, size_t depth, size_t rows, size_t cols, size_t orig_depth, size_t orig_rows, size_t orig_cols) {
    size_t d = blockIdx.z * blockDim.z + threadIdx.z;
    size_t r = blockIdx.y * blockDim.y + threadIdx.y;
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;

    if (d < depth && r < rows && c < cols) {
        d_final[d * orig_rows * orig_cols + r * orig_cols + c] = d_transformed[d * rows * cols + r * cols + c];
    }
}

#endif // KERNELS_CUH