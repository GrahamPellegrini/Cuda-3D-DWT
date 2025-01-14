#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda_runtime.h>
#include <stddef.h> // For size_t

// Define the max filter size (currently db4)
const int MAX_FILTER_SIZE = 8;

// Declare constant memory for coefficients (without __constant__)
__constant__ extern float lcf[MAX_FILTER_SIZE];
__constant__ extern float hcf[MAX_FILTER_SIZE];

// Declare kernel functions
__global__ void row_kernel(float* data, float* temp, size_t filter_size, size_t depth_limit, size_t row_limit, size_t col_limit);
__global__ void col_kernel(float* data, float* temp, size_t filter_size, size_t depth_limit, size_t row_limit, size_t col_limit);
__global__ void depth_kernel(float* data, float* temp, size_t filter_size, size_t depth_limit, size_t row_limit, size_t col_limit);
__global__ void map_kernel(float* d_volume, float* d_transformed, size_t depth, size_t rows, size_t cols);

#endif // KERNELS_CUH
