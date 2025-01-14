// Standard C++ headers
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <algorithm>

// CUDA headers
#include <cuda_runtime.h>
#include "../include/kernels.cu"


// Define the wavelet coefficients as floats
// Low coefficients
const std::vector<std::vector<float>> db_low = {
    {0.70710678f, 0.70710678f}, // db1
    {-0.12940952f, 0.22414387f, 0.83651630f, 0.48296291f}, // db2
    {0.03522629f, -0.08544127f, -0.13501102f, 0.45987750f, 0.80689151f, 0.33267055f}, // db3
    {-0.01059740f, 0.03288301f, 0.03084138f, -0.18703481f, -0.02798377f, 0.63088077f, 0.71484657f, 0.23037781f} // db4
};
// High coefficients
const std::vector<std::vector<float>> db_high = {
    {-0.70710678f, 0.70710678f}, // db1
    {-0.48296291f, 0.83651630f, -0.22414387f, -0.12940952f}, // db2
    {-0.33267055f, 0.80689151f, -0.45987750f, -0.13501102f, 0.08544127f, 0.03522629f}, // db3
    {-0.23037781f, 0.71484657f, -0.63088077f, -0.02798377f, 0.18703481f, 0.03084138f, -0.03288301f, -0.01059740f} // db4
};

void toGPU(std::vector<std::vector<std::vector<float>>> volume, size_t db_num, size_t depth, size_t rows, size_t cols, size_t&filter_size, float*& d_volume) {

    // Select the coefficients based on db_num
    std::vector<float> low_coeff = db_low[db_num - 1];
    std::vector<float> high_coeff = db_high[db_num - 1];

    // Calculate the filter size
    filter_size = low_coeff.size();
    if (filter_size > MAX_FILTER_SIZE) {
        throw std::runtime_error("Filter size exceeds constant memory capacity");
    }
    
    // Copy coefficients to constant memory
     cudaError_t err = cudaMemcpyToSymbol(lcf, low_coeff.data(), filter_size * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy low coefficients to constant memory: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpyToSymbol(hcf, high_coeff.data(), filter_size * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy high coefficients to constant memory: " + std::string(cudaGetErrorString(err)));
    }

    // Use of constant memory for coefficients is more efficient in the copy of the coefficients to the GPU since we do not need to allocate memory on the GPU and copy the coefficients to the GPU memory. But rather we can copy the coefficients directly to the constant memory on the GPU. This is more efficient since the constant memory is cached and has a lower latency compared to global memory.

    // Flatten the 3D volume into a 1D vector (row-major order)
    std::vector<float> flat_volume(depth * rows * cols);
    for (size_t d = 0; d < depth; ++d) {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                flat_volume[d * rows * cols + r * cols + c] = volume[d][r][c];
            }
        }
    }

    // Allocate memory on the GPU for the volume
    err = cudaMalloc(&d_volume, flat_volume.size() * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for volume: " + std::string(cudaGetErrorString(err)));
    }

    // Copy the flattened volume to the GPU
    err = cudaMemcpy(d_volume, flat_volume.data(), flat_volume.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_volume);
        throw std::runtime_error("Failed to copy volume data to GPU: " + std::string(cudaGetErrorString(err)));
    }

    // Clear the CPU memory after copying to GPU
    flat_volume.clear();
    volume.clear();

    // No need to synchronize here as the cudaMemcpytoSymbol, cudaMemcpy and cudaMalloc are synchronous block calls by default
}

std::vector<std::vector<std::vector<float>>> volCPU(float* d_volume, size_t depth, size_t rows, size_t cols) {
    // Allocate memory for the 3D volume on the CPU
    std::vector<std::vector<std::vector<float>>> volume(depth, std::vector<std::vector<float>>(rows, std::vector<float>(cols)));

    // Copy the data from the GPU to the CPU
    std::vector<float> flat_volume(depth * rows * cols);
    cudaError_t err = cudaMemcpy(flat_volume.data(), d_volume, flat_volume.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data from GPU to CPU: " + std::string(cudaGetErrorString(err)));
    }

    // Unflatten the 1D vector into a 3D volume
    for (size_t d = 0; d < depth; ++d) {
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                volume[d][r][c] = flat_volume[(d * rows * cols) + (r * cols) + c];
            }
        }
    }

    // Free the allocated GPU memory
    err = cudaFree(d_volume);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to free GPU memory: " + std::string(cudaGetErrorString(err)));
    }

    return volume;
}

void dwt_3d(float* d_volume, size_t depth, size_t rows, size_t cols, size_t filter_size) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    float* d_data1 = d_volume;
    float* d_data2 = nullptr;
    cudaError_t err = cudaMalloc(&d_data2, depth * rows * cols * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for temporary volume: " << cudaGetErrorString(err) << std::endl;
    }

    dim3 blockDim(16, 16, 4); // Updated block dimensions

    // Update grid dimensions based on the new blockDim
    dim3 row_grid((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y, (depth + blockDim.z - 1) / blockDim.z);
    dim3 col_grid((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, (depth + blockDim.z - 1) / blockDim.z);
    dim3 depth_grid((depth + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y, (rows + blockDim.z - 1) / blockDim.z);
    dim3 map_grid((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, (depth + blockDim.z - 1) / blockDim.z);

    // Perform convolution along the first dimension
    row_kernel<<<row_grid, blockDim, filter_size * sizeof(float) * 2>>>(d_data1, d_data2, filter_size, depth, rows, cols);
    // Synchronize the device
    cudaDeviceSynchronize();

    // Perform convolution along the second dimension
    col_kernel<<<col_grid, blockDim, filter_size * sizeof(float) * 2>>>(d_data2, d_data1,filter_size, depth, rows, cols);
    // Synchronize the device  
    cudaDeviceSynchronize();

    // Perform convolution along the third dimension
    depth_kernel<<<depth_grid, blockDim, filter_size * sizeof(float) * 2>>>(d_data1, d_data2, filter_size, depth, rows, cols);
    // Synchronize the device
    cudaDeviceSynchronize();

    // Copy the transformed data and map it back to the original volume
    map_kernel<<<map_grid, blockDim>>>(d_volume, d_data2, depth, rows, cols);
    // Synchronize the device
    cudaDeviceSynchronize();

    // Free the temporary volume
    err = cudaFree(d_data2);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free GPU memory for temporary volume: " << cudaGetErrorString(err) << std::endl;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cerr << "Time taken for DWT: " << milliseconds << "ms" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
