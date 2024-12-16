#ifndef LOADBIN_H
#define LOADBIN_H

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include "/usr/local/cuda/include/cuda_runtime.h"

// CUDA kernel to read data from the binary file
__global__ void readVolumeKernel(float* d_volume, float* d_data, int depth, int rows, int cols) {
    int d = blockIdx.z * blockDim.z + threadIdx.z;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (d < depth && r < rows && c < cols) {
        int index = d * rows * cols + r * cols + c;
        d_volume[index] = d_data[index];
    }
}

// Function to load a 3D volume from a binary file using CUDA
float* loadvolume(const std::string& filename, int& depth, int& rows, int& cols) {
    // Open the file
    std::ifstream file(filename, std::ios::binary);

    // Check if the file is open
    assert(file.is_open() && "File not found.");

    // Read the dimensions from the binary file to the variables
    file.read(reinterpret_cast<char*>(&depth), sizeof(depth));
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));

    // Allocate memory on the GPU
    float* d_volume;
    float* d_data;
    size_t size = depth * rows * cols * sizeof(float);
    cudaMalloc(&d_volume, size);
    cudaMalloc(&d_data, size);

    // Read the data from the binary file to the host memory
    std::vector<float> h_data(depth * rows * cols);
    file.read(reinterpret_cast<char*>(h_data.data()), size);

    // Copy the data from the host to the GPU
    cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);

    // Define the block and grid dimensions
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, (depth + blockDim.z - 1) / blockDim.z);

    // Launch the CUDA kernel
    readVolumeKernel<<<gridDim, blockDim>>>(d_volume, d_data, depth, rows, cols);

    // Free the temporary GPU memory
    cudaFree(d_data);

    // Check if the volume is empty
    assert(depth > 0 && rows > 0 && cols > 0 && "Invalid dimensions.");

    file.close();
    return d_volume;
}

#endif // LOADBIN_H