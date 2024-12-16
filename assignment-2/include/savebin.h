#ifndef SAVEBIN_H
#define SAVEBIN_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cassert>
#include "/usr/local/cuda/include/cuda_runtime.h"

// CUDA error checking utility
#include "cuda_utils.h"

// Save a 3D volume to a binary file using CUDA
void savevolume(const float* d_volume, const std::string& filename, int depth, int rows, int cols) {
    // Open the file
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Write the dimensions to the binary file
    file.write(reinterpret_cast<const char*>(&depth), sizeof(depth));
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    // Calculate size of the data in bytes
    size_t size = depth * rows * cols * sizeof(float);

    // Allocate host memory for the data
    std::vector<float> h_data(depth * rows * cols);

    // Copy data from device to host
    cudaError_t err = cudaMemcpy(h_data.data(), d_volume, size, cudaMemcpyDeviceToHost);
    cudaCheckError(err); // Use the utility function to check for errors

    // Write the data to the binary file
    file.write(reinterpret_cast<const char*>(h_data.data()), size);

    // Verify the correct writing to the file
    assert(file.tellp() == static_cast<std::streamoff>(sizeof(depth) + sizeof(rows) + sizeof(cols) + size) && "Error writing to file");

    file.close();
}

#endif // SAVEBIN_H
