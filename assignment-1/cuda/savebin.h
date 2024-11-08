#ifndef SAVEBIN_H
#define SAVEBIN_H

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cassert>
#include <cuda_runtime.h>

// Save a 3D volume to a binary file using CUDA
void savevolume(const float* d_volume, const std::string& filename, int depth, int rows, int cols) {
    // Open the file
    std::ofstream file(filename, std::ios::binary);

    // Check if the file is open
    assert(file.is_open() && "Error opening file");

    // Write the dimensions to the binary file
    file.write(reinterpret_cast<const char*>(&depth), sizeof(depth));
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    // Allocate host memory
    size_t size = depth * rows * cols * sizeof(float);
    std::vector<float> h_data(depth * rows * cols);

    // Copy the data from the GPU to the host
    cudaMemcpy(h_data.data(), d_volume, size, cudaMemcpyDeviceToHost);

    // Write the data from the host to the binary file
    file.write(reinterpret_cast<const char*>(h_data.data()), size);

    // Show the file dimensions written and check if the file is written correctly
    assert(file.tellp() == static_cast<std::streamoff>(sizeof(depth) + sizeof(rows) + sizeof(cols) + depth * rows * cols * sizeof(float)) && "Error writing to file");

    file.close();
}

#endif // SAVEBIN_H