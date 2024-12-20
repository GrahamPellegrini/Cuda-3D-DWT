#ifndef GPU_MEMORY_H
#define GPU_MEMORY_H

#include <cuda_runtime.h> // For GPU memory management
#include <iostream> // For error handling


// Function to transfer 3D volume to GPU
float* transferVolToGPU(const std::vector<std::vector<std::vector<float>>>& volume) {
    // Flatten the 3D volume into a 1D array
    size_t numElements = volume.size() * volume[0].size() * volume[0][0].size();
    float* h_volume = new float[numElements];
    size_t index = 0;

    // Flatten the volume
    for (size_t i = 0; i < volume.size(); ++i) {
        for (size_t j = 0; j < volume[i].size(); ++j) {
            for (size_t k = 0; k < volume[i][j].size(); ++k) {
                h_volume[index++] = volume[i][j][k];
            }
        }
    }

    float* d_volume = nullptr;
    // Allocate GPU memory
    cudaError_t err = cudaMalloc((void**)&d_volume, numElements * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Error allocating GPU memory: " << cudaGetErrorString(err) << std::endl;
        delete[] h_volume;
        return nullptr;
    }

    // Copy the flattened volume to the GPU
    err = cudaMemcpy(d_volume, h_volume, numElements * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Error copying to GPU memory: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_volume);
        delete[] h_volume;
        return nullptr;
    }

    // Free host memory
    delete[] h_volume;

    return d_volume;
}


// Function to retrieve 3D volume from GPU to CPU
std::vector<std::vector<std::vector<float>>> retrieveVolToCPU(float* d_volume, size_t dimX, size_t dimY, size_t dimZ) {
    size_t numElements = dimX * dimY * dimZ;
    std::vector<std::vector<std::vector<float>>> volume(dimX, std::vector<std::vector<float>>(dimY, std::vector<float>(dimZ)));

    float* h_volume = new float[numElements];

    // Copy data from GPU to host memory
    cudaError_t err = cudaMemcpy(h_volume, d_volume, numElements * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Error copying from GPU memory: " << cudaGetErrorString(err) << std::endl;
        delete[] h_volume;
        return volume;
    }

    size_t index = 0;

    // Rebuild the 3D volume from the 1D array
    for (size_t i = 0; i < dimX; ++i) {
        for (size_t j = 0; j < dimY; ++j) {
            for (size_t k = 0; k < dimZ; ++k) {
                volume[i][j][k] = h_volume[index++];
            }
        }
    }

    delete[] h_volume;

    return volume;
}


#endif // GPU_MEMORY_H