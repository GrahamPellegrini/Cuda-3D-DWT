// Standard C++ headers
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <chrono>
#include <algorithm>


// CUDA headers
#include <cuda_runtime.h>
#include "../include/cudaerr.h"
#include "../include/kernels.cu"

// Custom headers
#include "../include/loadbin.h"
#include "../include/savebin.h"


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

void toGPU(std::vector<std::vector<std::vector<float>>> volume, int db_num, int depth, int rows, int cols, float*& d_low_coeff, float*& d_high_coeff, int&filter_size, float*& d_volume) {

    // Select the coefficients based on db_num
    std::vector<float> low_coeff = db_low[db_num - 1];
    std::vector<float> high_coeff = db_high[db_num - 1];

    // Calculate the filter size
    filter_size = low_coeff.size();

    // Allocate memory for the low and high pass coefficients on the GPU
    cudaError_t err = cudaMalloc(&d_low_coeff, low_coeff.size() * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for low coefficients: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&d_high_coeff, high_coeff.size() * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_low_coeff); // Free previously allocated memory
        throw std::runtime_error("Failed to allocate GPU memory for high coefficients: " + std::string(cudaGetErrorString(err)));
    }

    // Copy the coefficients to the GPU
    err = cudaMemcpy(d_low_coeff, low_coeff.data(), low_coeff.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_low_coeff);
        cudaFree(d_high_coeff);
        throw std::runtime_error("Failed to copy low coefficients to GPU: " + std::string(cudaGetErrorString(err)));
    }

    err = cudaMemcpy(d_high_coeff, high_coeff.data(), high_coeff.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_low_coeff);
        cudaFree(d_high_coeff);
        throw std::runtime_error("Failed to copy high coefficients to GPU: " + std::string(cudaGetErrorString(err)));
    }

    // Flatten the 3D volume into a 1D vector (row-major order)
    std::vector<float> flat_volume(depth * rows * cols);
    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                flat_volume[d * rows * cols + r * cols + c] = volume[d][r][c];
            }
        }
    }

    // Allocate memory on the GPU for the volume
    err = cudaMalloc(&d_volume, flat_volume.size() * sizeof(float));
    if (err != cudaSuccess) {
        cudaFree(d_low_coeff);
        cudaFree(d_high_coeff);
        throw std::runtime_error("Failed to allocate GPU memory for volume: " + std::string(cudaGetErrorString(err)));
    }

    // Copy the flattened volume to the GPU
    err = cudaMemcpy(d_volume, flat_volume.data(), flat_volume.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_low_coeff);
        cudaFree(d_high_coeff);
        cudaFree(d_volume);
        throw std::runtime_error("Failed to copy volume data to GPU: " + std::string(cudaGetErrorString(err)));
    }

    // Clear the CPU memory after copying to GPU
    flat_volume.clear();
    volume.clear();

    // Synchronize the device with error checking
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to synchronize the device: " + std::string(cudaGetErrorString(err)));
    }
}

std::vector<std::vector<std::vector<float>>> volCPU(float* d_volume, int depth, int rows, int cols) {
    // Allocate memory for the 3D volume on the CPU
    std::vector<std::vector<std::vector<float>>> volume(depth, std::vector<std::vector<float>>(rows, std::vector<float>(cols)));

    // Copy the data from the GPU to the CPU
    std::vector<float> flat_volume(depth * rows * cols);
    cudaError_t err = cudaMemcpy(flat_volume.data(), d_volume, flat_volume.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy data from GPU to CPU: " + std::string(cudaGetErrorString(err)));
    }

    // Unflatten the 1D vector into a 3D volume
    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
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

__global__ void reduce(float* d_data, float* d_subband, int depth, int rows, int cols) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;

    if (d < depth && r < rows && c < cols) {
        d_subband[d * rows * cols + r * cols + c] = d_data[d * rows * cols + r * cols + c];
    }
}


void dwt_3d(float* d_volume, float* d_low_coeff, float* d_high_coeff, int depth, int rows, int cols, int filter_size) {
    float* d_data1 = d_volume;
    float* d_data2 = nullptr;
    cudaError_t err = cudaMalloc(&d_data2, depth * rows * cols * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for temporary volume: " << cudaGetErrorString(err) << std::endl;
    }

    dim3 blockDim(16, 8, 8);
    dim3 gridDim0((rows + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y, (depth + blockDim.z - 1) / blockDim.z);
    dim3 gridDim1((cols + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, (depth + blockDim.z - 1) / blockDim.z);
    dim3 gridDim2((depth + blockDim.x - 1) / blockDim.x, (cols + blockDim.y - 1) / blockDim.y, (rows + blockDim.z - 1) / blockDim.z);


    // Perform convolution along the first dimension
    dim0_kernel<<<gridDim0, blockDim, filter_size * sizeof(float) * 2>>>(d_data1, d_data2, d_low_coeff, d_high_coeff, filter_size, depth, rows, cols);
    // Synchronize the device
    cudaDeviceSynchronize();


    // Perform convolution along the second dimension
    dim1_kernel<<<gridDim1, blockDim, filter_size * sizeof(float) * 2>>>(d_data2, d_data1, d_low_coeff, d_high_coeff, filter_size, depth, rows, cols);
    // Synchronize the device  
    cudaDeviceSynchronize();

    // Perform convolution along the third dimension
    dim2_kernel<<<gridDim2, blockDim, filter_size * sizeof(float) * 2>>>(d_data1, d_data2, d_low_coeff, d_high_coeff, filter_size, depth, rows, cols);
    // Synchronize the device
    cudaDeviceSynchronize();

    // Free the original data
    err = cudaFree(d_data1);
    if (err != cudaSuccess) {
        std::cerr << "Failed to free GPU memory for original volume: " << cudaGetErrorString(err) << std
        ::endl;
    }

    // Swap the pointer
    d_volume = d_data2;
}


void multi_level(float* d_volume, float* d_low_coeff, float* d_high_coeff, int levels, int& depth, int& rows, int& cols, int filter_size) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    for (int i = 0; i < levels; i++) {
        dwt_3d(d_volume, d_low_coeff, d_high_coeff, depth, rows, cols, filter_size);

        // DWT at level i completed
        std::cerr << "DWT at level " << i << " completed" << std::endl;
        
        if (i != (levels - 1)){
            // New dimensions for the LLL subband
            depth = (depth + 1) / 2;
            rows = (rows + 1) / 2;
            cols = (cols + 1) / 2;

            // Allocate memory for the subband
            float* d_subband = nullptr;
            cudaError_t err = cudaMalloc(&d_subband, depth * rows * cols * sizeof(float));
            if (err != cudaSuccess) {
                std::cerr << "Failed to allocate GPU memory for subband: " << cudaGetErrorString(err) << std::endl;
            }
            // Message  
            std::cerr << "Allocated memory for subband at level " << i + 1 << std::endl;
            // Define a new grid and block size
            dim3 blockDim(16, 8, 8);
            dim3 gridDim((depth + blockDim.x - 1) / blockDim.x, (rows + blockDim.y - 1) / blockDim.y, (cols + blockDim.z - 1) / blockDim.z);
            // Reduce the volume to the subband
            reduce<<<gridDim, blockDim>>>(d_volume, d_subband, depth, rows, cols);
            // Synchronize the device
            err = cudaDeviceSynchronize();
            if (err != cudaSuccess) {
                std::cerr << "Failed to synchronize the device: " << cudaGetErrorString(err) << std::endl;
            }
            // Message
            std::cerr << "Reduced volume to subband at level " << i + 1 << std::endl;

            // Free the original volume
            err = cudaFree(d_volume);
            if (err != cudaSuccess) {
                std::cerr << "Failed to free GPU memory for original volume: " << cudaGetErrorString(err) << std::endl;
            }

            // Update the volume pointer
            d_volume = d_subband;
            
            // Print the level completed
            std::cerr << "Level " << i + 1 << " completed" << std::endl;
        }
        
    }
    

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Multi-level DWT completed in " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Main program entry point
int main(int argc, char *argv[]) {
    (void)argc; // Suppress unused parameter warning
    // Print the program title
    std::cerr << "Assignment 2: CUDA Implementation of 3D DWT" << std::endl;

    // Check if the number of arguments is correct
    assert(argc == 5 && "Usage: ./assignment-2 <input.bin> <output.bin> <db_num> <levels>");

    // Start the global timer
    auto start = std::chrono::high_resolution_clock::now();

    // Load the arguments into variables
    std::string bin_in = argv[1];
    std::string bin_out = argv[2];
    int db_num = std::stoi(argv[3]);
    int levels = std::stoi(argv[4]);

    // Load the 3D volume from the binary file
    std::vector<std::vector<std::vector<float>>> vol_in = loadvolume(bin_in);

    // Get the dimensions of the 3D volume
    int depth = vol_in.size();
    int rows = vol_in[0].size();
    int cols = vol_in[0][0].size();

    // Print the dimensions of the 3D volume
    std::cerr << "Volume dimensions: " << depth << "x" << rows << "x" << cols << std::endl;

    // Create pointers for the volume and coefficients on the GPU
    float* d_low_coeff = nullptr;
    float* d_high_coeff = nullptr;
    float* d_volume = nullptr;
    int filter_size;

    // Copy the volume and coefficients to the GPU
    toGPU(vol_in, db_num, depth, rows, cols, d_low_coeff, d_high_coeff, filter_size, d_volume);
    
    // Perform the multi-level DWT on the 3D volume
    multi_level(d_volume, d_low_coeff, d_high_coeff, levels, depth, rows, cols, filter_size);

    // Copy the data back to the CPU
    std::vector<std::vector<std::vector<float>>> vol_out = volCPU(d_volume, depth, rows, cols);

    // print the dimensions of volume after DWT
    std::cerr << "Volume dimensions after DWT: " << vol_out.size() << "x" << vol_out[0].size() << "x" << vol_out[0][0].size() << std::endl;

    // Save the modified 3D volume to the output binary file
    savevolume(vol_out, bin_out);

    // Stop the global timer
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the duration
    std::chrono::duration<double> t = end - start;

    // Log the time taken for the program
    std::cerr << "Total time taken: " << t.count() << " seconds" << std::endl;

    return EXIT_SUCCESS;
}