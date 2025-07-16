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
#include "../include/kernels.cuh"

// Custom headers
#include "../include/loadbin.h"
#include "../include/savebin.h"
#include "../include/idwt.h"


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

    // Start a timer to measure the time taken to copy coefficients to the GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Allocate memory for the low and high pass coefficients on the GPU
    cudaError_t err = cudaMalloc(&d_low_coeff, low_coeff.size() * sizeof(float));
    assert(err == cudaSuccess && "Failed to allocate GPU memory for low coefficients");

    err = cudaMalloc(&d_high_coeff, high_coeff.size() * sizeof(float));
    assert(err == cudaSuccess && "Failed to allocate GPU memory for high coefficients");

    // Copy the coefficients to the GPU
    err = cudaMemcpy(d_low_coeff, low_coeff.data(), low_coeff.size() * sizeof(float), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess && "Failed to copy low coefficients to GPU");

    err = cudaMemcpy(d_high_coeff, high_coeff.data(), high_coeff.size() * sizeof(float), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess && "Failed to copy high coefficients to GPU");

    // Stop the timer and calculate the time taken to copy coefficients to the GPU
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cerr << "Coefficients -> Global: " << milliseconds << " ms" << std::endl;

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
    assert(err == cudaSuccess && "Failed to allocate GPU memory for volume");

    // Copy the flattened volume to the GPU
    err = cudaMemcpy(d_volume, flat_volume.data(), flat_volume.size() * sizeof(float), cudaMemcpyHostToDevice);
    assert(err == cudaSuccess && "Failed to copy volume to GPU");

    // Clear the CPU memory after copying to GPU
    flat_volume.clear();
    volume.clear();

    // Synchronize the device with error checking
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "Failed to synchronize the device");
}

std::vector<std::vector<std::vector<float>>> volCPU(float* d_volume, int depth, int rows, int cols) {
    // Allocate memory for the 3D volume on the CPU
    std::vector<std::vector<std::vector<float>>> volume(depth, std::vector<std::vector<float>>(rows, std::vector<float>(cols)));

    // Copy the data from the GPU to the CPU
    std::vector<float> flat_volume(depth * rows * cols);
    cudaError_t err = cudaMemcpy(flat_volume.data(), d_volume, flat_volume.size() * sizeof(float), cudaMemcpyDeviceToHost);
    assert(err == cudaSuccess && "Failed to copy volume from GPU to CPU");

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
    assert(err == cudaSuccess && "Failed to free GPU memory for volume");

    return volume;
}


void dwt_3d(float* d_volume, float* d_low_coeff, float* d_high_coeff, int depth, int rows, int cols, int filter_size) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    float* d_data1 = d_volume;
    float* d_data2 = nullptr;
    cudaError_t err = cudaMalloc(&d_data2, depth * rows * cols * sizeof(float));
    assert(err == cudaSuccess && "Failed to allocate GPU memory for temporary volume");

    dim3 blockDim(256, 1, 1);
    dim3 row_grid((cols + blockDim.x - 1) / blockDim.x, rows, depth);
    dim3 col_grid((cols / 2 + blockDim.x - 1) / blockDim.x, rows, depth);
    dim3 depth_grid((depth / 2 + blockDim.x - 1) / blockDim.x, rows, cols);
    dim3 map_grid((cols + blockDim.x - 1) / blockDim.x, rows, depth);


    // Perform convolution along the first dimension
    row_kernel<<<row_grid, blockDim, filter_size * sizeof(float) * 2>>>(d_data1, d_data2, d_low_coeff, d_high_coeff, filter_size, depth, rows, cols);
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "Failed to synchronize the device at row kernel");


    // Perform convolution along the second dimension
    col_kernel<<<col_grid, blockDim, filter_size * sizeof(float) * 2>>>(d_data2, d_data1, d_low_coeff, d_high_coeff, filter_size, depth, rows, cols);
    // Synchronize the device  
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "Failed to synchronize the device at col kernel");

    // Perform convolution along the third dimension
    depth_kernel<<<depth_grid, blockDim, filter_size * sizeof(float) * 2>>>(d_data1, d_data2, d_low_coeff, d_high_coeff, filter_size, depth, rows, cols);
    // Synchronize the device
    err = cudaDeviceSynchronize();
    assert(err == cudaSuccess && "Failed to synchronize the device at depth kernel");



    // Copy the transformed data back to the original volume
    map_kernel<<<map_grid, blockDim>>>(d_data2, d_volume, depth, rows, cols, depth, rows, cols);
    // Synchronize the device
    cudaDeviceSynchronize();

    // Free the temporary volume
    err = cudaFree(d_data2);
    assert(err == cudaSuccess && "Failed to free GPU memory for temporary volume");

    // Stop the timer and calculate the time taken to perform the 3D DWT
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cerr << "DWT Kernels: " << milliseconds << " ms" << std::endl;
}


// Main program entry point
int main(int argc, char *argv[]) {
    (void)argc; // Suppress unused parameter warning
    // Print the program title
    std::cerr << "Assignment 2: CUDA Implementation of 3D DWT" << std::endl;

    // Check if the number of arguments is correct
    assert(argc == 5 && "Usage: ./assignment-2 <input.bin> <output.bin> <db_num> <inverse>");

    // Start the global timer
    auto start = std::chrono::high_resolution_clock::now();

    // Load the arguments into variables
    std::string bin_in = argv[1];
    std::string bin_out = argv[2];
    int db_num = std::stoi(argv[3]);
    int levels = 1;
    int inverse = std::stoi(argv[4]);

    // Load the 3D volume from the binary file
    std::vector<std::vector<std::vector<float>>> vol_in = loadvolume(bin_in);

    // Get the dimensions of the 3D volume
    int depth = vol_in.size();
    int rows = vol_in[0].size();
    int cols = vol_in[0][0].size();

    
    // Print the dimensions of the 3D volume
    std::cerr << "Volume dimensions: " << depth << "x" << rows << "x" << cols << std::endl;

    // Define volume for the output
    std::vector<std::vector<std::vector<float>>> vol_out;

    // Perform the inverse if inverse flag is set to 1
    if (inverse == 1) {
        std::cerr << "Performing inverse DWT" << std::endl;

        inverse_multi_level(vol_in, db_num, levels);

        // Swap vol_in and name it vol_out
        vol_out = vol_in;

        // print the dimensions of volume after inverse DWT
        std::cerr << "Volume dimensions after inverse DWT: " << vol_out.size() << "x" << vol_out[0].size() << "x" << vol_out[0][0].size() << std::endl;
    }
    else{
        
        // Create pointers for the volume and coefficients on the GPU
        float* d_low_coeff = nullptr;
        float* d_high_coeff = nullptr;
        float* d_volume = nullptr;
        int filter_size;

        // Copy the volume and coefficients to the GPU
        toGPU(vol_in, db_num, depth, rows, cols, d_low_coeff, d_high_coeff, filter_size, d_volume);
        
        // Perform the 3D DWT
        dwt_3d(d_volume, d_low_coeff, d_high_coeff, depth, rows, cols, filter_size);

        // Copy the data back to the CPU
        vol_out = volCPU(d_volume, depth, rows, cols);

        // print the dimensions of volume after DWT
        std::cerr << "Volume dimensions after DWT: " << vol_out.size() << "x" << vol_out[0].size() << "x" << vol_out[0][0].size() << std::endl;

    }

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
