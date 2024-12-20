#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
#include "../include/loadbin.h"
#include "../include/savebin.h"

// Define the wavelet coefficients as floats in constant memory
__constant__ float db_low[4][8] = {
    {0.70710678f, 0.70710678f}, // db1
    {-0.12940952f, 0.22414387f, 0.83651630f, 0.48296291f}, // db2
    {0.03522629f, -0.08544127f, -0.13501102f, 0.45987750f, 0.80689151f, 0.33267055f}, // db3
    {-0.01059740f, 0.03288301f, 0.03084138f, -0.18703481f, -0.02798377f, 0.63088077f, 0.71484657f, 0.23037781f} // db4
};

__constant__ float db_high[4][8] = {
    {-0.70710678f, 0.70710678f}, // db1
    {-0.48296291f, 0.83651630f, -0.22414387f, -0.12940952f}, // db2
    {-0.33267055f, 0.80689151f, -0.45987750f, -0.13501102f, 0.08544127f, 0.03522629f}, // db3
    {-0.23037781f, 0.71484657f, -0.63088077f, -0.02798377f, 0.18703481f, 0.03084138f, -0.03288301f, -0.01059740f} // db4
};

// Function to perform 1D DWT on a signal
__device__ void dwt_1d(float* signal, int signal_size, int db_num) {
    // Retrieve the low-pass and high-pass filters based on db_num
    const float* low_filter = db_low[db_num - 1];
    const float* high_filter = db_high[db_num - 1];
    int filter_length = (db_num == 1) ? 2 : (db_num == 2) ? 4 : (db_num == 3) ? 6 : 8;

    // Prepare array for DWT output
    float transformed[1024]; // Adjust size as needed

    // Perform convolution and downsampling
    for (int i = 0; i <= signal_size - filter_length; i += 2) {
        float low_sum = 0.0f, high_sum = 0.0f;

        // Convolve signal with filters
        for (int j = 0; j < filter_length; ++j) {
            low_sum += signal[i + j] * low_filter[j];
            high_sum += signal[i + j] * high_filter[j];
        }

        // Store results in transformed array
        transformed[i / 2] = low_sum;
        transformed[i / 2 + signal_size / 2] = high_sum;
    }

    // Copy transformed result back into signal
    for (int i = 0; i < signal_size; ++i) {
        signal[i] = transformed[i];
    }
}

// CUDA kernel to apply 1D DWT to rows
__global__ void dwt_1d_rows(float* volume, int rows, int cols, int depth, int db_num) {
    int d = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows) {
        // Apply 1D DWT to row
        dwt_1d(&volume[(d * rows + i) * cols], cols, db_num);
    }
}

// CUDA kernel to apply 1D DWT to columns
__global__ void dwt_1d_cols(float* volume, int rows, int cols, int depth, int db_num) {
    int d = blockIdx.z;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        // Declare column array
        float column[1024]; // Adjust size as needed
        // Apply 1D DWT to column
        for (int i = 0; i < rows; ++i) {
            column[i] = volume[(d * rows + i) * cols + j];
        }
        dwt_1d(column, rows, db_num);
        for (int i = 0; i < rows; ++i) {
            volume[(d * rows + i) * cols + j] = column[i];
        }
    }
}

// CUDA kernel to apply 1D DWT to depth slices
__global__ void dwt_1d_depth(float* volume, int rows, int cols, int depth, int db_num) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {
        // Declare signal array
        float signal[1024]; // Adjust size as needed
        // Apply 1D DWT to depth slice
        for (int d = 0; d < depth; ++d) {
            signal[d] = volume[(d * rows + i) * cols + j];
        }
        dwt_1d(signal, depth, db_num);
        for (int d = 0; d < depth; ++d) {
            volume[(d * rows + i) * cols + j] = signal[d];
        }
    }
}

// Function to perform 3D DWT on a 3D volume
void dwt_3D(std::vector<std::vector<std::vector<float>>>& volume, int db_num) {
    int depth = volume.size();
    int rows = volume[0].size();
    int cols = volume[0][0].size();

    // Allocate device memory
    float* d_volume;
    cudaMalloc(&d_volume, depth * rows * cols * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_volume, volume[0][0].data(), depth * rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y, depth);

    // Launch kernels
    dwt_1d_rows<<<gridSize, blockSize>>>(d_volume, rows, cols, depth, db_num);
    dwt_1d_cols<<<gridSize, blockSize>>>(d_volume, rows, cols, depth, db_num);
    dwt_1d_depth<<<gridSize, blockSize>>>(d_volume, rows, cols, depth, db_num);

    // Copy data back to host
    cudaMemcpy(volume[0][0].data(), d_volume, depth * rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_volume);
}

// Function to perform multi-level 3D DWT on a 3D volume
void multi_level(std::vector<std::vector<std::vector<float>>>& volume, int db_num, int levels) {
    // Start timer to measure time taken for DWT
    auto dwt_s = std::chrono::high_resolution_clock::now();

    // Get the shape of the volume
    int depth = volume.size();
    int rows = volume[0].size();
    int cols = volume[0][0].size();

    // Iterate over each level
    for (int i = 0; i < levels; i++) {
        // Perform 3D DWT on the volume
        dwt_3D(volume, db_num);

        // If it is not the last level
        if (i != levels - 1) {
            // Halve the dimensions of the volume
            depth = (depth + 1) / 2;
            rows = (rows + 1) / 2;
            cols = (cols + 1) / 2;
            // Note we add 1 before dividing by 2 to ensure that the dimensions are rounded up if they are odd

            // Resize the volume to the new dimensions which captures the approximation coefficients (LLL)
            volume.resize(depth);
            for (int d = 0; d < depth; ++d) {
                volume[d].resize(rows);
                for (int r = 0; r < rows; ++r) {
                    volume[d][r].resize(cols);
                }
            }
        }
    }

    // Stop timer
    auto dwt_e = std::chrono::high_resolution_clock::now();
    // Calculate the duration
    std::chrono::duration<double> dwt_d = dwt_e - dwt_s;
    // Log a success message
    std::cerr << "Multi-level DWT completed successfully with " << levels << " levels and db_num " << db_num << std::endl;

    // Log the time taken for the DWT
    std::cerr << "Time taken (DWT): " << dwt_d.count() << " seconds" << std::endl;
}

// Main program entry point
int main(int argc, char *argv[]) {
    (void)argc; // Suppress unused parameter warning
    // Print the program title
    std::cerr << "Assignment 1: Synchronous Multi DWT on 3D CT Image" << std::endl;

    // Check if the number of arguments is correct
    assert(argc == 5 && "Usage: ./assignment-1 <input.bin> <output.bin> <db_num> <levels>");

    // Start the global timer
    auto start = std::chrono::high_resolution_clock::now();

    // Load the arguments into variables
    std::string bin_in = argv[1];
    std::string bin_out = argv[2];
    int db_num = std::stoi(argv[3]);
    int levels = std::stoi(argv[4]);

    // Check if the db_num is between 1 and 4
    assert(db_num >= 1 && db_num <= 4 && "db_num must be between 1 and 4");

    // Load the 3D slice from the binary file
    std::vector<std::vector<std::vector<float>>> volume = loadvolume(bin_in);

    // Perform the multi-level DWT on the 3D volume
    multi_level(volume, db_num, levels);

    // Save the 3D volume to the binary file
    savevolume(volume, bin_out);

    // Stop the global timer
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the duration
    std::chrono::duration<double> t = end - start;

    // Log a success message
    std::cerr << "Program completed successfully" << std::endl;

    // Log the time taken for the program
    std::cerr << "Total time taken: " << t.count() << " seconds" << std::endl;

    return EXIT_SUCCESS;
}