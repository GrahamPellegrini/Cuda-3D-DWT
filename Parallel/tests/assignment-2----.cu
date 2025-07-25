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

__constant__ float d_low_coeff[8];   // Maximum size for low-pass filters
__constant__ float d_high_coeff[8]; // Maximum size for high-pass filters
__constant__ int d_filter_length;

void coeffGPU(int db_num) {
    // Ensure db_num is between 1 and 4
    assert(db_num >= 1 && db_num <= 4 && "db_num must be between 1 and 4");

    // Select the coefficients based on db_num
    std::vector<float> low_coeff = db_low[db_num - 1];
    std::vector<float> high_coeff = db_high[db_num - 1];

    // Fill the rest of the array with 0.0f if the filter is shorter than 8 coefficients
    std::vector<float> low_coeff_filled(8, 0.0f);
    std::vector<float> high_coeff_filled(8, 0.0f);

    // Copy the selected coefficients into the first part of the arrays
    std::copy(low_coeff.begin(), low_coeff.end(), low_coeff_filled.begin());
    std::copy(high_coeff.begin(), high_coeff.end(), high_coeff_filled.begin());

    // Copy the filled arrays to GPU's constant memory
    cudaError_t err = cudaMemcpyToSymbol(d_low_coeff, low_coeff_filled.data(), 8 * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Error copying low coefficients to GPU: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMemcpyToSymbol(d_high_coeff, high_coeff_filled.data(), 8 * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Error copying high coefficients to GPU: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Copy the filter length for the selected wavelet to GPU's constant memory
    int filter_length = low_coeff.size();
    err = cudaMemcpyToSymbol(d_filter_length, &filter_length, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Error copying filter length to GPU: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    std::cerr << "Coefficients and filter length for db" << db_num << " successfully copied to GPU" << std::endl;
}

float* volGPU(std::vector<std::vector<std::vector<float>>>& volume, int depth, int rows, int cols) {
    // Flatten the 3D volume into a 1D vector (row-major order)
    std::vector<float> flat_volume(depth * rows * cols);
    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                flat_volume[(d * rows * cols) + (r * cols) + c] = volume[d][r][c];
            }
        }
    }

    // Allocate memory on the GPU
    float* d_volume = nullptr;
    cudaError_t err = cudaMalloc(&d_volume, flat_volume.size() * sizeof(float));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate GPU memory for volume: " + std::string(cudaGetErrorString(err)));
    }

    // Copy data to the GPU
    err = cudaMemcpy(d_volume, flat_volume.data(), flat_volume.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        // Free the allocated memory in case of an error
        cudaFree(d_volume);
        throw std::runtime_error("Failed to copy data to GPU: " + std::string(cudaGetErrorString(err)));
    }

    // Clear the CPU memory after copying to GPU
    flat_volume.clear();
    volume.clear();

    // Synchronize the device with error checking
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to synchronize the device: " + std::string(cudaGetErrorString(err)));
    }

    return d_volume;
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

__global__ void reduce(float* volume, float* temp, int depth, int rows, int cols, int new_depth, int new_rows, int new_cols) {
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (z < new_depth && y < new_rows && x < new_cols) {
        // Calculate indices for LLL subband
        int old_idx = (z * 2) * rows * cols + (y * 2) * cols + (x * 2);
        int new_idx = z * new_rows * new_cols + y * new_cols + x;

        // Extract LLL subband
        temp[new_idx] = volume[old_idx];
    }
}


__global__ void dwt_1d_kernel(float* volume, int stride, int offset, int size, int dim_size){
    extern __shared__ float shared_mem[];

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;

    if (global_idx < dim_size) {
        // Load data into shared memory
        for (int i = 0; i < d_filter_length; ++i) {
            int shared_idx = tid + i * blockDim.x;
            int data_idx = global_idx + i * stride;
            shared_mem[shared_idx] = (data_idx < dim_size) ? volume[data_idx] : 0.0f;
        }
    }

    __syncthreads();

    // Perform convolution for low-pass and high-pass filters
    float sum_low = 0.0f;
    float sum_high = 0.0f;

    for (int i = 0; i < d_filter_length; ++i) {
        int idx = tid + i * blockDim.x;
        if (idx < size) {
            sum_low += shared_mem[idx] * d_low_coeff[i];
            sum_high += shared_mem[idx] * d_high_coeff[i];
        }
    }

    // Store results back to the volume
    if (global_idx < size / 2) {
        volume[global_idx * stride] = sum_low;
        volume[(global_idx + size / 2) * stride] = sum_high;
    }
}

void dwt_3d(float* volume, int depth, int rows, int cols) {
    dim3 block_dim(256); // Number of threads per block

    // Calculate grid dimensions for each axis
    dim3 grid_dim_x((cols + block_dim.x - 1) / block_dim.x, rows, depth);
    dim3 grid_dim_y((rows + block_dim.x - 1) / block_dim.x, cols, depth);
    dim3 grid_dim_z((depth + block_dim.x - 1) / block_dim.x, cols, rows);

    size_t shared_memory_size = 8 * block_dim.x * sizeof(float);

    // Perform 1D DWT along x-axis (columns)
    dwt_1d_kernel<<<grid_dim_x, block_dim, shared_memory_size>>>(volume, 1, 0, cols, cols * rows * depth);

    // Perform 1D DWT along y-axis (rows)
    dwt_1d_kernel<<<grid_dim_y, block_dim, shared_memory_size>>>(volume, cols, 0, rows, cols * rows * depth);

    // Perform 1D DWT along z-axis (depth)
    dwt_1d_kernel<<<grid_dim_z, block_dim, shared_memory_size>>>(volume, cols * rows, 0, depth, cols * rows * depth);
}

void multi_level(float* d_volume, int levels, int& depth, int& rows, int& cols) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < levels; i++) {
        
        // Perform the 3D DWT
        dwt_3d(d_volume, depth, rows, cols);

        // If last level skip the downsampling
        if(i == levels - 1) { break; }

        // Calculate the new dimensions after downsampling
        int new_depth = (depth + 1) / 2;
        int new_rows = (rows + 1) / 2;
        int new_cols = (cols + 1) / 2;

        // Allocate memory for the temporary volume
        float* d_temp = nullptr;
        cudaError_t err = cudaMalloc(&d_temp, new_depth * new_rows * new_cols * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate GPU memory for temporary volume: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        // Define the block and grid dimensions for the reduction kernel
        dim3 block_dim(8, 8, 8);
        dim3 grid_dim(
            (new_cols + block_dim.x - 1) / block_dim.x,
            (new_rows + block_dim.y - 1) / block_dim.y,
            (new_depth + block_dim.z - 1) / block_dim.z
        );

        // Perform the reduction
        reduce<<<grid_dim, block_dim>>>(d_volume, d_temp, depth, rows, cols, new_depth, new_rows, new_cols);

        // Free the old volume
        cudaFree(d_volume);

        // Update the dimensions
        depth = new_depth;
        rows = new_rows;
        cols = new_cols;

        // Update the volume pointer
        d_volume = d_temp;

        // Synchronize the device with error checking
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "Failed to synchronize the device: " << cudaGetErrorString(err) << std::endl;
            return;
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
    assert(argc == 5 && "Usage: ./assignment-1 <input.bin> <output.bin> <db_num> <levels>");

    // Start the global timer
    auto start = std::chrono::high_resolution_clock::now();

    // Load the arguments into variables
    std::string bin_in = argv[1];
    std::string bin_out = argv[2];
    int db_num = std::stoi(argv[3]);
    int levels = std::stoi(argv[4]);

    // Send the wavelet coefficients to the GPU
    coeffGPU(db_num);

    // Load the 3D volume from the binary file
    std::vector<std::vector<std::vector<float>>> vol_in = loadvolume(bin_in);

    // Get the dimensions of the 3D volume
    int depth = vol_in.size();
    int rows = vol_in[0].size();
    int cols = vol_in[0][0].size();

    // Print the dimensions of the 3D volume
    std::cerr << "Volume dimensions: " << depth << "x" << rows << "x" << cols << std::endl;

    // Allocate memory on the GPU for the 3D volume
    float* d_volume = volGPU(vol_in, depth, rows, cols);

    // Perform the multi-level DWT on the 3D volume
    multi_level(d_volume, levels, depth, rows, cols);
    
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