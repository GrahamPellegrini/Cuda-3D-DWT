// Standard C++ headers
#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <chrono>

// CUDA headers
#include <cuda_runtime.h>

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

float* volGPU(const std::vector<std::vector<std::vector<float>>>& volume, int depth, int rows, int cols) {
    // Flatten the 3D volume into a 1D vector
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

__global__ void dwt_1d_kernel(
    float* volume,
    int stride_outer,
    int stride_inner,
    int outer_size,
    int inner_size
) {
    // Calculate the outer index
    int outer_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // If the outer index is out of bounds, return
    if (outer_idx >= outer_size){
        printf("Outer index out of bounds\n");
        return;
    }

    // Allocate shared memory for the temporary results
    extern __shared__ float temp[];  
    int filter_length = d_filter_length;
    int low_offset = 0;
    int high_offset = inner_size / 2;

    for (int i = 0; i <= inner_size - filter_length; i += 2) {
        float low_sum = 0.0f, high_sum = 0.0f;

        for (int j = 0; j < filter_length; ++j) {
            int idx = outer_idx * stride_outer + (i + j) * stride_inner;
            // If the index is out of bounds, skip the calculation
            if (idx < outer_size * stride_outer) {
                low_sum += volume[idx] * d_low_coeff[j];
                high_sum += volume[idx] * d_high_coeff[j];
            }
        }

        temp[low_offset + i / 2] = low_sum;
        temp[high_offset + i / 2] = high_sum;
    }

    __syncthreads();

    for (int i = 0; i < inner_size; ++i) {
        int idx = outer_idx * stride_outer + i * stride_inner;
        if (idx < outer_size * stride_outer) { // Prevent out-of-bounds access
            volume[idx] = temp[i];
        }
    }
}

void dwt_3d(float* volume, int depth, int rows, int cols) {
    // Define CUDA launch parameters
    dim3 block_dim(256); // Threads per block
    int shared_mem_size = cols * sizeof(float); // Shared memory size for each thread block

    // Step 1: Apply 1D DWT along rows for each depth slice
    dim3 grid_dim_rows((rows + block_dim.x - 1) / block_dim.x, depth);
    dwt_1d_kernel<<<grid_dim_rows, block_dim, shared_mem_size>>>(
        volume, cols, 1, depth * rows, cols);
    cudaDeviceSynchronize();

    // Step 2: Apply 1D DWT along columns for each depth slice
    dim3 grid_dim_cols((cols + block_dim.x - 1) / block_dim.x, depth);
    dwt_1d_kernel<<<grid_dim_cols, block_dim, shared_mem_size>>>(
        volume, rows * cols, cols, depth * cols, rows);
    cudaDeviceSynchronize();

    // Step 3: Apply 1D DWT along depth for each row-column combination
    dim3 grid_dim_depth((depth + block_dim.x - 1) / block_dim.x);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float* depth_ptr = volume + i * cols + j;
            dwt_1d_kernel<<<grid_dim_depth, block_dim, shared_mem_size>>>(
                depth_ptr, rows * cols, rows * cols, depth, 1);
            cudaDeviceSynchronize();
        }
    }
}


void multi_level(float* d_volume, int levels, int& depth, int& rows, int& cols) {
    // Initialize CUDA timer events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Perform multi-level DWT
    for (int i = 0; i < levels; i++) {
        // Perform a single-level 3D DWT
        dwt_3d(d_volume, depth, rows, cols);

        // Update dimensions for the next level (reduce by half)
        depth = (depth + 1) / 2;
        rows = (rows + 1) / 2;
        cols = (cols + 1) / 2;

        // Ensure dimensions remain valid
        if (depth < 1 || rows < 1 || cols < 1) {
            throw std::runtime_error("Volume dimensions became invalid during multi-level decomposition.");
        }
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Multi-level DWT completed in " << milliseconds << " ms" << std::endl;

    // Cleanup CUDA events
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
    std::vector<std::vector<std::vector<float>>> volume = loadvolume(bin_in);

    // Get the dimensions of the 3D volume
    int depth = volume.size();
    int rows = volume[0].size();
    int cols = volume[0][0].size();

    // Allocate memory on the GPU for the 3D volume
    float* d_volume = volGPU(volume, depth, rows, cols);

    // Synchronize the device after memory transfer
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Error synchronizing the device: " << cudaGetErrorString(err) << std::endl;
        return EXIT_FAILURE;
    }

    // Perform the multi-level DWT on the 3D volume
    multi_level(d_volume, levels, depth, rows, cols);

    // Copy the data back to the CPU
    volume = volCPU(d_volume, depth, rows, cols);

    // Save the modified 3D volume to the output binary file
    savevolume(volume, bin_out);

    // Stop the global timer
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the duration
    std::chrono::duration<double> t = end - start;

    // Log the time taken for the program
    std::cerr << "Total time taken: " << t.count() << " seconds" << std::endl;

    return EXIT_SUCCESS;
}