#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <chrono>
#include <cuda_runtime.h>
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

__constant__ float low_filter_device[8];
__constant__ float high_filter_device[8];

__global__ void dwt_1d_kernel(float* signal, int signal_size, int db_num, float* transformed) {
    // Retrieve the low-pass and high-pass filters based on db_num
    __shared__ float low_filter[8]; // Adjust size as needed
    __shared__ float high_filter[8]; // Adjust size as needed

    if (threadIdx.x == 0) {
        int filter_length = (db_num == 1) ? 2 : (db_num == 2) ? 4 : (db_num == 3) ? 6 : 8;
        for (int i = 0; i < filter_length; ++i) {
            low_filter[i] = low_filter_device[i];
            high_filter[i] = high_filter_device[i];
        }
    }
    __syncthreads();

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < signal_size / 2) {
        int filter_length = (db_num == 1) ? 2 : (db_num == 2) ? 4 : (db_num == 3) ? 6 : 8;
        float low_sum = 0.0f, high_sum = 0.0f;

        // Convolve signal with filters
        for (int j = 0; j < filter_length; ++j) {
            low_sum += signal[2 * idx + j] * low_filter[j];
            high_sum += signal[2 * idx + j] * high_filter[j];
        }

        // Store results in transformed array
        transformed[idx] = low_sum;
        transformed[idx + signal_size / 2] = high_sum;
    }
}
// CUDA kernel to apply 1D DWT to rows
__global__ void dwt_1d_rows(float* volume, int rows, int cols, int depth, int db_num) {
    int d = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows) {
        int signal_size = cols;
        float* signal = &volume[(d * rows + i) * cols];
        float* transformed = &volume[(d * rows + i) * cols];

        // Retrieve the low-pass and high-pass filters from constant memory
        __shared__ float low_filter[8]; // Adjust size as needed
        __shared__ float high_filter[8]; // Adjust size as needed

        if (threadIdx.x == 0) {
            int filter_length = (db_num == 1) ? 2 : (db_num == 2) ? 4 : (db_num == 3) ? 6 : 8;
            for (int j = 0; j < filter_length; ++j) {
                low_filter[j] = low_filter_device[j];
                high_filter[j] = high_filter_device[j];
            }
        }
        __syncthreads();

        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < signal_size / 2) {
            int filter_length = (db_num == 1) ? 2 : (db_num == 2) ? 4 : (db_num == 3) ? 6 : 8;
            float low_sum = 0.0f, high_sum = 0.0f;

            // Convolve signal with filters
            for (int j = 0; j < filter_length; ++j) {
                low_sum += signal[2 * idx + j] * low_filter[j];
                high_sum += signal[2 * idx + j] * high_filter[j];
            }

            // Store results in transformed array
            transformed[idx] = low_sum;
            transformed[idx + signal_size / 2] = high_sum;
        }
    }
}

// CUDA kernel to apply 1D DWT to columns
__global__ void dwt_1d_cols(float* volume, int rows, int cols, int depth, int db_num) {
    int d = blockIdx.z;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < cols) {
        float column[1024]; // Adjust size as needed
        for (int i = 0; i < rows; ++i) {
            column[i] = volume[(d * rows + i) * cols + j];
        }

        // Retrieve the low-pass and high-pass filters from constant memory
        __shared__ float low_filter[8]; // Adjust size as needed
        __shared__ float high_filter[8]; // Adjust size as needed

        if (threadIdx.x == 0) {
            int filter_length = (db_num == 1) ? 2 : (db_num == 2) ? 4 : (db_num == 3) ? 6 : 8;
            for (int k = 0; k < filter_length; ++k) {
                low_filter[k] = low_filter_device[k];
                high_filter[k] = high_filter_device[k];
            }
        }
        __syncthreads();

        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < rows / 2) {
            int filter_length = (db_num == 1) ? 2 : (db_num == 2) ? 4 : (db_num == 3) ? 6 : 8;
            float low_sum = 0.0f, high_sum = 0.0f;

            // Convolve signal with filters
            for (int j = 0; j < filter_length; ++j) {
                low_sum += column[2 * idx + j] * low_filter[j];
                high_sum += column[2 * idx + j] * high_filter[j];
            }

            // Store results in transformed array
            column[idx] = low_sum;
            column[idx + rows / 2] = high_sum;
        }

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
        float signal[1024]; // Adjust size as needed
        for (int d = 0; d < depth; ++d) {
            signal[d] = volume[(d * rows + i) * cols + j];
        }

        // Retrieve the low-pass and high-pass filters from constant memory
        __shared__ float low_filter[8]; // Adjust size as needed
        __shared__ float high_filter[8]; // Adjust size as needed

        if (threadIdx.x == 0) {
            int filter_length = (db_num == 1) ? 2 : (db_num == 2) ? 4 : (db_num == 3) ? 6 : 8;
            for (int k = 0; k < filter_length; ++k) {
                low_filter[k] = low_filter_device[k];
                high_filter[k] = high_filter_device[k];
            }
        }
        __syncthreads();

        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx < depth / 2) {
            int filter_length = (db_num == 1) ? 2 : (db_num == 2) ? 4 : (db_num == 3) ? 6 : 8;
            float low_sum = 0.0f, high_sum = 0.0f;

            // Convolve signal with filters
            for (int j = 0; j < filter_length; ++j) {
                low_sum += signal[2 * idx + j] * low_filter[j];
                high_sum += signal[2 * idx + j] * high_filter[j];
            }

            // Store results in transformed array
            signal[idx] = low_sum;
            signal[idx + depth / 2] = high_sum;
        }

        for (int d = 0; d < depth; ++d) {
            volume[(d * rows + i) * cols + j] = signal[d];
        }
    }
}

void dwt_3D_cuda(std::vector<std::vector<std::vector<float>>>& volume, int db_num) {
    int depth = volume.size();
    int rows = volume[0].size();
    int cols = volume[0][0].size();

    // Copy the low-pass and high-pass filters to the device
    cudaMemcpyToSymbol(low_filter_device, db_low[db_num - 1].data(), db_low[db_num - 1].size() * sizeof(float));
    cudaMemcpyToSymbol(high_filter_device, db_high[db_num - 1].data(), db_high[db_num - 1].size() * sizeof(float));

    // Define a pointer to the device memory
    float* d_volume;
    // Allocate the memory on the device for the pointer to be the same size as the volume
    cudaError_t err = cudaMalloc(&d_volume, depth * rows * cols * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Create CUDA streams
    cudaStream_t stream1, stream2, stream3;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);

    // Flatten the volume into a 1D array and copy it to the device memory asynchronously
    for (int d = 0; d < depth; ++d) {
        for (int i = 0; i < rows; ++i) {
            err = cudaMemcpyAsync(&d_volume[(d * rows + i) * cols], volume[d][i].data(), cols * sizeof(float), cudaMemcpyHostToDevice, stream1);
            if (err != cudaSuccess) {
                std::cerr << "CUDA memcpy async failed: " << cudaGetErrorString(err) << std::endl;
                return;
            }
        }
    }

    // For block size we use a generalized block size of 16x16
    dim3 blockSize(16, 16);
    // The grid size is calculated based on the size of the volume
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y, depth);

    // Launch kernels asynchronously
    dwt_1d_rows<<<gridSize, blockSize, 0, stream1>>>(d_volume, rows, cols, depth, db_num);
    dwt_1d_cols<<<gridSize, blockSize, 0, stream2>>>(d_volume, rows, cols, depth, db_num);
    dwt_1d_depth<<<gridSize, blockSize, 0, stream3>>>(d_volume, rows, cols, depth, db_num);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Copy the processed data back from the device to the host asynchronously
    for (int d = 0; d < depth; ++d) {
        for (int i = 0; i < rows; ++i) {
            err = cudaMemcpyAsync(volume[d][i].data(), &d_volume[(d * rows + i) * cols], cols * sizeof(float), cudaMemcpyDeviceToHost, stream1);
            if (err != cudaSuccess) {
                std::cerr << "CUDA memcpy async failed: " << cudaGetErrorString(err) << std::endl;
                return;
            }
        }
    }

    // Synchronize streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);

    // Destroy streams
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);

    // Free device memory
    cudaFree(d_volume);
}

// Function to perform multi-level 3D DWT on a 3D volume using CUDA
void multi_level_cuda(std::vector<std::vector<std::vector<float>>>& volume, int db_num, int levels) {
    auto dwt_s = std::chrono::high_resolution_clock::now();

    int depth = volume.size();
    int rows = volume[0].size();
    int cols = volume[0][0].size();

    for (int i = 0; i < levels; i++) {
        dwt_3D_cuda(volume, db_num);

        if (i != levels - 1) {
            depth = (depth + 1) / 2;
            rows = (rows + 1) / 2;
            cols = (cols + 1) / 2;

            volume.resize(depth);
            for (int d = 0; d < depth; ++d) {
                volume[d].resize(rows);
                for (int r = 0; r < rows; ++r) {
                    volume[d][r].resize(cols);
                }
            }
        }
    }

    auto dwt_e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dwt_d = dwt_e - dwt_s;
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
    multi_level_cuda(volume, db_num, levels);

    // Save the 3D volume to the binary file
    savevolume(volume, bin_out);

    // Stop the global timer
    auto end = std::chrono::high_resolution_clock::now();
    // Calculate the duration
    std::chrono::duration<double> t = end - start;

    // Log a success message
    // std::cerr << "Program completed successfully" << std::endl;
    
    // Log the time taken for the program
    std::cerr << "Total time taken: " << t.count()<< " seconds" << std::endl;

    return EXIT_SUCCESS;
}