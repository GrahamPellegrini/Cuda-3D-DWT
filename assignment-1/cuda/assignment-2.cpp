#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include "../include/loadbin.h"
#include "../include/savebin.h"
#include "../../code/shared/jbutil.h"

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

// CUDA kernel to perform 1D DWT on a signal
__global__ void dwt_1d_kernel(float* signal, float* low_filter, float* high_filter, int signal_size, int filter_length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= signal_size - filter_length) {
        float low_sum = 0.0f, high_sum = 0.0f;
        for (int j = 0; j < filter_length; ++j) {
            low_sum += signal[i + j] * low_filter[j];
            high_sum += signal[i + j] * high_filter[j];
        }
        signal[i / 2] = low_sum;
        signal[i / 2 + signal_size / 2] = high_sum;
    }
}

// Function to perform 1D DWT on a signal using CUDA
void dwt_1d_cuda(float* d_signal, int signal_size, const std::vector<float>& low_filter, const std::vector<float>& high_filter) {
    int filter_length = low_filter.size();

    float* d_low_filter;
    float* d_high_filter;

    cudaMalloc(&d_low_filter, filter_length * sizeof(float));
    cudaMalloc(&d_high_filter, filter_length * sizeof(float));

    cudaMemcpy(d_low_filter, low_filter.data(), filter_length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_high_filter, high_filter.data(), filter_length * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (signal_size + blockSize - 1) / blockSize;
    dwt_1d_kernel<<<numBlocks, blockSize>>>(d_signal, d_low_filter, d_high_filter, signal_size, filter_length);

    cudaFree(d_low_filter);
    cudaFree(d_high_filter);
}

// Function to perform 3D DWT on a 3D volume using CUDA
void dwt_3D_cuda(float* d_volume, int depth, int rows, int cols, int db_num) {
    const auto& low_filter = db_low[db_num - 1];
    const auto& high_filter = db_high[db_num - 1];

    for (int d = 0; d < depth; ++d) {
        for (int r = 0; r < rows; ++r) {
            dwt_1d_cuda(d_volume + (d * rows * cols + r * cols), cols, low_filter, high_filter);
        }
        for (int c = 0; c < cols; ++c) {
            std::vector<float> column(rows);
            for (int r = 0; r < rows; ++r) {
                column[r] = d_volume[d * rows * cols + r * cols + c];
            }
            float* d_column;
            cudaMalloc(&d_column, rows * sizeof(float));
            cudaMemcpy(d_column, column.data(), rows * sizeof(float), cudaMemcpyHostToDevice);
            dwt_1d_cuda(d_column, rows, low_filter, high_filter);
            cudaMemcpy(column.data(), d_column, rows * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_column);
            for (int r = 0; r < rows; ++r) {
                d_volume[d * rows * cols + r * cols + c] = column[r];
            }
        }
    }

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::vector<float> signal(depth);
            for (int d = 0; d < depth; ++d) {
                signal[d] = d_volume[d * rows * cols + r * cols + c];
            }
            float* d_signal;
            cudaMalloc(&d_signal, depth * sizeof(float));
            cudaMemcpy(d_signal, signal.data(), depth * sizeof(float), cudaMemcpyHostToDevice);
            dwt_1d_cuda(d_signal, depth, low_filter, high_filter);
            cudaMemcpy(signal.data(), d_signal, depth * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_signal);
            for (int d = 0; d < depth; ++d) {
                d_volume[d * rows * cols + r * cols + c] = signal[d];
            }
        }
    }
}

// Function to perform multi-level 3D DWT on a 3D volume using CUDA
void multi_level_cuda(float* d_volume, int depth, int rows, int cols, int db_num, int levels) {
    double t = jbutil::gettime();

    for (int i = 0; i < levels; i++) {
        dwt_3D_cuda(d_volume, depth, rows, cols, db_num);

        if (i != levels - 1) {
            depth = (depth + 1) / 2;
            rows = (rows + 1) / 2;
            cols = (cols + 1) / 2;
        }
    }

    t = jbutil::gettime() - t;
    std::cerr << "Multi-level DWT completed successfully with " 
              << levels << " levels and db_num " << db_num << std::endl;
    std::cerr << "Time taken: " << t << " seconds" << std::endl;
    assert(t >= 0 && "Time taken should be non-negative.");
}

int main(int argc, char *argv[]) {
    (void)argc;
    std::cerr << "Assignment 1: Synchronous Multi DWT on 3D CT Image" << std::endl;
    assert(argc == 5 && "Usage: ./assignment-1 <input.bin> <output.bin> <db_num> <levels>");

    std::string bin_in = argv[1];
    std::string bin_out = argv[2];
    int db_num = std::stoi(argv[3]);
    int levels = std::stoi(argv[4]);

    assert(db_num >= 1 && db_num <= 4 && "db_num must be between 1 and 4");

    int depth, rows, cols;
    float* d_volume = loadvolume(bin_in, depth, rows, cols);

    multi_level_cuda(d_volume, depth, rows, cols, db_num, levels);

    savevolume(d_volume, bin_out, depth, rows, cols);

    cudaFree(d_volume);

    return EXIT_SUCCESS;
}