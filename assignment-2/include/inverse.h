// inverse.h
#ifndef INVERSE_H
#define INVERSE_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <chrono>
#include "../include/loadbin.h"
#include "../include/savebin.h"


// Redefine the wavelet coefficients for the inverse
// Low Inverse Coefficients
const std::vector<std::vector<float>> db_low_inv = {
    {0.70710678f, 0.70710678f}, // db1
    {0.48296291f, 0.83651630f, 0.22414387f, -0.12940952f}, // db2
    {0.33267055f, 0.80689151f, 0.45987750f, -0.13501102f, -0.08544127f, 0.03522629f}, // db3
    {0.23037781f, 0.71484657f, 0.63088077f, -0.02798377f, -0.18703481f, 0.03084138f, 0.03288301f, -0.01059740f} // db4
};
// High Inverse Coefficients
const std::vector<std::vector<float>> db_high_inv = {
     {0.70710678f, -0.70710678f}, // db1
    {-0.12940952f, -0.22414387f, 0.83651630f, -0.48296291f}, // db2
    {0.03522629f, 0.08544127f, -0.13501102f, -0.45987750f, 0.80689151f, -0.33267055f}, // db3
    {-0.01059740f, -0.03288301f, 0.03084138f, 0.18703481f, -0.02798377f, -0.63088077f, 0.71484657f, -0.23037781f} // db4
};

void idwt_1d(std::vector<float>& signal, int db_num) {
    // Fetch inverse low-pass and high-pass filters
    const auto& low_filter = db_low_inv[db_num - 1];
    const auto& high_filter = db_high_inv[db_num - 1];
    int filter_length = low_filter.size();

    int half_size = signal.size() / 2;  // Half size of the signal (number of coefficients)
    std::vector<float> reconstructed(signal.size(), 0.0f);

    // Loop over approximation and detail coefficients
    for (int i = 0; i < half_size; ++i) {
        for (int j = 0; j < filter_length; ++j) {
            int pos = (2 * i + j) % signal.size();  // Handle boundary wrapping

            // Reconstruct using low-pass and high-pass filters
            reconstructed[pos] += signal[i] * low_filter[j];            // Approximation contribution
            reconstructed[pos] += signal[i + half_size] * high_filter[j]; // Detail contribution
        }
    }

    signal = reconstructed;  // Update signal with reconstructed values
}

// Function to perform 3D inverse DWT on a 3D volume
void idwt_3D(std::vector<std::vector<std::vector<float>>>& volume, int db_num) {
    int depth = volume.size();
    int rows = volume[0].size();
    int cols = volume[0][0].size();

    // Iterate over each row and column
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::vector<float> signal(depth);
            for (int d = 0; d < depth; ++d) {
                signal[d] = volume[d][i][j];
            }
            idwt_1d(signal, db_num);
            for (int d = 0; d < depth; ++d) {
                volume[d][i][j] = signal[d];
            }
        }
    }

    // Iterate over each depth level
    for (int d = 0; d < depth; ++d) {
        for (int j = 0; j < cols; ++j) {
            std::vector<float> column(rows);
            for (int i = 0; i < rows; ++i) {
                column[i] = volume[d][i][j];
            }
            idwt_1d(column, db_num);
            for (int i = 0; i < rows; ++i) {
                volume[d][i][j] = column[i];
            }
        }

        for (int i = 0; i < rows; ++i) {
            idwt_1d(volume[d][i], db_num);
        }
    }
}

// Function to perform multi-level 3D inverse DWT on a 3D volume
void inverse_multi_level (std::vector<std::vector<std::vector<float>>>& volume, int db_num, int levels) {
    // Start timer to measure time taken for DWT
    auto dwt_s = std::chrono::high_resolution_clock::now();

    // Get the shape of the volume
    int depth = volume.size();
    int rows = volume[0].size();
    int cols = volume[0][0].size();

    for (int i = levels - 1; i >= 0; --i) {
        // print the level going to be performed
        std::cerr << "Performing level " << i << " DWT" << std::endl;
        if (i != levels - 1) {
            depth *= 2;
            rows *= 2;
            cols *= 2;
        }

        // Resize the volume to the original dimensions
        volume.resize(depth);
        for (int d = 0; d < depth; ++d) {
            volume[d].resize(rows);
            for (int r = 0; r < rows; ++r) {
                volume[d][r].resize(cols);
            }
        }

        // Perform inverse 3D DWT
        idwt_3D(volume, db_num);
    }

    
    // Stop timer
    auto dwt_e= std::chrono::high_resolution_clock::now();
    // Calculate the duration
    std::chrono::duration<double> dwt_d = dwt_e - dwt_s;
    // Log a success message
    //std::cerr << "Multi-level DWT completed successfully with "             << levels << " levels and db_num " << db_num << std::endl;

    // If there's a condition you want to assert, do that separately
    assert(levels > 0 && "Levels should be greater than 0 after processing.");

    // Log the time taken for the DWT
    std::cerr << "Time taken (DWT): " << dwt_d.count() << " seconds" << std::endl;

    // Assert a condition if necessary
    assert(dwt_d.count() >= 0 && "Time taken should be non-negative.");
}

#endif // INVERSE_H